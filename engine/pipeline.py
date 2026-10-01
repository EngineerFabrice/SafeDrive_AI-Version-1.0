"""Real-time monitoring pipeline.

camera frame -> person detection -> driver ROI -> face detection
             -> feature extraction -> [optional impairment model] -> state

The impairment model is injected (see engine.impairment.create_impairment_model);
the pipeline never chooses a concrete model itself.

Runs in its own thread and publishes a ``MonitoringSnapshot`` after every
frame (or every wait timeout when the camera delivers nothing). It has no
dependency on Flask; web code only reads ``pipeline.snapshot()``.

Run standalone for a live preview:  python -m engine.pipeline --preview
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .camera import Camera, CameraConfig, FpsMeter, Frame
from .detectors.driver_roi import DriverROI, DriverROIConfig, DriverROISelector
from .detectors.face import FaceDetection, FaceDetector, FaceDetectorConfig
from .detectors.person import PersonDetection, PersonDetector, PersonDetectorConfig
from .features import FaceFeatures, FeatureExtractor, FeatureExtractorConfig
from .impairment import ImpairmentInput, ImpairmentModel, ImpairmentResult, ImpairmentStatus
from .state import (CameraStatus, FeatureStatus, ModelStatus, MonitoringSnapshot,
                    MonitoringState, MonitoringStatus, resolve_status)

log = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    person: PersonDetectorConfig = field(default_factory=PersonDetectorConfig)
    driver_roi: DriverROIConfig = field(default_factory=DriverROIConfig)
    face: FaceDetectorConfig = field(default_factory=FaceDetectorConfig)
    features: FeatureExtractorConfig = field(default_factory=FeatureExtractorConfig)
    frame_wait_timeout: float = 0.5  # seconds; also the state refresh period when no frames arrive


@dataclass(frozen=True)
class FrameAnalysis:
    persons: Tuple[PersonDetection, ...]
    driver: Optional[DriverROI]
    face: Optional[FaceDetection]
    features: Optional[FaceFeatures]  # None when no face was found
    processing_time: float  # ms, all stages including feature extraction
    impairment: Optional[ImpairmentResult] = None  # None: no model, or features unavailable


class MonitoringPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None, camera: Optional[Camera] = None,
                 person_detector: Optional[PersonDetector] = None,
                 roi_selector: Optional[DriverROISelector] = None,
                 face_detector: Optional[FaceDetector] = None,
                 feature_extractor: Optional[FeatureExtractor] = None,
                 state: Optional[MonitoringState] = None,
                 impairment_model: Optional[ImpairmentModel] = None):
        self.config = config or PipelineConfig()
        self.camera = camera or Camera(self.config.camera)
        self.person_detector = person_detector or PersonDetector(self.config.person)
        self.roi_selector = roi_selector or DriverROISelector(self.config.driver_roi)
        self.face_detector = face_detector or FaceDetector(self.config.face)
        # One YuNet instance serves both face detection and Haar-fallback landmarking.
        self.feature_extractor = feature_extractor or FeatureExtractor(
            self.config.features, landmark_detector=self.face_detector.landmark_detector)
        self.impairment_model = impairment_model
        self.state = state or MonitoringState()
        self.state.update(impairment_model=impairment_model.info.to_dict() if impairment_model else None)
        self._fps = FpsMeter()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._latest: Optional[Tuple[Frame, FrameAnalysis]] = None
        self._last_feature_error_log = 0.0
        self._frames_processed = 0

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self.is_running:
            return
        self._stop_event.clear()
        self._fps.reset()
        self._frames_processed = 0
        self.state.update(frames_processed=0)
        self.roi_selector.reset()
        self.feature_extractor.reset()
        self.camera.start()
        self._thread = threading.Thread(target=self._run, name="safedrive-pipeline", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None
        self.camera.stop()
        self._latest = None
        self.state.update(status=MonitoringStatus.STOPPED, camera_status=CameraStatus.STOPPED,
                          model_status=self.model_status, driver_detected=False,
                          face_detected=False, fps=0.0, camera_fps=0.0, frame_latency=None,
                          processing_time=None, driver_bbox=None, driver_confidence=None,
                          face_bbox=None, features_status=FeatureStatus.FEATURES_UNAVAILABLE,
                          features=None, feature_time=None, impairment=None,
                          message="Monitoring stopped")

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def snapshot(self) -> MonitoringSnapshot:
        return self.state.snapshot()

    def latest_result(self) -> Optional[Tuple[Frame, FrameAnalysis]]:
        """Last processed frame and its analysis (for previews / later phases)."""
        return self._latest

    @property
    def model_status(self) -> ModelStatus:
        statuses = (self.person_detector.status, self.face_detector.status)
        if ModelStatus.UNAVAILABLE in statuses:
            return ModelStatus.UNAVAILABLE
        if all(s == ModelStatus.READY for s in statuses):
            return ModelStatus.READY
        if ModelStatus.LOADING in statuses:
            return ModelStatus.LOADING
        return ModelStatus.NOT_LOADED

    # ------------------------------------------------------------ processing
    def process_frame(self, image: np.ndarray, captured_at: Optional[float] = None,
                      frame_id: Optional[int] = None) -> FrameAnalysis:
        """Run all stages on one frame. Requires loaded detection models.

        ``captured_at`` is the frame's perf_counter capture time, used for
        motion rates; it defaults to now.
        """
        t0 = time.perf_counter()
        captured_at = t0 if captured_at is None else captured_at
        persons = tuple(self.person_detector.detect(image))
        driver = self.roi_selector.select(persons, image.shape)
        face = self.face_detector.detect(image, driver.roi) if driver else None
        if face is not None:
            features = self._extract_features(image, face, captured_at, frame_id)
        else:
            features = None
            self.feature_extractor.reset()  # no motion measured across a lost face
        impairment = self._predict_impairment(features)
        return FrameAnalysis(persons=persons, driver=driver, face=face, features=features,
                             processing_time=(time.perf_counter() - t0) * 1000,
                             impairment=impairment)

    def _predict_impairment(self, features: Optional[FaceFeatures]) -> Optional[ImpairmentResult]:
        """Run the configured model on available features only; never raises."""
        if self.impairment_model is None or features is None or not features.valid:
            return None
        try:
            return self.impairment_model.predict(ImpairmentInput.from_features(features))
        except Exception as exc:  # predict() already guards model code; this guards the interface
            log.exception("Impairment model call failed")
            return ImpairmentResult(status=ImpairmentStatus.MODEL_ERROR, model=self.impairment_model.info,
                                    error=f"{type(exc).__name__}: {exc}")

    def _extract_features(self, image: np.ndarray, face: FaceDetection, captured_at: float,
                          frame_id: Optional[int]) -> FaceFeatures:
        timestamp = time.time() - (time.perf_counter() - captured_at)  # wall clock of the capture
        try:
            return self.feature_extractor.extract(image, face, captured_at, timestamp, frame_id)
        except Exception as exc:
            # A feature failure must not stop the camera engine; report it as unavailable.
            now = time.monotonic()
            if now - self._last_feature_error_log > 10:
                log.exception("Feature extraction failed")
                self._last_feature_error_log = now
            self.feature_extractor.reset()
            return FaceFeatures.unavailable(f"feature extraction error: {exc}", timestamp, frame_id)

    def _load_models(self) -> None:
        camera_status = self.camera.status
        self.state.update(status=resolve_status(True, camera_status, ModelStatus.LOADING,
                                                False, False, False),
                          camera_status=camera_status, model_status=ModelStatus.LOADING,
                          message="Loading detection models")
        self.face_detector.load()
        self.feature_extractor.load()  # failure only makes features unavailable
        self.person_detector.load()  # loads at most once per detector instance

    def _model_error(self) -> str:
        return "; ".join(e for e in (self.person_detector.error, self.face_detector.error) if e)

    def _run(self) -> None:
        try:
            self._load_models()
        except Exception:
            log.exception("Unexpected error while loading models")

        last_id = None
        while not self._stop_event.is_set():
            frame = self.camera.wait_for_frame(last_id, timeout=self.config.frame_wait_timeout)
            camera_status = self.camera.status
            model_status = self.model_status

            if frame is None or camera_status != CameraStatus.CONNECTED:
                self._publish(camera_status, model_status, None, None,
                              self.camera.last_error or "Waiting for camera frames")
                continue
            last_id = frame.frame_id

            if model_status != ModelStatus.READY:
                self._publish(camera_status, model_status, frame, None,
                              self._model_error() or "Detection model not ready")
                continue

            try:
                analysis = self.process_frame(frame.image, frame.captured_at, frame.frame_id)
            except Exception as exc:
                # A failing detector must surface as unavailable, never as a clean result.
                log.exception("Frame processing failed")
                self._publish(camera_status, ModelStatus.UNAVAILABLE, frame, None,
                              f"Detection error: {exc}")
                continue

            self._fps.tick()
            self._frames_processed += 1
            self._latest = (frame, analysis)
            self._publish(camera_status, model_status, frame, analysis, "")

    def _publish(self, camera_status: CameraStatus, model_status: ModelStatus,
                 frame: Optional[Frame], analysis: Optional[FrameAnalysis], message: str) -> None:
        driver = analysis.driver if analysis else None
        face = analysis.face if analysis else None
        features = analysis.features if analysis else None
        features_ok = features is not None and features.valid
        status = resolve_status(True, camera_status, model_status, driver is not None,
                                face is not None, features_ok)
        if features is not None and not features_ok and not message:
            message = features.reason
        latency = (time.perf_counter() - frame.captured_at) * 1000 if (frame and analysis) else None
        self.state.update(
            status=status,
            camera_status=camera_status,
            model_status=model_status,
            driver_detected=driver is not None,
            face_detected=face is not None,
            fps=round(self._fps.fps, 2),
            frames_processed=self._frames_processed,
            camera_fps=round(self.camera.fps, 2),
            frame_latency=round(latency, 1) if latency is not None else None,
            processing_time=round(analysis.processing_time, 1) if analysis else None,
            frame_id=frame.frame_id if frame else None,
            frame_size=(frame.image.shape[1], frame.image.shape[0]) if frame else None,
            driver_bbox=driver.person.bbox.as_tuple() if driver else None,
            driver_confidence=round(driver.person.confidence, 3) if driver else None,
            face_bbox=face.bbox.as_tuple() if face else None,
            features_status=(FeatureStatus.FEATURES_AVAILABLE if features_ok
                             else FeatureStatus.FEATURES_UNAVAILABLE),
            features=features.to_dict() if features is not None else None,
            feature_time=round(features.processing_time, 2) if features is not None else None,
            impairment=analysis.impairment.to_dict() if analysis and analysis.impairment else None,
            message=message,
        )


def draw_overlay(image: np.ndarray, analysis: Optional[FrameAnalysis],
                 snapshot: MonitoringSnapshot) -> np.ndarray:
    """Debug visualisation; returns an annotated copy of ``image``."""
    import cv2

    out = image.copy()
    if analysis:
        for p in analysis.persons:
            cv2.rectangle(out, p.bbox.as_tuple()[:2], p.bbox.as_tuple()[2:], (128, 128, 128), 1)
        if analysis.driver:
            r = analysis.driver.roi
            cv2.rectangle(out, (r.x1, r.y1), (r.x2, r.y2), (255, 160, 0), 2)
        if analysis.face:
            f = analysis.face.bbox
            cv2.rectangle(out, (f.x1, f.y1), (f.x2, f.y2), (0, 220, 0), 2)
    latency = f"{snapshot.frame_latency:.0f}ms" if snapshot.frame_latency is not None else "-"
    lines = [f"{snapshot.status.value}  fps {snapshot.fps:.1f}  latency {latency}"]
    feats = analysis.features if analysis else None
    if feats is not None and feats.head_pose.valid:
        hp, gz = feats.head_pose, feats.gaze
        gaze = f"{gz.horizontal:+.2f}" if gz.valid else "-"
        lines.append(f"pitch {hp.pitch:+.0f} yaw {hp.yaw:+.0f} roll {hp.roll:+.0f} gaze-h {gaze}")
    for i, text in enumerate(lines):
        y = 28 + 28 * i
        cv2.putText(out, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(out, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def _main() -> None:
    import argparse

    from .impairment import create_impairment_model

    parser = argparse.ArgumentParser(description="SafeDrive AI monitoring engine")
    parser.add_argument("--source", default="0", help="camera index or video path")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--preview", action="store_true", help="show annotated OpenCV window")
    parser.add_argument("--model-provider", default=None,
                        help="impairment model provider (default: $MODEL_PROVIDER, else none)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    source = int(args.source) if args.source.isdigit() else args.source
    pipeline = MonitoringPipeline(PipelineConfig(camera=CameraConfig(source=source),
                                                 person=PersonDetectorConfig(weights=args.weights)),
                                  impairment_model=create_impairment_model(args.model_provider))
    pipeline.start()
    try:
        if args.preview:
            import cv2
            while True:
                result = pipeline.latest_result()
                snap = pipeline.snapshot()
                if result:
                    cv2.imshow("SafeDrive AI - Phase 1", draw_overlay(result[0].image, result[1], snap))
                if cv2.waitKey(30) & 0xFF in (27, ord("q")):
                    break
            cv2.destroyAllWindows()
        else:
            while True:
                s = pipeline.snapshot()
                print(f"{s.status.value:<20} cam={s.camera_status.value:<11} model={s.model_status.value:<11} "
                      f"fps={s.fps:5.1f} latency={s.frame_latency} ms {s.message}")
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()


if __name__ == "__main__":
    _main()
