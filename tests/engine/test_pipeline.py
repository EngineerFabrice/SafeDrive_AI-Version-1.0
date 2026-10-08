"""Pipeline orchestration with fake camera/detectors, plus a real face check when files exist."""
import glob
import os
import time
from types import SimpleNamespace

import numpy as np
import pytest

from engine.camera import Frame
from engine.detectors import BoundingBox
from engine.detectors.face import FaceDetector, FaceDetectorConfig
from engine.detectors.landmarks import DEFAULT_YUNET_PATH, LandmarkDetectorConfig
from engine.detectors.person import PersonDetection
from engine.features import FEATURE_NAMES
from engine.impairment import ImpairmentStatus, ModelInfo
from engine.pipeline import MonitoringPipeline
from engine.state import CameraStatus, ModelStatus, MonitoringStatus

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FakeCamera:
    def __init__(self):
        self.running, self.n, self.fps, self.last_error = False, 0, 30.0, ""
        self.status = CameraStatus.STOPPED

    def start(self):
        self.running, self.status = True, CameraStatus.CONNECTED

    def stop(self):
        self.running, self.status = False, CameraStatus.STOPPED

    def wait_for_frame(self, after_id=None, timeout=1.0):
        time.sleep(0.01)
        self.n += 1
        return Frame(np.zeros((480, 640, 3), np.uint8), self.n, time.perf_counter())


class FakePersonDetector:
    status, error = ModelStatus.READY, ""

    def __init__(self, persons=()):
        self.persons = list(persons)

    def load(self):
        return True

    def detect(self, image):
        return self.persons


class NoFaceDetector:
    status, error, landmark_detector = ModelStatus.READY, "", None

    def load(self):
        return True

    def detect(self, image, roi):
        return None


def wait_status(pipeline, wanted, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pipeline.snapshot().status == wanted:
            return True
        time.sleep(0.02)
    return False


def make_pipeline(persons=()):
    return MonitoringPipeline(camera=FakeCamera(), person_detector=FakePersonDetector(persons),
                              face_detector=NoFaceDetector())


def test_pipeline_reports_no_driver_then_stops_cleanly():
    p = make_pipeline()
    p.start()
    try:
        assert wait_status(p, MonitoringStatus.NO_DRIVER)
        assert p.snapshot().frames_processed > 0
    finally:
        p.stop()
    snap = p.snapshot()
    assert snap.status == MonitoringStatus.STOPPED and not p.is_running and snap.impairment is None


def test_pipeline_reports_face_not_detected_for_a_driver():
    p = make_pipeline([PersonDetection(BoundingBox(200, 50, 440, 480), 0.9)])
    p.start()
    try:
        assert wait_status(p, MonitoringStatus.FACE_NOT_DETECTED)
        assert p.snapshot().driver_detected
    finally:
        p.stop()


def test_model_failure_is_isolated():
    class Exploding:
        info = ModelInfo(name="x", version="1", provider="test", schema_version=1, classes=())

        def predict(self, inp):
            raise RuntimeError("model crashed")

    p = make_pipeline()
    p.impairment_model = Exploding()
    features = SimpleNamespace(valid=True, to_vector=lambda: np.zeros(len(FEATURE_NAMES), np.float32),
                               timestamp=0.0, frame_id=1)
    result = p._predict_impairment(features)
    assert result.status == ImpairmentStatus.MODEL_ERROR and "model crashed" in result.error


def test_face_detector_falls_back_to_haar_when_yunet_missing():
    det = FaceDetector(FaceDetectorConfig(yunet=LandmarkDetectorConfig(model_path="missing.onnx")))
    assert det.load() and det.backend == "haar"
    assert det.detect(np.zeros((240, 320, 3), np.uint8), BoundingBox(0, 0, 320, 240)) is None


SAMPLES = sorted(glob.glob(os.path.join(ROOT, "AlcoholDetectionDataset", "*", "*.png")))


@pytest.mark.model_files
@pytest.mark.skipif(not (os.path.isfile(DEFAULT_YUNET_PATH) and SAMPLES), reason="YuNet model or sample image missing")
def test_real_face_detection_and_feature_vector():
    """Detector/feature regression check on one bundled frame (not used as an alcohol label)."""
    import cv2

    image = cv2.imread(SAMPLES[0])
    det = FaceDetector()
    assert det.load() and det.backend == "yunet"
    face = det.detect(image, BoundingBox(0, 0, image.shape[1], image.shape[0]))
    assert face is not None and face.landmarks is not None

    p = MonitoringPipeline(camera=FakeCamera(), person_detector=FakePersonDetector(), face_detector=det)
    feats = p.feature_extractor.extract(image, face, time.perf_counter())
    vector = feats.to_vector()
    assert vector.shape == (len(FEATURE_NAMES),)
    assert feats.head_pose.valid and np.isnan(vector[-9:]).all()   # no motion on a single frame
