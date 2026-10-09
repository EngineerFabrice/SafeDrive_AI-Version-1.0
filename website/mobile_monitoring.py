# website/mobile_monitoring.py
"""Phone-camera monitoring for the mobile app.

The web dashboard's engine reads a camera attached to the server. A phone instead uploads single
JPEG frames from its front camera; each frame goes through the SAME engine code path as the live
camera thread (``MonitoringPipeline.analyze_frame``):

    person detection -> driver ROI -> face detection -> aligned crop + quality gate
    -> configured impairment model (MODEL_PROVIDER) -> temporal decision engine

Every driver has their own session (own driver-ROI tracking, motion features and temporal window);
the loaded models are shared and used under one lock. Frames are decoded in memory and discarded:
no image, crop, feature vector or probability is ever stored.

Differences from the live camera, both deliberate:
* Frames arrive over the network at ~1-3 FPS instead of ~10 FPS, so the temporal window spans up to
  PHONE_WINDOW_SECONDS instead of 3 s (same frame count and the same thresholds/hysteresis).
* The capture time is the server's receive time; client timestamps are never trusted.
"""
import logging
import os
import threading
import time

import numpy as np

log = logging.getLogger(__name__)
if os.environ.get("SAFEDRIVE_PHONE_DEBUG_LOG", "").strip() == "1":
    # One line per phone frame (user id, face found, quality score, counted, assessment); never image data.
    # Development only: assessments are sensitive, so this stays off unless explicitly enabled.
    log.setLevel(logging.DEBUG)

PHONE_WINDOW_SECONDS = 10.0
MIN_FRAME_INTERVAL_S = 0.25      # at most 4 frames per second per driver
SESSION_IDLE_S = 120             # a session without frames for this long is closed
MAX_SESSIONS = 20                # concurrent phone sessions on one development server
MAX_FRAME_BYTES = 900_000
MIN_SIDE, MAX_SIDE = 96, 4096    # accepted decoded image size; larger frames are downscaled
PROCESS_MAX_WIDTH = 960
ASSESSMENT_FRESH_S = 30          # an AI_TRIGGERED claim needs an assessment at most this old


class FrameRejected(Exception):
    def __init__(self, http_status, code, message):
        super().__init__(message)
        self.http_status, self.code, self.message = http_status, code, message


class _Components:
    """Detectors and impairment model shared by all phone sessions (loaded once, on first use)."""

    def __init__(self, person_detector=None, face_detector=None, impairment_model=None, model_set=False):
        self.lock = threading.Lock()
        self._person, self._face = person_detector, face_detector
        self._model, self._model_set = impairment_model, model_set

    def _build(self):
        from engine.detectors.face import FaceDetector
        from engine.detectors.person import PersonDetector, PersonDetectorConfig
        from engine.impairment import create_impairment_model
        if self._person is None:
            weights = os.environ.get("SAFEDRIVE_YOLO_WEIGHTS", "yolov8n.pt")
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if not os.path.isabs(weights) and os.path.isfile(os.path.join(root, weights)):
                weights = os.path.join(root, weights)
            self._person = PersonDetector(PersonDetectorConfig(weights=weights))
        if self._face is None:
            self._face = FaceDetector()
        if not self._model_set:
            self._model, self._model_set = create_impairment_model(), True

    def ready(self):
        """Load the detectors (once); returns (ok, error message). Call with ``lock`` held."""
        self._build()
        self._face.load()
        self._person.load()
        errors = [e for e in (self._person.error, self._face.error) if e]
        from engine.state import ModelStatus
        ok = self._person.status == ModelStatus.READY and self._face.status == ModelStatus.READY
        return ok, "; ".join(errors) or ("" if ok else "Detection models are not ready.")

    @property
    def model(self):
        self._build()
        return self._model

    def pipeline(self):
        from engine.decision import TemporalConfig
        from engine.pipeline import MonitoringPipeline, PipelineConfig
        self._build()
        return MonitoringPipeline(PipelineConfig(decision=TemporalConfig(max_window_seconds=PHONE_WINDOW_SECONDS)),
                                  person_detector=self._person, face_detector=self._face,
                                  impairment_model=self._model)


class PhoneSession:
    def __init__(self, user_id, components):
        self.user_id = user_id
        self.pipeline = components.pipeline()
        self.started_at = time.time()
        self.last_frame_at = 0.0          # monotonic
        self.frames = 0
        self.decision = None
        self.decision_at = None           # monotonic time of the latest assessment

    def to_dict(self):
        return {"active": True, "started_at": self.started_at, "frames_processed": self.frames,
                "assessment": self.decision.to_dict() if self.decision is not None else None}


_components = _Components()
_sessions = {}
_sessions_lock = threading.Lock()


def set_components(person_detector=None, face_detector=None, impairment_model=None):
    """Replace the shared detectors/model (tests) and close every session."""
    global _components
    with _sessions_lock:
        _components = _Components(person_detector, face_detector, impairment_model, model_set=True)
        _sessions.clear()


def reset_components():
    global _components
    with _sessions_lock:
        _components = _Components()
        _sessions.clear()


def _expire(now):
    for uid in [u for u, s in _sessions.items() if s.last_frame_at and now - s.last_frame_at > SESSION_IDLE_S]:
        _sessions.pop(uid, None)


def model_info():
    """The configured impairment model, as the web /monitoring/model endpoint describes it."""
    model = _components.model
    info = model.info.to_dict() if model is not None else None
    return {"enabled": info is not None, "available": bool(model is not None and model.available),
            "model": info, "development_only": bool(info and info.get("development_only")),
            "is_mock": bool(info and info.get("is_mock")),
            "unavailable_reason": model.unavailable_reason if model is not None and not model.available else ""}


def start(user_id):
    """Start (or restart) the driver's phone session with a fresh temporal window."""
    now = time.monotonic()
    with _sessions_lock:
        _expire(now)
        if user_id not in _sessions and len(_sessions) >= MAX_SESSIONS:
            raise FrameRejected(503, "BUSY", "Too many phone monitoring sessions are running. Try again shortly.")
        session = PhoneSession(user_id, _components)
        _sessions[user_id] = session
    return session.to_dict()


def stop(user_id):
    with _sessions_lock:
        return _sessions.pop(user_id, None) is not None


def status(user_id):
    with _sessions_lock:
        _expire(time.monotonic())
        session = _sessions.get(user_id)
        return session.to_dict() if session else {"active": False, "frames_processed": 0, "assessment": None}


def fresh_assessment(user_id):
    """The driver's latest phone assessment label if it is at most ASSESSMENT_FRESH_S old, else None."""
    with _sessions_lock:
        session = _sessions.get(user_id)
        if session is None or session.decision is None or session.decision_at is None:
            return None
        if time.monotonic() - session.decision_at > ASSESSMENT_FRESH_S:
            return None
        return session.decision.assessment.value


def decode_frame(data):
    """JPEG/PNG bytes -> BGR uint8 image (downscaled to PROCESS_MAX_WIDTH); raises FrameRejected."""
    import cv2
    if not data:
        raise FrameRejected(400, "NO_FRAME", "No camera frame was received.")
    if len(data) > MAX_FRAME_BYTES:
        raise FrameRejected(413, "FRAME_TOO_LARGE", "The camera frame is too large.")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3:
        raise FrameRejected(400, "INVALID_FRAME", "The camera frame is not a valid JPEG or PNG image.")
    h, w = image.shape[:2]
    if min(h, w) < MIN_SIDE or max(h, w) > MAX_SIDE:
        raise FrameRejected(400, "INVALID_FRAME", f"The camera frame must be between {MIN_SIDE} and {MAX_SIDE} pixels.")
    if w > PROCESS_MAX_WIDTH:
        scale = PROCESS_MAX_WIDTH / w
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return image


def analyze(user_id, data):
    """Run one uploaded frame through the engine for the driver's session; returns the result dict."""
    now = time.monotonic()
    with _sessions_lock:
        _expire(now)
        session = _sessions.get(user_id)
        components = _components
    if session is None:
        raise FrameRejected(409, "NOT_STARTED", "Start phone monitoring first.")
    if session.last_frame_at and now - session.last_frame_at < MIN_FRAME_INTERVAL_S:
        raise FrameRejected(429, "TOO_FAST", "Frames are being sent too quickly.")
    session.last_frame_at = now
    image = decode_frame(data)

    with components.lock:
        ok, error = components.ready()
        if not ok:
            raise FrameRejected(503, "DETECTION_UNAVAILABLE", f"Detection models are unavailable: {error}")
        try:
            analysis, decision = session.pipeline.analyze_frame(image, time.perf_counter(), session.frames + 1)
        except Exception as exc:     # a failing detector must never look like a clean result
            log.exception("Phone frame processing failed")
            raise FrameRejected(500, "PROCESSING_ERROR", f"The frame could not be analysed ({type(exc).__name__}).")
    session.frames += 1
    if decision is not None:
        session.decision, session.decision_at = decision, time.monotonic()

    model = components.model
    quality = analysis.face_quality
    min_quality = session.pipeline.decision.config.min_quality
    # Exactly the temporal engine's validity rule: a usable prediction AND quality >= min_quality.
    counted = bool(analysis.impairment is not None and analysis.impairment.valid and quality is not None
                   and quality.score >= min_quality)
    if model is None:
        message = "No impairment model is configured on the server (MODEL_PROVIDER=none): detection only."
    elif not model.available:
        message = f"The impairment model is unavailable: {model.unavailable_reason}"
    elif analysis.driver is None:
        message = "No driver detected. Hold the phone so your head and shoulders are in view."
    elif analysis.face is None:
        message = "No face detected. Face the camera."
    elif quality is not None and not quality.ok:
        message = (f"Face image rejected ({', '.join(quality.reasons)}). Improve lighting, hold the phone still "
                   "and closer. This frame was not scored.")
    elif not counted and quality is not None:
        message = (f"Face visible, but its quality score {quality.score:.2f} is below the {min_quality:.2f} the "
                   "assessment requires (usually lighting or blur). This frame was not counted.")
    else:
        message = ""
    log.debug("phone frame user=%s frame=%s face=%s quality=%s counted=%s assessment=%s", user_id, session.frames,
              analysis.face is not None, round(quality.score, 3) if quality else None, counted,
              session.decision.assessment.value if session.decision is not None else None)
    return {
        "frame": session.frames,
        "driver_detected": analysis.driver is not None,
        "face_detected": analysis.face is not None,
        "face_quality": quality.to_dict() if quality is not None else None,
        "counted": counted,
        "min_quality": min_quality,
        "impairment": analysis.impairment.to_dict() if analysis.impairment is not None else None,
        "assessment": session.decision.to_dict() if session.decision is not None else None,
        "processing_ms": round(analysis.processing_time, 1),
        "frame_size": [int(image.shape[1]), int(image.shape[0])],
        "message": message,
    }
