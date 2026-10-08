"""Pipeline -> face quality -> classifier -> temporal decision engine (fake camera, detectors and model)."""
import time

import numpy as np
import pytest

from engine.camera import Frame
from engine.decision import TemporalConfig, TemporalDecisionEngine
from engine.detectors import BoundingBox
from engine.detectors.face import FaceDetection
from engine.detectors.person import PersonDetection
from engine.impairment import ImpairmentModel, ModelInfo
from engine.pipeline import MonitoringPipeline
from engine.state import CameraStatus, ModelStatus


class FakeCamera:
    def __init__(self, image):
        self.image, self.n, self.fps, self.last_error = image, 0, 30.0, ""
        self.status = CameraStatus.STOPPED

    def start(self):
        self.status = CameraStatus.CONNECTED

    def stop(self):
        self.status = CameraStatus.STOPPED

    def wait_for_frame(self, after_id=None, timeout=1.0):
        time.sleep(0.01)
        self.n += 1
        return Frame(self.image, self.n, time.perf_counter())


class Person:
    status, error = ModelStatus.READY, ""

    def load(self):
        return True

    def detect(self, image):
        return [PersonDetection(BoundingBox(160, 40, 480, 480), 0.9)]


class Face:
    status, error, landmark_detector = ModelStatus.READY, "", None

    def __init__(self, box=BoundingBox(220, 100, 420, 340), score=0.95):
        self.box, self.score = box, score

    def load(self):
        return True

    def detect(self, image, roi):
        return FaceDetection(bbox=self.box, score=self.score, backend="yunet")


class FixedModel(ImpairmentModel):
    """Image model returning a fixed P(alcoholic); records what it was given."""
    def __init__(self, p, positive="alcoholic"):
        self.p, self.calls = p, []
        self._info = ModelInfo(name="fixed", version="1", provider="test", schema_version=1,
                               classes=("non_alcoholic", "alcoholic"), input_type="face_image",
                               positive_class=positive)

    @property
    def info(self):
        return self._info

    def _predict(self, face):
        self.calls.append(face.shape)
        return ("alcoholic" if self.p >= 0.5 else "non_alcoholic"), {"non_alcoholic": 1 - self.p, "alcoholic": self.p}


def textured():
    rng = np.random.default_rng(1)
    return np.clip(120 + rng.integers(-50, 50, (480, 640, 3)), 0, 255).astype(np.uint8)


def run(model, face=None, image=None, frames=12, timeout=5.0):
    p = MonitoringPipeline(camera=FakeCamera(textured() if image is None else image), person_detector=Person(),
                           face_detector=face or Face(), impairment_model=model,
                           decision_engine=TemporalDecisionEngine(TemporalConfig(max_window_seconds=30)))
    p.start()
    end = time.monotonic() + timeout
    while p.snapshot().frames_processed < frames and time.monotonic() < end:
        time.sleep(0.02)
    snap = p.snapshot()
    p.stop()
    return p, snap


def test_consistent_high_probability_reaches_potentially_not_sober():
    model = FixedModel(0.95)
    _, snap = run(model)
    assert snap.assessment["assessment"] == "POTENTIALLY_NOT_SOBER"
    assert snap.assessment["confidence"] >= 0.9 and snap.face_quality["ok"]
    assert model.calls and model.calls[0] == (224, 224, 3)          # aligned crop, model input size


def test_consistent_low_probability_reaches_sober():
    _, snap = run(FixedModel(0.05))
    assert snap.assessment["assessment"] == "SOBER"


def test_poor_face_quality_never_reaches_the_model():
    model = FixedModel(0.95)
    _, snap = run(model, face=Face(score=0.2), frames=16)             # > window_size (15) invalid frames
    assert model.calls == []
    assert snap.assessment["assessment"] == "UNCERTAIN"              # window full of invalid frames
    assert not snap.face_quality["ok"]


def test_model_without_positive_class_never_drives_safety_decision():
    _, snap = run(FixedModel(0.99, positive=None))
    assert snap.assessment is None


def test_stop_clears_and_start_resets_assessment():
    p, snap = run(FixedModel(0.95))
    assert snap.assessment is not None
    after_stop = p.snapshot()
    assert after_stop.assessment is None and after_stop.face_quality is None
    assert p.decision.assessment.value == "POTENTIALLY_NOT_SOBER"   # engine keeps state until restart
    p.camera.wait_for_frame = lambda after_id=None, timeout=1.0: time.sleep(0.01)   # no frames after restart
    p.start()                                                        # a new session starts a new assessment
    try:
        time.sleep(0.1)
        assert p.decision.assessment.value == "ASSESSING" and p.decision.last_decision is None
        assert p.snapshot().assessment is None
    finally:
        p.stop()
