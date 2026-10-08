"""Face crop alignment and face-quality gate (shared by training and live inference)."""
import cv2
import numpy as np
import pytest

from engine.detectors import BoundingBox
from engine.detectors.landmarks import FaceLandmarks
from engine.preprocessing import CROP_SIZE, FaceQualityConfig, align_face_crop, assess_face_quality


def textured(h=480, w=640, value=128):
    rng = np.random.default_rng(0)
    img = np.full((h, w, 3), value, np.uint8)
    noise = rng.integers(-40, 40, (h, w, 1))
    return np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)


def landmarks(eye_l, eye_r, box):
    pts = lambda *p: np.array(p, np.float32)   # noqa: E731
    return FaceLandmarks(eye_left=pts(*eye_l), eye_right=pts(*eye_r), nose=pts(320, 260),
                         mouth_left=pts(290, 300), mouth_right=pts(350, 300), score=0.9, bbox=box)


def test_crop_has_model_input_size_and_handles_borders():
    img = textured()
    for box in (BoundingBox(250, 150, 390, 330), BoundingBox(0, 0, 200, 220), BoundingBox(560, 400, 640, 480)):
        crop = align_face_crop(img, box)
        assert crop.shape == (CROP_SIZE, CROP_SIZE, 3) and crop.dtype == np.uint8


def test_crop_levels_a_tilted_eye_line():
    """A marker on the right eye of a tilted face must end up level with the left eye after alignment."""
    img = np.zeros((480, 640, 3), np.uint8)
    box = BoundingBox(240, 140, 400, 340)
    eye_l, eye_r = (280, 220), (360, 250)                    # eye line tilted ~20 degrees
    cv2.circle(img, eye_l, 6, (0, 0, 255), -1)
    cv2.circle(img, eye_r, 6, (0, 255, 0), -1)
    crop = align_face_crop(img, box, landmarks(eye_l, eye_r, box))
    red = np.argwhere((crop[..., 2] > 150) & (crop[..., 1] < 100))
    green = np.argwhere((crop[..., 1] > 150) & (crop[..., 2] < 100))
    assert abs(red[:, 0].mean() - green[:, 0].mean()) < 3   # same row (y) after rotation


def test_empty_box_rejected():
    with pytest.raises(ValueError):
        align_face_crop(textured(), BoundingBox(10, 10, 10, 10))


def test_good_face_passes_quality():
    q = assess_face_quality(align_face_crop(textured(), BoundingBox(200, 100, 440, 380)),
                            BoundingBox(200, 100, 440, 380), detection_score=0.95)
    assert q.ok and q.score >= 0.5 and q.reasons == ()


@pytest.mark.parametrize("make, box, score, reason", [
    (lambda: textured(value=10), BoundingBox(200, 100, 440, 380), 0.9, "too_dark"),
    (lambda: np.full((480, 640, 3), 245, np.uint8), BoundingBox(200, 100, 440, 380), 0.9, "too_bright"),
    (lambda: cv2.GaussianBlur(textured(), (0, 0), 12), BoundingBox(200, 100, 440, 380), 0.9, "blurred"),
    (textured, BoundingBox(300, 200, 330, 230), 0.9, "face_too_small"),
    (textured, BoundingBox(200, 100, 440, 380), 0.3, "low_detection_confidence"),
])
def test_poor_faces_rejected_and_never_usable_by_temporal_engine(make, box, score, reason):
    q = assess_face_quality(align_face_crop(make(), box), box, score)
    assert not q.ok and reason in q.reasons
    assert q.score < 0.5            # below TemporalConfig.min_quality: such frames never count


def test_quality_thresholds_configurable():
    box = BoundingBox(300, 200, 360, 260)
    crop = align_face_crop(textured(), box)
    assert not assess_face_quality(crop, box, 0.9).ok
    assert assess_face_quality(crop, box, 0.9, FaceQualityConfig(min_face_pixels=40)).ok
