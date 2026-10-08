"""Face quality check and face-crop preprocessing for the image classifier.

Shared by offline training (``training/prepare_dataset.py``) and live inference
(``engine/pipeline.py``) so the model always sees crops produced by exactly the
same geometry. A mismatch here would silently degrade a trained model.

align_face_crop
    Square crop centred on the face box, rotated so the eyes are level when
    landmarks are available (YuNet), padded by mirroring the border when the
    crop leaves the image (dataset images are tight face crops).

assess_face_quality
    Rejects faces that are too small, too dark/bright, too blurred or detected
    with low confidence, so the sobriety assessment never relies on a frame in
    which the face cannot be seen properly. The score (0..1) is passed to the
    temporal decision engine, which ignores low-quality frames.
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

from .detectors import BoundingBox
from .detectors.landmarks import FaceLandmarks

CROP_SIZE = 224          # MobileNetV3 / EfficientNet-B0 ImageNet input size
CROP_MARGIN = 0.05       # small context around the detector box; keeps background (a known shortcut) out


def align_face_crop(image: np.ndarray, bbox: BoundingBox, landmarks: Optional[FaceLandmarks] = None,
                    size: int = CROP_SIZE, margin: float = CROP_MARGIN) -> np.ndarray:
    """Return a ``size`` x ``size`` BGR face crop (uint8)."""
    if bbox.area == 0:
        raise ValueError("empty face box")
    cx, cy = bbox.center
    side = max(bbox.width, bbox.height) * (1.0 + 2 * margin)
    angle = 0.0
    if landmarks is not None:
        dx, dy = landmarks.eye_right - landmarks.eye_left
        angle = math.degrees(math.atan2(float(dy), float(dx)))   # rotate so the eye line is horizontal
    m = cv2.getRotationMatrix2D((float(cx), float(cy)), angle, size / side)
    m[0, 2] += size / 2.0 - cx
    m[1, 2] += size / 2.0 - cy
    return cv2.warpAffine(image, m, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


@dataclass
class FaceQualityConfig:
    min_face_pixels: int = 64           # shorter side of the detector box in the source frame
    min_detection_score: float = 0.6
    min_brightness: float = 40.0        # mean grey level of the crop, 0..255
    max_brightness: float = 220.0
    min_sharpness: float = 15.0         # variance of the Laplacian on the 224 px crop
    good_sharpness: float = 120.0       # sharpness at which the sharpness term saturates


@dataclass(frozen=True)
class FaceQuality:
    ok: bool
    score: float                        # 0..1, used as a weight by the temporal engine
    reasons: Tuple[str, ...] = ()
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "score": round(self.score, 3), "reasons": list(self.reasons),
                "metrics": {k: round(float(v), 2) for k, v in self.metrics.items()}}


def assess_face_quality(crop: np.ndarray, bbox: BoundingBox, detection_score: Optional[float] = None,
                        config: Optional[FaceQualityConfig] = None) -> FaceQuality:
    """Quality of an aligned face crop (BGR) whose detector box in the source frame was ``bbox``."""
    cfg = config or FaceQualityConfig()
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    face_px = float(min(bbox.width, bbox.height))
    score_det = 1.0 if detection_score is None else float(detection_score)

    reasons = []
    if face_px < cfg.min_face_pixels:
        reasons.append("face_too_small")
    if score_det < cfg.min_detection_score:
        reasons.append("low_detection_confidence")
    if brightness < cfg.min_brightness:
        reasons.append("too_dark")
    elif brightness > cfg.max_brightness:
        reasons.append("too_bright")
    if sharpness < cfg.min_sharpness:
        reasons.append("blurred")

    # Smooth 0..1 terms so the temporal engine can down-weight marginal frames.
    size_term = min(1.0, face_px / (2.0 * cfg.min_face_pixels))
    light_mid = (cfg.min_brightness + cfg.max_brightness) / 2.0
    light_term = max(0.0, 1.0 - abs(brightness - light_mid) / (light_mid - cfg.min_brightness + 1e-6) * 0.5)
    sharp_term = min(1.0, sharpness / cfg.good_sharpness)
    score = float(np.clip(size_term * light_term * sharp_term * min(1.0, score_det / 0.9), 0.0, 1.0))
    if reasons:
        score = min(score, 0.3)          # below the engine's default min_quality (0.5): never used
    return FaceQuality(ok=not reasons, score=score, reasons=tuple(reasons),
                       metrics={"face_pixels": face_px, "brightness": brightness, "sharpness": sharpness,
                                "detection_score": score_det})
