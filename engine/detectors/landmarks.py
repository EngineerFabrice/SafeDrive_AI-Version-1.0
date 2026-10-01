"""YuNet face detection with five landmarks (eyes, nose tip, mouth corners).

Uses OpenCV's YuNet model (``cv2.FaceDetectorYN``, OpenCV Zoo, MIT licence; see models/README.md),
which needs no extra Python dependency, only a ~230 KB ONNX file. The face
detector uses it as its primary backend inside the driver ROI, and the
feature extractor uses it to add landmarks to a face box from the Haar
fallback.
"""

import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from . import BoundingBox
from ..state import ModelStatus

log = logging.getLogger(__name__)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_YUNET_PATH = os.path.join(_REPO_ROOT, "models", "face_detection_yunet_2023mar.onnx")


@dataclass
class LandmarkDetectorConfig:
    model_path: str = DEFAULT_YUNET_PATH
    score_threshold: float = 0.6
    crop_margin: float = 0.5      # context added around the face box (YuNet needs some)
    max_input_size: int = 320     # crop is downscaled to at most this side length


@dataclass(frozen=True)
class FaceLandmarks:
    """Landmarks in full-frame pixel coordinates.

    Eyes and mouth corners are ordered by image x, so ``*_left`` means the
    point on the left of the image (the subject's right side).
    """
    eye_left: np.ndarray
    eye_right: np.ndarray
    nose: np.ndarray
    mouth_left: np.ndarray
    mouth_right: np.ndarray
    score: float
    bbox: BoundingBox

    @property
    def points(self) -> np.ndarray:
        """(5, 2) float array in the order above."""
        return np.stack([self.eye_left, self.eye_right, self.nose, self.mouth_left, self.mouth_right])

    @property
    def interocular(self) -> float:
        return float(np.linalg.norm(self.eye_right - self.eye_left))


class LandmarkDetector:
    def __init__(self, config: Optional[LandmarkDetectorConfig] = None):
        self.config = config or LandmarkDetectorConfig()
        self._net = None
        self._status = ModelStatus.NOT_LOADED
        self._error = ""

    @property
    def status(self) -> ModelStatus:
        return self._status

    @property
    def error(self) -> str:
        return self._error

    def load(self) -> bool:
        if self._status in (ModelStatus.READY, ModelStatus.UNAVAILABLE):
            return self._status == ModelStatus.READY
        path = self.config.model_path
        try:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"YuNet model not found: {path}")
            self._net = cv2.FaceDetectorYN.create(path, "", (self.config.max_input_size,) * 2,
                                                  self.config.score_threshold)
            self._status = ModelStatus.READY
        except Exception as exc:
            self._status = ModelStatus.UNAVAILABLE
            self._error = f"{type(exc).__name__}: {exc}"
            log.error("Landmark detector unavailable: %s", self._error)
        return self._status == ModelStatus.READY

    def detect(self, frame: np.ndarray, face: BoundingBox) -> Optional[FaceLandmarks]:
        """Landmarks for the face overlapping ``face`` most; None if YuNet finds none there."""
        h, w = frame.shape[:2]
        faces = self.detect_all(frame, face.expand(self.config.crop_margin).clip(w, h))
        best = max(faces, key=lambda f: f.bbox.iou(face), default=None)
        return best if best is not None and best.bbox.iou(face) > 0 else None

    def detect_all(self, frame: np.ndarray, region: BoundingBox) -> List[FaceLandmarks]:
        """All faces YuNet finds inside ``region`` (full-frame coordinates)."""
        if self._net is None or region.area == 0:
            return []
        crop = frame[region.y1:region.y2, region.x1:region.x2]
        scale = min(1.0, self.config.max_input_size / max(crop.shape[:2]))
        if scale < 1.0:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        self._net.setInputSize((crop.shape[1], crop.shape[0]))
        _, faces = self._net.detect(crop)
        if faces is None:
            return []

        offset = np.array([region.x1, region.y1], np.float32)
        results = []
        for row in faces:
            x, y, bw, bh = row[:4] / scale
            box = BoundingBox(int(x + offset[0]), int(y + offset[1]),
                              int(x + bw + offset[0]), int(y + bh + offset[1])).clip(*frame.shape[1::-1])
            pts = row[4:14].reshape(5, 2) / scale + offset
            eyes = sorted(pts[0:2], key=lambda p: p[0])
            mouth = sorted(pts[3:5], key=lambda p: p[0])
            results.append(FaceLandmarks(eye_left=eyes[0], eye_right=eyes[1], nose=pts[2],
                                         mouth_left=mouth[0], mouth_right=mouth[1],
                                         score=float(row[14]), bbox=box))
        return results
