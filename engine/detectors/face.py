"""Locate the driver's face inside the driver ROI.

Primary backend: YuNet (``cv2.FaceDetectorYN``, see landmarks.py), a small CNN
that also returns five landmarks. In live testing it found a backlit
driver's face that the Haar cascade missed in nearly every frame; Haar
cascades are known to degrade with strong backlight and darker skin tones.

Fallback backend: OpenCV's bundled frontal-face Haar cascade. It is used only
when the YuNet model file is missing, so the engine still localises faces,
less robustly.

This module only localises the face. Feature extraction is a separate stage
that consumes the returned box (and landmarks, when the backend provides them).
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from . import BoundingBox
from .landmarks import FaceLandmarks, LandmarkDetector, LandmarkDetectorConfig
from ..state import ModelStatus

log = logging.getLogger(__name__)


@dataclass
class FaceDetectorConfig:
    backend: str = "auto"                # "auto" (YuNet, else Haar), "yunet" or "haar"
    yunet: LandmarkDetectorConfig = field(default_factory=LandmarkDetectorConfig)
    cascade_path: Optional[str] = None   # defaults to OpenCV's haarcascade_frontalface_default.xml
    scale_factor: float = 1.1
    min_neighbors: int = 5
    min_face_fraction: float = 0.08      # min face width relative to ROI width
    min_face_pixels: int = 24            # Haar only: min face side in the downscaled ROI
    max_search_width: int = 320          # Haar only: ROI is downscaled to this width for speed
    upper_region_fraction: float = 0.75  # face centre must lie in the upper part of the person ROI


@dataclass(frozen=True)
class FaceDetection:
    bbox: BoundingBox                      # full-frame coordinates
    score: float                           # YuNet confidence, or Haar level weight (relative)
    landmarks: Optional[FaceLandmarks] = None
    backend: str = ""


class FaceDetector:
    def __init__(self, config: Optional[FaceDetectorConfig] = None,
                 landmark_detector: Optional[LandmarkDetector] = None):
        self.config = config or FaceDetectorConfig()
        self.landmark_detector = landmark_detector or LandmarkDetector(self.config.yunet)
        self._cascade: Optional[cv2.CascadeClassifier] = None
        self._backend = ""
        self._status = ModelStatus.NOT_LOADED
        self._error = ""
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

    @property
    def status(self) -> ModelStatus:
        return self._status

    @property
    def error(self) -> str:
        return self._error

    @property
    def backend(self) -> str:
        return self._backend

    def load(self) -> bool:
        if self._status in (ModelStatus.READY, ModelStatus.UNAVAILABLE):
            return self._status == ModelStatus.READY
        backend = self.config.backend
        if backend in ("auto", "yunet") and self.landmark_detector.load():
            self._backend = "yunet"
        elif backend in ("auto", "haar") and self._load_haar():
            self._backend = "haar"
            if backend == "auto":
                log.warning("YuNet unavailable (%s); falling back to Haar face detection",
                            self.landmark_detector.error)
        else:
            self._status = ModelStatus.UNAVAILABLE
            self._error = self._error or self.landmark_detector.error or f"No face backend for {backend!r}"
            log.error("Face detector unavailable: %s", self._error)
            return False
        self._status = ModelStatus.READY
        return True

    def _load_haar(self) -> bool:
        path = self.config.cascade_path or os.path.join(
            cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        cascade = cv2.CascadeClassifier(path)
        if cascade.empty():
            self._error = f"Face cascade could not be loaded: {path}"
            return False
        self._cascade = cascade
        return True

    def detect(self, frame: np.ndarray, roi: BoundingBox) -> Optional[FaceDetection]:
        """Return the most plausible driver face inside ``roi``, or None."""
        if roi.area == 0:
            return None
        if self._backend == "yunet":
            return self._detect_yunet(frame, roi)
        if self._backend == "haar":
            return self._detect_haar(frame, roi)
        return None

    def _plausible(self, box: BoundingBox, roi: BoundingBox) -> bool:
        if box.width < roi.width * self.config.min_face_fraction:
            return False
        # A "face" low in the body box is almost certainly a false positive.
        return box.center[1] - roi.y1 <= roi.height * self.config.upper_region_fraction

    def _detect_yunet(self, frame: np.ndarray, roi: BoundingBox) -> Optional[FaceDetection]:
        faces = [f for f in self.landmark_detector.detect_all(frame, roi) if self._plausible(f.bbox, roi)]
        if not faces:
            return None
        best = max(faces, key=lambda f: f.bbox.area)
        return FaceDetection(bbox=best.bbox, score=best.score, landmarks=best, backend="yunet")

    def _detect_haar(self, frame: np.ndarray, roi: BoundingBox) -> Optional[FaceDetection]:
        cfg = self.config
        crop = frame[roi.y1:roi.y2, roi.x1:roi.x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        scale = min(1.0, cfg.max_search_width / gray.shape[1])
        if scale < 1.0:
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        # Local contrast equalisation copes with in-car backlight (bright windows behind the driver).
        gray = self._clahe.apply(gray)

        min_side = max(cfg.min_face_pixels, int(gray.shape[1] * cfg.min_face_fraction))
        boxes, _, weights = self._cascade.detectMultiScale3(
            gray, scaleFactor=cfg.scale_factor, minNeighbors=cfg.min_neighbors,
            minSize=(min_side, min_side), outputRejectLevels=True)

        inv = 1.0 / scale
        best = None
        for (x, y, fw, fh), weight in zip(boxes, np.ravel(weights)):
            box = BoundingBox(roi.x1 + int(x * inv), roi.y1 + int(y * inv),
                              roi.x1 + int((x + fw) * inv), roi.y1 + int((y + fh) * inv))
            if self._plausible(box, roi) and (best is None or box.area > best[0].area):
                best = (box, float(weight))
        if best is None:
            return None
        return FaceDetection(bbox=best[0], score=best[1], backend="haar")
