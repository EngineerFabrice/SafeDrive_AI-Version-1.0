"""Person detection with a COCO-pretrained YOLO model (YOLOv8n by default).

Reuses the YOLOv8n choice of the original ``website/yolo_detector.py`` but,
unlike it, keeps only the ``person`` class, loads the model once, and reports
missing or unusable weights as an explicit unavailable state instead of
raising at import time.
"""

import logging
import os
import threading
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from . import BoundingBox
from ..state import ModelStatus

log = logging.getLogger(__name__)

# Official ultralytics asset names; only these can be auto-downloaded.
_DOWNLOADABLE = {"yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolo11n.pt", "yolo11s.pt"}


class ModelUnavailableError(RuntimeError):
    pass


@dataclass
class PersonDetectorConfig:
    weights: str = "yolov8n.pt"
    confidence: float = 0.4
    image_size: int = 640
    device: Optional[str] = None    # None lets ultralytics choose (CPU here)
    allow_download: bool = True     # let ultralytics fetch official weights if missing


@dataclass(frozen=True)
class PersonDetection:
    bbox: BoundingBox
    confidence: float


class PersonDetector:
    def __init__(self, config: Optional[PersonDetectorConfig] = None):
        self.config = config or PersonDetectorConfig()
        self._model = None
        self._person_class: Optional[int] = None
        self._status = ModelStatus.NOT_LOADED
        self._error = ""
        self._lock = threading.Lock()
        self.load_count = 0  # number of times weights were actually loaded

    @property
    def status(self) -> ModelStatus:
        return self._status

    @property
    def error(self) -> str:
        return self._error

    def load(self) -> bool:
        """Load the model once. Safe to call repeatedly; returns True when ready."""
        with self._lock:
            if self._status == ModelStatus.READY:
                return True
            if self._status == ModelStatus.UNAVAILABLE:
                return False
            self._status = ModelStatus.LOADING
            try:
                self._load_locked()
                self._status = ModelStatus.READY
                log.info("Person detector ready (%s)", self.config.weights)
                return True
            except Exception as exc:  # any failure leaves the detector explicitly unavailable
                self._model = None
                self._status = ModelStatus.UNAVAILABLE
                self._error = f"{type(exc).__name__}: {exc}"
                log.error("Person detector unavailable: %s", self._error)
                return False

    def _load_locked(self) -> None:
        weights = self.config.weights
        if not os.path.isfile(weights):
            downloadable = os.path.basename(weights) == weights and weights in _DOWNLOADABLE
            if not (self.config.allow_download and downloadable):
                raise ModelUnavailableError(f"YOLO weights not found: {weights}")

        from ultralytics import YOLO  # heavy import; deferred so the engine imports without it

        model = YOLO(weights)
        names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
        person_ids = [i for i, n in names.items() if str(n).lower() == "person"]
        if not person_ids:
            raise ModelUnavailableError(f"{weights} has no 'person' class")
        self._person_class = person_ids[0]

        # Warm-up so the first live frame is not slowed by lazy initialisation.
        model.predict(np.zeros((self.config.image_size, self.config.image_size, 3), np.uint8),
                      imgsz=self.config.image_size, device=self.config.device, verbose=False)
        self._model = model
        self.load_count += 1

    def detect(self, frame: np.ndarray) -> List[PersonDetection]:
        """Return person detections only, highest confidence first."""
        if self._status != ModelStatus.READY or self._model is None:
            raise ModelUnavailableError(self._error or "Person detector not loaded")

        results = self._model.predict(
            frame, classes=[self._person_class], conf=self.config.confidence,
            imgsz=self.config.image_size, device=self.config.device, verbose=False)

        h, w = frame.shape[:2]
        detections = []
        for result in results:
            boxes = result.boxes
            for xyxy, conf, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
                if int(cls) != self._person_class:
                    continue
                bbox = BoundingBox(*(int(round(v)) for v in xyxy)).clip(w, h)
                if bbox.area > 0:
                    detections.append(PersonDetection(bbox=bbox, confidence=float(conf)))
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
