"""Thread-safe monitoring state for the SafeDrive AI computer-vision engine.

The state reports whether the system is *able* to monitor the driver
(camera, models, driver and face localisation, feature availability). It
never reports an impairment verdict, and in particular never reports "SAFE":
no impairment model exists yet, so the absence of a detection must never be
read as a safe result.
"""

import threading
import time
from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Optional, Tuple


class CameraStatus(str, Enum):
    STOPPED = "STOPPED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    UNAVAILABLE = "UNAVAILABLE"


class ModelStatus(str, Enum):
    NOT_LOADED = "NOT_LOADED"
    LOADING = "LOADING"
    READY = "READY"
    UNAVAILABLE = "UNAVAILABLE"


class FeatureStatus(str, Enum):
    FEATURES_AVAILABLE = "FEATURES_AVAILABLE"
    FEATURES_UNAVAILABLE = "FEATURES_UNAVAILABLE"


class MonitoringStatus(str, Enum):
    STOPPED = "STOPPED"
    CAMERA_UNAVAILABLE = "CAMERA_UNAVAILABLE"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    NO_DRIVER = "NO_DRIVER"
    FACE_NOT_DETECTED = "FACE_NOT_DETECTED"
    FEATURES_UNAVAILABLE = "FEATURES_UNAVAILABLE"
    MONITORING_READY = "MONITORING_READY"   # driver, face and features available


def resolve_status(running: bool, camera_status: CameraStatus, model_status: ModelStatus,
                   driver_detected: bool, face_detected: bool,
                   features_available: bool) -> MonitoringStatus:
    """Map component conditions to a single monitoring status.

    Checks run from the most fundamental failure to the least, so a missing
    camera or model is always reported as such and is never masked by a
    later stage.
    """
    if not running:
        return MonitoringStatus.STOPPED
    if camera_status != CameraStatus.CONNECTED:
        return MonitoringStatus.CAMERA_UNAVAILABLE
    if model_status != ModelStatus.READY:
        return MonitoringStatus.MODEL_UNAVAILABLE
    if not driver_detected:
        return MonitoringStatus.NO_DRIVER
    if not face_detected:
        return MonitoringStatus.FACE_NOT_DETECTED
    if not features_available:
        return MonitoringStatus.FEATURES_UNAVAILABLE
    return MonitoringStatus.MONITORING_READY


BBox = Tuple[int, int, int, int]


@dataclass(frozen=True)
class MonitoringSnapshot:
    """Immutable view of the engine at one point in time."""
    status: MonitoringStatus = MonitoringStatus.STOPPED
    camera_status: CameraStatus = CameraStatus.STOPPED
    model_status: ModelStatus = ModelStatus.NOT_LOADED
    driver_detected: bool = False
    face_detected: bool = False
    fps: float = 0.0                       # frames processed by the pipeline per second
    camera_fps: float = 0.0                # frames delivered by the camera per second
    frame_latency: Optional[float] = None  # ms from frame capture to state update
    processing_time: Optional[float] = None  # ms spent in detection for the last frame
    timestamp: float = 0.0                 # wall-clock time (epoch seconds) of this snapshot
    frame_id: Optional[int] = None
    frames_processed: int = 0              # frames fully analysed since the last start()
    frame_size: Optional[Tuple[int, int]] = None  # (width, height)
    driver_bbox: Optional[BBox] = None
    driver_confidence: Optional[float] = None
    face_bbox: Optional[BBox] = None
    features_status: FeatureStatus = FeatureStatus.FEATURES_UNAVAILABLE
    features: Optional[dict] = None        # JSON-safe FaceFeatures.to_dict() of the last frame
    feature_time: Optional[float] = None   # ms spent extracting features for the last frame
    # Phase 3 model output for the last frame (ImpairmentResult.to_dict()). Informational
    # only: it never changes `status`, and a mock model is flagged development_only.
    impairment: Optional[dict] = None
    impairment_model: Optional[dict] = None  # ModelInfo of the configured model; None = disabled
    # Temporal decision (TemporalDecision.to_dict()): SOBER / UNCERTAIN / POTENTIALLY_NOT_SOBER, or
    # ASSESSING. None when no alcohol classifier is configured. Separate from `status`, which
    # only describes whether the system is able to monitor.
    assessment: Optional[dict] = None
    face_quality: Optional[dict] = None      # FaceQuality.to_dict() of the last face
    message: str = ""

    def to_dict(self) -> dict:
        data = asdict(self)
        for key in ("status", "camera_status", "model_status", "features_status"):
            data[key] = getattr(self, key).value
        return data


class MonitoringState:
    """Holds the latest snapshot; safe to read from any thread."""

    def __init__(self):
        self._lock = threading.Lock()
        self._snapshot = MonitoringSnapshot(timestamp=time.time())

    def update(self, **fields) -> MonitoringSnapshot:
        with self._lock:
            self._snapshot = replace(self._snapshot, timestamp=time.time(), **fields)
            return self._snapshot

    def snapshot(self) -> MonitoringSnapshot:
        with self._lock:
            return self._snapshot
