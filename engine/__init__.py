"""SafeDrive AI real-time computer-vision engine (Phase 1: camera + localisation).

Independent of Flask; the web layer only reads ``MonitoringPipeline.snapshot()``.
"""

from .camera import Camera, CameraConfig
from .pipeline import MonitoringPipeline, PipelineConfig
from .state import CameraStatus, ModelStatus, MonitoringSnapshot, MonitoringStatus

__all__ = [
    "Camera", "CameraConfig", "MonitoringPipeline", "PipelineConfig",
    "CameraStatus", "ModelStatus", "MonitoringSnapshot", "MonitoringStatus",
]
