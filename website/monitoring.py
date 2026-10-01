# website/monitoring.py
"""Thin Flask access to the SafeDrive CV engine.

The engine captures and processes frames in its own threads; these endpoints
only start/stop it and read its latest state. No frames travel over HTTP.

Environment variables:
  SAFEDRIVE_CAMERA_SOURCE   camera index or video path (default 0)
  SAFEDRIVE_YOLO_WEIGHTS    YOLO weights path (default yolov8n.pt)
"""
import atexit
import os
import threading

from flask import Blueprint, jsonify
from flask_login import login_required

from engine import CameraConfig, MonitoringPipeline, PipelineConfig
from engine.detectors.person import PersonDetectorConfig

monitoring = Blueprint("monitoring", __name__, url_prefix="/monitoring")

_engine = None
_engine_lock = threading.Lock()


def get_engine():
    """Create the engine on first use (keeps the debug reloader's parent process off the camera)."""
    global _engine
    with _engine_lock:
        if _engine is None:
            source = os.environ.get("SAFEDRIVE_CAMERA_SOURCE", "0")
            config = PipelineConfig(
                camera=CameraConfig(source=int(source) if source.isdigit() else source),
                person=PersonDetectorConfig(weights=os.environ.get("SAFEDRIVE_YOLO_WEIGHTS", "yolov8n.pt")),
            )
            _engine = MonitoringPipeline(config)
            atexit.register(_engine.stop)
        return _engine


@monitoring.route("/status")
@login_required
def status():
    snapshot = get_engine().snapshot().to_dict()
    snapshot["running"] = get_engine().is_running
    return jsonify(snapshot)


@monitoring.route("/start", methods=["POST"])
@login_required
def start():
    get_engine().start()
    return jsonify({"running": True})


@monitoring.route("/stop", methods=["POST"])
@login_required
def stop():
    get_engine().stop()
    return jsonify({"running": False})
