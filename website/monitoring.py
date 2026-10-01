# website/monitoring.py
"""Thin Flask access to the SafeDrive CV engine.

The engine captures and processes frames in its own threads; these endpoints
only start/stop it and read its latest state. The dashboard's video endpoint
re-encodes frames the pipeline has already processed (display only, no extra
computer vision); frames are never sent *to* the server over HTTP.

Environment variables:
  SAFEDRIVE_CAMERA_SOURCE   camera index or video path (default 0)
  SAFEDRIVE_YOLO_WEIGHTS    YOLO weights path (default yolov8n.pt)
  MODEL_PROVIDER            impairment model: none (default) | mock (DEVELOPMENT_ONLY)
"""
import atexit
import os
import re
import threading
import time

import cv2
from flask import Blueprint, Response, jsonify, render_template, request
from flask_login import current_user, login_required

from engine import CameraConfig, MonitoringPipeline, PipelineConfig
from engine.detectors.person import PersonDetectorConfig
from engine.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from engine.impairment import create_impairment_model
from engine.pipeline import draw_overlay

from .history import HistoryRecorder, HistoryUnavailable, event_to_dict, session_to_dict

monitoring = Blueprint("monitoring", __name__, url_prefix="/monitoring")

# Shown on the dashboard at all times. Update this text (not the dashboard) when a
# validated model is introduced.
SYSTEM_NOTICE = ("Development system: model outputs shown here are experimental/mock results "
                 "and must not be interpreted as a determination of alcohol impairment.")

VIDEO_MAX_FPS = 10       # dashboard preview rate; independent of the processing rate
VIDEO_MAX_WIDTH = 960    # preview is downscaled to keep bandwidth low

_engine = None
_recorder = None
_engine_lock = threading.Lock()
_SESSION_ID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def get_engine():
    """Create the engine on first use (keeps the debug reloader's parent process off the camera)."""
    global _engine, _recorder
    with _engine_lock:
        if _engine is None:
            source = os.environ.get("SAFEDRIVE_CAMERA_SOURCE", "0")
            config = PipelineConfig(
                camera=CameraConfig(source=int(source) if source.isdigit() else source),
                person=PersonDetectorConfig(weights=os.environ.get("SAFEDRIVE_YOLO_WEIGHTS", "yolov8n.pt")),
            )
            _engine = MonitoringPipeline(config, impairment_model=create_impairment_model())
            _recorder = HistoryRecorder(_engine)
            atexit.register(_engine.stop)
            atexit.register(_recorder.shutdown)   # runs first (LIFO): closes the open session
        return _engine


def get_recorder():
    get_engine()
    return _recorder


@monitoring.route("/")
@login_required
def dashboard():
    return render_template("monitoring-dashboard.html", username=current_user.username,
                           feature_schema_version=FEATURE_SCHEMA_VERSION,
                           feature_count=len(FEATURE_NAMES), system_notice=SYSTEM_NOTICE)


@monitoring.route("/status")
@login_required
def status():
    snapshot = get_engine().snapshot().to_dict()
    snapshot["running"] = get_engine().is_running
    return jsonify(snapshot)


@monitoring.route("/model")
@login_required
def model():
    """Which impairment model is configured, and whether it is development-only."""
    engine = get_engine()
    info = engine.snapshot().impairment_model
    return jsonify({
        "enabled": info is not None,
        "available": bool(engine.impairment_model and engine.impairment_model.available),
        "model": info,
        "development_only": bool(info and info.get("development_only")),
        "notice": ("MOCK / DEVELOPMENT_ONLY model: outputs are synthetic and are not an "
                   "impairment determination." if info and info.get("is_mock") else ""),
    })


@monitoring.route("/video")
@login_required
def video():
    """MJPEG preview of the frames the pipeline has already processed, with its existing overlay."""
    engine = get_engine()

    def frames():
        last_id = None
        while True:
            result = engine.latest_result()
            if result is None or result[0].frame_id == last_id:
                time.sleep(0.05)
                continue
            frame, analysis = result
            last_id = frame.frame_id
            image = draw_overlay(frame.image, analysis, engine.snapshot())
            if image.shape[1] > VIDEO_MAX_WIDTH:
                scale = VIDEO_MAX_WIDTH / image.shape[1]
                image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
            time.sleep(1.0 / VIDEO_MAX_FPS)

    return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-store"})


@monitoring.route("/start", methods=["POST"])
@login_required
def start():
    engine = get_engine()
    if not engine.is_running:
        engine.start()
        get_recorder().start_session(started_by=_user_id())   # in-memory; written by its own thread
    return jsonify({"running": True})


@monitoring.route("/stop", methods=["POST"])
@login_required
def stop():
    engine = get_engine()
    if engine.is_running:
        get_recorder().end_session("COMPLETED")   # before stop(), so the session is not seen as interrupted
    engine.stop()
    return jsonify({"running": False})


def _user_id():
    try:
        return int(current_user.get_id())
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- history (read-only)
@monitoring.route("/history")
@login_required
def history():
    """Recent monitoring sessions (technical statistics only)."""
    limit = max(1, min(request.args.get("limit", 20, type=int), 100))
    recorder = get_recorder()
    try:
        sessions = [session_to_dict(r) for r in recorder.store.list_sessions(limit)]
    except HistoryUnavailable:
        return jsonify({"sessions": [], "persistence": recorder.status(),
                        "error": "Monitoring history is temporarily unavailable."}), 503
    return jsonify({"sessions": sessions, "persistence": recorder.status()})


@monitoring.route("/history/<session_id>")
@login_required
def history_session(session_id):
    """One session and its technical events. No feature vectors or model outputs are stored."""
    if not _SESSION_ID.match(session_id):
        return jsonify({"error": "Invalid session id."}), 400
    try:
        session, events = get_recorder().store.get_session(session_id)
    except HistoryUnavailable:
        return jsonify({"error": "Monitoring history is temporarily unavailable."}), 503
    if session is None:
        return jsonify({"error": "Session not found."}), 404
    return jsonify({"session": session_to_dict(session), "events": [event_to_dict(e) for e in events]})
