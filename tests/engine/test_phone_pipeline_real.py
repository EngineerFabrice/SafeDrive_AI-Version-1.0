"""SOFTWARE PIPELINE TEST with the real detectors and model (no database, no HTTP).

Checks that phone frames flow through YOLOv8n -> YuNet -> alignment -> quality gate -> MobileNetV3 ->
temporal decision exactly as the mobile API runs them, and that the response fields are consistent.

It deliberately uses training-dataset images, so it says NOTHING about whether the model detects alcohol.
Model evaluation on consented, independent images is scripts/evaluate_independent_faces.py.
"""
import glob
import io
import os

import cv2
import numpy as np
import pytest

from website import mobile_monitoring

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = os.path.join(ROOT, "models", "alcohol_mobilenetv3", "model.pt")
YOLO = os.path.join(ROOT, "yolov8n.pt")
SOBER = sorted(glob.glob(os.path.join(ROOT, "AlcoholDetectionDataset", "non_alcoholic", "*.png")))
ALCOHOLIC = sorted(glob.glob(os.path.join(ROOT, "AlcoholDetectionDataset", "alcoholic", "*.png")))
LABELS = {"ASSESSING", "SOBER", "UNCERTAIN", "POTENTIALLY_NOT_SOBER"}

needs_models = pytest.mark.skipif(not (os.path.isfile(MODEL) and os.path.isfile(YOLO) and SOBER and ALCOHOLIC),
                                  reason="model.pt / yolov8n.pt / dataset images missing")


def jpeg(path, quality=90):
    return cv2.imencode(".jpg", cv2.imread(path), [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


@pytest.fixture(scope="module")
def real():
    from engine.impairment import create_impairment_model
    mobile_monitoring.set_components(impairment_model=create_impairment_model("alcohol_mobilenetv3"))
    yield mobile_monitoring
    mobile_monitoring.reset_components()


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(mobile_monitoring, "MIN_FRAME_INTERVAL_S", 0.0)


@pytest.mark.model_files
@needs_models
def test_real_frames_produce_consistent_results(real, fast):
    real.start(1)
    results = [real.analyze(1, jpeg(p)) for p in SOBER[::20][:8]]
    first = results[0]
    assert first["face_detected"] and first["assessment"]["assessment"] == "ASSESSING"   # one frame never decides
    for r in results:
        assert r["assessment"]["assessment"] in LABELS
        if r["impairment"] is not None:
            assert r["impairment"]["model_name"] == "AlcoholMobileNetV3" and r["impairment"]["development_only"]
            assert sum(r["impairment"]["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
        q = r["face_quality"]
        # `counted` is exactly the temporal engine's rule
        assert r["counted"] == bool(r["impairment"] and r["impairment"]["valid"] and q and q["score"] >= r["min_quality"])
    counted = sum(r["counted"] for r in results)
    if counted >= 5:
        assert results[-1]["assessment"]["assessment"] != "ASSESSING"
    real.stop(1)


@pytest.mark.model_files
@needs_models
def test_quality_ok_but_below_temporal_minimum_is_reported_as_not_counted(real, fast):
    """Documents the gate seen with the darker, blurrier 'alcoholic' recording: such frames are predicted
    but not counted, and the API says so instead of claiming the frame was used."""
    real.start(2)
    found = None
    for p in ALCOHOLIC:
        r = real.analyze(2, jpeg(p))
        q = r["face_quality"]
        if q and q["ok"] and q["score"] < r["min_quality"]:
            found = r
            break
    real.stop(2)
    assert found is not None, "expected at least one ok-but-low-score frame in the alcoholic recording"
    assert found["counted"] is False and "not counted" in found["message"]


@pytest.mark.model_files
@needs_models
def test_evaluation_harness_exclusion_reasons_match_the_live_counted_rule(real):
    """Software check of scripts/evaluate_independent_faces.run_pipeline (training images: no accuracy meaning)."""
    from scripts import evaluate_independent_faces as ev
    accepted = [({"file": os.path.basename(p), "subject_id": "T", "session_id": f"T-{label}", "label": label}, cv2.imread(p))
                for label, paths in (("sober", SOBER[:6]), ("alcohol", ALCOHOLIC[:6])) for p in paths]
    frames, sessions, _ = ev.run_pipeline(accepted, real._components)
    assert len(frames) == 12 and len(sessions) == 2
    for f in frames:
        assert f["counted"] == (f["excluded"] is None)
        if f["excluded"] == "quality_score_below_min":
            assert f["quality_ok"] and f["quality_score"] < 0.5
        if f["counted"]:
            assert f["quality_score"] >= 0.5 and f["p_positive"] is not None


def test_exif_orientation_is_applied_to_phone_jpegs():
    """Android camera JPEGs may carry an EXIF rotation; the face must reach the detector upright."""
    from PIL import Image
    img = np.full((200, 400, 3), 128, np.uint8)
    pil = Image.fromarray(img)
    exif = pil.getexif()
    exif[0x0112] = 6                                   # rotate 90 degrees clockwise when displayed
    buf = io.BytesIO()
    pil.save(buf, "JPEG", exif=exif.tobytes())
    decoded = mobile_monitoring.decode_frame(buf.getvalue())
    assert decoded.shape[:2] == (400, 200)


def test_large_phone_frames_are_downscaled_for_processing():
    big = cv2.imencode(".jpg", np.full((1080, 1920, 3), 90, np.uint8))[1].tobytes()
    assert mobile_monitoring.decode_frame(big).shape[1] == mobile_monitoring.PROCESS_MAX_WIDTH
