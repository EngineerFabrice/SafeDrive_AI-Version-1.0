"""The independent-evaluation harness: consent / label / independence refusals and honest status wording.

These test the harness mechanics only; they contain no model evaluation.
"""
import csv
import os
import shutil

import cv2
import numpy as np
import pytest

from scripts import evaluate_independent_faces as ev

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_IMG = os.path.join(ROOT, "AlcoholDetectionDataset", "non_alcoholic", "a0255.png")


def write_set(tmp_path, rows):
    with open(tmp_path / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "subject_id", "session_id", "label", "label_source", "consent"])
        w.writeheader()
        w.writerows(rows)


def noise(tmp_path, name, seed):
    img = np.random.default_rng(seed).integers(0, 255, (240, 240, 3)).astype(np.uint8)
    cv2.imwrite(str(tmp_path / name), img)


def row(file, **over):
    return {"file": file, "subject_id": "S01", "session_id": "S01-sober", "label": "sober",
            "label_source": "breath_test", "consent": "yes", **over}


@pytest.mark.skipif(not (os.path.isfile(ev.TRAINING_MANIFEST) and os.path.isfile(TRAIN_IMG)),
                    reason="training manifest / dataset missing")
def test_refuses_training_images_missing_consent_bad_labels_and_path_escape(tmp_path):
    shutil.copy(TRAIN_IMG, tmp_path / "copied.png")
    near = cv2.imread(TRAIN_IMG)
    cv2.imwrite(str(tmp_path / "recompressed.jpg"), near, [cv2.IMWRITE_JPEG_QUALITY, 70])   # same frame, new bytes
    for i, name in enumerate(("ok.png", "noconsent.png", "badlabel.png", "badsource.png")):
        noise(tmp_path, name, i)
    write_set(tmp_path, [row("copied.png"), row("recompressed.jpg"), row("ok.png"),
                         row("noconsent.png", consent="no"), row("badlabel.png", label="drunk"),
                         row("badsource.png", label_source="guess"), row("../outside.png"), row("missing.png")])
    md5s, hashes = ev.training_fingerprints()
    accepted, refused = ev.load_rows(str(tmp_path), md5s, hashes)
    assert [r["file"] for r, _ in accepted] == ["ok.png"]
    reasons = {r["file"]: r["reasons"] for r in refused}
    assert reasons["copied.png"] == ["identical_to_training_image"]
    assert reasons["recompressed.jpg"] == ["near_duplicate_of_training_image"]
    assert reasons["noconsent.png"] == ["no_consent"]
    assert reasons["badlabel.png"] == ["unknown_label"]
    assert reasons["badsource.png"] == ["unknown_label_source"]
    assert reasons["../outside.png"] == ["file_missing"] and reasons["missing.png"] == ["file_missing"]


def frames_for(subjects, counted=True):
    frames, sessions, accepted = [], [], []
    for s in range(subjects):
        for label, p in (("sober", 0.1), ("alcohol", 0.9)):
            sid = f"S{s}-{label}"
            accepted.append(({"subject_id": f"S{s}", "label": label, "label_source": "breath_test"}, None))
            frames.append({"label": label, "face": True, "quality_ok": True, "quality_score": 0.8, "brightness": 100,
                           "sharpness": 150, "p_positive": p, "counted": counted})
            sessions.append({"session": sid, "subject": f"S{s}", "label": label, "frames": 1,
                             "final_assessment": "SOBER" if label == "sober" else "POTENTIALLY_NOT_SOBER"})
    return accepted, frames, sessions


def test_small_set_is_never_called_validation():
    accepted, frames, sessions = frames_for(3)
    report = ev.summarize(accepted, [], frames, sessions, {"name": "m"})
    assert report["status"].startswith("EXPLORATORY EVALUATION ONLY")
    assert report["design_checks"]["min_subjects_met"] is False
    assert "blood alcohol" in report["disclaimer"]


def test_even_a_well_designed_set_is_not_declared_validated():
    accepted, frames, sessions = frames_for(30)
    report = ev.summarize(accepted, [], frames, sessions, {"name": "m"})
    assert "NOT a validation" in report["status"] and "independent scientific review" in report["status"]
    assert report["frame_metrics_on_counted_frames"]["n"] == 60


def test_session_level_metrics_report_abstentions_and_intervals():
    accepted, frames, sessions = frames_for(3)
    sessions[0]["final_assessment"] = "UNCERTAIN"                       # S0-sober abstains
    sessions[1]["final_assessment"] = "SOBER"                           # S0-alcohol missed
    m = ev.summarize(accepted, [], frames, sessions, {})["session_level_metrics"]
    assert m["sessions"] == 6 and m["decided"] == 5 and m["abstained"] == 1 and m["participants"] == 3
    assert m["confusion_matrix"] == [[2, 0], [1, 2]]
    assert m["sensitivity"] == pytest.approx(2 / 3, abs=1e-3) and m["specificity"] == 1.0
    lo, hi = m["sensitivity_95ci"]
    assert lo < 2 / 3 < hi and hi - lo > 0.5                            # 3 sessions: a very wide interval


def test_wilson_interval():
    assert ev.wilson(0, 0) is None
    lo, hi = ev.wilson(50, 100)
    assert lo == pytest.approx(0.404, abs=1e-3) and hi == pytest.approx(0.596, abs=1e-3)


def test_excluded_frames_and_gate_comparison_are_reported():
    accepted, frames, sessions = frames_for(3)
    low = next(f for f in frames if f["label"] == "alcohol")
    low.update(counted=False, excluded="quality_score_below_min", quality_score=0.41)
    report = ev.summarize(accepted, [], frames, sessions, {})
    assert report["frames_by_label"]["alcohol"]["excluded_by_reason"] == {"quality_score_below_min": 1}
    gates = report["quality_gate_comparison"]
    assert gates["live_gate_score_ge_min_quality"]["frames"] == 5
    assert gates["training_gate_quality_ok"]["frames"] == 6


def test_label_correlated_recording_conditions_are_flagged():
    accepted, frames, sessions = frames_for(3)
    for f in frames:
        if f["label"] == "alcohol":
            f.update(brightness=60, sharpness=40)
    assert any("recording conditions" in w for w in ev.summarize(accepted, [], frames, sessions, {})["warnings"])
