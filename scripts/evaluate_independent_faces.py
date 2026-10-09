"""Evaluate the live phone pipeline + alcohol model on CONSENTED face images that are independent of training.

    python scripts/evaluate_independent_faces.py --data path/to/evaluation_set [--out report.json]

This is a MODEL EVALUATION tool, not a software test (the software pipeline is tested in
tests/engine/test_phone_pipeline_real.py and tests/web/test_mobile_api.py). Its output is never a
"validation" claim: see VALIDATION_CRITERIA and the "status" field of the report.

Evaluation set layout
---------------------
    <data>/manifest.csv     one row per image, columns:
        file          image path relative to <data> (JPEG/PNG), e.g. s01/sober/0001.jpg
        subject_id    pseudonymous person id (never a name), e.g. S01
        session_id    one recording session (frames in file-name order, ~1 per second), e.g. S01-sober
        label         sober | alcohol
        label_source  breath_test | blood_test | controlled_dose | self_report
        consent       yes  (written consent for this use was obtained and is stored outside this repository)
    <data>/<files>

Rows are refused (and listed in the report) when consent is not "yes", the label or label source is unknown,
the file is missing or unreadable, or the image is identical (MD5) or a near-duplicate (dHash distance <= 4
bits) of any image in the training manifest. The hash check only catches copied / near-identical frames: it
cannot tell whether a person also appears in the training data. The curator must make sure no evaluation
subject is the person in AlcoholDetectionDataset.

The report gives: refused rows with reasons; per label, frames excluded by the pipeline with reasons (no_driver,
no_face, quality_rejected:<reasons>, no_prediction, quality_score_below_min); frame-level metrics on counted
frames (no intervals: frames of a session are correlated); session-level (participant-level) confusion matrix,
sensitivity, specificity and precision with 95 % Wilson intervals, abstentions (UNCERTAIN / ASSESSING) reported
separately; and the same frames scored under the training-preparation quality gate for comparison.

Every image goes through exactly the code used for phone frames (engine.pipeline.MonitoringPipeline.analyze_frame
with the phone temporal configuration of website/mobile_monitoring.py). Images, crops and per-frame outputs
stay in memory; the report holds only aggregate numbers, pseudonymous ids and file names.
"""
import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

LABELS = {"sober": 0, "alcohol": 1}                 # model classes: 0 non_alcoholic, 1 alcoholic
LABEL_SOURCES = ("breath_test", "blood_test", "controlled_dose", "self_report")
OBJECTIVE_SOURCES = ("breath_test", "blood_test", "controlled_dose")
DHASH_DUPLICATE_BITS = 4
FRAME_INTERVAL_S = 1.0                              # phone upload rate used for the temporal engine
TRAINING_MANIFEST = os.path.join(ROOT, "data", "processed", "alcohol_v1", "manifest.csv")
DEFAULT_OUT_DIR = os.path.join(ROOT, "instance", "evaluations")   # git-ignored

# Minimum conditions before results could even be submitted for independent scientific review.
VALIDATION_CRITERIA = {
    "min_subjects": 30,
    "subjects_with_both_conditions_fraction": 0.8,  # within-subject sober + alcohol sessions
    "objective_label_fraction": 1.0,                # breath/blood test or controlled dose for every row
}


def dhash(gray, size=8):          # same definition as training/prepare_dataset.py
    g = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    return int("".join("1" if b else "0" for b in (g[:, 1:] > g[:, :-1]).flatten()), 2)


def training_fingerprints(path=TRAINING_MANIFEST):
    if not os.path.isfile(path):
        raise SystemExit(f"training manifest not found: {path} (needed for the independence check)")
    md5s, hashes = set(), []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("md5"):
                md5s.add(r["md5"])
            if r.get("dhash"):
                hashes.append(int(r["dhash"]))
    return md5s, hashes


def load_rows(data_dir, md5s, hashes):
    """(accepted rows with decoded images, refused rows with reasons)."""
    manifest = os.path.join(data_dir, "manifest.csv")
    if not os.path.isfile(manifest):
        raise SystemExit(f"{manifest} not found (see the layout in this script's docstring)")
    accepted, refused = [], []
    with open(manifest, newline="", encoding="utf-8") as fh:
        for i, r in enumerate(csv.DictReader(fh), start=2):
            r = {k: (v or "").strip() for k, v in r.items()}
            why = []
            if r.get("consent", "").lower() != "yes":
                why.append("no_consent")
            if r.get("label", "").lower() not in LABELS:
                why.append("unknown_label")
            if r.get("label_source", "").lower() not in LABEL_SOURCES:
                why.append("unknown_label_source")
            if not r.get("subject_id") or not r.get("session_id"):
                why.append("missing_subject_or_session")
            path = os.path.normpath(os.path.join(data_dir, r.get("file", "")))
            if not path.startswith(os.path.normpath(data_dir) + os.sep) or not os.path.isfile(path):
                why.append("file_missing")
            image = None
            if not why:
                with open(path, "rb") as f:
                    data = f.read()
                image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    why.append("unreadable_image")
                elif hashlib.md5(data).hexdigest() in md5s:
                    why.append("identical_to_training_image")
                else:
                    h = dhash(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
                    if any(bin(h ^ t).count("1") <= DHASH_DUPLICATE_BITS for t in hashes):
                        why.append("near_duplicate_of_training_image")
            if why:
                refused.append({"line": i, "file": r.get("file"), "reasons": why})
            else:
                r["label"], r["label_source"] = r["label"].lower(), r["label_source"].lower()
                accepted.append((r, image))
    return accepted, refused


def run_pipeline(accepted, components=None):
    """Feed every session through a fresh phone pipeline; returns per-frame and per-session results."""
    from website.mobile_monitoring import _Components
    components = components or _Components()
    ok, error = components.ready()
    if not ok:
        raise SystemExit(f"detection models unavailable: {error}")
    model = components.model
    if model is None or not model.available:
        raise SystemExit("no usable impairment model (check MODEL_PROVIDER / ALCOHOL_MODEL_PATH)")
    positive = model.info.positive_class
    sessions = defaultdict(list)
    for r, image in accepted:
        sessions[r["session_id"]].append((r, image))
    frames, session_results = [], []
    for sid, items in sorted(sessions.items()):
        items.sort(key=lambda x: x[0]["file"])
        pipeline = components.pipeline()
        decision, t0 = None, time.perf_counter()
        for k, (r, image) in enumerate(items):
            analysis, d = pipeline.analyze_frame(image, t0 + k * FRAME_INTERVAL_S, k + 1)
            decision = d or decision
            q, imp = analysis.face_quality, analysis.impairment
            p = imp.probabilities.get(positive) if imp is not None and imp.valid and imp.probabilities else None
            min_q = pipeline.decision.config.min_quality
            if analysis.face is None:
                excluded = "no_face" if analysis.driver is not None else "no_driver"
            elif q is None or not q.ok:
                excluded = "quality_rejected:" + ";".join(q.reasons) if q else "quality_rejected"
            elif p is None:
                excluded = "no_prediction"
            elif q.score < min_q:
                excluded = "quality_score_below_min"
            else:
                excluded = None
            frames.append({"excluded": excluded, "label": r["label"], "subject": r["subject_id"], "session": sid,
                           "driver": analysis.driver is not None, "face": analysis.face is not None,
                           "quality_ok": bool(q and q.ok), "quality_score": q.score if q else None,
                           "brightness": q.metrics.get("brightness") if q else None,
                           "sharpness": q.metrics.get("sharpness") if q else None,
                           "p_positive": p,
                           "counted": excluded is None})
        labels = {r["label"] for r, _ in items}
        session_results.append({"session": sid, "subject": items[0][0]["subject_id"],
                                "label": labels.pop() if len(labels) == 1 else "mixed",
                                "frames": len(items),
                                "final_assessment": decision.assessment.value if decision else "NO_DECISION"})
    return frames, session_results, model.info.to_dict()


def _median(values):
    values = [v for v in values if v is not None]
    return round(statistics.median(values), 3) if values else None


def wilson(k, n, z=1.96):
    """95 % Wilson score interval for k successes out of n (None when n == 0)."""
    if not n:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5 / (1 + z * z / n)
    return [round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)]


def session_metrics(sessions):
    """Participant-level results: one final temporal decision per recording session.

    POTENTIALLY_NOT_SOBER is a positive call and SOBER a negative call; UNCERTAIN, ASSESSING and NO_DECISION are
    abstentions, reported separately and not counted as correct. Sessions of the same participant are not fully
    independent, so the intervals are optimistic when participants contribute several sessions."""
    decided = [s for s in sessions if s["label"] in LABELS and s["final_assessment"] in ("SOBER", "POTENTIALLY_NOT_SOBER")]
    tp = sum(s["label"] == "alcohol" and s["final_assessment"] == "POTENTIALLY_NOT_SOBER" for s in decided)
    fn = sum(s["label"] == "alcohol" and s["final_assessment"] == "SOBER" for s in decided)
    tn = sum(s["label"] == "sober" and s["final_assessment"] == "SOBER" for s in decided)
    fp = sum(s["label"] == "sober" and s["final_assessment"] == "POTENTIALLY_NOT_SOBER" for s in decided)
    labelled = [s for s in sessions if s["label"] in LABELS]
    return {
        "sessions": len(labelled), "decided": len(decided), "abstained": len(labelled) - len(decided),
        "abstention_rate": round((len(labelled) - len(decided)) / len(labelled), 3) if labelled else None,
        "confusion_matrix": [[tn, fp], [fn, tp]],      # rows = true [sober, alcohol], cols = decided
        "sensitivity": round(tp / (tp + fn), 3) if tp + fn else None, "sensitivity_95ci": wilson(tp, tp + fn),
        "specificity": round(tn / (tn + fp), 3) if tn + fp else None, "specificity_95ci": wilson(tn, tn + fp),
        "precision": round(tp / (tp + fp), 3) if tp + fp else None, "precision_95ci": wilson(tp, tp + fp),
        "participants": len({s["subject"] for s in labelled}),
    }


def summarize(accepted, refused, frames, sessions, model_info):
    from training.metrics import binary_metrics
    by_label = {}
    for label in LABELS:
        fs = [f for f in frames if f["label"] == label]
        by_label[label] = {
            "frames": len(fs), "face_detected": sum(f["face"] for f in fs),
            "quality_ok": sum(f["quality_ok"] for f in fs), "counted": sum(f["counted"] for f in fs),
            "excluded_by_reason": dict(Counter(f.get("excluded") for f in fs if f.get("excluded"))),
            "median_quality_score": _median(f["quality_score"] for f in fs),
            "median_brightness": _median(f["brightness"] for f in fs),
            "median_sharpness": _median(f["sharpness"] for f in fs),
            "median_p_positive": _median(f["p_positive"] for f in fs),
        }
    counted = [f for f in frames if f["counted"]]
    frame_metrics = (binary_metrics([LABELS[f["label"]] for f in counted], [f["p_positive"] for f in counted])
                     if counted else None)
    # Sensitivity analysis of the quality gate, on the SAME frames: the live rule (score >= min_quality) versus
    # the training-preparation rule (quality ok). Reported only; the app keeps the live rule.
    training_gate = [f for f in frames if f["quality_ok"] and f["p_positive"] is not None]
    gate_comparison = {
        "live_gate_score_ge_min_quality": {"frames": len(counted), "metrics": frame_metrics},
        "training_gate_quality_ok": {
            "frames": len(training_gate),
            "metrics": (binary_metrics([LABELS[f["label"]] for f in training_gate],
                                       [f["p_positive"] for f in training_gate]) if training_gate else None)},
        "note": "Frame-level only and frames of one session are correlated; use it to see whether the gate removes "
                "one label's frames, not as evidence that either gate is more accurate.",
    }
    session_table = defaultdict(Counter)
    for s in sessions:
        session_table[s["label"]][s["final_assessment"]] += 1

    rows = [r for r, _ in accepted]
    subjects = {r["subject_id"] for r in rows}
    conditions = defaultdict(set)
    for r in rows:
        conditions[r["subject_id"]].add(r["label"])
    both = sum(1 for s in subjects if conditions[s] == set(LABELS))
    objective = sum(r["label_source"] in OBJECTIVE_SOURCES for r in rows)
    checks = {
        "subjects": len(subjects),
        "min_subjects_met": len(subjects) >= VALIDATION_CRITERIA["min_subjects"],
        "subjects_with_both_conditions": both,
        "both_conditions_met": bool(subjects) and both / len(subjects)
        >= VALIDATION_CRITERIA["subjects_with_both_conditions_fraction"],
        "objective_label_fraction": round(objective / len(rows), 3) if rows else 0.0,
        "objective_labels_met": bool(rows) and objective / len(rows) >= VALIDATION_CRITERIA["objective_label_fraction"],
    }
    warnings = []
    rates = {k: (v["counted"] / v["frames"]) for k, v in by_label.items() if v["frames"]}
    if len(rates) == 2 and abs(rates["sober"] - rates["alcohol"]) > 0.2:
        warnings.append("The share of frames counted differs by more than 20 points between labels: the quality gate "
                        "is correlated with the label (lighting / sharpness confound), as in the training dataset.")
    b = [by_label[k]["median_brightness"] for k in LABELS]
    s = [by_label[k]["median_sharpness"] for k in LABELS]
    if all(b) and abs(b[0] - b[1]) > 15 or all(s) and max(s) > 2 * min(s):
        warnings.append("Median brightness or sharpness differs strongly between labels: the classes can be "
                        "separated by recording conditions alone, so accuracy here is not evidence of alcohol detection.")
    criteria_met = checks["min_subjects_met"] and checks["both_conditions_met"] and checks["objective_labels_met"]
    status = ("EXPLORATORY EVALUATION - meets the minimum design criteria for a validation study; the results still "
              "require independent scientific review and are NOT a validation."
              if criteria_met else
              "EXPLORATORY EVALUATION ONLY - does not meet the minimum criteria for scientific validation.")
    return {
        "status": status,
        "disclaimer": "The model estimates visual patterns. It does not measure blood alcohol concentration and does "
                      "not prove intoxication.",
        "model": {k: model_info.get(k) for k in ("name", "version", "provider", "development_only")},
        "rows": {"accepted": len(rows), "refused": len(refused)}, "refused": refused,
        "design_checks": checks, "validation_criteria": VALIDATION_CRITERIA,
        "frames_by_label": by_label,
        # frame level: frames of one session are strongly correlated, so no confidence intervals are given here
        "frame_metrics_on_counted_frames": frame_metrics,
        "quality_gate_comparison": gate_comparison,
        "session_level_metrics": session_metrics(sessions),
        "sessions_final_assessment_by_label": {k: dict(v) for k, v in session_table.items()},
        "per_subject": sorted(({"subject": s["subject"], "session": s["session"], "label": s["label"],
                                "frames": s["frames"], "final_assessment": s["final_assessment"]} for s in sessions),
                              key=lambda x: (x["subject"], x["session"])),
        "warnings": warnings,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", required=True, help="evaluation set directory containing manifest.csv")
    parser.add_argument("--out", help=f"report path (default: {DEFAULT_OUT_DIR}/eval-<time>.json, git-ignored)")
    args = parser.parse_args(argv)
    data_dir = os.path.abspath(args.data)
    md5s, hashes = training_fingerprints()
    accepted, refused = load_rows(data_dir, md5s, hashes)
    print(f"accepted {len(accepted)} rows, refused {len(refused)}")
    if not accepted:
        print(json.dumps(refused[:20], indent=2))
        return 2
    frames, sessions, info = run_pipeline(accepted)
    report = summarize(accepted, refused, frames, sessions, info)
    out = args.out or os.path.join(DEFAULT_OUT_DIR, time.strftime("eval-%Y%m%d-%H%M%S.json"))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(report["status"])
    for w in report["warnings"]:
        print("WARNING:", w)
    print(json.dumps({"frames_by_label": report["frames_by_label"],
                      "sessions": report["sessions_final_assessment_by_label"],
                      "session_level_metrics": report["session_level_metrics"],
                      "design_checks": report["design_checks"]}, indent=2))
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
