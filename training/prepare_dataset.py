"""Audit AlcoholDetectionDataset and build the processed face-crop dataset + manifest.

    python -m training.prepare_dataset

Reads AlcoholDetectionDataset/ (never modified) and writes data/processed/alcohol_v1/:
    images/<class>/<stem>.png   aligned 224x224 face crops (engine.preprocessing, same as live inference)
    manifest.csv                one row per source image, with split and quality status
    audit/audit.json            all measured numbers
    audit/AUDIT.md              human-readable findings

Split strategy
--------------
The dataset has no subject or session metadata. Each class is a single
recording (consecutive frame numbers, near-identical frames), so a subject- or
session-grouped split is impossible. To stop near-identical consecutive frames
from crossing splits, each class is split into contiguous blocks in recording
order (70/15/15) and SPLIT_GAP frames are dropped at every boundary. The test
set therefore measures generalisation to *later moments of the same recording*,
not to new people. This limitation is recorded in the audit and in the model
metadata.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.detectors import BoundingBox                         # noqa: E402
from engine.detectors.face import FaceDetector                    # noqa: E402
from engine.preprocessing import CROP_SIZE, align_face_crop, assess_face_quality  # noqa: E402
from training import config                                       # noqa: E402

MANIFEST_FIELDS = ("image_path", "original_path", "label", "label_type", "source_id", "session_id",
                   "subject_id", "frame_number", "split", "original_resolution", "face_detection_status",
                   "face_score", "quality_status", "quality_score", "quality_reasons", "brightness",
                   "sharpness", "md5", "dhash")


# ---------------------------------------------------------------- pure helpers (unit-tested)
def frame_number(filename):
    digits = re.sub(r"\D", "", os.path.splitext(filename)[0])
    return int(digits) if digits else None


def dhash(gray, size=8):
    g = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    return int("".join("1" if b else "0" for b in (g[:, 1:] > g[:, :-1]).flatten()), 2)


def hamming(a, b):
    return bin(a ^ b).count("1")


def temporal_block_split(keys, fractions=config.SPLIT_FRACTIONS, gap=config.SPLIT_GAP):
    """Assign ordered keys to contiguous train/val/test blocks with ``gap`` keys dropped at each boundary.

    Returns {key: "train" | "val" | "test" | "buffer"}.
    """
    keys = sorted(keys)
    n = len(keys)
    b1 = int(round(n * fractions[0]))
    b2 = int(round(n * (fractions[0] + fractions[1])))
    out = {}
    for i, k in enumerate(keys):
        if i < b1:
            split = "train"
        elif i < b2:
            split = "val"
        else:
            split = "test"
        # drop `gap` frames on each side of every boundary
        if any(b - gap <= i < b + gap for b in (b1, b2)) and gap > 0:
            split = "buffer"
        out[k] = split
    return out


def shortcut_features(img):
    """Image statistics that carry no facial information beyond colour/light."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    ring = np.ones_like(gray, bool)
    ring[h // 8: h - h // 8, w // 8: w - w // 8] = False
    b, g, r = (img[..., c].astype(np.float32) for c in range(3))
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "mean_r": float(r.mean()), "mean_g": float(g.mean()), "mean_b": float(b.mean()),
        "lab_a": float(lab[..., 1].mean()), "lab_b": float(lab[..., 2].mean()),
        "border_ring_brightness": float(gray[ring].mean()),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }


def probe(train_rows, test_rows, feature_names):
    """Logistic-regression accuracy on test from the given shortcut features (trained on train only)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def xy(rows):
        return (np.array([[r["probe"][f] for f in feature_names] for r in rows]),
                np.array([config.CLASSES.index(r["label"]) for r in rows]))
    xtr, ytr = xy(train_rows)
    xte, yte = xy(test_rows)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(xtr, ytr)
    return round(float((clf.predict(xte) == yte).mean()), 4)


# ---------------------------------------------------------------- main
def build(source_dir=config.SOURCE_DIR, out_dir=config.PROCESSED_DIR, detector=None):
    images_dir = os.path.join(out_dir, "images")
    audit_dir = os.path.join(out_dir, "audit")
    os.makedirs(audit_dir, exist_ok=True)
    detector = detector or FaceDetector()
    if not detector.load() or detector.backend != "yunet":
        raise RuntimeError(f"YuNet face detector required (got {detector.backend!r}: {detector.error})")

    classes_found = sorted(d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d)))
    if tuple(sorted(classes_found)) != tuple(sorted(config.CLASSES)):
        raise RuntimeError(f"unexpected class folders {classes_found}")

    rows, corrupt = [], []
    for label in config.CLASSES:
        cls_dir = os.path.join(source_dir, label)
        for name in sorted(os.listdir(cls_dir)):
            path = os.path.join(cls_dir, name)
            img = cv2.imread(path)
            if img is None:
                corrupt.append(os.path.relpath(path, config.ROOT))
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            row = {"original_path": os.path.relpath(path, config.ROOT).replace("\\", "/"), "label": label,
                   "label_type": config.LABEL_TYPE, "source_id": config.SOURCE_ID,
                   # one continuous recording per class (Phase 2A): the class folder is the session
                   "session_id": f"{config.SOURCE_ID}:{label}", "subject_id": "unknown",
                   "frame_number": frame_number(name), "original_resolution": f"{img.shape[1]}x{img.shape[0]}",
                   "md5": hashlib.md5(open(path, "rb").read()).hexdigest(), "dhash": dhash(gray),
                   "format": os.path.splitext(name)[1].lower().lstrip(".")}

            faces = detector.landmark_detector.detect_all(img, BoundingBox(0, 0, img.shape[1], img.shape[0]))
            if not faces:
                row.update(face_detection_status="no_face", face_score="", quality_status="rejected",
                           quality_score=0.0, quality_reasons="no_face", brightness="", sharpness="",
                           image_path="")
                rows.append(row)
                continue
            face = max(faces, key=lambda f: f.bbox.area)
            crop = align_face_crop(img, face.bbox, face)
            q = assess_face_quality(crop, face.bbox, face.score)
            out_path = os.path.join(images_dir, label, os.path.splitext(name)[0] + ".png")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cv2.imwrite(out_path, crop)
            row.update(face_detection_status="multiple_faces" if len(faces) > 1 else "ok",
                       face_score=round(face.score, 4), quality_status="ok" if q.ok else "rejected",
                       quality_score=round(q.score, 4), quality_reasons=";".join(q.reasons),
                       brightness=round(q.metrics["brightness"], 2), sharpness=round(q.metrics["sharpness"], 2),
                       image_path=os.path.relpath(out_path, config.ROOT).replace("\\", "/"))
            row["probe"] = shortcut_features(crop)
            row["probe_original"] = shortcut_features(img)
            rows.append(row)

    # split per class in recording order; rejected images are excluded but still listed
    for label in config.CLASSES:
        cls_rows = [r for r in rows if r["label"] == label]
        assignment = temporal_block_split([r["frame_number"] for r in cls_rows])
        for r in cls_rows:
            r["split"] = assignment[r["frame_number"]] if r["quality_status"] == "ok" else "excluded"

    audit = make_audit(rows, corrupt)
    write_outputs(rows, audit, out_dir)
    return rows, audit


def make_audit(rows, corrupt):
    usable = [r for r in rows if r["split"] in ("train", "val", "test")]
    by_split = {s: [r for r in usable if r["split"] == s] for s in ("train", "val", "test")}

    # leakage: nearest perceptual-hash distance between every test/val image and any train image
    def nearest(a_rows, b_rows):
        if not a_rows or not b_rows:
            return None
        return int(min(min(hamming(a["dhash"], b["dhash"]) for b in b_rows) for a in a_rows))
    md5s = Counter(r["md5"] for r in rows)
    dup_across = sum(1 for r in usable if md5s[r["md5"]] > 1)

    stats = {}
    for label in config.CLASSES:
        cr = [r for r in rows if r["label"] == label and "probe" in r]
        stats[label] = {k: round(float(np.mean([r["probe"][k] for r in cr])), 2) for k in cr[0]["probe"]} if cr else {}

    probes = {}
    if all(by_split[s] for s in ("train", "test")):
        probes = {
            "brightness_only": probe(by_split["train"], by_split["test"], ["brightness"]),
            "colour_statistics": probe(by_split["train"], by_split["test"],
                                       ["mean_r", "mean_g", "mean_b", "lab_a", "lab_b"]),
            "border_ring_only": probe(by_split["train"], by_split["test"], ["border_ring_brightness"]),
            "sharpness_only": probe(by_split["train"], by_split["test"], ["sharpness"]),
            "all_image_statistics": probe(by_split["train"], by_split["test"], list(by_split["train"][0]["probe"])),
        }

    return {
        "source": config.SOURCE_ID,
        "label_type": config.LABEL_TYPE,
        "total_images": len(rows) + len(corrupt),
        "readable_images": len(rows),
        "corrupt_images": corrupt,
        "images_per_class": dict(Counter(r["label"] for r in rows)),
        "formats": dict(Counter(r["format"] for r in rows)),
        "resolutions": dict(Counter(r["original_resolution"] for r in rows)),
        "exact_duplicate_files": sum(v - 1 for v in md5s.values() if v > 1),
        "exact_duplicates_in_usable_set": dup_across,
        "subject_metadata": "none (no subject or session identifiers in the dataset)",
        "sessions": "one continuous recording per class (frame-number sequences, near-identical frames)",
        "face_detection": dict(Counter(r["face_detection_status"] for r in rows)),
        "quality": dict(Counter(r["quality_status"] for r in rows)),
        "quality_rejection_reasons": dict(Counter(x for r in rows for x in r["quality_reasons"].split(";") if x)),
        "split_counts": {s: dict(Counter(r["label"] for r in rows if r["split"] == s))
                         for s in ("train", "val", "test", "buffer", "excluded")},
        "split_method": f"per-class contiguous blocks in recording order {config.SPLIT_FRACTIONS}, "
                        f"{config.SPLIT_GAP} frames dropped at each boundary",
        "nearest_dhash_bits_val_to_train": nearest(by_split["val"], by_split["train"]),
        "nearest_dhash_bits_test_to_train": nearest(by_split["test"], by_split["train"]),
        "class_mean_image_statistics_of_crops": stats,
        "shortcut_probe_test_accuracy": probes,
    }


def write_outputs(rows, audit, out_dir):
    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["label"], r["frame_number"])):
            w.writerow(r)
    audit_dir = os.path.join(out_dir, "audit")
    with open(os.path.join(audit_dir, "audit.json"), "w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2)
    with open(os.path.join(audit_dir, "AUDIT.md"), "w", encoding="utf-8") as fh:
        fh.write(render_markdown(audit))


def render_markdown(a):
    p = a["shortcut_probe_test_accuracy"]
    s = a["class_mean_image_statistics_of_crops"]
    lines = [
        "# AlcoholDetectionDataset audit (generated by training/prepare_dataset.py)",
        "",
        "> **Use limitation.** Each class is one recording of one person (no subject/session metadata,",
        "> undocumented labels). Test results measure later frames of the *same* recordings, not new people,",
        "> and cannot support any claim about detecting alcohol in general. Never report them as BAC or",
        "> intoxication accuracy.",
        "",
        f"- Images: {a['readable_images']} readable / {a['total_images']} total; corrupt: {len(a['corrupt_images'])}",
        f"- Per class: {a['images_per_class']}; formats {a['formats']}; resolutions {a['resolutions']}",
        f"- Exact duplicate files: {a['exact_duplicate_files']}",
        f"- Subject metadata: {a['subject_metadata']}",
        f"- Sessions: {a['sessions']}",
        f"- Face detection: {a['face_detection']}; quality: {a['quality']} {a['quality_rejection_reasons']}",
        f"- Split ({a['split_method']}): {a['split_counts']}",
        f"- Nearest perceptual-hash distance to any training image: val {a['nearest_dhash_bits_val_to_train']} bits, "
        f"test {a['nearest_dhash_bits_test_to_train']} bits (of 64; <=6 means near-duplicate)",
        "",
        "## Shortcut probes (logistic regression, trained on train, accuracy on test)",
        "",
        "| Features (no facial information) | Test accuracy |",
        "|---|---|",
    ] + [f"| {k} | {v} |" for k, v in p.items()] + [
        "",
        "Accuracy near 1.0 means the classes can be separated without looking at the face.",
        "",
        "## Mean crop statistics per class",
        "",
        "| Statistic | " + " | ".join(s) + " |",
        "|---|" + "---|" * len(s),
    ] + [f"| {k} | " + " | ".join(str(s[c][k]) for c in s) + " |" for k in next(iter(s.values()))] + [
        "",
        "Visual inspection (Phase 2A contact sheet): `non_alcoholic` frames show glasses, cooler/brighter",
        "lighting and a different background; `alcoholic` frames show no glasses and warm, dimmer lighting.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=config.PROCESSED_DIR)
    args = parser.parse_args()
    _, audit = build(out_dir=args.out)
    print(json.dumps({k: audit[k] for k in ("images_per_class", "quality", "split_counts",
                                            "nearest_dhash_bits_test_to_train",
                                            "shortcut_probe_test_accuracy")}, indent=1))
