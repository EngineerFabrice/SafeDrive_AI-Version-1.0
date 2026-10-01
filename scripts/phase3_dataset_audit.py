"""Phase 3 dataset audit: can the bundled datasets support an impairment model?

Read-only. Measures duplicates, augmentation groups, frame sequences, source
shortcuts and Phase 2 feature availability, and prints a JSON summary.
No model is trained here.

    python scripts/phase3_dataset_audit.py [--out audit.json]
"""

import argparse
import collections
import glob
import hashlib
import json
import os
import re
import sys
import warnings

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.detectors import BoundingBox                      # noqa: E402
from engine.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION  # noqa: E402
from engine.pipeline import MonitoringPipeline                # noqa: E402

DATASETS = {
    "AlcoholDetectionDataset": ("alcoholic", "non_alcoholic"),
    "DrunkingDetectionDataset": ("drunk", "sober"),
}
NEAR_DUP_BITS = 6  # dHash Hamming distance (of 64) treated as near-duplicate


def dhash(img, size=8):
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (size + 1, size))
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a, b):
    return bin(a ^ b).count("1")


def group_key(dataset, path):
    """Best available grouping of images that share an origin."""
    name = os.path.basename(path)
    if dataset == "DrunkingDetectionDataset" and ".rf." in name:
        # Roboflow export: "<base><n>_<ext>.rf.<hash>.jpg"; B, B1, B2, B3 are copies of one image
        base = re.sub(r"\.rf\..*$", "", name)
        base = re.sub(r"_(jpe?g|png)$", "", base, flags=re.I)
        return re.sub(r"(B)\d*$", r"\1", base).lower()
    return None


def clusters(hashes, max_bits):
    """Union-find clusters of near-duplicate images."""
    parent = list(range(len(hashes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            if hamming(hashes[i], hashes[j]) <= max_bits:
                parent[find(i)] = find(j)
    return [find(i) for i in range(len(hashes))]


def audit_dataset(dataset, classes, pipeline):
    rows = []
    for cls in classes:
        for path in sorted(glob.glob(os.path.join(dataset, cls, "*"))):
            img = cv2.imread(path)
            rows.append(dict(path=path, cls=cls, img=img))

    report = {"classes": {}, "invalid_files": [r["path"] for r in rows if r["img"] is None]}
    rows = [r for r in rows if r["img"] is not None]
    for r in rows:
        r["md5"] = hashlib.md5(open(r["path"], "rb").read()).hexdigest()
        r["dhash"] = dhash(r["img"])
        r["shape"] = r["img"].shape[:2]
        r["group"] = group_key(dataset, r["path"])

    # exact and near duplicates (within and across classes)
    by_md5 = collections.defaultdict(list)
    for r in rows:
        by_md5[r["md5"]].append(r)
    cl = clusters([r["dhash"] for r in rows], NEAR_DUP_BITS)
    by_cluster = collections.defaultdict(list)
    for r, c in zip(rows, cl):
        by_cluster[c].append(r)
    cross_class = sum(1 for g in by_cluster.values() if len({r["cls"] for r in g}) > 1)

    # Phase 2 features through the real pipeline, and directly on the full image
    for r in rows:
        img = r["img"]
        pipeline.roi_selector.reset()
        pipeline.feature_extractor.reset()
        a = pipeline.process_frame(img)
        r["person"] = a.driver is not None
        face = a.face
        if face is None:  # dataset images are often tight face crops: search the whole image
            face = pipeline.face_detector.detect(img, BoundingBox(0, 0, img.shape[1], img.shape[0]))
        r["face"] = face is not None
        if face is not None:
            pipeline.feature_extractor.reset()
            f = pipeline.feature_extractor.extract(img, face, 0.0)
            r["features_valid"] = f.valid
            r["groups_valid"] = {g: getattr(f, g).valid for g in ("head_pose", "gaze", "appearance", "facial_motion")}
            r["vector"] = f.to_vector()
        else:
            r["features_valid"] = False

    for cls in classes:
        cr = [r for r in rows if r["cls"] == cls]
        groups = {r["group"] for r in cr if r["group"]}
        vectors = np.array([r["vector"] for r in cr if r.get("features_valid")])
        with warnings.catch_warnings():  # all-NaN columns (motion on stills) are expected
            warnings.simplefilter("ignore", RuntimeWarning)
            std_median = float(np.nanmedian(np.nanstd(vectors, axis=0))) if len(vectors) else None
        group_counts = collections.Counter()
        for r in cr:
            for g, ok in r.get("groups_valid", {}).items():
                group_counts[g] += ok
        report["classes"][cls] = {
            "files": len(cr),
            "image_shapes": {f"{h}x{w}": n for (h, w), n in collections.Counter(r["shape"] for r in cr).most_common(5)},
            "distinct_shapes": len({r["shape"] for r in cr}),
            "exact_duplicates": len(cr) - len({r["md5"] for r in cr}),
            "near_duplicate_clusters": len({c for r, c in zip(rows, cl) if r["cls"] == cls}),
            "augmentation_groups": len(groups) or None,
            "person_detected": sum(r["person"] for r in cr),
            "face_detected": sum(r["face"] for r in cr),
            "features_available": sum(r["features_valid"] for r in cr),
            "feature_groups_valid": dict(group_counts),
            "feature_std_median": std_median,
            "all_nan_features": [n for n, col in zip(FEATURE_NAMES, vectors.T) if np.all(np.isnan(col))] if len(vectors) else None,
        }
    report["cross_class_near_duplicate_clusters"] = cross_class

    # Source shortcut: can image size alone separate the classes?
    a, b = classes
    shapes_a = {r["shape"] for r in rows if r["cls"] == a}
    shapes_b = {r["shape"] for r in rows if r["cls"] == b}
    report["image_shape_overlap_between_classes"] = len(shapes_a & shapes_b)
    if len(shapes_b) == 1:
        only = next(iter(shapes_b))
        hits = sum((r["shape"] == only) == (r["cls"] == b) for r in rows)
        report["shape_rule_accuracy"] = round(hits / len(rows), 4)
        report["shape_rule"] = f"predict '{b}' iff image is {only[0]}x{only[1]}"

    # Frame-sequence check: consecutive file numbers that are near-identical
    if dataset == "AlcoholDetectionDataset":
        seq = {}
        for cls in classes:
            cr = sorted((r for r in rows if r["cls"] == cls), key=lambda r: r["path"])
            d = [hamming(x["dhash"], y["dhash"]) for x, y in zip(cr, cr[1:])]
            nums = [int(re.sub(r"\D", "", os.path.basename(r["path"]))) for r in cr]
            seq[cls] = {"file_number_range": [min(nums), max(nums)],
                        "median_dhash_distance_consecutive": float(np.median(d))}
        report["sequence_check"] = seq
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out")
    args = parser.parse_args()
    pipeline = MonitoringPipeline()
    pipeline.person_detector.load()
    pipeline.face_detector.load()
    pipeline.feature_extractor.load()
    result = {"feature_schema_version": FEATURE_SCHEMA_VERSION,
              "datasets": {d: audit_dataset(d, c, pipeline) for d, c in DATASETS.items()}}
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)


if __name__ == "__main__":
    main()
