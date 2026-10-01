"""DEVELOPMENT-ONLY synthetic dataset in the Phase 2 feature schema.

The values are random numbers in plausible ranges, generated from a fixed
seed. No real person, face or measurement is involved. ``MOCK_LEVEL_k``
targets are arbitrary. A small artificial shift is added to a few features
per level so that training and evaluation code can be smoke-tested end to
end; it has no relation to alcohol or impairment.

DATASET_TYPE=MOCK, NOT_FOR_RESEARCH_CLAIMS=true.
"""

import csv
import json
import math
import os
from typing import Optional

import numpy as np

from ..features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from ..impairment.interface import SchemaMismatchError
from .interface import DatasetInfo, DatasetProvider, FeatureDataset

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MOCK_DIR = os.path.join(_REPO_ROOT, "data", "mock", "phase3")
FEATURES_FILE = "features.csv"
METADATA_FILE = "metadata.json"
META_COLUMNS = ("subject_id", "session_id", "timestamp", "frame_id", "condition", "bac", "target")

MOCK_CLASSES = ("MOCK_LEVEL_0", "MOCK_LEVEL_1", "MOCK_LEVEL_2")
MOCK_CLASS_DEFINITIONS = {c: f"Synthetic label {i}; no real-world meaning." for i, c in enumerate(MOCK_CLASSES)}

# Plausible (mean, std) per feature for random generation; same order as FEATURE_NAMES.
_RANGES = {
    "head_pose.pitch": (0, 8), "head_pose.yaw": (0, 10), "head_pose.roll": (0, 5),
    "head_pose.yaw_ratio": (0, 0.08), "head_pose.pitch_ratio": (0.5, 0.06), "head_pose.roll_2d": (0, 5),
    "head_pose.reprojection_error": (0.03, 0.01), "head_pose.confidence": (0.7, 0.1),
    "gaze.horizontal": (0, 0.3), "gaze.vertical": (0.3, 0.2), "gaze.horizontal_left": (0, 0.3),
    "gaze.horizontal_right": (0, 0.3), "gaze.eye_disagreement": (0.2, 0.1), "gaze.eye_contrast": (0.2, 0.05),
    "appearance.skin_a": (15, 5), "appearance.skin_red_ratio": (0.45, 0.04),
    "appearance.eye_region_a_rel": (-5, 5), "appearance.brightness": (40, 10),
    "appearance.contrast": (12, 3), "appearance.sharpness": (150, 60),
    "facial_motion.face_speed": (0.1, 0.05), "facial_motion.face_dx": (0, 0.01),
    "facial_motion.face_dy": (0, 0.01), "facial_motion.scale_change": (0, 0.1),
    "facial_motion.landmark_speed": (0.2, 0.1), "facial_motion.landmark_deformation": (0.1, 0.05),
    "facial_motion.pitch_rate": (0, 10), "facial_motion.yaw_rate": (0, 10), "facial_motion.roll_rate": (0, 5),
}
# Arbitrary artificial per-level shift (in std units) so pipelines have something to learn.
_ARTIFICIAL_SHIFT = {"head_pose.pitch": 0.6, "gaze.horizontal": -0.5, "facial_motion.landmark_speed": 0.7}


def generate_mock_dataset(out_dir: str = DEFAULT_MOCK_DIR, seed: int = 20261001,
                          n_subjects: int = 8, frames_per_session: int = 30, fps: float = 30.0) -> str:
    """Write the synthetic dataset (CSV + metadata JSON) and return the directory."""
    assert tuple(_RANGES) == FEATURE_NAMES, "mock ranges must follow FEATURE_NAMES order"
    rng = np.random.default_rng(seed)
    means = np.array([m for m, _ in _RANGES.values()])
    stds = np.array([s for _, s in _RANGES.values()])
    shift = np.array([_ARTIFICIAL_SHIFT.get(n, 0.0) for n in FEATURE_NAMES])
    gaze = np.array([n.startswith("gaze.") for n in FEATURE_NAMES])
    motion = np.array([n.startswith("facial_motion.") for n in FEATURE_NAMES])

    os.makedirs(out_dir, exist_ok=True)
    base_time = 1_767_225_600.0  # 2026-01-01T00:00:00Z, synthetic
    rows = []
    for s in range(1, n_subjects + 1):
        subject = f"MOCK_S{s:02d}"
        subject_offset = rng.normal(0, 0.5, len(FEATURE_NAMES)) * stds   # per-subject baseline
        for level, target in enumerate(MOCK_CLASSES):
            session = f"{subject}_T{level + 1}"
            start = base_time + s * 86_400 + level * 3_600
            for f in range(frames_per_session):
                x = means + subject_offset + level * shift * stds + rng.normal(0, 1, len(FEATURE_NAMES)) * stds
                if f == 0:
                    x[motion] = np.nan          # no previous frame, as in Phase 2
                if rng.random() < 0.1:
                    x[gaze] = np.nan            # gaze group sometimes unavailable, as in Phase 2
                rows.append([subject, session, f"{start + f / fps:.3f}", f, f"MOCK_CONDITION_{level}", "",
                             target] + ["" if math.isnan(v) else f"{v:.6g}" for v in x])

    with open(os.path.join(out_dir, FEATURES_FILE), "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(list(META_COLUMNS) + list(FEATURE_NAMES))
        writer.writerows(rows)

    metadata = {
        "DATASET_TYPE": "MOCK",
        "NOT_FOR_RESEARCH_CLAIMS": True,
        "name": "safedrive-phase3-mock",
        "description": "Synthetic random feature vectors for development and testing only. "
                       "No real people, faces or measurements. Targets and the artificial "
                       "per-level shift are arbitrary and carry no meaning about alcohol or impairment.",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "target_name": "impairment_class",
        "classes": list(MOCK_CLASSES),
        "class_definitions": MOCK_CLASS_DEFINITIONS,
        "bac": "not applicable (empty) in the mock dataset",
        "generator": {"function": "engine.datasets.mock_provider.generate_mock_dataset", "seed": seed,
                      "n_subjects": n_subjects, "sessions_per_subject": len(MOCK_CLASSES),
                      "frames_per_session": frames_per_session, "fps": fps,
                      "artificial_shift_std_units_per_level": _ARTIFICIAL_SHIFT},
        "rows": len(rows),
    }
    with open(os.path.join(out_dir, METADATA_FILE), "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    return out_dir


class MockDatasetProvider(DatasetProvider):
    name = "mock"

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir or os.environ.get("MOCK_DATASET_DIR", DEFAULT_MOCK_DIR)
        self._metadata: Optional[dict] = None

    def _read_metadata(self) -> dict:
        if self._metadata is None:
            with open(os.path.join(self.data_dir, METADATA_FILE), encoding="utf-8") as fh:
                meta = json.load(fh)
            if meta.get("DATASET_TYPE") != "MOCK" or meta.get("NOT_FOR_RESEARCH_CLAIMS") is not True:
                raise ValueError("mock dataset metadata must declare DATASET_TYPE=MOCK and "
                                 "NOT_FOR_RESEARCH_CLAIMS=true")
            self._metadata = meta
        return self._metadata

    def info(self) -> DatasetInfo:
        meta = self._read_metadata()
        return DatasetInfo(
            name=meta["name"], dataset_type=meta["DATASET_TYPE"],
            not_for_research_claims=meta["NOT_FOR_RESEARCH_CLAIMS"],
            schema_version=meta["feature_schema_version"], feature_names=tuple(meta["feature_names"]),
            classes=tuple(meta["classes"]), class_definitions=dict(meta["class_definitions"]),
            target_name=meta["target_name"], source="synthetic (generate_mock_dataset)",
            license="project-internal synthetic data", description=meta["description"],
            extra={"generator": meta.get("generator", {})})

    def _load(self) -> FeatureDataset:
        info = self.info()
        with open(os.path.join(self.data_dir, FEATURES_FILE), newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = tuple(next(reader))
            rows = list(reader)
        if header[:len(META_COLUMNS)] != META_COLUMNS or header[len(META_COLUMNS):] != FEATURE_NAMES:
            raise SchemaMismatchError("features.csv columns do not match META_COLUMNS + FEATURE_NAMES")

        def num(v):
            return float(v) if v != "" else np.nan

        cols = list(zip(*rows)) if rows else [()] * len(header)
        k = len(META_COLUMNS)
        return FeatureDataset(
            info=info,
            X=np.array([[num(v) for v in r[k:]] for r in rows], dtype=np.float32).reshape(len(rows), -1),
            y=np.array(cols[6], dtype=object),
            subject_ids=np.array(cols[0], dtype=object),
            session_ids=np.array(cols[1], dtype=object),
            timestamps=np.array([float(v) for v in cols[2]]),
            frame_ids=np.array([int(v) for v in cols[3]]),
            conditions=np.array(cols[4], dtype=object),
            bac=np.array([num(v) for v in cols[5]]),
        )
