"""Dataset split rules and metrics used by the training scripts."""
import csv
import os

import numpy as np
import pytest

from training import config
from training.metrics import binary_metrics
from training.prepare_dataset import frame_number, hamming, temporal_block_split


def test_temporal_block_split_is_contiguous_with_gaps():
    keys = list(range(100, 300))                    # 200 consecutive frames
    split = temporal_block_split(keys, (0.7, 0.15, 0.15), gap=5)
    order = [split[k] for k in sorted(keys)]
    # contiguous blocks in recording order, separated by buffers
    compact = [s for i, s in enumerate(order) if i == 0 or s != order[i - 1]]
    assert compact == ["train", "buffer", "val", "buffer", "test"]
    assert order.count("buffer") == 20               # 5 frames on each side of 2 boundaries
    # no training frame is within `gap` frames of a validation/test frame
    train = [k for k in keys if split[k] == "train"]
    held = [k for k in keys if split[k] in ("val", "test")]
    assert min(abs(a - b) for a in train for b in held) > 5


def test_split_without_gap_and_tiny_inputs():
    split = temporal_block_split(range(10), (0.7, 0.15, 0.15), gap=0)
    assert set(split.values()) <= {"train", "val", "test"} and "buffer" not in split.values()


def test_frame_number_parsing_and_hamming():
    assert frame_number("A0523.png") == 523 and frame_number("a0007.png") == 7 and frame_number("x.png") is None
    assert hamming(0b1011, 0b0001) == 2


def test_binary_metrics_positive_class_is_alcoholic():
    m = binary_metrics([0, 0, 1, 1, 1], [0.1, 0.8, 0.9, 0.7, 0.2])
    assert m["confusion_matrix"] == [[1, 1], [1, 2]]
    assert m["precision"] == pytest.approx(2 / 3, abs=1e-4) and m["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["specificity_negative"] == 0.5 and "roc_auc" in m


@pytest.mark.skipif(not os.path.isfile(config.MANIFEST), reason="processed dataset not built")
def test_generated_manifest_has_no_frame_crossing_splits():
    rows = list(csv.DictReader(open(config.MANIFEST, encoding="utf-8")))
    assert {"image_path", "label", "split", "session_id", "subject_id", "quality_status"} <= set(rows[0])
    for label in config.CLASSES:
        by_split = {s: [int(r["frame_number"]) for r in rows if r["label"] == label and r["split"] == s]
                    for s in ("train", "val", "test")}
        assert all(by_split.values())
        assert max(by_split["train"]) < min(by_split["val"]) <= max(by_split["val"]) < min(by_split["test"])
    assert all(r["subject_id"] == "unknown" for r in rows)          # limitation stays explicit
    assert np.isclose(sum(r["split"] == "train" for r in rows) / len(rows), 0.675, atol=0.05)
