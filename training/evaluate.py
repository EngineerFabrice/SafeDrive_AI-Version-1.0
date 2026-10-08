"""Evaluate the exported model ONCE on the held-out test split, and measure CPU latency.

    python -m training.evaluate

Writes models/alcohol_mobilenetv3/test_report.json. Run after model selection only.
"""
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.impairment import ImpairmentInput                     # noqa: E402
from engine.impairment.alcohol_model import AlcoholImageModel     # noqa: E402
from training import config                                       # noqa: E402
from training.data import read_manifest                           # noqa: E402
from training.metrics import binary_metrics                       # noqa: E402


def main():
    model = AlcoholImageModel(os.path.join(config.MODEL_DIR, "model.pt"))
    rows = read_manifest(split="test")
    labels, probs, latencies = [], [], []
    for r in rows:
        crop = cv2.imread(os.path.join(config.ROOT, r["image_path"]))      # same BGR crops as live inference
        t0 = time.perf_counter()
        res = model.predict(ImpairmentInput.from_face(crop))
        latencies.append((time.perf_counter() - t0) * 1000)
        if not res.valid:
            raise RuntimeError(f"prediction failed for {r['image_path']}: {res.status} {res.error}")
        labels.append(config.CLASSES.index(r["label"]))
        probs.append(res.probabilities[config.POSITIVE_CLASS])

    metrics = binary_metrics(labels, probs)
    lat = np.array(latencies[3:] or latencies)                            # skip warm-up
    report = {
        "dataset": config.SOURCE_ID, "split": "test", "n": len(rows),
        "selected_stage": model.metadata.get("selected_stage"), "arch": model.metadata.get("arch"),
        "metrics": metrics,
        "latency_ms_cpu": {"mean": round(float(lat.mean()), 2), "p95": round(float(np.percentile(lat, 95)), 2)},
        "model_file_mb": round(os.path.getsize(os.path.join(config.MODEL_DIR, "model.pt")) / 1e6, 2),
        "interpretation": ("Test frames come from later parts of the SAME two recordings used for training "
                           "(one person per class). These numbers measure separation of those two recordings, "
                           "which image statistics alone already achieve (see data/processed/alcohol_v1/audit). "
                           "They are not evidence of detecting alcohol in new people and never a BAC measurement."),
    }
    with open(os.path.join(config.MODEL_DIR, "test_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
