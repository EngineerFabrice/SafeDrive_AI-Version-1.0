"""Paths and constants shared by the training scripts."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCE_ID = "AlcoholDetectionDataset"
SOURCE_DIR = os.path.join(ROOT, "AlcoholDetectionDataset")
CLASSES = ("non_alcoholic", "alcoholic")      # index 0 / 1; "alcoholic" is the positive class
POSITIVE_CLASS = "alcoholic"
# The dataset ships no documentation of how labels were assigned (see docs/phase3_dataset_audit.md).
LABEL_TYPE = "dataset_label_unverified"

PROCESSED_DIR = os.path.join(ROOT, "data", "processed", "alcohol_v1")
IMAGES_DIR = os.path.join(PROCESSED_DIR, "images")
MANIFEST = os.path.join(PROCESSED_DIR, "manifest.csv")
AUDIT_DIR = os.path.join(PROCESSED_DIR, "audit")

MODEL_DIR = os.path.join(ROOT, "models", "alcohol_mobilenetv3")
RUNS_DIR = os.path.join(MODEL_DIR, "runs")

SEED = 20261007
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)          # train / val / test, per class, in recording order
SPLIT_GAP = 5                                 # frames dropped at every split boundary (temporal buffer)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
