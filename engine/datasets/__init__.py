"""Dataset-provider interface (Phase 3 integration foundation).

Training/evaluation code depends on ``DatasetProvider`` / ``FeatureDataset`` and
``create_dataset_provider`` only. See docs/phase3_integration.md.
"""

from .interface import DatasetInfo, DatasetProvider, FeatureDataset, validate_dataset
from .registry import create_dataset_provider

__all__ = ["DatasetInfo", "DatasetProvider", "FeatureDataset", "validate_dataset",
           "create_dataset_provider"]
