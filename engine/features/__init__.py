"""Alcohol-related visual feature extraction (Phase 2).

Produces numeric per-frame features for a later trained impairment model.
No module here classifies impairment; no single feature indicates intoxication.
"""

from .extractor import (FEATURE_NAMES, FEATURE_SCHEMA_VERSION, FaceFeatures, FeatureExtractor,
                        FeatureExtractorConfig, FeatureStatus)

__all__ = ["FEATURE_NAMES", "FEATURE_SCHEMA_VERSION", "FaceFeatures", "FeatureExtractor",
           "FeatureExtractorConfig", "FeatureStatus"]
