"""Contract for datasets that supply Phase 2 feature vectors for model work.

A provider turns some source (a synthetic file today; later an audited
research dataset) into the same internal representation: one row per frame
with subject, session, time, the 29-feature vector in FEATURE_NAMES order,
and the target. Training and evaluation code only ever sees
``FeatureDataset``, so swapping the provider does not change it.

``load()`` validates the schema, so an incompatible dataset is rejected.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from ..features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from ..impairment.interface import TARGET_NAME, SchemaMismatchError


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    dataset_type: str                       # "MOCK" or "RESEARCH"
    not_for_research_claims: bool
    schema_version: int
    feature_names: Tuple[str, ...]
    classes: Tuple[str, ...]
    class_definitions: Dict[str, str]       # what each target value means, in plain words
    target_name: str = TARGET_NAME
    source: str = ""
    license: str = ""
    description: str = ""
    extra: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class FeatureDataset:
    """Frame-level feature table. Row i of every array describes the same frame."""
    info: DatasetInfo
    X: np.ndarray                   # (n, 29) float32, NaN = missing
    y: np.ndarray                   # (n,) target class strings
    subject_ids: np.ndarray         # (n,) str: the grouping unit for splits
    session_ids: np.ndarray         # (n,) str: trip / recording
    timestamps: np.ndarray          # (n,) float seconds
    frame_ids: np.ndarray           # (n,) int
    conditions: np.ndarray          # (n,) str: protocol condition as recorded by the source
    bac: np.ndarray                 # (n,) float g/100ml; NaN when not measured / not applicable

    def __len__(self) -> int:
        return len(self.y)

    @property
    def subjects(self) -> Tuple[str, ...]:
        return tuple(sorted(set(self.subject_ids.tolist())))

    def sessions(self, subject_id: Optional[str] = None) -> Tuple[str, ...]:
        mask = self.subject_ids == subject_id if subject_id is not None else slice(None)
        return tuple(sorted(set(self.session_ids[mask].tolist())))


def validate_dataset(ds: FeatureDataset) -> FeatureDataset:
    info = ds.info
    if info.schema_version != FEATURE_SCHEMA_VERSION:
        raise SchemaMismatchError(
            f"dataset schema v{info.schema_version} != engine schema v{FEATURE_SCHEMA_VERSION}")
    if tuple(info.feature_names) != FEATURE_NAMES:
        raise SchemaMismatchError("dataset feature names/order differ from FEATURE_NAMES")
    n = len(ds.y)
    if ds.X.shape != (n, len(FEATURE_NAMES)):
        raise SchemaMismatchError(f"X has shape {ds.X.shape}, expected ({n}, {len(FEATURE_NAMES)})")
    for name in ("subject_ids", "session_ids", "timestamps", "frame_ids", "conditions", "bac"):
        if len(getattr(ds, name)) != n:
            raise ValueError(f"{name} has {len(getattr(ds, name))} rows, expected {n}")
    if np.isinf(ds.X).any():
        raise ValueError("X contains infinite values")
    unknown = set(ds.y.tolist()) - set(info.classes)
    if unknown:
        raise ValueError(f"targets not declared in info.classes: {sorted(unknown)}")
    return ds


class DatasetProvider(ABC):
    """Source of a FeatureDataset. Implementations: MockDatasetProvider (now),
    KeshtkaranDatasetProvider (future, after the dataset is obtained and audited)."""

    name: str = ""

    @abstractmethod
    def info(self) -> DatasetInfo:
        ...

    @abstractmethod
    def _load(self) -> FeatureDataset:
        ...

    def load(self) -> FeatureDataset:
        return validate_dataset(self._load())
