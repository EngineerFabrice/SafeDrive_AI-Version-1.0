"""Contract for impairment models that consume Phase 2 feature vectors.

Every model, whether the development mock or a future model trained on an
audited dataset, implements ``ImpairmentModel`` and is called only through
``predict()``. The base class checks the input against the frozen Phase 2
schema before any model code runs, so an incompatible vector is rejected
instead of silently scored.

A result is a model output for one frame. It is not a risk level, alert or
final verdict; those belong to later phases.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ..features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, FaceFeatures

TARGET_NAME = "impairment_class"   # neutral name; class meanings come from the model/dataset metadata


class ImpairmentStatus(str, Enum):
    PREDICTION_AVAILABLE = "PREDICTION_AVAILABLE"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
    INVALID_INPUT = "INVALID_INPUT"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    MODEL_ERROR = "MODEL_ERROR"


@dataclass(frozen=True)
class ModelInfo:
    name: str
    version: str
    provider: str                       # value of MODEL_PROVIDER that selected it
    schema_version: int                 # feature schema the model was built for
    classes: Tuple[str, ...]
    target_name: str = TARGET_NAME
    is_mock: bool = False
    development_only: bool = False
    description: str = ""
    input_type: str = "features"        # "features" (29-value vector) or "face_image" (aligned BGR crop)
    # Class whose probability means "potentially not sober". Only models that declare it feed the
    # temporal decision engine; the mock model leaves it None, so its output can never raise a warning.
    positive_class: Optional[str] = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["classes"] = list(self.classes)
        return data


@dataclass(frozen=True)
class ImpairmentInput:
    """One frame's model input: the Phase 2 feature vector and/or the aligned face crop."""
    vector: np.ndarray                  # shape (29,), FEATURE_NAMES order, NaN = missing
    schema_version: int
    timestamp: Optional[float] = None   # wall clock (epoch s) of the source frame
    frame_id: Optional[int] = None
    feature_names: Optional[Tuple[str, ...]] = None  # optional extra check of the order
    face_image: Optional[np.ndarray] = None          # aligned BGR uint8 crop (engine.preprocessing)

    @classmethod
    def from_features(cls, features: FaceFeatures, face_image: Optional[np.ndarray] = None) -> "ImpairmentInput":
        return cls(vector=features.to_vector(), schema_version=FEATURE_SCHEMA_VERSION,
                   timestamp=features.timestamp, frame_id=features.frame_id,
                   feature_names=FEATURE_NAMES, face_image=face_image)

    @classmethod
    def from_face(cls, face_image: np.ndarray, timestamp: Optional[float] = None,
                  frame_id: Optional[int] = None) -> "ImpairmentInput":
        """Input for an image model when no feature vector is available."""
        return cls(vector=np.full(len(FEATURE_NAMES), np.nan, np.float32), schema_version=FEATURE_SCHEMA_VERSION,
                   timestamp=timestamp, frame_id=frame_id, face_image=face_image)


@dataclass(frozen=True)
class ImpairmentResult:
    status: ImpairmentStatus
    model: ModelInfo
    input_schema_version: Optional[int] = None
    prediction: Optional[str] = None
    probabilities: Optional[Dict[str, float]] = None
    timestamp: Optional[float] = None
    frame_id: Optional[int] = None
    error: str = ""
    note: str = ""
    created_at: float = field(default_factory=time.time)

    @property
    def valid(self) -> bool:
        return self.status == ImpairmentStatus.PREDICTION_AVAILABLE

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "valid": self.valid,
            "target": self.model.target_name,
            "prediction": self.prediction,
            "probabilities": ({k: round(float(v), 4) for k, v in self.probabilities.items()}
                              if self.probabilities else None),
            "model_name": self.model.name,
            "model_version": self.model.version,
            "model_provider": self.model.provider,
            "is_mock": self.model.is_mock,
            "development_only": self.model.development_only,
            "schema_version": self.model.schema_version,
            "input_schema_version": self.input_schema_version,
            "timestamp": self.timestamp,
            "frame_id": self.frame_id,
            "error": self.error,
            "note": self.note,
        }


class SchemaMismatchError(ValueError):
    pass


def check_face_image(image) -> np.ndarray:
    """Validate an aligned face crop: HxWx3 uint8, at least 16x16."""
    if image is None:
        raise ValueError("face image missing")
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.shape[0] < 16 or arr.shape[1] < 16:
        raise ValueError(f"face image must be HxWx3 (got shape {arr.shape})")
    if arr.dtype != np.uint8:
        raise ValueError(f"face image must be uint8 (got {arr.dtype})")
    return arr


def check_vector(vector, schema_version: int,
                 feature_names: Optional[Sequence[str]] = None) -> np.ndarray:
    """Validate a feature vector against the frozen Phase 2 schema; returns it as float32.

    Raises SchemaMismatchError (wrong version / names / length) or ValueError (bad values).
    """
    if schema_version != FEATURE_SCHEMA_VERSION:
        raise SchemaMismatchError(
            f"feature schema v{schema_version} is not supported (expected v{FEATURE_SCHEMA_VERSION})")
    if feature_names is not None and tuple(feature_names) != FEATURE_NAMES:
        raise SchemaMismatchError("feature names/order differ from FEATURE_NAMES")
    try:
        arr = np.asarray(vector, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"feature vector is not numeric: {exc}") from None
    if arr.shape != (len(FEATURE_NAMES),):
        raise SchemaMismatchError(f"expected shape ({len(FEATURE_NAMES)},), got {arr.shape}")
    if np.isinf(arr).any():
        raise ValueError("feature vector contains infinite values")
    if np.isnan(arr).all():
        raise ValueError("feature vector has no available values")
    return arr


class ImpairmentModel(ABC):
    """Base class for all impairment models. Subclasses implement ``_predict`` only."""

    @property
    @abstractmethod
    def info(self) -> ModelInfo:
        ...

    @property
    def available(self) -> bool:
        return True

    def predict(self, inp: ImpairmentInput) -> ImpairmentResult:
        base = dict(model=self.info, input_schema_version=inp.schema_version,
                    timestamp=inp.timestamp, frame_id=inp.frame_id)
        if not self.available:
            return ImpairmentResult(status=ImpairmentStatus.MODEL_UNAVAILABLE,
                                    error=self.unavailable_reason, **base)
        try:
            if self.info.input_type == "face_image":
                model_input = check_face_image(inp.face_image)
            else:
                model_input = check_vector(inp.vector, inp.schema_version, inp.feature_names)
                if self.info.schema_version != FEATURE_SCHEMA_VERSION:
                    raise SchemaMismatchError(
                        f"model expects schema v{self.info.schema_version}, engine produces v{FEATURE_SCHEMA_VERSION}")
        except SchemaMismatchError as exc:
            return ImpairmentResult(status=ImpairmentStatus.SCHEMA_MISMATCH, error=str(exc), **base)
        except ValueError as exc:
            return ImpairmentResult(status=ImpairmentStatus.INVALID_INPUT, error=str(exc), **base)
        try:
            prediction, probabilities = self._predict(model_input)
        except Exception as exc:  # a model fault must never look like a prediction
            return ImpairmentResult(status=ImpairmentStatus.MODEL_ERROR,
                                    error=f"{type(exc).__name__}: {exc}", **base)
        return ImpairmentResult(status=ImpairmentStatus.PREDICTION_AVAILABLE, prediction=prediction,
                                probabilities=probabilities, note=self.result_note, **base)

    @property
    def unavailable_reason(self) -> str:
        return ""

    @property
    def result_note(self) -> str:
        return ""

    @abstractmethod
    def _predict(self, model_input: np.ndarray) -> Tuple[str, Optional[Dict[str, float]]]:
        """Return (predicted class, class probabilities or None) for validated input.

        ``model_input`` is the feature vector (may contain NaN for missing features) for
        ``input_type="features"`` models, or the aligned BGR face crop for ``"face_image"`` models.
        """


class UnavailableImpairmentModel(ImpairmentModel):
    """Stand-in when the configured provider cannot be loaded; always MODEL_UNAVAILABLE."""

    def __init__(self, provider: str, reason: str):
        self._info = ModelInfo(name="unavailable", version="-", provider=provider,
                               schema_version=FEATURE_SCHEMA_VERSION, classes=())
        self._reason = reason

    @property
    def info(self) -> ModelInfo:
        return self._info

    @property
    def available(self) -> bool:
        return False

    @property
    def unavailable_reason(self) -> str:
        return self._reason

    def _predict(self, vector):  # never called: predict() returns MODEL_UNAVAILABLE first
        raise RuntimeError(self._reason)
