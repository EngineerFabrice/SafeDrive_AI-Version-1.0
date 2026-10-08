"""Model interface contract: schema checks, error isolation, provider registry, mock flags."""
import math

import numpy as np
import pytest

from engine.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from engine.impairment import (ImpairmentInput, ImpairmentModel, ImpairmentStatus, ModelInfo,
                               create_impairment_model)

N = len(FEATURE_NAMES)


def vec(value=0.1):
    return np.full(N, value, np.float32)


class Constant(ImpairmentModel):
    def __init__(self, fail=False):
        self.fail = fail

    @property
    def info(self):
        return ModelInfo(name="const", version="1", provider="test",
                         schema_version=FEATURE_SCHEMA_VERSION, classes=("A", "B"))

    def _predict(self, vector):
        if self.fail:
            raise RuntimeError("boom")
        return "A", {"A": 0.7, "B": 0.3}


def test_valid_prediction():
    r = Constant().predict(ImpairmentInput(vec(), FEATURE_SCHEMA_VERSION, frame_id=7))
    assert r.valid and r.prediction == "A" and r.frame_id == 7


@pytest.mark.parametrize("inp, status", [
    (ImpairmentInput(vec(), FEATURE_SCHEMA_VERSION + 1), ImpairmentStatus.SCHEMA_MISMATCH),
    (ImpairmentInput(np.zeros(N - 1, np.float32), FEATURE_SCHEMA_VERSION), ImpairmentStatus.SCHEMA_MISMATCH),
    (ImpairmentInput(vec(), FEATURE_SCHEMA_VERSION, feature_names=tuple(reversed(FEATURE_NAMES))),
     ImpairmentStatus.SCHEMA_MISMATCH),
    (ImpairmentInput(vec(math.nan), FEATURE_SCHEMA_VERSION), ImpairmentStatus.INVALID_INPUT),
    (ImpairmentInput(vec(math.inf), FEATURE_SCHEMA_VERSION), ImpairmentStatus.INVALID_INPUT),
])
def test_bad_input_is_rejected_before_the_model(inp, status):
    r = Constant().predict(inp)
    assert r.status == status and not r.valid and r.prediction is None


def test_model_exception_never_looks_like_a_prediction():
    r = Constant(fail=True).predict(ImpairmentInput(vec(), FEATURE_SCHEMA_VERSION))
    assert r.status == ImpairmentStatus.MODEL_ERROR and "boom" in r.error and r.prediction is None


def test_registry_disabled_unknown_and_mock():
    assert create_impairment_model("none") is None
    unknown = create_impairment_model("does-not-exist")
    assert not unknown.available
    r = unknown.predict(ImpairmentInput(vec(), FEATURE_SCHEMA_VERSION))
    assert r.status == ImpairmentStatus.MODEL_UNAVAILABLE

    mock = create_impairment_model("mock")
    assert mock.info.is_mock and mock.info.development_only
    out = mock.predict(ImpairmentInput(vec(), FEATURE_SCHEMA_VERSION))
    assert out.valid and out.prediction.startswith("MOCK_") and "DEVELOPMENT_ONLY" in out.note
    assert out.to_dict()["is_mock"] is True
