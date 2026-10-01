"""DEVELOPMENT-ONLY mock impairment model.

Produces deterministic, clearly synthetic outputs so the application flow
(pipeline -> model interface -> API) can be built and tested before a real,
validated model exists. Its outputs are a fixed mathematical function of the
input vector. They carry **no information about alcohol or impairment** and
must never be used or reported as such.
"""

from typing import Dict, Tuple

import numpy as np

from ..features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from .interface import ImpairmentModel, ModelInfo

MOCK_CLASSES = ("MOCK_LEVEL_0", "MOCK_LEVEL_1", "MOCK_LEVEL_2")
MOCK_NOTE = "MOCK / DEVELOPMENT_ONLY: synthetic output, not an impairment determination."

# Fixed, arbitrary weights: weight[i, k] = sin((i + 1) * (k + 1)). Deterministic on every platform.
_WEIGHTS = np.sin(np.outer(np.arange(1, len(FEATURE_NAMES) + 1),
                           np.arange(1, len(MOCK_CLASSES) + 1))).astype(np.float64)


class MockImpairmentModel(ImpairmentModel):
    def __init__(self, provider: str = "mock"):
        self._info = ModelInfo(
            name="MockImpairmentModel", version="mock-1", provider=provider,
            schema_version=FEATURE_SCHEMA_VERSION, classes=MOCK_CLASSES,
            is_mock=True, development_only=True,
            description="MOCK / DEVELOPMENT_ONLY. Deterministic synthetic output for testing "
                        "the application flow; not trained on any data.")

    @property
    def info(self) -> ModelInfo:
        return self._info

    @property
    def result_note(self) -> str:
        return MOCK_NOTE

    def _predict(self, vector: np.ndarray) -> Tuple[str, Dict[str, float]]:
        x = np.nan_to_num(vector.astype(np.float64), nan=0.0)
        logits = np.tanh(0.1 * x) @ _WEIGHTS          # bounded, deterministic
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        return MOCK_CLASSES[int(np.argmax(probs))], dict(zip(MOCK_CLASSES, map(float, probs)))
