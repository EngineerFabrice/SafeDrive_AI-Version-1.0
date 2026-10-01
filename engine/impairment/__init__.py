"""Impairment-model interface (Phase 3 integration foundation).

Application code depends on ``ImpairmentModel`` and ``create_impairment_model``
only, never on a concrete model. See docs/phase3_integration.md.
"""

from .interface import (TARGET_NAME, ImpairmentInput, ImpairmentModel, ImpairmentResult,
                        ImpairmentStatus, ModelInfo, SchemaMismatchError, check_vector)
from .registry import create_impairment_model

__all__ = ["TARGET_NAME", "ImpairmentInput", "ImpairmentModel", "ImpairmentResult",
           "ImpairmentStatus", "ModelInfo", "SchemaMismatchError", "check_vector",
           "create_impairment_model"]
