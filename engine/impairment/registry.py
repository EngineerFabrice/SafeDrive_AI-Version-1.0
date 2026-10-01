"""Select the impairment model from configuration (``MODEL_PROVIDER``).

    MODEL_PROVIDER=none   (default) no model; the live pipeline runs without predictions
    MODEL_PROVIDER=mock   MockImpairmentModel (DEVELOPMENT_ONLY)

A future real model is added by registering one more provider here (for
example ``artifact``, loading a validated model file). Nothing else in the
application changes.
"""

import logging
import os
from typing import Callable, Dict, Optional

from .interface import ImpairmentModel, UnavailableImpairmentModel

log = logging.getLogger(__name__)

ENV_VAR = "MODEL_PROVIDER"
DISABLED = ("", "none", "off", "disabled")


def _mock(provider: str) -> ImpairmentModel:
    from .mock_model import MockImpairmentModel
    return MockImpairmentModel(provider=provider)


PROVIDERS: Dict[str, Callable[[str], ImpairmentModel]] = {
    "mock": _mock,
    # "artifact": lambda p: ArtifactImpairmentModel(os.environ["MODEL_PATH"], provider=p),  # future
}


def create_impairment_model(provider: Optional[str] = None) -> Optional[ImpairmentModel]:
    """Model for ``provider`` (default: $MODEL_PROVIDER). None when disabled.

    Never raises for configuration problems: an unknown or failing provider
    yields an UnavailableImpairmentModel so the live pipeline keeps running.
    """
    name = (os.environ.get(ENV_VAR, "none") if provider is None else provider).strip().lower()
    if name in DISABLED:
        return None
    factory = PROVIDERS.get(name)
    if factory is None:
        reason = f"unknown {ENV_VAR} {name!r} (available: {', '.join(sorted(PROVIDERS))}, none)"
        log.error(reason)
        return UnavailableImpairmentModel(name, reason)
    try:
        model = factory(name)
    except Exception as exc:
        log.exception("Impairment model provider %r failed to load", name)
        return UnavailableImpairmentModel(name, f"{type(exc).__name__}: {exc}")
    if model.info.development_only:
        log.warning("Impairment model %s is DEVELOPMENT_ONLY; its output is synthetic", model.info.name)
    return model
