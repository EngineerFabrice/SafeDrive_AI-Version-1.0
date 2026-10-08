"""Select the impairment model from configuration (``MODEL_PROVIDER``).

    MODEL_PROVIDER=alcohol_mobilenetv3  (default) MobileNetV3 face classifier trained by
                                        training/train_mobilenet.py (ALCOHOL_MODEL_PATH overrides the file)
    MODEL_PROVIDER=mock                 MockImpairmentModel (DEVELOPMENT_ONLY; never feeds safety decisions)
    MODEL_PROVIDER=none                 no model; the live pipeline runs without predictions

A missing artefact or a failing provider yields an UnavailableImpairmentModel,
so monitoring keeps running and reports the model as unavailable.
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


def _alcohol_mobilenetv3(provider: str) -> ImpairmentModel:
    from .alcohol_model import AlcoholImageModel
    return AlcoholImageModel(provider=provider)


DEFAULT_PROVIDER = "alcohol_mobilenetv3"

PROVIDERS: Dict[str, Callable[[str], ImpairmentModel]] = {
    "mock": _mock,
    "alcohol_mobilenetv3": _alcohol_mobilenetv3,
}


def create_impairment_model(provider: Optional[str] = None) -> Optional[ImpairmentModel]:
    """Model for ``provider`` (default: $MODEL_PROVIDER). None when disabled.

    Never raises for configuration problems: an unknown or failing provider
    yields an UnavailableImpairmentModel so the live pipeline keeps running.
    """
    name = (os.environ.get(ENV_VAR, DEFAULT_PROVIDER) if provider is None else provider).strip().lower()
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
    if model.info.is_mock:
        log.warning("Impairment model %s is a MOCK; its output is synthetic", model.info.name)
    elif model.info.development_only:
        log.warning("Impairment model %s is a DEVELOPMENT_ONLY prototype; not validated on independent subjects",
                    model.info.name)
    return model
