"""Select the dataset provider from configuration (``DATASET_PROVIDER``).

    DATASET_PROVIDER=mock   (default) MockDatasetProvider: synthetic, DEVELOPMENT_ONLY

A future ``keshtkaran`` provider is added here once the dataset has been
obtained under its transfer agreement and audited.
"""

import os
from typing import Callable, Dict, Optional

from .interface import DatasetProvider

ENV_VAR = "DATASET_PROVIDER"


def _mock() -> DatasetProvider:
    from .mock_provider import MockDatasetProvider
    return MockDatasetProvider()


PROVIDERS: Dict[str, Callable[[], DatasetProvider]] = {
    "mock": _mock,
    # "keshtkaran": lambda: KeshtkaranDatasetProvider(os.environ["KESHTKARAN_DATA_DIR"]),  # future
}


def create_dataset_provider(provider: Optional[str] = None) -> DatasetProvider:
    name = (os.environ.get(ENV_VAR, "mock") if provider is None else provider).strip().lower()
    factory = PROVIDERS.get(name)
    if factory is None:
        raise ValueError(f"unknown {ENV_VAR} {name!r} (available: {', '.join(sorted(PROVIDERS))})")
    return factory()
