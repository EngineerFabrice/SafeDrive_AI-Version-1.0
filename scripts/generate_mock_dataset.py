"""Regenerate the DEVELOPMENT-ONLY synthetic Phase 3 dataset (data/mock/phase3/).

    python scripts/generate_mock_dataset.py [--out DIR] [--seed N]

Output is deterministic for a given seed. DATASET_TYPE=MOCK, NOT_FOR_RESEARCH_CLAIMS=true.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.datasets.mock_provider import DEFAULT_MOCK_DIR, generate_mock_dataset  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=DEFAULT_MOCK_DIR)
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()
    print("wrote", generate_mock_dataset(args.out, seed=args.seed))
