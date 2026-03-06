from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest
import torch


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture(autouse=True)
def _set_deterministic_seeds() -> Iterator[None]:
    """Set deterministic seeds for unit tests.

    Note: Some stochasticity remains due to multinomial sampling in model code,
    but fixed seeds make failures reproducible.
    """
    torch.manual_seed(0)
    np.random.seed(0)
    yield

