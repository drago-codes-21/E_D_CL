from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def set_global_seed(seed: int, deterministic_torch: bool = False) -> None:
    """
    Set seeds across random generators to improve reproducibility.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
