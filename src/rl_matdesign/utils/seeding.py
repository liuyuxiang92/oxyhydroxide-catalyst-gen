"""Global seed utilities for reproducible experiments."""
from __future__ import annotations

import random


def set_global_seed(seed: int) -> None:
    """Seed Python random, NumPy, and PyTorch (CPU + GPU) for reproducibility.

    Call this once at the start of each experiment run, before creating any
    models or environments.

    Parameters
    ----------
    seed:
        Integer seed value.  Use distinct values across runs (e.g. 0–4) for
        multi-seed statistical reporting.
    """
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
