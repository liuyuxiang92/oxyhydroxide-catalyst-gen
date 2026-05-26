"""Global seed utilities for reproducible experiments."""
from __future__ import annotations

import random


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python random, NumPy, and PyTorch (CPU + GPU) for reproducibility.

    Parameters
    ----------
    seed:
        Integer seed value.
    deterministic:
        If True, enable full GPU determinism (cudnn.deterministic,
        use_deterministic_algorithms).  Slower but fully reproducible on GPU.
        Mirror of feat/classical-dqn's behaviour when --train-seed is explicit.
    """
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.use_deterministic_algorithms(True)
            torch.set_num_threads(1)
    except ImportError:
        pass
