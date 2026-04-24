from __future__ import annotations

from typing import Sequence

import numpy as np


def one_hot_index(index: int, size: int) -> np.ndarray:
    v = np.zeros(size, dtype=float)
    v[index] = 1.0
    return v


def encode_choice(choice: str, choices: Sequence[str]) -> np.ndarray:
    try:
        idx = list(choices).index(choice)
    except ValueError as e:
        raise ValueError(f"Unknown choice '{choice}'.") from e
    return one_hot_index(idx, len(choices))


def decode_one_hot(vec: Sequence[float], choices: Sequence[str]) -> str:
    arr = np.asarray(vec, dtype=float)
    if arr.ndim != 1 or arr.size != len(choices):
        raise ValueError("Bad one-hot shape.")
    idx = int(arr.argmax())
    return list(choices)[idx]
