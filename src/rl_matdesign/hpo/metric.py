"""Metric helpers for the HPO driver — read a run's `generated.csv` and
compute the scalar objective Optuna will maximize.

The `reward` column in `generated.csv` is always "higher is better" regardless
of the predictor's `objective` setting (mean / mean_minus_kstd / mean_plus_kstd) —
the env reward sign and `objective_from_mean_std()` already canonicalize that
direction. The driver therefore always passes ``direction="maximize"`` to Optuna.
"""

from __future__ import annotations

import csv
import logging
import math
import os
from typing import Tuple


logger = logging.getLogger(__name__)


def top_k_mean_from_csv(csv_path: str, k: int) -> Tuple[float, int]:
    """Return ``(top_k_mean, n_rows_used)``.

    Reads the ``reward`` column from a `generated.csv`, drops NaNs, sorts
    descending, averages the top ``min(k, n)`` values. Logs a warning when
    fewer than ``k`` rows are available.

    Raises FileNotFoundError if the CSV doesn't exist (caller decides whether
    to treat that as a trial failure). Raises ValueError on an empty/malformed
    file with no usable reward values.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    rewards = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "reward" not in reader.fieldnames:
            raise ValueError(f"{csv_path}: missing 'reward' column (fields={reader.fieldnames})")
        for row in reader:
            raw = row.get("reward", "")
            if raw == "" or raw is None:
                continue
            try:
                v = float(raw)
            except ValueError:
                continue
            if math.isnan(v) or math.isinf(v):
                continue
            rewards.append(v)
    if not rewards:
        raise ValueError(f"{csv_path}: no usable reward values found")
    rewards.sort(reverse=True)
    used = min(k, len(rewards))
    if len(rewards) < k:
        logger.warning(
            "top_k_mean: requested K=%d but %s has only %d usable rows; using top-%d",
            k, csv_path, len(rewards), used,
        )
    return (sum(rewards[:used]) / used, used)
