"""RunMetrics — structured metrics collector for RL experiments.

Replaces the ad-hoc ``list[dict]`` pattern in the training/generation loops.
All evaluation metrics (Top-K, diversity, hit rate, Pareto front for
exploitation vs. exploration trade-off) are computed from a single collector.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RunMetrics:
    """Collect per-episode / per-step metrics and compute summaries.

    Usage
    -----
        metrics = RunMetrics()
        # inside training loop:
        metrics.log(phase="train", episode=i, reward=r, entropy=h, ...)
        # after generation:
        metrics.log(phase="generate", formula=f, dp_mean=m, dp_std=s, purpose="exploit")
        metrics.to_csv("training_log.csv")
        print(metrics.top_k("dp_mean", k=10))
        print(metrics.pareto_front())
    """

    rows: List[Dict[str, Any]] = field(default_factory=list)

    def log(self, **kwargs: Any) -> None:
        """Append one row of metrics."""
        self.rows.append(kwargs)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_csv(self, path: str | Path) -> None:
        """Write all rows to a CSV file."""
        if not self.rows:
            return
        fieldnames = list(self.rows[0].keys())
        # Collect any extra keys that appeared in later rows.
        for row in self.rows[1:]:
            for k in row:
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.rows)

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def _values(self, key: str, phase: Optional[str] = None) -> List[float]:
        rows = self.rows if phase is None else [r for r in self.rows if r.get("phase") == phase]
        return [float(r[key]) for r in rows if key in r]

    def rolling_mean(self, key: str, window: int = 50, phase: Optional[str] = None) -> List[float]:
        """Rolling mean of *key* values over a sliding window."""
        import numpy as np
        vals = self._values(key, phase)
        return [float(np.mean(vals[max(0, i - window): i + 1])) for i in range(len(vals))]

    def top_k(self, key: str = "dp_mean", k: int = 10, phase: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return the *k* rows with the highest value of *key*."""
        rows = self.rows if phase is None else [r for r in self.rows if r.get("phase") == phase]
        eligible = [r for r in rows if key in r]
        return sorted(eligible, key=lambda r: float(r[key]), reverse=True)[:k]

    def diversity(self, formula_key: str = "formula", phase: Optional[str] = None) -> int:
        """Count unique canonical formula strings."""
        rows = self.rows if phase is None else [r for r in self.rows if r.get("phase") == phase]
        return len({r[formula_key] for r in rows if formula_key in r})

    def hit_rate(self, key: str, threshold: float, phase: Optional[str] = None) -> float:
        """Fraction of rows where *key* exceeds *threshold*."""
        vals = self._values(key, phase)
        if not vals:
            return 0.0
        return sum(v >= threshold for v in vals) / len(vals)

    # ------------------------------------------------------------------
    # Pareto front: exploitation vs. exploration trade-off
    # ------------------------------------------------------------------

    def pareto_front(
        self,
        exploit_key: str = "dp_mean",
        explore_key: str = "dp_std",
        phase: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return Pareto-optimal candidates in (exploit_key, explore_key) space.

        A candidate is Pareto-optimal if no other candidate is strictly better
        in *both* exploit_key and explore_key simultaneously.

        This gives a ranked list of compositions that are valuable either as:
        - Synthesis targets (high exploit_key = high predicted reward), OR
        - DFT validation targets (high explore_key = high model uncertainty), OR
        - Both (most valuable of all).

        Parameters
        ----------
        exploit_key:
            Column name for exploitation metric, e.g. ``"dp_mean"``.
        explore_key:
            Column name for exploration metric (uncertainty), e.g. ``"dp_std"``.
        phase:
            If set, restrict to rows with matching ``phase`` value.
        """
        rows = self.rows if phase is None else [r for r in self.rows if r.get("phase") == phase]
        eligible = [r for r in rows if exploit_key in r and explore_key in r]
        if not eligible:
            return []

        pareto: List[Dict[str, Any]] = []
        for cand in eligible:
            dominated = False
            for other in eligible:
                if other is cand:
                    continue
                if (
                    float(other[exploit_key]) >= float(cand[exploit_key])
                    and float(other[explore_key]) >= float(cand[explore_key])
                    and (
                        float(other[exploit_key]) > float(cand[exploit_key])
                        or float(other[explore_key]) > float(cand[explore_key])
                    )
                ):
                    dominated = True
                    break
            if not dominated:
                pareto.append(cand)

        # Sort Pareto front by exploit_key descending for readability.
        pareto.sort(key=lambda r: float(r[exploit_key]), reverse=True)
        return pareto
