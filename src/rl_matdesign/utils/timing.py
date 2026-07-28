"""Wall-clock + predictor-cost instrumentation for method comparisons.

The point of this module is to answer "which RL method reaches a good candidate
for the least *total* cost", where cost includes the reward-model call — which,
for a DeepMD ensemble or a geometry-optimised structure score, dominates
everything the RL machinery does.

Two pieces:

* :class:`PredictorTimer` — a transparent proxy around any ``PropertyPredictor``.
  Every training reward flows through ``predictor.predict`` (the ``reward_fn``
  closures in ``run_experiment.build_env``, invoked only at an episode's terminal
  step), so wrapping the predictor object once captures every reward call in
  every phase, for every env type and every method.
* :class:`PhaseTimer` — accumulates wall-clock seconds per named phase
  (``setup`` / ``warmup`` / ``train`` / ``generate``).

Both are written to ``<out>/timing.json`` by ``run_experiment.py``; the
per-episode cumulative counters also land in ``training_log.csv`` so a
best-reward-vs-time curve can be plotted after the fact without re-running.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple


def candidate_key(candidate: Any) -> tuple:
    """Hashable canonical key for a predictor input.

    Handles both the flat ``{element: fraction}`` mapping and the nested
    ``{group: {element: fraction}}`` mapping that :class:`MultiGroupEnv` hands
    to the predictor.  Mirrors ``StructureScorePredictor._key`` in shape, but
    rounds fractions like ``baselines/_common.score_composition`` does so that
    float round-off doesn't inflate the unique count.

    Non-dict inputs (a custom predictor taking some other candidate object) fall
    back to ``repr``, which keeps counting sane rather than raising.
    """
    if not isinstance(candidate, dict):
        return ("__repr__", repr(candidate))
    items = []
    for el, v in candidate.items():
        if isinstance(v, dict):
            items.append((str(el), tuple(sorted(
                (str(e), round(float(f), 6)) for e, f in v.items()))))
        else:
            items.append((str(el), round(float(v), 6)))
    return tuple(sorted(items, key=lambda t: t[0]))


class PredictorTimer:
    """Counting/timing proxy around a ``PropertyPredictor``.

    Wraps ``predict`` / ``predict_raw`` / ``batch_predict`` and delegates every
    other attribute to the wrapped predictor.  Delegation is load-bearing:
    ``run_experiment`` reads ``predictor._cache`` for DQN checkpointing and
    ``training.generate_candidates`` probes ``predict_raw`` /
    ``per_objective_stats`` / ``check_phase``.  ``__getattr__`` hands back the
    inner objects, so those paths keep working untouched.

    ``n_unique`` is counted against this wrapper's own key set rather than the
    predictor's internal cache, so it is meaningful even for predictors that
    cache under a different attribute name (``StructureScorePredictor``) or
    don't cache at all (the ``dummy`` predictor).  ``n_calls - n_unique`` is
    therefore the number of calls that a correctly-keyed cache could serve.
    """

    def __init__(self, inner: Any, t0: Optional[float] = None) -> None:
        self._inner = inner
        self._t0 = time.perf_counter() if t0 is None else float(t0)
        self.n_calls = 0
        self.n_unique = 0
        self.t_predict_s = 0.0
        self.best_mean: Optional[float] = None
        self._seen: set = set()
        self._marks: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Timed pass-throughs
    # ------------------------------------------------------------------

    def _record(self, candidate: Any, mean: float, dt: float) -> None:
        self.n_calls += 1
        self.t_predict_s += dt
        key = candidate_key(candidate)
        if key not in self._seen:
            self._seen.add(key)
            self.n_unique += 1
        try:
            m = float(mean)
        except (TypeError, ValueError):
            return
        # NaN never wins a comparison, so a bad prediction can't poison the max.
        if m == m and (self.best_mean is None or m > self.best_mean):
            self.best_mean = m

    def predict(self, candidate: Any) -> Tuple[float, float]:
        t = time.perf_counter()
        out = self._inner.predict(candidate)
        dt = time.perf_counter() - t
        self._record(candidate, out[0], dt)
        return out

    def batch_predict(self, candidates: List[Any]) -> List[Tuple[float, float]]:
        t = time.perf_counter()
        out = self._inner.batch_predict(candidates)
        dt = time.perf_counter() - t
        # Attribute the batch cost evenly; the per-call figure stays honest and
        # the total is exact, which is what the comparison needs.
        share = dt / max(1, len(candidates))
        for cand, res in zip(candidates, out):
            self._record(cand, res[0], share)
        return out

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes this wrapper does not define itself, so
        # `_inner`, the counters and the methods above never land here.
        # `predict_raw` is routed through here on purpose: if the inner
        # predictor lacks it, AttributeError propagates and `hasattr` stays
        # correctly False (training.generate_candidates branches on that).
        inner = object.__getattribute__(self, "_inner")
        attr = getattr(inner, name)
        if name == "predict_raw":
            def _timed_predict_raw(candidate: Any) -> Tuple[float, float]:
                t = time.perf_counter()
                out = attr(candidate)
                self.n_calls += 1
                self.t_predict_s += time.perf_counter() - t
                key = candidate_key(candidate)
                if key not in self._seen:
                    self._seen.add(key)
                    self.n_unique += 1
                return out
            return _timed_predict_raw
        return attr

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @property
    def t_wall_s(self) -> float:
        """Seconds since the timer was created (i.e. since run start)."""
        return time.perf_counter() - self._t0

    def snapshot(self) -> Dict[str, Any]:
        """Live counters, for embedding in a metrics row."""
        return {
            "t_wall": round(self.t_wall_s, 4),
            "t_predict_cum": round(self.t_predict_s, 4),
            "n_predict_calls": self.n_calls,
            "n_predict_unique": self.n_unique,
            "best_reward_so_far": self.best_mean,
        }

    def mark(self, phase: str) -> None:
        """Record a named boundary (end of warmup, end of training, ...)."""
        snap = self.snapshot()
        snap["phase"] = phase
        self._marks.append(snap)

    def summary(self) -> Dict[str, Any]:
        """Aggregate predictor-cost report for ``timing.json``."""
        n = self.n_calls
        return {
            "t_predict_s": round(self.t_predict_s, 4),
            "n_calls": n,
            "n_unique": self.n_unique,
            "n_cache_hits": n - self.n_unique,
            "cache_hit_rate": round((n - self.n_unique) / n, 6) if n else 0.0,
            "mean_s_per_call": round(self.t_predict_s / n, 6) if n else 0.0,
            "mean_s_per_unique": (round(self.t_predict_s / self.n_unique, 6)
                                  if self.n_unique else 0.0),
            "best_reward": self.best_mean,
        }

    @property
    def marks(self) -> List[Dict[str, Any]]:
        return list(self._marks)


class PhaseTimer:
    """Accumulate wall-clock seconds under named phases.

    Usage::

        phases = PhaseTimer()
        with phases("setup"):
            ...
        phases.totals  # {"setup": 12.4, ...}

    Re-entering a phase name adds to its total rather than replacing it.
    """

    def __init__(self) -> None:
        self.totals: Dict[str, float] = {}
        self._stack: List[Tuple[str, float]] = []

    def __call__(self, name: str) -> "PhaseTimer":
        self._stack.append((name, time.perf_counter()))
        return self

    def __enter__(self) -> "PhaseTimer":
        return self

    def __exit__(self, *_exc: Any) -> bool:
        name, t = self._stack.pop()
        self.totals[name] = round(self.totals.get(name, 0.0) + (time.perf_counter() - t), 4)
        return False
