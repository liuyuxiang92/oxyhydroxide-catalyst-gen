"""Timing / predictor-cost instrumentation.

Two layers:

* Unit tests for :class:`PredictorTimer` — the counting is what the whole
  method comparison rests on, and the attribute delegation is load-bearing
  (``run_experiment`` reads ``predictor._cache``, ``generate_candidates`` probes
  ``predict_raw``), so both are pinned here.
* End-to-end smoke via subprocess on ``configs/test_dummy.yaml`` for all three
  comparison arms (DQN bootstrap, DQN mc, A2C), asserting that ``timing.json``
  and the new ``training_log.csv`` columns are produced.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from rl_matdesign.utils.timing import PhaseTimer, PredictorTimer, candidate_key

_REPO = Path(__file__).resolve().parent.parent
_CONFIG = _REPO / "configs" / "test_dummy.yaml"
_RUNNER = _REPO / "scripts" / "run_experiment.py"

_TIMING_COLS = {"t_wall", "t_predict_cum", "n_predict_calls",
                "n_predict_unique", "best_reward_so_far"}


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _Bare:
    """Minimal predictor: `predict` only, no cache, no predict_raw."""

    def __init__(self):
        self.calls = 0

    def predict(self, comp):
        self.calls += 1
        return (sum(comp.values()), 0.1)

    def batch_predict(self, comps):
        return [self.predict(c) for c in comps]


class _Rich(_Bare):
    """Predictor with the optional surface the framework probes for."""

    def __init__(self):
        super().__init__()
        self._cache = {"seeded": (1.0, 0.0)}

    def predict_raw(self, comp):
        return (99.0, 0.5)

    def per_objective_stats(self, comp):
        return {"energy": (1.0, 0.1)}

    def check_phase(self, comp):
        return True, "ok"


# ---------------------------------------------------------------------------
# candidate_key
# ---------------------------------------------------------------------------

def test_candidate_key_is_order_invariant():
    assert candidate_key({"Ni": 0.7, "Fe": 0.3}) == candidate_key({"Fe": 0.3, "Ni": 0.7})


def test_candidate_key_handles_nested_multigroup_mapping():
    a = {"P_site": {"P": 0.9, "Si": 0.1}, "S_site": {"S": 1.0}}
    b = {"S_site": {"S": 1.0}, "P_site": {"Si": 0.1, "P": 0.9}}
    assert candidate_key(a) == candidate_key(b)
    assert candidate_key(a) != candidate_key({"P_site": {"P": 1.0}, "S_site": {"S": 1.0}})


def test_candidate_key_survives_non_dict_candidates():
    # A custom predictor may take something else entirely; counting must not blow up.
    assert candidate_key("Ni0.7Fe0.3") == candidate_key("Ni0.7Fe0.3")


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------

def test_counts_calls_and_uniques_separately():
    t = PredictorTimer(_Bare())
    t.predict({"Ni": 0.7, "Fe": 0.3})
    t.predict({"Fe": 0.3, "Ni": 0.7})   # same multiset, different order
    t.predict({"Ni": 0.5, "Fe": 0.5})
    assert t.n_calls == 3
    assert t.n_unique == 2
    s = t.summary()
    assert s["n_cache_hits"] == 1
    assert s["cache_hit_rate"] == pytest.approx(1 / 3)


def test_tracks_running_best_mean():
    t = PredictorTimer(_Bare())
    t.predict({"A": 1.0})
    t.predict({"A": 3.0})
    t.predict({"A": 2.0})
    assert t.best_mean == pytest.approx(3.0)
    assert t.snapshot()["best_reward_so_far"] == pytest.approx(3.0)


def test_nan_prediction_cannot_poison_best_mean():
    class _Nan(_Bare):
        def predict(self, comp):
            return (float("nan"), 0.0)

    t = PredictorTimer(_Bare())
    t.predict({"A": 2.0})
    t._inner = _Nan()
    t.predict({"B": 1.0})
    assert t.best_mean == pytest.approx(2.0)


def test_accumulates_predict_time():
    t = PredictorTimer(_Bare())
    t.predict({"A": 1.0})
    assert t.t_predict_s >= 0.0
    assert t.summary()["t_predict_s"] >= 0.0


def test_batch_predict_counts_every_candidate():
    t = PredictorTimer(_Bare())
    t.batch_predict([{"A": 1.0}, {"B": 1.0}, {"A": 1.0}])
    assert t.n_calls == 3
    assert t.n_unique == 2


def test_marks_record_named_boundaries():
    t = PredictorTimer(_Bare())
    t.predict({"A": 1.0})
    t.mark("warmup_end")
    t.predict({"B": 1.0})
    t.mark("run_end")
    assert [m["phase"] for m in t.marks] == ["warmup_end", "run_end"]
    assert t.marks[0]["n_predict_calls"] == 1
    assert t.marks[1]["n_predict_calls"] == 2


def test_snapshot_has_exactly_the_training_log_columns():
    assert set(PredictorTimer(_Bare()).snapshot()) == _TIMING_COLS


# ---------------------------------------------------------------------------
# Delegation — required by run_experiment / generate_candidates
# ---------------------------------------------------------------------------

def test_delegates_cache_attribute_for_dqn_checkpointing():
    inner = _Rich()
    t = PredictorTimer(inner)
    # run_experiment.py does exactly this, then mutates the result in place.
    cache = getattr(t, "_cache", None)
    assert cache is inner._cache
    cache["added"] = (2.0, 0.0)
    assert inner._cache["added"] == (2.0, 0.0)


def test_delegates_optional_methods():
    t = PredictorTimer(_Rich())
    assert t.per_objective_stats({"A": 1.0}) == {"energy": (1.0, 0.1)}
    assert t.check_phase({"A": 1.0}) == (True, "ok")


def test_predict_raw_is_timed_and_counted():
    t = PredictorTimer(_Rich())
    assert t.predict_raw({"A": 1.0}) == (99.0, 0.5)
    assert t.n_calls == 1
    assert t.n_unique == 1


def test_missing_predict_raw_stays_missing():
    # generate_candidates branches on hasattr; the proxy must not fake it.
    t = PredictorTimer(_Bare())
    assert not hasattr(t, "predict_raw")
    assert hasattr(PredictorTimer(_Rich()), "predict_raw")


def test_summary_is_safe_with_zero_calls():
    s = PredictorTimer(_Bare()).summary()
    assert s["n_calls"] == 0
    assert s["cache_hit_rate"] == 0.0
    assert s["mean_s_per_call"] == 0.0
    assert s["best_reward"] is None


# ---------------------------------------------------------------------------
# PhaseTimer
# ---------------------------------------------------------------------------

def test_phase_timer_accumulates_repeated_phases():
    p = PhaseTimer()
    with p("train"):
        pass
    with p("train"):
        pass
    with p("generate"):
        pass
    assert set(p.totals) == {"train", "generate"}
    assert all(v >= 0.0 for v in p.totals.values())


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

def _run(tmp_path: Path, method: str, target_mode: str | None) -> Path:
    out = tmp_path / f"timing_{method}_{target_mode or 'default'}"
    cmd = [
        sys.executable, str(_RUNNER),
        "--config", str(_CONFIG),
        "--method", method,
        "--out", str(out),
        "--dp-seed", "0", "--train-seed", "0", "--gen-seed", "0",
    ]
    if target_mode is not None:
        cmd += ["--dqn-target-mode", target_mode]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=_REPO)
    assert proc.returncode == 0, (
        f"runner failed for method={method} target_mode={target_mode}\n"
        f"stdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}"
    )
    return out


@pytest.mark.parametrize("method,target_mode", [
    ("dqn", "bootstrap"),
    ("dqn", "mc"),
    ("a2c", None),
])
def test_run_writes_timing_json(tmp_path, method, target_mode):
    out = _run(tmp_path, method, target_mode)
    timing = json.loads((out / "timing.json").read_text())

    assert timing["method"] == method
    assert timing["dqn_target_mode"] == target_mode
    assert timing["total_s"] > 0.0

    pred = timing["predictor"]
    assert pred["n_calls"] > 0
    assert pred["n_unique"] <= pred["n_calls"]
    assert pred["n_cache_hits"] == pred["n_calls"] - pred["n_unique"]
    assert pred["t_predict_s"] >= 0.0
    # The predictor is part of the run, so it cannot have taken longer than it.
    assert pred["t_predict_s"] <= timing["total_s"]
    assert timing["overhead_s"] == pytest.approx(
        timing["total_s"] - pred["t_predict_s"], abs=1e-3)

    assert "train" in timing["phases_s"]
    assert "generate" in timing["phases_s"]
    assert {m["phase"] for m in timing["marks"]} >= {"warmup_end", "run_end"}


@pytest.mark.parametrize("method,phase", [("dqn", "dqn_train"), ("a2c", "pg_train")])
def test_training_log_carries_cumulative_cost_columns(tmp_path, method, phase):
    out = _run(tmp_path, method, "bootstrap" if method == "dqn" else None)
    with (out / "training_log.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if r.get("phase") == phase]

    assert rows, f"no {phase} rows in training_log.csv"
    assert _TIMING_COLS <= set(rows[0]), (
        f"missing timing columns: {_TIMING_COLS - set(rows[0])}")

    t_wall = [float(r["t_wall"]) for r in rows]
    calls = [int(r["n_predict_calls"]) for r in rows]
    uniq = [int(r["n_predict_unique"]) for r in rows]
    best = [float(r["best_reward_so_far"]) for r in rows]

    # Every cumulative column must be monotonically non-decreasing; that is what
    # makes them usable as a plot x-axis.
    for name, seq in (("t_wall", t_wall), ("n_predict_calls", calls),
                      ("n_predict_unique", uniq), ("best_reward_so_far", best)):
        assert all(b >= a for a, b in zip(seq, seq[1:])), f"{name} is not monotonic"
    assert all(u <= c for u, c in zip(uniq, calls))


def test_generated_csv_schema_is_untouched(tmp_path):
    # Generation cost lives in timing.json; generated.csv must keep the schema
    # compare_methods.py / plot_figures.py already read.
    out = _run(tmp_path, "a2c", None)
    with (out / "generated.csv").open() as f:
        cols = set(csv.DictReader(f).fieldnames or [])
    assert {"formula", "reward", "dp_mean", "dp_std"} <= cols
    assert not (_TIMING_COLS & cols)
