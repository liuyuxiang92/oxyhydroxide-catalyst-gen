"""Every episode that costs a predictor call must appear in training_log.csv.

Three wrong conclusions on the sinter/calcine benchmark came from this file being
incomplete rather than from the algorithms:

* A2C logged one ``pg_train`` row per *iteration* holding only the batch mean — the
  very statistic A2C optimises. It improved monotonically while the best candidate
  froze, so the failure was invisible in the log.
* DQN's warmup pays a real predictor call per episode (``_rollout_random_episode``
  -> ``env.step`` -> ``reward_fn``) and logged none of them. At a 2,500-episode
  budget that is 40% of the run.
* Both loggers used the name ``return`` for different quantities: DQN's is the
  undiscounted terminal reward, PG's the discounted ``G0 = gamma^(n-1) * r_T``.
  Plotting them on one axis made A2C look ~200 K better than it was.

These pin the row counts, the ``terminal_reward`` column that resolves the units,
and the append-mode schema safety that the warmup rows put at risk.
"""
from __future__ import annotations

import collections
import csv
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_CONFIG = _REPO / "configs" / "test_dummy.yaml"
_RUNNER = _REPO / "scripts" / "run_experiment.py"

# Mirrors configs/test_dummy.yaml. Kept explicit so a config edit fails loudly here
# rather than silently weakening the assertions.
_DQN_WARMUP_EPS = 20
_DQN_TRAIN_EPS = 100
_PG_NUM_ITERS = 10
_PG_BATCH_EPS = 5
_GAMMA = 0.99
_N_COMPONENTS = 5


def _run(tmp_path: Path, method: str) -> list[dict]:
    out = tmp_path / f"log_{method}"
    proc = subprocess.run(
        [
            sys.executable, str(_RUNNER),
            "--config", str(_CONFIG),
            "--method", method,
            "--out", str(out),
            "--dp-seed", "0", "--train-seed", "0", "--gen-seed", "0",
        ],
        capture_output=True, text=True, timeout=300, cwd=_REPO,
    )
    assert proc.returncode == 0, (
        f"runner failed for method={method}\nstdout:\n{proc.stdout[-2000:]}\n"
        f"stderr:\n{proc.stderr[-2000:]}"
    )
    with open(out / "training_log.csv") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def dqn_rows(tmp_path_factory) -> list[dict]:
    return _run(tmp_path_factory.mktemp("dqn"), "dqn")


@pytest.fixture(scope="module")
def a2c_rows(tmp_path_factory) -> list[dict]:
    return _run(tmp_path_factory.mktemp("a2c"), "a2c")


# ---------------------------------------------------------------------------
# Row counts — the episodes that cost predictor calls are all present
# ---------------------------------------------------------------------------

def test_dqn_logs_every_warmup_episode(dqn_rows):
    """Warmup pays a real predictor call per episode, so it belongs in the log."""
    counts = collections.Counter(r["phase"] for r in dqn_rows)
    assert counts["dqn_warmup"] == _DQN_WARMUP_EPS
    assert counts["dqn_train"] == _DQN_TRAIN_EPS


def test_pg_logs_every_sampled_episode(a2c_rows):
    """One row per episode, not per batch — the batch mean hides the tail."""
    counts = collections.Counter(r["phase"] for r in a2c_rows)
    assert counts["pg_episode"] == _PG_NUM_ITERS * _PG_BATCH_EPS
    assert counts["pg_train"] == _PG_NUM_ITERS


def test_warmup_rows_precede_training_rows(dqn_rows):
    """Warmup happens first, so the episode axis is the cumulative budget spent."""
    phases = [r["phase"] for r in dqn_rows if r["phase"].startswith("dqn_")]
    assert phases[:_DQN_WARMUP_EPS] == ["dqn_warmup"] * _DQN_WARMUP_EPS
    assert set(phases[_DQN_WARMUP_EPS:]) == {"dqn_train"}


# ---------------------------------------------------------------------------
# terminal_reward — the column that makes DQN and PG comparable
# ---------------------------------------------------------------------------

def test_pg_rows_carry_terminal_reward(a2c_rows):
    for phase in ("pg_episode", "pg_train"):
        rows = [r for r in a2c_rows if r["phase"] == phase]
        assert rows, f"no {phase} rows"
        assert all(r.get("terminal_reward") not in (None, "") for r in rows)


def test_pg_terminal_reward_undiscounts_the_return(a2c_rows):
    """``return`` is G0 = gamma^(n-1) * r_T; ``terminal_reward`` is r_T itself.

    Exact for these envs because the reward is terminal-only. This is the
    regression that put A2C ~200 K off on a shared axis with DQN.
    """
    rows = [r for r in a2c_rows if r["phase"] == "pg_episode"]
    factor = _GAMMA ** (_N_COMPONENTS - 1)
    assert factor < 1.0, "test is vacuous unless gamma < 1"
    for r in rows[:20]:
        assert float(r["return"]) == pytest.approx(
            float(r["terminal_reward"]) * factor, rel=1e-6
        )


def test_dqn_return_is_already_undiscounted(dqn_rows):
    """DQN's ``return`` is the terminal reward, so the two columns agree."""
    rows = [r for r in dqn_rows if r["phase"] == "dqn_warmup"]
    for r in rows[:20]:
        assert float(r["return"]) == pytest.approx(float(r["terminal_reward"]))


# ---------------------------------------------------------------------------
# Append-mode schema safety (RunMetrics.to_csv)
# ---------------------------------------------------------------------------

def test_append_uses_the_existing_header_not_the_first_row(tmp_path):
    """Resume must not reorder columns under a header already on disk.

    A fresh DQN run opens with a ``dqn_warmup`` row; a resumed one skips warmup and
    opens with a ``dqn_train`` row whose keys are in a different order. Deriving the
    writer's column order from ``rows[0]`` would append misaligned numbers with no
    error at all.
    """
    from rl_matdesign.utils.metrics import RunMetrics

    path = tmp_path / "training_log.csv"

    first = RunMetrics()
    first.log(phase="dqn_warmup", episode=1, ret=10.0, terminal_reward=10.0)
    first.log(phase="dqn_train", episode=1, ret=11.0, terminal_reward=11.0, loss=0.5)
    first.to_csv(path, mode="w")

    # Resumed run: different leading key order, and one column the first run lacked.
    second = RunMetrics()
    second.log(loss=0.25, terminal_reward=12.0, ret=12.0, episode=2, phase="dqn_train")
    second.to_csv(path, mode="a")

    with open(path) as f:
        rows = list(csv.DictReader(f))
    assert [r["phase"] for r in rows] == ["dqn_warmup", "dqn_train", "dqn_train"]
    assert [float(r["ret"]) for r in rows] == [10.0, 11.0, 12.0]
    assert [float(r["terminal_reward"]) for r in rows] == [10.0, 11.0, 12.0]
    # The warmup row has no loss; it must be blank, not a shifted neighbour's value.
    assert rows[0]["loss"] == ""


def test_append_keeps_columns_absent_from_the_header_out_of_the_rows(tmp_path):
    """A key the on-disk header lacks is dropped, never inserted mid-row.

    Inserting it would shift every later column for that row only — the same silent
    misalignment, one field further along.
    """
    from rl_matdesign.utils.metrics import RunMetrics

    path = tmp_path / "log.csv"
    RunMetrics(rows=[{"phase": "a", "x": 1}]).to_csv(path, mode="w")
    RunMetrics(rows=[{"phase": "b", "x": 2, "brand_new": 9}]).to_csv(path, mode="a")

    with open(path) as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["phase", "x"]
        rows = list(reader)
    assert [(r["phase"], r["x"]) for r in rows] == [("a", "1"), ("b", "2")]
