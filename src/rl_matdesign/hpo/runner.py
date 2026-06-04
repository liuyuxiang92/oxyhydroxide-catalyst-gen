"""Subprocess runner for one (trial × seed) of `run_experiment.py`.

Process isolation matters here: DeepMD / TF / matminer globals would accumulate
across in-process trials and OOM after ~50 runs. Spawning a fresh process per
seed amortizes import overhead (~10-30 s) over a multi-minute DP-bound run, so
the cost is <1% of trial wall-clock.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_RUN_EXPERIMENT = os.path.join(_REPO_ROOT, "scripts", "run_experiment.py")


class TrialRunError(RuntimeError):
    """Subprocess exited nonzero or produced no usable output."""


@dataclass
class TrialResult:
    returncode: int
    stdout_tail: str
    stderr_tail: str
    generated_csv: str


def _tail(s: str, n: int = 2000) -> str:
    """Return the last ``n`` chars of ``s`` (for log surfacing)."""
    return s[-n:] if len(s) > n else s


def run_trial_subprocess(
    *,
    config_yaml: str,
    out_dir: str,
    method: str,
    train_seed: int,
    dp_seed: int,
    gen_seed: int,
    timeout_s: Optional[int] = None,
    env_extra: Optional[dict] = None,
) -> TrialResult:
    """Spawn `run_experiment.py` once; return the result without raising on nonzero exit.

    The caller decides whether to mark the Optuna trial FAIL — that lets the
    driver distinguish "missing generated.csv with exit 0" (bug, raise loudly)
    from "nonzero exit" (deterministic config issue, FAIL the trial).
    """
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        sys.executable, _RUN_EXPERIMENT,
        "--config", config_yaml,
        "--method", method,
        "--out", out_dir,
        "--train-seed", str(train_seed),
        "--dp-seed", str(dp_seed),
        "--gen-seed", str(gen_seed),
    ]
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    try:
        proc = subprocess.run(
            cmd,
            cwd=_REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as e:
        raise TrialRunError(
            f"trial timed out after {timeout_s}s: {' '.join(cmd)}"
        ) from e
    return TrialResult(
        returncode=proc.returncode,
        stdout_tail=_tail(proc.stdout or ""),
        stderr_tail=_tail(proc.stderr or ""),
        generated_csv=os.path.join(out_dir, "generated.csv"),
    )
