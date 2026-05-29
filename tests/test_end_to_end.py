"""End-to-end smoke tests via subprocess.

Runs `scripts/run_experiment.py` with `configs/test_dummy.yaml` (no DeepMD
needed) for both A2C and DQN. Asserts only structural properties (exit
code, file existence, generated.csv has expected columns) — not numeric
values — so it survives RNG drift.
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


_REPO = Path(__file__).resolve().parent.parent
_CONFIG = _REPO / "configs" / "test_dummy.yaml"
_RUNNER = _REPO / "scripts" / "run_experiment.py"

# Columns we expect to find in generated.csv regardless of method.
_EXPECTED_COLS = {"formula", "reward", "dp_mean", "dp_std"}


def _run_smoke(tmp_path: Path, method: str) -> Path:
    out = tmp_path / f"smoke_{method}"
    proc = subprocess.run(
        [
            sys.executable, str(_RUNNER),
            "--config", str(_CONFIG),
            "--method", method,
            "--out", str(out),
            "--dp-seed", "0",
            "--train-seed", "0",
            "--gen-seed", "0",
        ],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=_REPO,
    )
    assert proc.returncode == 0, (
        f"runner failed for method={method}\nstdout:\n{proc.stdout[-2000:]}\n"
        f"stderr:\n{proc.stderr[-2000:]}"
    )
    return out


def _assert_generated_csv_has_columns(out_dir: Path) -> None:
    csv_path = out_dir / "generated.csv"
    assert csv_path.exists(), f"generated.csv missing in {out_dir}"
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        rows = list(reader)
    missing = _EXPECTED_COLS - cols
    assert not missing, f"generated.csv missing columns {missing}; have {cols}"
    assert rows, "generated.csv has no rows"


def test_end_to_end_a2c(tmp_path):
    out = _run_smoke(tmp_path, "a2c")
    _assert_generated_csv_has_columns(out)


def test_end_to_end_dqn(tmp_path):
    out = _run_smoke(tmp_path, "dqn")
    _assert_generated_csv_has_columns(out)
