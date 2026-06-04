"""Tests for the HPO top-K-mean metric helper."""

from __future__ import annotations

import csv
import math
import os

import pytest

from rl_matdesign.hpo.metric import top_k_mean_from_csv


def _write_generated_csv(path: str, rewards):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["formula", "reward", "dp_mean"])
        w.writeheader()
        for i, r in enumerate(rewards):
            w.writerow({"formula": f"F{i}", "reward": r, "dp_mean": ""})


def test_top_k_mean_returns_average_of_top_k(tmp_path):
    p = str(tmp_path / "generated.csv")
    _write_generated_csv(p, [1.0, 5.0, 3.0, 2.0, 4.0])
    mean, used = top_k_mean_from_csv(p, k=3)
    assert used == 3
    assert mean == pytest.approx((5.0 + 4.0 + 3.0) / 3.0)


def test_top_k_mean_caps_when_n_lt_k(tmp_path, caplog):
    p = str(tmp_path / "generated.csv")
    _write_generated_csv(p, [1.0, 2.0])
    with caplog.at_level("WARNING"):
        mean, used = top_k_mean_from_csv(p, k=10)
    assert used == 2
    assert mean == pytest.approx(1.5)
    assert "only 2 usable rows" in caplog.text


def test_top_k_mean_handles_k_equal_to_n(tmp_path):
    p = str(tmp_path / "generated.csv")
    _write_generated_csv(p, [3.0, 1.0, 2.0])
    mean, used = top_k_mean_from_csv(p, k=3)
    assert used == 3
    assert mean == pytest.approx(2.0)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        top_k_mean_from_csv(str(tmp_path / "missing.csv"), k=5)


def test_missing_reward_column_raises(tmp_path):
    p = str(tmp_path / "generated.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["formula", "dp_mean"])
        w.writeheader()
        w.writerow({"formula": "F0", "dp_mean": "1.0"})
    with pytest.raises(ValueError, match="missing 'reward' column"):
        top_k_mean_from_csv(p, k=3)


def test_empty_csv_with_only_header_raises(tmp_path):
    p = str(tmp_path / "generated.csv")
    _write_generated_csv(p, [])
    with pytest.raises(ValueError, match="no usable reward values"):
        top_k_mean_from_csv(p, k=3)


def test_nan_and_inf_rows_skipped(tmp_path):
    p = str(tmp_path / "generated.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["formula", "reward", "dp_mean"])
        w.writeheader()
        for i, r in enumerate(["1.0", "nan", "inf", "-inf", "2.0", "3.0"]):
            w.writerow({"formula": f"F{i}", "reward": r, "dp_mean": ""})
    mean, used = top_k_mean_from_csv(p, k=3)
    assert used == 3
    assert mean == pytest.approx(2.0)


def test_blank_reward_rows_skipped(tmp_path):
    p = str(tmp_path / "generated.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["formula", "reward", "dp_mean"])
        w.writeheader()
        for i, r in enumerate(["1.0", "", "2.0"]):
            w.writerow({"formula": f"F{i}", "reward": r, "dp_mean": ""})
    mean, used = top_k_mean_from_csv(p, k=10)
    assert used == 2
    assert mean == pytest.approx(1.5)
