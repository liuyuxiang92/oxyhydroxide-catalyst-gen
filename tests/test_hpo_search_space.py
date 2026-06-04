"""Tests for the HPO search-space sampler.

We construct an Optuna trial via a fresh in-memory study and check that:
- each distribution shape returns values of the right type within bounds
- frac_of resolves against a passed-in base
- validation rejects malformed specs
"""

from __future__ import annotations

import math

import optuna
import pytest

from rl_matdesign.hpo.search_space import (
    SearchSpaceError,
    sample_from_search_space,
    validate_search_space,
)


def _trial():
    study = optuna.create_study(direction="maximize")
    return study.ask()


def test_uniform_within_bounds():
    spec = {"x": {"dist": "uniform", "low": 0.0, "high": 1.0}}
    for _ in range(20):
        out = sample_from_search_space(_trial(), spec)
        assert 0.0 <= out["x"] <= 1.0


def test_loguniform_within_bounds():
    spec = {"lr": {"dist": "loguniform", "low": 1e-5, "high": 1e-2}}
    for _ in range(20):
        out = sample_from_search_space(_trial(), spec)
        assert 1e-5 <= out["lr"] <= 1e-2


def test_int_within_bounds_and_typed():
    spec = {"k": {"dist": "int", "low": 1, "high": 10}}
    for _ in range(20):
        out = sample_from_search_space(_trial(), spec)
        assert isinstance(out["k"], int)
        assert 1 <= out["k"] <= 10


def test_int_log_within_bounds():
    spec = {"freq": {"dist": "int_log", "low": 1, "high": 1000}}
    out = sample_from_search_space(_trial(), spec)
    assert isinstance(out["freq"], int)
    assert 1 <= out["freq"] <= 1000


def test_categorical_picks_from_list():
    spec = {"bs": {"dist": "categorical", "choices": [32, 64, 128]}}
    for _ in range(20):
        out = sample_from_search_space(_trial(), spec)
        assert out["bs"] in (32, 64, 128)


def test_frac_of_int_typed_base_rounds_to_int():
    spec = {
        "dqn_eps_anneal_eps": {
            "dist": "frac_of",
            "base": "dqn_num_train_eps",
            "low": 0.3,
            "high": 0.8,
        }
    }
    resolved = {"dqn_num_train_eps": 1000}
    for _ in range(20):
        out = sample_from_search_space(_trial(), spec, resolved_values=resolved)
        val = out["dqn_eps_anneal_eps"]
        assert isinstance(val, int)
        assert 300 <= val <= 800


def test_frac_of_float_base_stays_float():
    # A non-int-typed base (e.g. lr-scale dial) should produce a float.
    spec = {
        "lr_scale": {
            "dist": "frac_of",
            "base": "dqn_lr",
            "low": 0.1,
            "high": 1.0,
        }
    }
    resolved = {"dqn_lr": 0.001}
    out = sample_from_search_space(_trial(), spec, resolved_values=resolved)
    val = out["lr_scale"]
    assert isinstance(val, float)
    assert 0.0001 <= val <= 0.001 + 1e-12


# ---------------- validation ---------------- #


def test_validate_rejects_missing_dist():
    with pytest.raises(SearchSpaceError, match="missing required 'dist'"):
        validate_search_space({"x": {"low": 0.0, "high": 1.0}})


def test_validate_rejects_unknown_dist():
    with pytest.raises(SearchSpaceError, match="unknown dist"):
        validate_search_space({"x": {"dist": "moon", "low": 0, "high": 1}})


def test_validate_rejects_inverted_bounds():
    with pytest.raises(SearchSpaceError, match="low < high"):
        validate_search_space({"x": {"dist": "uniform", "low": 1.0, "high": 0.0}})


def test_validate_rejects_loguniform_with_nonpositive():
    with pytest.raises(SearchSpaceError, match="loguniform requires low > 0"):
        validate_search_space({"x": {"dist": "loguniform", "low": 0.0, "high": 1.0}})


def test_validate_rejects_empty_categorical():
    with pytest.raises(SearchSpaceError, match="non-empty 'choices'"):
        validate_search_space({"x": {"dist": "categorical", "choices": []}})


def test_validate_frac_of_unknown_base():
    spec = {
        "anneal": {
            "dist": "frac_of",
            "base": "nonexistent_key",
            "low": 0.3,
            "high": 0.8,
        }
    }
    with pytest.raises(SearchSpaceError, match="not found in fixed_overrides or base"):
        validate_search_space(spec, fixed_overrides={}, base_values={})


def test_validate_frac_of_base_found_via_fixed_overrides():
    spec = {
        "anneal": {
            "dist": "frac_of",
            "base": "dqn_num_train_eps",
            "low": 0.3,
            "high": 0.8,
        }
    }
    validate_search_space(spec, fixed_overrides={"dqn_num_train_eps": 1000}, base_values={})


def test_validate_frac_of_base_found_via_base_values():
    spec = {
        "anneal": {
            "dist": "frac_of",
            "base": "dqn_num_train_eps",
            "low": 0.3,
            "high": 0.8,
        }
    }
    validate_search_space(spec, fixed_overrides={}, base_values={"dqn_num_train_eps": 20000})


def test_sample_frac_of_missing_base_raises():
    spec = {
        "anneal": {
            "dist": "frac_of",
            "base": "dqn_num_train_eps",
            "low": 0.3,
            "high": 0.8,
        }
    }
    with pytest.raises(SearchSpaceError, match="not resolvable at sample time"):
        sample_from_search_space(_trial(), spec, resolved_values={})


def test_multi_param_spec_returns_all_keys():
    spec = {
        "lr": {"dist": "loguniform", "low": 1e-5, "high": 1e-2},
        "bs": {"dist": "categorical", "choices": [32, 64]},
        "grad_steps": {"dist": "int", "low": 1, "high": 5},
    }
    out = sample_from_search_space(_trial(), spec)
    assert set(out) == {"lr", "bs", "grad_steps"}
    assert isinstance(out["lr"], float)
    assert out["bs"] in (32, 64)
    assert isinstance(out["grad_steps"], int)
