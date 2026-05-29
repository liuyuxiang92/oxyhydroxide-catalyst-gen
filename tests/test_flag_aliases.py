"""Flag-naming alias tests.

Guards two invariants:
  1. Loading any deprecated YAML key emits exactly one DeprecationWarning
     naming the new key, and the value is remapped.
  2. None of the bundled `configs/*.yaml` files contain old keys — they
     must load with zero deprecation warnings (regression guard against
     accidental re-introduction).
"""
from __future__ import annotations

import glob
import warnings
from pathlib import Path

import pytest


_ALIASES = {
    "buffer_size":          "dqn_buffer_size",
    "num_train_eps":        "dqn_num_train_eps",
    "grad_steps_per_ep":    "dqn_grad_steps_per_ep",
    "target_update_freq":   "dqn_target_update_freq",
    "eps_anneal_eps":       "dqn_eps_anneal_eps",
    "eps_min":              "dqn_eps_min",
    "entropy_coef":         "pg_entropy_coef",
    "repeat_penalty_coef":  "pg_repeat_penalty_coef",
    "repeat_penalty_shape": "pg_repeat_penalty_shape",
}


@pytest.mark.parametrize("old_key,new_key", list(_ALIASES.items()))
def test_old_key_remapped_with_deprecation_warning(old_key, new_key):
    from run_experiment import _apply_flag_aliases
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = _apply_flag_aliases({old_key: "sentinel_value"})
    assert out == {new_key: "sentinel_value"}
    dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(dep) == 1
    msg = str(dep[0].message)
    assert old_key in msg and new_key in msg


def test_old_and_new_both_present_new_wins_with_conflict_warning():
    from run_experiment import _apply_flag_aliases
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = _apply_flag_aliases({"buffer_size": 999, "dqn_buffer_size": 50000})
    assert out == {"dqn_buffer_size": 50000}
    dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(dep) == 1
    assert "both" in str(dep[0].message).lower() or "ignoring" in str(dep[0].message).lower()


def test_all_bundled_configs_load_without_deprecation_warnings():
    from run_experiment import load_config
    repo_root = Path(__file__).resolve().parent.parent
    cfg_paths = sorted(glob.glob(str(repo_root / "configs" / "*.yaml")))
    assert cfg_paths, "no configs/*.yaml found"
    for p in cfg_paths:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            load_config(p)
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert not dep, f"{p} loads with deprecation warnings: {[str(w.message) for w in dep]}"
