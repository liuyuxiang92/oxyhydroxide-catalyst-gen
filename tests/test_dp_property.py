"""DPPropertyPredictor cfg-parsing tests (no DeepMD/ASE imports needed).

The .predict() path requires deepmd-kit and is exercised at runtime; here we
only cover the cfg surface that fails fast without the heavy backend.
"""
from __future__ import annotations

import pytest


def _min_cfg(**overrides):
    cfg = {
        "base_poscar": "/tmp/fake.POSCAR",
        "dp_models": ["/tmp/m1.pt", "/tmp/m2.pt"],
    }
    cfg.update(overrides)
    return cfg


def test_minimal_cfg_uses_expected_defaults():
    from rl_matdesign.predictors.dp_property import DPPropertyPredictor
    p = DPPropertyPredictor(_min_cfg())
    assert p.site_symbol == "X"
    assert p.dp_head == "property"
    assert p.output_index == 0
    assert p.output_aggregator == "index"
    assert p.maximize is False
    assert p.objective == "mean_minus_kstd"
    assert p.k == 1.0
    assert p.n_random_configs == 5


def test_missing_dp_models_raises():
    from rl_matdesign.predictors.dp_property import DPPropertyPredictor
    with pytest.raises(ValueError) as info:
        DPPropertyPredictor({"base_poscar": "/tmp/x"})
    assert "dp_models" in str(info.value)


def test_missing_poscar_raises():
    from rl_matdesign.predictors.dp_property import DPPropertyPredictor
    with pytest.raises(ValueError) as info:
        DPPropertyPredictor({"dp_models": ["/tmp/m.pt"]})
    assert "POSCAR" in str(info.value) or "base_poscar" in str(info.value)


def test_legacy_poscar_key_accepted():
    from rl_matdesign.predictors.dp_property import DPPropertyPredictor
    p = DPPropertyPredictor({"poscar": "/tmp/x", "dp_models": ["/tmp/m.pt"]})
    assert p.poscar_template == "/tmp/x"


def test_maximize_flips_value_sign():
    from rl_matdesign.predictors.dp_property import DPPropertyPredictor
    assert DPPropertyPredictor(_min_cfg())._value_sign() == -1.0
    assert DPPropertyPredictor(_min_cfg(maximize=True))._value_sign() == 1.0


def test_registry_returns_dp_property_predictor():
    from rl_matdesign.registry import resolve_predictor
    from rl_matdesign.predictors.dp_property import DPPropertyPredictor
    p = resolve_predictor("dp_property", _min_cfg())
    assert isinstance(p, DPPropertyPredictor)
