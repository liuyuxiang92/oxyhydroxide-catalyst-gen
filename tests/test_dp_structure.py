"""DPStructurePredictor + HEA / perovskite subclass tests.

Tests focus on construction-time behavior (cfg parsing, defaults, error
messages). Actual DeepMD eval is not exercised — those paths require
deepmd-kit and POSCAR/ckpt files.
"""
from __future__ import annotations

import pytest


def test_dp_structure_minimal_cfg():
    from rl_matdesign.predictors.dp_structure import DPStructurePredictor
    p = DPStructurePredictor(
        {"base_poscar": "fake.POSCAR", "dp_models": ["m1.pt", "m2.pt"]},
        seed=0,
    )
    assert p.poscar_template == "fake.POSCAR"
    assert p.dp_models == ["m1.pt", "m2.pt"]
    assert p.site_symbol == "X"          # base-class default
    assert p.objective == "mean_minus_kstd"
    assert p.k == 1.0
    assert p.n_random_configs == 5
    assert p.energy_per_atom is True


def test_dp_structure_accepts_legacy_poscar_key():
    from rl_matdesign.predictors.dp_structure import DPStructurePredictor
    p = DPStructurePredictor({"poscar": "old.POSCAR", "dp_models": ["m.pt"]})
    assert p.poscar_template == "old.POSCAR"


def test_dp_structure_accepts_legacy_poscar_template_key():
    from rl_matdesign.predictors.dp_structure import DPStructurePredictor
    p = DPStructurePredictor({"poscar_template": "older.POSCAR", "dp_models": ["m.pt"]})
    assert p.poscar_template == "older.POSCAR"


def test_dp_structure_missing_poscar_raises():
    from rl_matdesign.predictors.dp_structure import DPStructurePredictor
    with pytest.raises(ValueError) as info:
        DPStructurePredictor({"dp_models": ["m.pt"]})
    assert "POSCAR" in str(info.value)


def test_dp_structure_missing_models_raises():
    from rl_matdesign.predictors.dp_structure import DPStructurePredictor
    with pytest.raises(ValueError) as info:
        DPStructurePredictor({"base_poscar": "x"})
    assert "dp_models" in str(info.value)


def test_hea_default_site_symbol_is_X():
    from rl_matdesign.predictors.hea import HEAPropertyPredictor
    from rl_matdesign.predictors.dp_structure import DPStructurePredictor
    assert HEAPropertyPredictor.DEFAULT_SITE_SYMBOL == "X"
    assert issubclass(HEAPropertyPredictor, DPStructurePredictor)
    h = HEAPropertyPredictor({"base_poscar": "x", "dp_models": ["m"]})
    assert h.site_symbol == "X"


def test_perovskite_default_site_symbol_is_Fe():
    from rl_matdesign.predictors.perovskite import PerovskitePropertyPredictor
    from rl_matdesign.predictors.dp_structure import DPStructurePredictor
    assert PerovskitePropertyPredictor.DEFAULT_SITE_SYMBOL == "Fe"
    assert issubclass(PerovskitePropertyPredictor, DPStructurePredictor)
    pv = PerovskitePropertyPredictor({"base_poscar": "x", "dp_models": ["m"]})
    assert pv.site_symbol == "Fe"


def test_explicit_site_symbol_overrides_subclass_default():
    from rl_matdesign.predictors.hea import HEAPropertyPredictor
    h = HEAPropertyPredictor({"base_poscar": "x", "dp_models": ["m"], "site_symbol": "Cu"})
    assert h.site_symbol == "Cu"
