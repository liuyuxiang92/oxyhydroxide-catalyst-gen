"""StructureScorePredictor — the unified structure-based predictor.

Covers cfg parsing/validation, the weighted-signed-scaled combine in both
structural regimes (share_structure true = old structure_pipeline; false = old
composite), and the consumer contract (predict / predict_raw /
per_objective_stats). DeepMD eval + geo-opt are GPU-gated and stubbed here.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rl_matdesign.predictors.structure_score import StructureScorePredictor  # noqa: E402


# --------------------------------------------------------------------------- #
# cfg parsing / validation
# --------------------------------------------------------------------------- #

def _shared_cfg(**over):
    cfg = {
        "builder": "substitute",
        "base_poscar": "x",            # SubstituteBuilder reads it lazily at build()
        "site_symbol": "X",
        "k": 1.0,
        "properties": [
            {"name": "conductivity", "backend": "property", "models": ["m1.pt"],
             "head": "experiment", "direction": "max", "weight": 2.0, "scale": 1.0,
             "objective": "mean_minus_kstd"},
            {"name": "stability", "backend": "property", "models": ["s1.pt"],
             "head": None, "direction": "max", "weight": 1.0, "scale": 5.0,
             "objective": "mean_minus_kstd"},
        ],
    }
    cfg.update(over)
    return cfg


def test_empty_properties_raises():
    with pytest.raises(ValueError) as info:
        StructureScorePredictor({"properties": []})
    assert "properties" in str(info.value)


def test_missing_models_raises():
    with pytest.raises(ValueError) as info:
        StructureScorePredictor({"base_poscar": "x",
                                 "properties": [{"name": "p", "backend": "energy"}]})
    assert "models" in str(info.value)


def test_bad_backend_raises():
    with pytest.raises(ValueError) as info:
        StructureScorePredictor({"base_poscar": "x", "properties": [
            {"name": "p", "backend": "magic", "models": ["m"]}]})
    assert "backend" in str(info.value)


def test_bad_direction_raises():
    with pytest.raises(ValueError) as info:
        StructureScorePredictor({"base_poscar": "x", "properties": [
            {"name": "p", "backend": "energy", "models": ["m"], "direction": "down"}]})
    assert "direction" in str(info.value)


def test_duplicate_names_raise():
    with pytest.raises(ValueError) as info:
        StructureScorePredictor({"base_poscar": "x", "properties": [
            {"name": "p", "backend": "energy", "models": ["m"]},
            {"name": "p", "backend": "energy", "models": ["m"]}]})
    assert "duplicates" in str(info.value)


def test_zero_scale_raises():
    with pytest.raises(ValueError) as info:
        StructureScorePredictor({"base_poscar": "x", "properties": [
            {"name": "p", "backend": "energy", "models": ["m"], "scale": 0.0}]})
    assert "scale" in str(info.value)


def test_legacy_dp_models_key_accepted():
    p = StructureScorePredictor({"base_poscar": "x", "properties": [
        {"name": "p", "backend": "property", "dp_models": ["a.pt", "b.pt"]}]})
    assert p.properties[0]["models"] == ["a.pt", "b.pt"]


def test_default_builder_is_substitute():
    from rl_matdesign.predictors.builders.substitute import SubstituteBuilder
    p = StructureScorePredictor({"base_poscar": "x", "properties": [
        {"name": "p", "backend": "energy", "models": ["m"], "direction": "min"}]})
    assert isinstance(p._shared_builder, SubstituteBuilder)


# --------------------------------------------------------------------------- #
# share_structure: true  (old structure_pipeline)
# --------------------------------------------------------------------------- #

def test_shared_structure_weighted_combine():
    p = StructureScorePredictor(_shared_cfg())
    assert p.share_structure is True
    assert p.geo_opt_enabled is False

    # Stub: one shared build; per-property score fixed by name.
    p._shared_builder = type("B", (), {
        "build": lambda self, cand, *, n_configs, rng: [object()]})()
    fixed = {"conductivity": (10.0, 1.0), "stability": (5.0, 0.5)}
    p._score = lambda prop, structures: fixed[prop["name"]]

    reward, std = p.predict({"P_site": {"P": 0.95, "Mn": 0.05}})
    # direction=max, mean_minus_kstd, k=1 → v = mean - std.
    #   conductivity: 2 * (10 - 1) / 1   = 18.0
    #   stability:    1 * (5  - 0.5) / 5 = 0.9
    assert abs(reward - 18.9) < 1e-9
    assert abs(std - (1.0 + 0.5) / 2) < 1e-9   # mean of per-property stds


def test_per_objective_stats_and_predict_raw():
    p = StructureScorePredictor(_shared_cfg())
    p._shared_builder = type("B", (), {
        "build": lambda self, cand, *, n_configs, rng: [object()]})()
    fixed = {"conductivity": (10.0, 1.0), "stability": (5.0, 0.5)}
    p._score = lambda prop, structures: fixed[prop["name"]]

    stats = p.per_objective_stats({"x": 1.0})
    assert stats == {"conductivity": (10.0, 1.0), "stability": (5.0, 0.5)}

    # predict_raw ignores std: 2 * (+1)*10 / 1 + 1 * (+1)*5 / 5 = 20 + 1 = 21.0
    raw, slot = p.predict_raw({"x": 1.0})
    assert slot == 0.0
    assert abs(raw - 21.0) < 1e-9


# --------------------------------------------------------------------------- #
# share_structure: false  (old composite)
# --------------------------------------------------------------------------- #

def _independent_cfg(child_objective="mean"):
    return {
        "share_structure": False,
        "builder": "substitute",
        "base_poscar": "x",
        "site_symbol": "Ti",
        "k": 1.0,
        "properties": [
            {"name": "energy", "backend": "property", "models": ["m"],
             "direction": "min", "objective": child_objective, "weight": 1.0, "scale": 2.0},
            {"name": "bulk", "backend": "property", "models": ["m"],
             "direction": "max", "objective": child_objective, "weight": 0.5, "scale": 100.0},
        ],
    }


def test_independent_structures_built_per_objective():
    p = StructureScorePredictor(_independent_cfg())
    assert p.share_structure is False
    assert len(p._builders) == 2                       # one builder per objective
    assert p._builders[0] is not p._builders[1]

    calls = {"energy": 0, "bulk": 0}

    def _mk(tag):
        def _build(self, cand, *, n_configs, rng):
            calls[tag] += 1
            return [object()]
        return type("B", (), {"build": _build})()

    p._builders = [_mk("energy"), _mk("bulk")]
    fixed = {"energy": (4.0, 0.1), "bulk": (80.0, 5.0)}
    p._score = lambda prop, structures: fixed[prop["name"]]

    # objective=mean: reward = 1*(-1*4)/2 + 0.5*(1*80)/100 = -2 + 0.4 = -1.6
    reward, _ = p.predict({"Ti": 1.0})
    assert abs(reward - (-1.6)) < 1e-9
    assert calls == {"energy": 1, "bulk": 1}           # each objective built its own


def test_independent_mean_minus_kstd_matches_old_composite():
    p = StructureScorePredictor(_independent_cfg(child_objective="mean_minus_kstd"))
    p._builders = [type("B", (), {"build": lambda self, c, *, n_configs, rng: [object()]})()
                   for _ in range(2)]
    fixed = {"energy": (4.0, 0.1), "bulk": (80.0, 5.0)}
    p._score = lambda prop, structures: fixed[prop["name"]]
    # v_energy = (-1)*4 - 0.1 = -4.1 ; v_bulk = (+1)*80 - 5 = 75
    # reward = 1*-4.1/2 + 0.5*75/100 = -2.05 + 0.375 = -1.675
    reward, _ = p.predict({"Ti": 1.0})
    assert abs(reward - (-1.675)) < 1e-9


# --------------------------------------------------------------------------- #
# sweep: per-composition operating-condition optimization (e.g. temperature)
# --------------------------------------------------------------------------- #

def _sweep_cfg(**over):
    cfg = {
        "builder": "substitute",
        "base_poscar": "x",
        "site_symbol": "X",
        "k": 1.0,
        "sweep": {"name": "temperature", "values": [460, 470, 480, 490]},
        "properties": [
            {"name": "conductivity", "backend": "property", "models": ["m"],
             "fparam": [0.0, None, 6], "direction": "max", "weight": 2.0,
             "scale": 1.0, "objective": "mean"},
            {"name": "stability", "backend": "property", "models": ["s"],
             "fparam": [0.0, None, 6, 4], "direction": "max", "weight": 1.0,
             "scale": 1.0, "objective": "mean"},
        ],
    }
    cfg.update(over)
    return cfg


def _bind_sweep_stub(p):
    """Stub: build one shared cell; score depends on the swept T (fparam[1])."""
    p._shared_builder = type("B", (), {
        "build": lambda self, cand, *, n_configs, rng: [object()]})()

    def _score(prop, structures, *, fparam=None):
        T = fparam[1]
        if prop["name"] == "conductivity":
            return (float(T), 0.0)         # rises with T
        return (1000.0 - float(T), 0.0)    # stability falls with T
    p._score = _score


def test_fparam_null_without_sweep_raises():
    with pytest.raises(ValueError) as info:
        StructureScorePredictor({"base_poscar": "x", "properties": [
            {"name": "p", "backend": "property", "models": ["m"],
             "fparam": [0.0, None, 6]}]})
    assert "sweep" in str(info.value)


def test_sweep_picks_temperature_maximizing_combined_reward():
    p = StructureScorePredictor(_sweep_cfg())
    _bind_sweep_stub(p)

    # combined(T) = 2*T + (1000 - T) = 1000 + T  -> maximized at T = 490.
    reward, _ = p.predict({"P_site": {"P": 0.95, "Mn": 0.05}})
    assert abs(reward - (1000.0 + 490.0)) < 1e-9


def test_sweep_reports_chosen_temperature_in_stats():
    p = StructureScorePredictor(_sweep_cfg())
    _bind_sweep_stub(p)

    stats = p.per_objective_stats({"x": 1.0})
    assert stats["temperature"] == (490.0, 0.0)        # chosen operating T
    assert stats["conductivity"] == (490.0, 0.0)       # scored at that T
    assert stats["stability"] == (510.0, 0.0)


def test_sweep_with_temperature_independent_property():
    # One swept property + one T-independent property (no null in fparam):
    # the constant property is scored once and contributes equally at every T.
    cfg = _sweep_cfg(properties=[
        {"name": "conductivity", "backend": "property", "models": ["m"],
         "fparam": [0.0, None, 6], "direction": "max", "weight": 1.0,
         "scale": 1.0, "objective": "mean"},
        {"name": "fixed", "backend": "property", "models": ["m"],
         "fparam": [7.0], "direction": "max", "weight": 1.0, "scale": 1.0,
         "objective": "mean"},
    ])
    p = StructureScorePredictor(cfg)
    assert p.properties[1]["sweep_slots"] == []        # T-independent

    p._shared_builder = type("B", (), {
        "build": lambda self, cand, *, n_configs, rng: [object()]})()

    def _score(prop, structures, *, fparam=None):
        if prop["name"] == "conductivity":
            return (float(fparam[1]), 0.0)             # rises with T
        return (3.0, 0.0)                              # constant
    p._score = _score

    # combined(T) = T + 3  -> max at 490; reward = 493; chosen T = 490.
    reward, _ = p.predict({"x": 1.0})
    assert abs(reward - 493.0) < 1e-9
    assert p.per_objective_stats({"x": 1.0})["temperature"] == (490.0, 0.0)


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #

def test_registry_resolves_structure_score_and_old_names_gone():
    from rl_matdesign.registry import resolve_predictor, PREDICTORS
    assert "structure_score" in PREDICTORS
    for gone in ("dp_structure", "dp_property", "composite", "structure_pipeline"):
        assert gone not in PREDICTORS
    pred = resolve_predictor("structure_score", _shared_cfg(), seed=0)
    assert isinstance(pred, StructureScorePredictor)


def test_old_predictor_modules_deleted():
    import importlib
    for mod in ("dp_structure", "dp_property", "composite", "structure_pipeline"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(f"rl_matdesign.predictors.{mod}")
