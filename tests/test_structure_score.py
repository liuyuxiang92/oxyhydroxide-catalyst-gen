"""StructureScorePredictor — the unified structure-based predictor.

Covers cfg parsing/validation, the weighted-signed-scaled combine in both
structural regimes (share_structure true = old structure_pipeline; false = old
composite), and the consumer contract (predict / predict_raw /
per_objective_stats). DeepMD eval + geo-opt are GPU-gated and stubbed here.
"""
import os
import sys

import numpy as np
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
            {"name": "conductivity", "predictor": "dp_property", "models": ["m1.pt"],
             "head": "experiment", "direction": "max", "weight": 2.0, "scale": 1.0,
             "objective": "mean_minus_kstd"},
            {"name": "stability", "predictor": "dp_property", "models": ["s1.pt"],
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
        StructureScorePredictor({"base_poscar": "x", "properties": [
            {"name": "p", "predictor": "dp_energy", "direction": "min"}]})
    assert "models" in str(info.value)


def test_missing_predictor_raises():
    with pytest.raises(ValueError) as info:
        StructureScorePredictor({"base_poscar": "x", "properties": [
            {"name": "p", "models": ["m"], "direction": "min"}]})
    assert "predictor" in str(info.value)


def test_unknown_predictor_raises():
    # A non-structure predictor name is resolved as a composition leaf; an
    # unregistered name surfaces the registry's "Unknown predictor" error.
    with pytest.raises(ValueError) as info:
        StructureScorePredictor({"base_poscar": "x", "properties": [
            {"name": "p", "predictor": "magic", "direction": "max"}]})
    assert "magic" in str(info.value)


def test_missing_direction_raises():
    with pytest.raises(ValueError) as info:
        StructureScorePredictor({"base_poscar": "x", "properties": [
            {"name": "p", "predictor": "dp_energy", "models": ["m"]}]})
    assert "direction" in str(info.value)


def test_bad_direction_raises():
    with pytest.raises(ValueError) as info:
        StructureScorePredictor({"base_poscar": "x", "properties": [
            {"name": "p", "predictor": "dp_energy", "models": ["m"], "direction": "down"}]})
    assert "direction" in str(info.value)


def test_duplicate_names_raise():
    with pytest.raises(ValueError) as info:
        StructureScorePredictor({"base_poscar": "x", "properties": [
            {"name": "p", "predictor": "dp_energy", "models": ["m"], "direction": "min"},
            {"name": "p", "predictor": "dp_energy", "models": ["m"], "direction": "min"}]})
    assert "duplicates" in str(info.value)


def test_zero_scale_raises():
    with pytest.raises(ValueError) as info:
        StructureScorePredictor({"base_poscar": "x", "properties": [
            {"name": "p", "predictor": "dp_energy", "models": ["m"],
             "direction": "min", "scale": 0.0}]})
    assert "scale" in str(info.value)


def test_legacy_dp_models_key_accepted():
    p = StructureScorePredictor({"base_poscar": "x", "properties": [
        {"name": "p", "predictor": "dp_property", "dp_models": ["a.pt", "b.pt"],
         "direction": "max"}]})
    assert p.properties[0]["models"] == ["a.pt", "b.pt"]


def test_bad_transform_raises():
    with pytest.raises(ValueError) as info:
        StructureScorePredictor({"base_poscar": "x", "properties": [
            {"name": "p", "predictor": "dp_property", "models": ["m"],
             "direction": "max", "transform": "sqrt"}]})
    assert "transform" in str(info.value)


def test_transform_exp_maps_log_outputs_to_real_units(monkeypatch):
    import numpy as np
    import rl_matdesign.utils.dp_eval as dp_eval

    p = StructureScorePredictor({"base_poscar": "x", "properties": [
        {"name": "conductivity", "predictor": "dp_property", "models": ["m1.pt", "m2.pt"],
         "direction": "max", "transform": "exp", "objective": "mean"}]})
    # Avoid loading real DeepProperty models; ensemble emits *log* values.
    p._get_prop_models = lambda prop: ([object(), object()], {})
    # Shape is (n_models, n_structures) -- here 2 models x 1 structure, so the
    # spread below is genuine model disagreement, not decoration scatter.
    monkeypatch.setattr(
        dp_eval, "eval_property_ensemble",
        lambda *a, **k: np.array([[float(np.log(12.0))], [float(np.log(8.0))]]),
    )
    mean, std = p._score(p.properties[0], [object()])
    # exp applied per ensemble member, then mean/std on the real distribution.
    assert abs(mean - float(np.mean([12.0, 8.0]))) < 1e-9
    assert abs(std - float(np.std([12.0, 8.0]))) < 1e-9


def test_default_builder_is_substitute():
    from rl_matdesign.predictors.builders.substitute import SubstituteBuilder
    p = StructureScorePredictor({"base_poscar": "x", "properties": [
        {"name": "p", "predictor": "dp_energy", "models": ["m"], "direction": "min"}]})
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


def test_persistent_cache_recomputes_each_composition_once():
    p = StructureScorePredictor(_shared_cfg())
    calls = {"build": 0}

    def _build(self, cand, *, n_configs, rng):
        calls["build"] += 1
        return [object()]

    p._shared_builder = type("B", (), {"build": _build})()
    fixed = {"conductivity": (10.0, 1.0), "stability": (5.0, 0.5)}
    p._score = lambda prop, structures: fixed[prop["name"]]

    A = {"P_site": {"Mn": 0.05}}
    B = {"P_site": {"Fe": 0.05}}
    p.predict(A); p.predict(A)        # second A is a cache hit
    p.predict(B)                      # different composition -> miss
    p.predict(A)                      # A STILL cached (multi-entry, not single)
    # Old single-entry cache would rebuild A here (3 builds); persistent cache: 2.
    assert calls["build"] == 2
    assert len(p._stats_cache) == 2


def test_cache_size_cap_evicts_lru():
    p = StructureScorePredictor(_shared_cfg(predict_cache_size=2))
    p._shared_builder = type("B", (), {
        "build": lambda self, cand, *, n_configs, rng: [object()]})()
    fixed = {"conductivity": (10.0, 1.0), "stability": (5.0, 0.5)}
    p._score = lambda prop, structures: fixed[prop["name"]]
    for el in ("Mn", "Fe", "Co"):     # 3 comps, cap 2 -> oldest (Mn) evicted
        p.predict({"P_site": {el: 0.05}})
    assert len(p._stats_cache) == 2


def test_score_breakdown_returns_structures_and_raw_combine_matches():
    p = StructureScorePredictor(_shared_cfg())
    sentinel = object()
    p._shared_builder = type("B", (), {
        "build": lambda self, cand, *, n_configs, rng: [sentinel]})()
    fixed = {"conductivity": (10.0, 1.0), "stability": (5.0, 0.5)}
    p._score = lambda prop, structures: fixed[prop["name"]]

    bd = p.score_breakdown({"x": 1.0})
    # The exact (relaxed) cell that was scored is returned for reuse (no re-build).
    assert bd["structures"]["conductivity"] == [sentinel]
    # raw_combine on the breakdown stats == predict_raw (no extra scoring needed).
    assert abs(p.raw_combine(bd["best"]["stats"]) - p.predict_raw({"x": 1.0})[0]) < 1e-9
    # best reward == predict's full objective.
    assert abs(bd["best"]["reward"] - p.predict({"x": 1.0})[0]) < 1e-9


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
            {"name": "energy", "predictor": "dp_property", "models": ["m"],
             "direction": "min", "objective": child_objective, "weight": 1.0, "scale": 2.0},
            {"name": "bulk", "predictor": "dp_property", "models": ["m"],
             "direction": "max", "objective": child_objective, "weight": 0.5, "scale": 100.0},
        ],
    }


def test_independent_structures_built_per_objective():
    p = StructureScorePredictor(_independent_cfg())
    assert p.share_structure is False
    assert len(p._builders) == 2                       # one builder per objective
    assert p._builders["energy"] is not p._builders["bulk"]

    calls = {"energy": 0, "bulk": 0}

    def _mk(tag):
        def _build(self, cand, *, n_configs, rng):
            calls[tag] += 1
            return [object()]
        return type("B", (), {"build": _build})()

    p._builders = {"energy": _mk("energy"), "bulk": _mk("bulk")}
    fixed = {"energy": (4.0, 0.1), "bulk": (80.0, 5.0)}
    p._score = lambda prop, structures: fixed[prop["name"]]

    # objective=mean: reward = 1*(-1*4)/2 + 0.5*(1*80)/100 = -2 + 0.4 = -1.6
    reward, _ = p.predict({"Ti": 1.0})
    assert abs(reward - (-1.6)) < 1e-9
    assert calls == {"energy": 1, "bulk": 1}           # each objective built its own


def test_independent_mean_minus_kstd_matches_old_composite():
    p = StructureScorePredictor(_independent_cfg(child_objective="mean_minus_kstd"))
    _b = type("B", (), {"build": lambda self, c, *, n_configs, rng: [object()]})
    p._builders = {"energy": _b(), "bulk": _b()}
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
            {"name": "conductivity", "predictor": "dp_property", "models": ["m"],
             "fparam": [0.0, None, 6], "direction": "max", "weight": 2.0,
             "scale": 1.0, "objective": "mean"},
            {"name": "stability", "predictor": "dp_property", "models": ["s"],
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
            {"name": "p", "predictor": "dp_property", "models": ["m"],
             "direction": "max", "fparam": [0.0, None, 6]}]})
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
        {"name": "conductivity", "predictor": "dp_property", "models": ["m"],
         "fparam": [0.0, None, 6], "direction": "max", "weight": 1.0,
         "scale": 1.0, "objective": "mean"},
        {"name": "fixed", "predictor": "dp_property", "models": ["m"],
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


# --------------------------------------------------------------------------- #
# FQN leaf opting into structure scoring via predict_structures()
# --------------------------------------------------------------------------- #

def test_fqn_with_predict_structures_is_a_structure_objective(tmp_path, monkeypatch):
    mod = tmp_path / "fake_struct_pred.py"
    mod.write_text(
        "class FakeStructPredictor:\n"
        "    def __init__(self, cfg, *, seed=None):\n"
        "        self.cfg = cfg\n"
        "    def predict_structures(self, atoms_list):\n"
        "        return 7.0, 0.3\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    p = StructureScorePredictor({"base_poscar": "x", "properties": [
        {"name": "p", "predictor": "fake_struct_pred:FakeStructPredictor",
         "direction": "min"}]})

    prop = p.properties[0]
    assert prop["backend"] == "structure_fqn"
    assert p._has_structure is True          # a shared builder is required
    assert p._shared_builder is not None

    mean, std = p._score(prop, [object(), object()])
    assert (mean, std) == (7.0, 0.3)


def test_fqn_without_predict_structures_stays_composition(tmp_path, monkeypatch):
    # Regression pin: a leaf exposing only predict() must be unaffected by the
    # new dispatch — still composition-backed, no builder required.
    mod = tmp_path / "fake_comp_pred.py"
    mod.write_text(
        "class FakeCompPredictor:\n"
        "    def __init__(self, cfg, *, seed=None):\n"
        "        pass\n"
        "    def predict(self, composition):\n"
        "        return 1.0, 0.0\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    p = StructureScorePredictor({"properties": [
        {"name": "p", "predictor": "fake_comp_pred:FakeCompPredictor",
         "direction": "min"}]})

    prop = p.properties[0]
    assert prop["backend"] == "composition"
    assert p._has_structure is False
    assert p._shared_builder is None

    mean, std = p._score(prop, {"Fe": 1.0})
    assert (mean, std) == (1.0, 0.0)


def test_shared_structure_routes_built_cells_to_predict_structures(tmp_path, monkeypatch):
    # End-to-end through predict(): the shared builder's output must be exactly
    # what predict_structures() receives.
    mod = tmp_path / "fake_struct_pred2.py"
    mod.write_text(
        "class FakeStructPredictor2:\n"
        "    def __init__(self, cfg, *, seed=None):\n"
        "        pass\n"
        "    def predict_structures(self, atoms_list):\n"
        "        return float(len(atoms_list)), 0.0\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    p = StructureScorePredictor({
        "base_poscar": "x", "n_random_configs": 3,
        "properties": [
            {"name": "p", "predictor": "fake_struct_pred2:FakeStructPredictor2",
             "direction": "min", "objective": "mean"},
        ],
    })
    sentinel_structures = [object(), object(), object()]
    p._shared_builder = type(
        "B", (), {"build": lambda self, cand, *, n_configs, rng: sentinel_structures}
    )()

    stats = p.per_objective_stats({"Fe": 1.0})
    assert stats["p"] == (3.0, 0.0)          # len(sentinel_structures) == 3


# --------------------------------------------------------------------------- #
# direction: target — hit a value rather than push it up or down
# --------------------------------------------------------------------------- #

def _target_cfg(**over):
    prop = {"name": "bandgap", "predictor": "dp_property", "models": ["m"],
            "direction": "target", "target_value": 1.34, "objective": "mean"}
    prop.update(over.pop("prop", {}))
    cfg = {"base_poscar": "x", "properties": [prop]}
    cfg.update(over)
    return cfg


def test_target_direction_requires_target_value():
    with pytest.raises(ValueError) as info:
        StructureScorePredictor(_target_cfg(prop={"target_value": None}))
    msg = str(info.value)
    assert "target_value" in msg
    # Must disambiguate from the leaf's own checkpoint key, the whole reason for
    # the name: a user reading this should not reach for `model`/`target`.
    assert "checkpoint" in msg


def test_target_keys_without_target_direction_raise():
    with pytest.raises(ValueError) as info:
        StructureScorePredictor({"base_poscar": "x", "properties": [
            {"name": "p", "predictor": "dp_property", "models": ["m"],
             "direction": "min", "target_value": 1.0}]})
    assert "target_value" in str(info.value)
    assert "silently ignored" in str(info.value)


def test_negative_target_tolerance_raises():
    with pytest.raises(ValueError) as info:
        StructureScorePredictor(_target_cfg(prop={"target_tolerance": -0.1}))
    assert "target_tolerance" in str(info.value)


def test_unknown_direction_lists_target():
    with pytest.raises(ValueError) as info:
        StructureScorePredictor(_target_cfg(prop={"direction": "sideways"}))
    assert "target" in str(info.value)


def test_target_reward_is_negated_distance_and_symmetric():
    p = StructureScorePredictor(_target_cfg())
    prop = p.properties[0]
    assert p._property_value(prop, 1.34, 0.0) == pytest.approx(0.0)
    assert p._property_value(prop, 2.34, 0.0) == pytest.approx(-1.0)
    assert p._property_value(prop, 0.34, 0.0) == pytest.approx(-1.0)   # symmetric
    # strictly worse the farther away
    assert p._property_value(prop, 3.34, 0.0) < p._property_value(prop, 2.34, 0.0)


def test_target_tolerance_is_a_flat_deadband():
    p = StructureScorePredictor(_target_cfg(prop={"target_tolerance": 0.10}))
    prop = p.properties[0]
    for v in (1.34, 1.40, 1.44, 1.24):
        assert p._property_value(prop, v, 0.0) == pytest.approx(0.0)
    assert p._property_value(prop, 1.54, 0.0) == pytest.approx(-0.10)


def test_target_with_mean_minus_kstd_penalises_uncertainty():
    # Dead on target but uncertain must still be penalised: the objective applies
    # to the DISTANCE, so mean_minus_kstd reads as "effectively farther away".
    p = StructureScorePredictor(_target_cfg(
        prop={"objective": "mean_minus_kstd"}, k=1.0))
    prop = p.properties[0]
    assert p._property_value(prop, 1.34, 0.2) == pytest.approx(-0.2)
    assert p._property_value(prop, 1.84, 0.2) == pytest.approx(-0.7)


def test_target_and_min_objectives_combine():
    p = StructureScorePredictor({"base_poscar": "x", "k": 1.0, "properties": [
        {"name": "bandgap", "predictor": "dp_property", "models": ["m"],
         "direction": "target", "target_value": 1.34, "target_tolerance": 0.10,
         "objective": "mean", "weight": 1.0, "scale": 1.0},
        {"name": "ehull", "predictor": "dp_property", "models": ["m"],
         "direction": "min", "objective": "mean", "weight": 1.0, "scale": 2.0},
    ]})
    stats = {"bandgap": (1.54, 0.0), "ehull": (0.8, 0.0)}
    # bandgap: -(0.20 - 0.10) = -0.10 ; ehull: -0.8 / 2.0 = -0.40
    assert p._combine(stats) == pytest.approx(-0.5)
    assert p.raw_combine(stats) == pytest.approx(-0.5)   # std=0 => same


# --------------------------------------------------------------------------- #
# direction: target — hit a value rather than push it up or down
# --------------------------------------------------------------------------- #

def _target_cfg(**over):
    prop = {"name": "bandgap", "predictor": "dp_property", "models": ["m"],
            "direction": "target", "target_value": 1.34, "objective": "mean"}
    prop.update(over.pop("prop", {}))
    cfg = {"base_poscar": "x", "properties": [prop]}
    cfg.update(over)
    return cfg


def test_target_direction_requires_target_value():
    with pytest.raises(ValueError) as info:
        StructureScorePredictor(_target_cfg(prop={"target_value": None}))
    msg = str(info.value)
    assert "target_value" in msg
    # Must disambiguate from the leaf's own checkpoint key — the whole reason for
    # the name: a reader should not reach for `model:` here.
    assert "checkpoint" in msg


def test_target_keys_without_target_direction_raise():
    with pytest.raises(ValueError) as info:
        StructureScorePredictor({"base_poscar": "x", "properties": [
            {"name": "p", "predictor": "dp_property", "models": ["m"],
             "direction": "min", "target_value": 1.0}]})
    assert "target_value" in str(info.value)
    assert "silently ignored" in str(info.value)


def test_negative_target_tolerance_raises():
    with pytest.raises(ValueError) as info:
        StructureScorePredictor(_target_cfg(prop={"target_tolerance": -0.1}))
    assert "target_tolerance" in str(info.value)


def test_unknown_direction_lists_target():
    with pytest.raises(ValueError) as info:
        StructureScorePredictor(_target_cfg(prop={"direction": "sideways"}))
    assert "target" in str(info.value)


def test_target_reward_is_negated_distance_and_symmetric():
    p = StructureScorePredictor(_target_cfg())
    prop = p.properties[0]
    assert p._property_value(prop, 1.34, 0.0) == pytest.approx(0.0)
    assert p._property_value(prop, 2.34, 0.0) == pytest.approx(-1.0)
    assert p._property_value(prop, 0.34, 0.0) == pytest.approx(-1.0)   # symmetric
    assert p._property_value(prop, 3.34, 0.0) < p._property_value(prop, 2.34, 0.0)


def test_target_tolerance_is_a_flat_deadband():
    p = StructureScorePredictor(_target_cfg(prop={"target_tolerance": 0.10}))
    prop = p.properties[0]
    for v in (1.34, 1.40, 1.44, 1.24):
        assert p._property_value(prop, v, 0.0) == pytest.approx(0.0)
    assert p._property_value(prop, 1.54, 0.0) == pytest.approx(-0.10)


def test_target_with_mean_minus_kstd_penalises_uncertainty():
    # Dead on target but uncertain is still penalised: the objective applies to the
    # DISTANCE, so mean_minus_kstd reads as "effectively farther away".
    p = StructureScorePredictor(_target_cfg(prop={"objective": "mean_minus_kstd"}, k=1.0))
    prop = p.properties[0]
    assert p._property_value(prop, 1.34, 0.2) == pytest.approx(-0.2)
    assert p._property_value(prop, 1.84, 0.2) == pytest.approx(-0.7)


def test_target_and_min_objectives_combine():
    p = StructureScorePredictor({"base_poscar": "x", "k": 1.0, "properties": [
        {"name": "bandgap", "predictor": "dp_property", "models": ["m"],
         "direction": "target", "target_value": 1.34, "target_tolerance": 0.10,
         "objective": "mean", "weight": 1.0, "scale": 1.0},
        {"name": "ehull", "predictor": "dp_property", "models": ["m"],
         "direction": "min", "objective": "mean", "weight": 1.0, "scale": 2.0},
    ]})
    stats = {"bandgap": (1.54, 0.0), "ehull": (0.8, 0.0)}
    # bandgap: -(0.20 - 0.10) = -0.10 ; ehull: -0.8 / 2.0 = -0.40
    assert p._combine(stats) == pytest.approx(-0.5)
    assert p.raw_combine(stats) == pytest.approx(-0.5)   # std = 0 => identical


# --------------------------------------------------------------------------- #
# The predictor contract — general, not MGTransformer-specific.
#
# These use TOY leaves defined right here, on purpose: the guarantee under test is
# "any leaf wrapping one model gets std == 0 automatically", so exercising it
# through a real predictor would prove much less.
# --------------------------------------------------------------------------- #

class _ToyStructureLeaf:
    """One model. Scores each structure as offset + its `value` attribute."""

    def __init__(self, cfg, *, seed=None):
        self.offset = float(cfg.get("model", 0.0))

    def score_structures(self, atoms_list):
        return [self.offset + float(getattr(a, "value", 0.0)) for a in atoms_list]


class _ToyCompositionLeaf:
    def __init__(self, cfg, *, seed=None):
        self.offset = float(cfg.get("model", 0.0))

    def score(self, composition):
        return self.offset + float(sum(composition.values()))


class _ToySelfFoldingLeaf:
    """Owns a genuine internal ensemble, so it reports its own (mean, std)."""

    def __init__(self, cfg, *, seed=None):
        pass

    def predict_structures(self, atoms_list):
        return 7.0, 3.0


class _Struct:
    def __init__(self, value):
        self.value = value


_TOY_STRUCT = f"{__name__}:_ToyStructureLeaf"
_TOY_COMP = f"{__name__}:_ToyCompositionLeaf"
_TOY_SELF = f"{__name__}:_ToySelfFoldingLeaf"


def _leaf_cfg(predictor, model, **over):
    prop = {"name": "p", "predictor": predictor, "model": model,
            "direction": "max", "objective": "mean"}
    prop.update(over)
    return {"base_poscar": "x", "properties": [prop]}


def test_single_model_leaf_has_exactly_zero_std():
    # THE guarantee: one model => no uncertainty, whatever the structure scatter.
    p = StructureScorePredictor(_leaf_cfg(_TOY_STRUCT, 10.0))
    structures = [_Struct(0.0), _Struct(2.0), _Struct(10.0)]   # wildly scattered
    mean, std = p._score(p.properties[0], structures)
    assert mean == pytest.approx(14.0)        # 10 + mean(0, 2, 10)
    assert std == 0.0                          # exactly zero, not merely small


def test_model_list_builds_one_leaf_per_model_and_std_is_their_spread():
    p = StructureScorePredictor(_leaf_cfg(_TOY_STRUCT, [1.0, 3.0, 5.0]))
    prop = p.properties[0]
    assert len(prop["instance"]) == 3          # one instance per model path
    structures = [_Struct(0.0), _Struct(4.0)]  # per-model mean adds +2.0
    mean, std = p._score(prop, structures)
    assert mean == pytest.approx(5.0)          # mean(3, 5, 7)
    assert std == pytest.approx(float(np.std([3.0, 5.0, 7.0])))


def test_structure_scatter_does_not_leak_into_std():
    # Same models, very different within-model scatter => identical std. This is the
    # bug the nested fold fixes: pooling would have inflated std here.
    p = StructureScorePredictor(_leaf_cfg(_TOY_STRUCT, [1.0, 3.0]))
    prop = p.properties[0]
    _m1, s1 = p._score(prop, [_Struct(0.0), _Struct(0.0)])
    _m2, s2 = p._score(prop, [_Struct(-50.0), _Struct(50.0)])
    assert s1 == pytest.approx(s2)


def test_score_leaf_is_routed_as_a_composition_objective():
    p = StructureScorePredictor(_leaf_cfg(_TOY_COMP, 1.0))
    assert p.properties[0]["backend"] == "composition"
    assert p._score(p.properties[0], {"Fe": 0.25, "Ni": 0.75}) == (pytest.approx(2.0), 0.0)


def test_score_structures_leaf_is_routed_as_a_structure_objective():
    p = StructureScorePredictor(_leaf_cfg(_TOY_STRUCT, 0.0))
    assert p.properties[0]["backend"] == "structure_fqn"


def test_self_folding_leaf_passes_its_own_mean_and_std_through():
    # rf_magpie's guarantee: a leaf with real internal uncertainty keeps it.
    p = StructureScorePredictor(_leaf_cfg(_TOY_SELF, None))
    mean, std = p._score(p.properties[0], [_Struct(0.0)])
    assert (mean, std) == (pytest.approx(7.0), pytest.approx(3.0))


def test_close_is_forwarded_to_leaves_and_is_safe_without_one():
    closed = []

    class _Closable(_ToyStructureLeaf):
        def close(self):
            closed.append(True)

    import sys as _sys
    setattr(_sys.modules[__name__], "_Closable", _Closable)
    p = StructureScorePredictor(_leaf_cfg(f"{__name__}:_Closable", [1.0, 2.0]))
    p.close()
    assert len(closed) == 2                    # one per model instance
    StructureScorePredictor(_leaf_cfg(_TOY_STRUCT, 1.0)).close()   # no close() -> no error
