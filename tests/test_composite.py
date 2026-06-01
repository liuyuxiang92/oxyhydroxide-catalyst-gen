"""CompositePredictor cfg-parsing + math tests (no DeepMD imports needed).

Children are stubbed via a custom FQN-resolved class that returns hard-coded
(mean, std) per composition. The composite math is exercised end-to-end without
loading any DP backend.
"""
from __future__ import annotations

import pytest


class StubChild:
    """Returns hard-coded (mean, std) keyed by element-sum of composition.

    The composite cfg passes through ``hardcoded`` as a dict mapping a string
    key to ``(mean, std)``; the key is built per call from the composition's
    sorted element symbols joined with '-'. Predictor signature matches what
    ``resolve_predictor`` expects: ``__init__(cfg, *, seed=None)``.
    """

    def __init__(self, cfg, *, seed=None):
        self.fixed_mean = float(cfg.get("mean", 0.0))
        self.fixed_std = float(cfg.get("std", 0.0))
        self.seed = seed
        self.calls = 0

    def raw_mean_std(self, composition):
        self.calls += 1
        return self.fixed_mean, self.fixed_std

    def predict(self, composition):
        m, s = self.raw_mean_std(composition)
        return m, s


_STUB_FQN = "tests.test_composite:StubChild"


def _two_objective_cfg(child_objective="mean", **overrides):
    cfg = {
        "k": 1.0,
        "objectives": [
            {
                "name": "energy",
                "predictor": _STUB_FQN,
                "direction": "min",
                "objective": child_objective,
                "weight": 1.0,
                "scale": 2.0,
                "mean": 4.0,
                "std": 0.1,
            },
            {
                "name": "bulk",
                "predictor": _STUB_FQN,
                "direction": "max",
                "objective": child_objective,
                "weight": 0.5,
                "scale": 100.0,
                "mean": 80.0,
                "std": 5.0,
            },
        ],
    }
    cfg.update(overrides)
    return cfg


def test_empty_objectives_raises():
    from rl_matdesign.predictors.composite import CompositePredictor
    with pytest.raises(ValueError) as info:
        CompositePredictor({"objectives": []})
    assert "objectives" in str(info.value)


def test_missing_predictor_key_raises():
    from rl_matdesign.predictors.composite import CompositePredictor
    with pytest.raises(ValueError) as info:
        CompositePredictor({"objectives": [{"name": "x", "direction": "min"}]})
    assert "predictor" in str(info.value)


def test_bad_direction_raises():
    from rl_matdesign.predictors.composite import CompositePredictor
    with pytest.raises(ValueError) as info:
        CompositePredictor({"objectives": [{
            "predictor": _STUB_FQN,
            "direction": "down",
        }]})
    assert "direction" in str(info.value)


def test_duplicate_names_raise():
    from rl_matdesign.predictors.composite import CompositePredictor
    cfg = {"objectives": [
        {"name": "p", "predictor": _STUB_FQN, "direction": "min"},
        {"name": "p", "predictor": _STUB_FQN, "direction": "max"},
    ]}
    with pytest.raises(ValueError) as info:
        CompositePredictor(cfg)
    assert "duplicates" in str(info.value)


def test_zero_scale_raises():
    from rl_matdesign.predictors.composite import CompositePredictor
    with pytest.raises(ValueError) as info:
        CompositePredictor({"objectives": [{
            "predictor": _STUB_FQN, "direction": "min", "scale": 0.0,
        }]})
    assert "scale" in str(info.value)


def test_weighted_sum_math_objective_mean():
    """With per-child objective=mean, v_k = dir*m_k → reward = Σ w*dir*m/scale."""
    from rl_matdesign.predictors.composite import CompositePredictor
    p = CompositePredictor(_two_objective_cfg(child_objective="mean"))
    # dir(min)=-1, dir(max)=+1
    # contrib_energy = 1.0 * -1 * 4.0 / 2.0 = -2.0
    # contrib_bulk   = 0.5 * +1 * 80.0 / 100.0 = +0.4
    # reward = -1.6, std slot = 0.0
    reward, std = p.predict({"Ti": 1.0})
    assert std == 0.0
    assert reward == pytest.approx(-1.6, rel=1e-9)


def test_weighted_sum_math_objective_mean_minus_kstd():
    """With per-child objective=mean_minus_kstd, v_k = dir*m_k - k*s_k.

    Direction is applied to mean BEFORE the std fold so the std term acts as
    an uncertainty penalty regardless of direction (exploit semantics in
    both min and max). Matches single-predictor convention.
    """
    from rl_matdesign.predictors.composite import CompositePredictor
    p = CompositePredictor(_two_objective_cfg(child_objective="mean_minus_kstd", k=1.0))
    # v_energy = (-1)*4.0 - 1.0*0.1 = -4.1   (direction: min applied to mean)
    # v_bulk   = (+1)*80.0 - 1.0*5.0 = 75.0  (direction: max applied to mean)
    # contrib_energy = 1.0 * -4.1 / 2.0 = -2.05
    # contrib_bulk   = 0.5 * 75.0 / 100.0 = +0.375
    # reward = -1.675
    reward, _ = p.predict({"Ti": 1.0})
    assert reward == pytest.approx(-1.675, rel=1e-9)


def test_per_child_objective_mix():
    """One child exploit (mean_minus_kstd), another explore (mean_plus_kstd)."""
    from rl_matdesign.predictors.composite import CompositePredictor
    cfg = {
        "k": 1.0,
        "objectives": [
            {"name": "energy", "predictor": _STUB_FQN, "direction": "min",
             "objective": "mean_minus_kstd", "weight": 1.0, "scale": 2.0,
             "mean": 4.0, "std": 0.1},
            {"name": "bulk", "predictor": _STUB_FQN, "direction": "max",
             "objective": "mean_plus_kstd", "weight": 0.5, "scale": 100.0,
             "mean": 80.0, "std": 5.0},
        ],
    }
    p = CompositePredictor(cfg)
    # v_energy = (-1)*4.0 - 1.0*0.1 = -4.1                       (exploit-min)
    # v_bulk   = (+1)*80.0 + 1.0*5.0 = 85.0                      (explore-max)
    # contrib_energy = 1.0 * -4.1 / 2.0   = -2.05
    # contrib_bulk   = 0.5 * 85.0 / 100.0 = +0.425
    # reward = -1.625
    reward, _ = p.predict({"Ti": 1.0})
    assert reward == pytest.approx(-1.625, rel=1e-9)


def test_legacy_top_level_objective_rejected():
    """A top-level objective: key should be flagged with a migration hint."""
    from rl_matdesign.predictors.composite import CompositePredictor
    cfg = {
        "objective": "mean_minus_kstd",  # legacy — no longer accepted
        "objectives": [{"name": "x", "predictor": _STUB_FQN, "direction": "min"}],
    }
    with pytest.raises(ValueError) as info:
        CompositePredictor(cfg)
    msg = str(info.value)
    assert "top-level" in msg
    assert "per entry" in msg


def test_per_child_objective_defaults_to_mean_minus_kstd():
    """When objective: is omitted per child, default is exploit (mean_minus_kstd)."""
    from rl_matdesign.predictors.composite import CompositePredictor
    cfg = {
        "k": 1.0,
        "objectives": [
            {"name": "x", "predictor": _STUB_FQN, "direction": "max",
             "weight": 1.0, "scale": 1.0, "mean": 10.0, "std": 2.0},
        ],
    }
    p = CompositePredictor(cfg)
    assert p.objectives == ["mean_minus_kstd"]
    # v = (+1)*10.0 - 1.0*2.0 = 8.0 → reward = 1.0 * 8.0 / 1.0 = 8.0
    reward, _ = p.predict({"Ti": 1.0})
    assert reward == pytest.approx(8.0, rel=1e-9)


def test_invalid_per_child_objective_raises():
    from rl_matdesign.predictors.composite import CompositePredictor
    cfg = {
        "objectives": [{
            "name": "x", "predictor": _STUB_FQN, "direction": "min",
            "objective": "ucb",  # not a real option
        }],
    }
    with pytest.raises(ValueError) as info:
        CompositePredictor(cfg)
    assert "objective" in str(info.value)
    assert "ucb" in str(info.value)


def test_predict_raw_skips_std_penalty():
    """predict_raw uses raw m_k regardless of objective."""
    from rl_matdesign.predictors.composite import CompositePredictor
    p = CompositePredictor(_two_objective_cfg(child_objective="mean_minus_kstd", k=2.0))
    # predict_raw should ignore k, just use means.
    # contrib_energy = 1.0 * -1 * 4.0 / 2.0 = -2.0
    # contrib_bulk   = 0.5 * +1 * 80.0 / 100.0 = +0.4
    raw, _ = p.predict_raw({"Ti": 1.0})
    assert raw == pytest.approx(-1.6, rel=1e-9)


def test_per_objective_stats_returns_each_child():
    from rl_matdesign.predictors.composite import CompositePredictor
    p = CompositePredictor(_two_objective_cfg())
    stats = p.per_objective_stats({"Ti": 1.0})
    assert set(stats) == {"energy", "bulk"}
    assert stats["energy"] == (4.0, 0.1)
    assert stats["bulk"] == (80.0, 5.0)


def test_memoization_one_call_per_child_per_composition():
    """Calling predict + predict_raw + per_objective_stats for one composition
    should evaluate each child exactly once."""
    from rl_matdesign.predictors.composite import CompositePredictor
    p = CompositePredictor(_two_objective_cfg())
    comp = {"Ti": 1.0}
    p.predict(comp)
    p.predict_raw(comp)
    p.per_objective_stats(comp)
    for child in p.children:
        assert child.calls == 1


def test_minimize_maximize_aliases_accepted():
    from rl_matdesign.predictors.composite import CompositePredictor
    cfg = {"objectives": [
        {"name": "a", "predictor": _STUB_FQN, "direction": "minimize",
         "objective": "mean", "mean": 1.0},
        {"name": "b", "predictor": _STUB_FQN, "direction": "maximize",
         "objective": "mean", "mean": 2.0},
    ]}
    p = CompositePredictor(cfg)
    assert p.dirs == [-1.0, 1.0]


def test_negative_weight_flips_contribution():
    """weight=-1 is equivalent to flipping direction."""
    from rl_matdesign.predictors.composite import CompositePredictor
    cfg = {"objectives": [
        {"name": "x", "predictor": _STUB_FQN, "direction": "min",
         "objective": "mean", "weight": -1.0, "scale": 1.0, "mean": 5.0},
    ]}
    p = CompositePredictor(cfg)
    # contrib = -1 * -1 * 5.0 / 1.0 = +5.0
    reward, _ = p.predict({"Ti": 1.0})
    assert reward == pytest.approx(5.0, rel=1e-9)


def test_child_seed_is_offset_per_index():
    """Composite seed S → child k seed = S + k."""
    from rl_matdesign.predictors.composite import CompositePredictor
    cfg = _two_objective_cfg()
    p = CompositePredictor(cfg, seed=100)
    assert p.children[0].seed == 100
    assert p.children[1].seed == 101


def test_shared_defaults_inherited_by_subcfgs():
    """Top-level base_poscar / site_symbol / n_random_configs inherit into
    sub-cfgs that don't set them. Sub-cfg overrides win."""
    from rl_matdesign.predictors.composite import CompositePredictor

    captured_cfgs = []

    class CaptureChild:
        def __init__(self, cfg, *, seed=None):
            captured_cfgs.append(dict(cfg))
        def raw_mean_std(self, composition):
            return 0.0, 0.0

    # Register temporarily via FQN. Use a side-channel: put the class in a
    # module-level name so FQN resolution finds it.
    import sys, types
    mod = types.ModuleType("_test_capture_mod")
    mod.CaptureChild = CaptureChild
    sys.modules["_test_capture_mod"] = mod
    try:
        cfg = {
            "base_poscar": "/shared/poscar",
            "site_symbol": "X",
            "n_random_configs": 7,
            "objectives": [
                {"name": "a", "predictor": "_test_capture_mod:CaptureChild",
                 "direction": "min"},
                {"name": "b", "predictor": "_test_capture_mod:CaptureChild",
                 "direction": "max", "n_random_configs": 3},  # override
            ],
        }
        CompositePredictor(cfg)
        assert captured_cfgs[0]["base_poscar"] == "/shared/poscar"
        assert captured_cfgs[0]["n_random_configs"] == 7
        assert captured_cfgs[1]["base_poscar"] == "/shared/poscar"
        # sub-cfg override wins
        assert captured_cfgs[1]["n_random_configs"] == 3
    finally:
        del sys.modules["_test_capture_mod"]


def test_registry_resolves_composite_short_name():
    from rl_matdesign.registry import resolve_predictor
    from rl_matdesign.predictors.composite import CompositePredictor
    p = resolve_predictor("composite", _two_objective_cfg())
    assert isinstance(p, CompositePredictor)


def test_child_without_raw_mean_std_rejected():
    """A child that lacks raw_mean_std should fail composite init early."""
    from rl_matdesign.predictors.composite import CompositePredictor

    class NoRawChild:
        def __init__(self, cfg, *, seed=None):
            pass
        def predict(self, composition):
            return 0.0, 0.0

    import sys, types
    mod = types.ModuleType("_test_noraw_mod")
    mod.NoRawChild = NoRawChild
    sys.modules["_test_noraw_mod"] = mod
    try:
        cfg = {"objectives": [
            {"name": "x", "predictor": "_test_noraw_mod:NoRawChild",
             "direction": "min"},
        ]}
        with pytest.raises(TypeError) as info:
            CompositePredictor(cfg)
        assert "raw_mean_std" in str(info.value)
    finally:
        del sys.modules["_test_noraw_mod"]
