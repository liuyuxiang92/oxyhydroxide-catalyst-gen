"""CompositionEnv per-element bounds + fixed_order_amount tests.

The default OOH-style path (no element_bounds, no episode_style override)
must remain byte-identical to the pre-refactor behavior.
"""
from __future__ import annotations

import random
import numpy as np
import pytest


def _zero_feats(_state: str) -> np.ndarray:
    """Featurizer stub so tests don't pull in matminer at import time."""
    return np.zeros(4)


def test_existing_ooh_path_unchanged():
    """Default constructor produces the same env behavior as before the refactor."""
    from rl_matdesign.env import CompositionEnv
    env = CompositionEnv(
        species_set=["Ni", "Fe", "Co", "Mn", "Cu", "Al"],
        fraction_set=["0.05", "0.10", "0.15", "0.20", "0.25", "0.30",
                      "0.35", "0.40", "0.45", "0.50"],
        anion_formula="O2H1",
        n_components=5,
        state_featurizer=_zero_feats,
    )
    assert env.episode_style == "element_then_amount"
    assert env._element_unit_bounds is None
    assert env._total_units == 20
    env.initialize()
    for _ in range(5):
        env.step(env.allowed_actions()[0])
    assert env.counter == 5
    assert env.terminal_formula.endswith("O2H1")


def _ti_alloy_env(state_featurizer=_zero_feats):
    """Helper: build the Ti-alloy-style env used across several tests."""
    from rl_matdesign.env import CompositionEnv
    return CompositionEnv(
        species_set=["Ti", "Al", "V", "Cr", "Zr"],
        fraction_set=[f"{i/100:.2f}" for i in range(0, 101)],
        anion_formula="",
        n_components=5,
        total_units=100,
        element_bounds={
            "Ti": (0.45, 0.90), "Al": (0.00, 0.25),
            "V": (0.00, 0.15), "Cr": (0.00, 0.10), "Zr": (0.00, 0.10),
        },
        episode_style="fixed_order_amount",
        state_featurizer=state_featurizer,
    )


def test_ti_alloy_env_construction():
    env = _ti_alloy_env()
    assert env.n_components == 5
    assert env.episode_style == "fixed_order_amount"
    assert env._element_unit_bounds == [(45, 90), (0, 25), (0, 15), (0, 10), (0, 10)]
    assert env._total_units == 100


def test_ti_alloy_100_random_episodes_respect_bounds():
    """The strict invariant: any random rollout must satisfy bounds AND sum=1.0."""
    env = _ti_alloy_env()
    bounds = {"Ti": (0.45, 0.90), "Al": (0, 0.25), "V": (0, 0.15),
              "Cr": (0, 0.10), "Zr": (0, 0.10)}
    random.seed(0)
    for _ in range(100):
        env.initialize()
        for _ in range(env.n_components):
            actions = env.allowed_actions()
            assert actions, "no actions should always be available in a feasible Ti env"
            env.step(random.choice(actions))
        comp = env.terminal_cation_fractions()
        for el, (lo, hi) in bounds.items():
            v = comp.get(el, 0.0)
            assert lo - 1e-9 <= v <= hi + 1e-9
        assert abs(sum(comp.values()) - 1.0) < 1e-9


def test_fixed_order_amount_emits_element_at_step_index():
    """Each step's available actions are pinned to species_set[counter]."""
    from rl_matdesign.encoding import decode_one_hot
    env = _ti_alloy_env()
    env.initialize()
    for step_idx in range(env.n_components):
        actions = env.allowed_actions()
        for elem_oh, _ in actions:
            elem = decode_one_hot(elem_oh, env.species_set)
            assert elem == env.species_set[step_idx]
        env.step(actions[0])


def test_infeasible_bounds_raise():
    from rl_matdesign.env import CompositionEnv
    with pytest.raises(ValueError) as info:
        CompositionEnv(
            species_set=["A", "B"],
            fraction_set=["0.1", "0.2"],
            n_components=2,
            total_units=10,
            element_bounds={"A": (0.0, 0.2), "B": (0.0, 0.2)},  # max sum = 0.4 < 1.0
            episode_style="fixed_order_amount",
        )
    assert "infeasible" in str(info.value).lower()


def _ti_alloy_eta_env(n_components=4, state_featurizer=_zero_feats):
    """Ti-alloy-style env in element_then_amount mode with per-element bounds."""
    from rl_matdesign.env import CompositionEnv
    return CompositionEnv(
        species_set=["Ti", "Al", "V", "Cr", "Zr"],
        fraction_set=[f"{i/100:.2f}" for i in range(0, 101)],
        anion_formula="",
        n_components=n_components,
        total_units=100,
        element_bounds={
            "Ti": (0.45, 0.90), "Al": (0.00, 0.25),
            "V": (0.00, 0.15), "Cr": (0.00, 0.10), "Zr": (0.05, 0.45),
        },
        episode_style="element_then_amount",
        state_featurizer=state_featurizer,
    )


def test_element_then_amount_with_bounds_constructs():
    env = _ti_alloy_eta_env()
    assert env.episode_style == "element_then_amount"
    assert env._element_unit_bounds_by_sym["Ti"] == (45, 90)
    assert env._element_unit_bounds_by_sym["Zr"] == (5, 45)


def test_element_then_amount_random_episodes_respect_bounds():
    """Free element choice + per-element bounds: every rollout obeys bounds,
    sums to 1.0, and includes the mandatory (lo>0) cations Ti and Zr."""
    env = _ti_alloy_eta_env(n_components=4)
    bounds = {"Ti": (0.45, 0.90), "Al": (0, 0.25), "V": (0, 0.15),
              "Cr": (0, 0.10), "Zr": (0.05, 0.45)}
    random.seed(0)
    for _ in range(200):
        env.initialize()
        for _ in range(env.n_components):
            actions = env.allowed_actions()
            assert actions, "feasible env must always offer a legal action"
            env.step(random.choice(actions))
        comp = env.terminal_cation_fractions()
        assert abs(sum(comp.values()) - 1.0) < 1e-9
        for el, f in comp.items():
            lo, hi = bounds[el]
            assert lo - 1e-9 <= f <= hi + 1e-9
        assert comp.get("Ti", 0.0) > 0 and comp.get("Zr", 0.0) > 0   # mandatory


def test_element_then_amount_too_many_mandatory_raises():
    from rl_matdesign.env import CompositionEnv
    with pytest.raises(ValueError, match="mandatory"):
        CompositionEnv(
            species_set=["A", "B", "C"],
            fraction_set=[f"{i/100:.2f}" for i in range(0, 101)],
            n_components=2,
            total_units=100,
            element_bounds={"A": (0.2, 0.5), "B": (0.2, 0.5), "C": (0.2, 0.5)},  # 3 mandatory > 2 slots
            episode_style="element_then_amount",
        )


def test_element_then_amount_infeasible_total_raises():
    from rl_matdesign.env import CompositionEnv
    with pytest.raises(ValueError, match="infeasible"):
        CompositionEnv(
            species_set=["A", "B", "C"],
            fraction_set=[f"{i/100:.2f}" for i in range(0, 101)],
            n_components=2,
            total_units=100,
            element_bounds={"A": (0.0, 0.2), "B": (0.0, 0.2), "C": (0.0, 0.2)},  # max 2-subset = 0.4 < 1.0
            episode_style="element_then_amount",
        )


def test_step_rejects_wrong_element_in_fixed_order_mode():
    """In fixed_order_amount mode, the element at step k must be species_set[k]."""
    from rl_matdesign.encoding import encode_choice
    env = _ti_alloy_env()
    env.initialize()
    wrong_elem_oh = tuple(encode_choice("Al", env.species_set).tolist())   # step 0 expects Ti
    comp_oh = tuple(encode_choice("0.10", env.fraction_set).tolist())
    with pytest.raises(ValueError) as info:
        env.step((wrong_elem_oh, comp_oh))
    assert "Ti" in str(info.value) or "fixed_order_amount" in str(info.value)


def test_lo_greater_than_hi_raises():
    from rl_matdesign.env import CompositionEnv
    with pytest.raises(ValueError):
        CompositionEnv(
            species_set=["A", "B"],
            fraction_set=["0.5"],
            n_components=2,
            element_bounds={"A": (0.6, 0.4), "B": (0.0, 1.0)},
            episode_style="fixed_order_amount",
        )
