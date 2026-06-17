"""Order-invariance guarantees for state, featurizer, and predictor contract.

The framework's design enforces that picking elements in any order yields the
same per-step state features (whenever the partial multisets match) and the
same terminal reward. These tests pin that property so future refactors of
the featurizer or predictor wiring can't silently break it.

Three layers are covered separately so a regression points at the right place:
- ``featurize_formula`` is order-invariant on the input formula string.
- ``CompositionEnv`` / ``IntegerRatioEnv`` produce identical per-step features
  at any step index where the *partial multisets* match between two episodes.
- The predictor contract (``predict(composition) -> (mean, std)``) treats the
  input dict as an unordered mapping.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pytest


# ============================================================================
# Featurizer layer
# ============================================================================

@pytest.mark.parametrize(
    "perm_strings",
    [
        # 2-element
        ("Fe0.50Ni0.50", "Ni0.50Fe0.50"),
        # 3-element with one duplicate fraction (catches tie-break leaks)
        ("Fe0.20Ti0.30Ni0.50", "Ni0.50Ti0.30Fe0.20", "Ti0.30Ni0.50Fe0.20"),
        # 5-element with all equal fractions (worst-case for any
        # ordering-by-fraction implementations)
        ("Fe0.20Co0.20Ni0.20Mn0.20Cr0.20", "Cr0.20Mn0.20Ni0.20Co0.20Fe0.20"),
    ],
)
def test_featurize_formula_is_order_invariant_fractional(perm_strings):
    """Multiset-equivalent fractional formula strings produce identical features."""
    from rl_matdesign.featurization import featurize_formula

    feats = [featurize_formula(s) for s in perm_strings]
    # Tolerance allows for benign floating-point round-off in downstream
    # matminer aggregations (mean / std / weighted sum can reorder operands
    # depending on the dict insertion order coming out of pymatgen). The
    # architectural invariance is "same composition → same features up to
    # floating-point precision" — we test exactly that.
    for k, other in enumerate(feats[1:], start=1):
        np.testing.assert_allclose(
            feats[0], other, rtol=1e-10, atol=1e-12,
            err_msg=f"featurize_formula({perm_strings[k]!r}) drifted from "
                    f"featurize_formula({perm_strings[0]!r}) beyond fp noise"
        )


@pytest.mark.parametrize(
    "perm_strings",
    [
        ("Fe2Ti3", "Ti3Fe2"),
        ("Ba1Ti2O3", "Ti2Ba1O3", "O3Ba1Ti2"),
        ("Ni3Fe3Co2Mn1O5", "Mn1Co2Fe3Ni3O5", "O5Mn1Co2Ni3Fe3"),
    ],
)
def test_featurize_formula_is_order_invariant_integer(perm_strings):
    """Integer-ratio formula strings are equally normalized via pymatgen."""
    from rl_matdesign.featurization import featurize_formula

    feats = [featurize_formula(s) for s in perm_strings]
    # Tolerance allows for benign floating-point round-off in downstream
    # matminer aggregations (mean / std / weighted sum can reorder operands
    # depending on the dict insertion order coming out of pymatgen). The
    # architectural invariance is "same composition → same features up to
    # floating-point precision" — we test exactly that.
    for k, other in enumerate(feats[1:], start=1):
        np.testing.assert_allclose(
            feats[0], other, rtol=1e-10, atol=1e-12,
            err_msg=f"featurize_formula({perm_strings[k]!r}) drifted from "
                    f"featurize_formula({perm_strings[0]!r}) beyond fp noise"
        )


# ============================================================================
# CompositionEnv layer
# ============================================================================

def _oh(idx: int, n: int) -> Tuple[float, ...]:
    v = [0.0] * n
    v[idx] = 1.0
    return tuple(v)


def _walk(env, action_seq: Sequence[Tuple[str, str]]) -> None:
    """Roll *env* through a sequence of (element, fraction-string) pairs."""
    env.initialize()
    species_set = env.species_set
    fraction_set = env.fraction_set
    for elem, frac_str in action_seq:
        elem_oh = _oh(species_set.index(elem), len(species_set))
        comp_oh = _oh(fraction_set.index(frac_str), len(fraction_set))
        env.step((elem_oh, comp_oh))


def _multiset(prefix: Sequence[Tuple[str, str]]) -> Tuple[Tuple[str, str], ...]:
    """Canonical sorted multiset of (elem, frac_str) pairs."""
    return tuple(sorted(prefix))


@pytest.fixture
def composition_env_factory():
    """Build a CompositionEnv with a deterministic reward."""
    from rl_matdesign.env import CompositionEnv

    def _build(reward_fn=None):
        return CompositionEnv(
            species_set=["Fe", "Co", "Ni", "Mn", "Cr"],
            fraction_set=["0.05", "0.10", "0.15", "0.20", "0.25",
                          "0.30", "0.35", "0.40"],
            anion_formula="",
            n_components=5,
            reward_fn=reward_fn or (lambda _f: 1.0),
        )

    return _build


def _deterministic_reward(formula: str) -> float:
    """Reward that depends only on the (canonical) Composition, not on order.

    Used to verify terminal-reward invariance without relying on a real
    predictor. Hashing the sorted (element, fraction) tuple guarantees the
    reward function itself is order-invariant by construction.
    """
    from pymatgen.core.composition import Composition

    c = Composition(formula).fractional_composition
    items = tuple(sorted((str(el), round(frac, 6)) for el, frac in c.items()))
    return float(sum(idx * f for idx, (_, f) in enumerate(items, start=1)))


@pytest.mark.parametrize(
    "perm",
    [
        [("Fe", "0.30"), ("Ni", "0.30"), ("Co", "0.20"), ("Mn", "0.10"), ("Cr", "0.10")],
        [("Cr", "0.10"), ("Mn", "0.10"), ("Co", "0.20"), ("Ni", "0.30"), ("Fe", "0.30")],
        [("Ni", "0.30"), ("Cr", "0.10"), ("Fe", "0.30"), ("Co", "0.20"), ("Mn", "0.10")],
        [("Co", "0.20"), ("Fe", "0.30"), ("Mn", "0.10"), ("Cr", "0.10"), ("Ni", "0.30")],
    ],
)
def test_composition_env_terminal_reward_invariant(perm, composition_env_factory):
    """Terminal reward depends only on the final multiset, not order."""
    env = composition_env_factory(reward_fn=_deterministic_reward)
    _walk(env, perm)
    expected = _deterministic_reward(env.terminal_formula)
    assert env.path[-1].reward == pytest.approx(expected)
    # And cross-check: ANY permutation reaching this multiset gives the same R.
    canonical_R = _deterministic_reward(
        "Cr0.10Mn0.10Co0.20Ni0.30Fe0.30"
    )
    assert env.path[-1].reward == pytest.approx(canonical_R)


def test_composition_env_features_match_when_partial_multisets_match(
    composition_env_factory,
):
    """Per-step features match at every step k whose partial multiset matches."""
    a_seq = [("Fe", "0.30"), ("Ni", "0.30"), ("Co", "0.20"),
             ("Mn", "0.10"), ("Cr", "0.10")]
    b_seq = [("Ni", "0.30"), ("Fe", "0.30"), ("Co", "0.20"),
             ("Mn", "0.10"), ("Cr", "0.10")]

    env_a = composition_env_factory()
    env_b = composition_env_factory()
    _walk(env_a, a_seq)
    _walk(env_b, b_seq)

    for k in range(5):
        prefix_a = _multiset(a_seq[:k])
        prefix_b = _multiset(b_seq[:k])
        if prefix_a == prefix_b:
            np.testing.assert_allclose(
                env_a.path[k].state_material_features,
                env_b.path[k].state_material_features,
                rtol=1e-10, atol=1e-12,
                err_msg=(
                    f"step {k}: partial multisets are equal ({prefix_a}) "
                    "but per-step state features differ beyond fp noise"
                ),
            )


def test_composition_env_features_first_step_always_equal(composition_env_factory):
    """Step 0 features are always equal (empty bag for every permutation)."""
    env_a = composition_env_factory()
    env_b = composition_env_factory()
    _walk(env_a, [("Fe", "0.30"), ("Ni", "0.30"), ("Co", "0.20"),
                  ("Mn", "0.10"), ("Cr", "0.10")])
    _walk(env_b, [("Cr", "0.10"), ("Mn", "0.10"), ("Co", "0.20"),
                  ("Ni", "0.30"), ("Fe", "0.30")])
    # Empty bag has no operands to reorder, so this can be bit-exact.
    np.testing.assert_array_equal(
        env_a.path[0].state_material_features,
        env_b.path[0].state_material_features,
    )


# ============================================================================
# IntegerRatioEnv layer
# ============================================================================

def _walk_int(env, action_seq: Sequence[Tuple[str, str]]) -> None:
    env.initialize()
    species_set = env.species_set
    ratio_set = env.ratio_set
    for elem, digit_str in action_seq:
        elem_oh = _oh(species_set.index(elem), len(species_set))
        comp_oh = _oh(ratio_set.index(digit_str), len(ratio_set))
        env.step((elem_oh, comp_oh))


@pytest.fixture
def integer_env_factory():
    from rl_matdesign.env_integer import IntegerRatioEnv

    def _build(reward_fn=None):
        return IntegerRatioEnv(
            species_set=["Fe", "Co", "Ni", "Mn", "Cr"],
            ratio_set=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            n_components=5,
            reward_fn=reward_fn or (lambda _f: 1.0),
        )

    return _build


@pytest.mark.parametrize(
    "perm",
    [
        [("Fe", "3"), ("Ni", "3"), ("Co", "2"), ("Mn", "1"), ("Cr", "1")],
        [("Cr", "1"), ("Mn", "1"), ("Co", "2"), ("Ni", "3"), ("Fe", "3")],
        [("Ni", "3"), ("Cr", "1"), ("Fe", "3"), ("Co", "2"), ("Mn", "1")],
    ],
)
def test_integer_env_terminal_reward_invariant(perm, integer_env_factory):
    env = integer_env_factory(reward_fn=_deterministic_reward)
    _walk_int(env, perm)
    canonical_R = _deterministic_reward("Fe3Ni3Co2Mn1Cr1")
    assert env.path[-1].reward == pytest.approx(canonical_R)


def test_integer_env_features_match_when_partial_multisets_match(
    integer_env_factory,
):
    a_seq = [("Fe", "3"), ("Ni", "3"), ("Co", "2"), ("Mn", "1"), ("Cr", "1")]
    b_seq = [("Ni", "3"), ("Fe", "3"), ("Co", "2"), ("Mn", "1"), ("Cr", "1")]

    env_a = integer_env_factory()
    env_b = integer_env_factory()
    _walk_int(env_a, a_seq)
    _walk_int(env_b, b_seq)

    for k in range(5):
        prefix_a = _multiset(a_seq[:k])
        prefix_b = _multiset(b_seq[:k])
        if prefix_a == prefix_b:
            np.testing.assert_array_equal(
                env_a.path[k].state_material_features,
                env_b.path[k].state_material_features,
                err_msg=(
                    f"IntegerRatioEnv step {k}: partial multisets equal "
                    f"({prefix_a}) but features differ"
                ),
            )


# ============================================================================
# Predictor contract layer
# ============================================================================

def _permute_dict(d: Dict[str, float], shift: int = 1) -> Dict[str, float]:
    """Rebuild *d* with keys in a rotated insertion order."""
    items = list(d.items())
    rotated = items[shift:] + items[:shift]
    return {k: v for k, v in rotated}


@pytest.mark.parametrize(
    "composition",
    [
        {"Fe": 0.5, "Ni": 0.5},
        {"Fe": 0.20, "Ti": 0.30, "Ni": 0.50},
        {"Fe": 0.20, "Co": 0.20, "Ni": 0.20, "Mn": 0.20, "Cr": 0.20},
    ],
)
def test_contract_example_predictor_is_order_invariant(composition):
    """A predictor that follows the contract returns the same value
    for differently-ordered dicts representing the same composition.

    This documents the contract via test: user-written predictors should
    pass this same shape of assertion."""

    class _ContractPredictor:
        """Example of a contract-compliant predictor: it treats the input
        dict as an unordered mapping and only uses sorted access."""

        def predict(self, comp):
            items = sorted(comp.items(), key=lambda kv: str(kv[0]))
            # Order-invariant operation: weighted sum with alphabet rank.
            mean = sum(rank * frac for rank, (_, frac) in enumerate(items, 1))
            std = max(comp.values()) - min(comp.values())
            return float(mean), float(std)

    p = _ContractPredictor()
    base = p.predict(composition)
    for shift in (1, 2, len(composition) - 1):
        permuted = _permute_dict(composition, shift)
        assert p.predict(permuted) == base


def test_contract_violation_is_detected():
    """A predictor that *does* depend on dict insertion order should fail
    the same assertion. This documents what a buggy predictor looks like."""

    class _BadPredictor:
        """Anti-pattern: assumes the first key is the 'major' element."""

        def predict(self, comp):
            first_elem = next(iter(comp))
            return float(comp[first_elem]), 0.0

    bad = _BadPredictor()
    comp = {"Fe": 0.5, "Ni": 0.3, "Co": 0.2}
    assert bad.predict(comp) != bad.predict(_permute_dict(comp, 1))


@pytest.mark.parametrize("filename", [
    "optimal_sinter_RF.joblib",
    "optimal_calcine_RF.joblib",
])
def test_rf_magpie_predictor_is_order_invariant(filename):
    """RFMagpiePredictor returns identical (mean, std) for permuted dicts."""
    pytest.importorskip("matminer")
    pytest.importorskip("joblib")

    rf_path = (
        Path(__file__).resolve().parent.parent
        / "models" / "sinter_calcine" / filename
    )
    if not rf_path.exists():
        pytest.skip(f"model file not found at {rf_path}")

    from rl_matdesign.predictors.rf_magpie import RFMagpiePredictor

    pred = RFMagpiePredictor(model_path=str(rf_path))
    comp = {"Fe": 0.3, "Ni": 0.3, "Co": 0.2, "Mn": 0.1, "Cr": 0.1}
    base = pred.predict(comp)
    # Clear the per-call cache so we genuinely re-evaluate, not return cache hit.
    pred._cache.clear()
    permuted = pred.predict(_permute_dict(comp, 2))
    assert base == pytest.approx(permuted)
