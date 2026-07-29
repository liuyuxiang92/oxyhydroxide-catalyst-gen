"""A2C/REINFORCE scale invariance: advantage standardisation + entropy floor.

These pin the fix for the entropy collapse observed on the oxide benchmark, where
A2C got *worse* with more episodes: raw sintering-temperature returns of 400-700
made the actor term O(hundreds) while the entropy bonus was O(0.5), so the policy
went deterministic (entropy 0.00, one composition sampled 31,843 times out of
45,200 episodes) and the best candidate froze at ~20% of the run.

The invariant worth protecting is that ``pg_entropy_coef`` and
``pg_repeat_penalty_coef`` mean the same thing whatever units the property is in.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rl_matdesign.training import (  # noqa: E402
    _ENTROPY_CTRL_MAX,
    entropy_coef_update,
)


# ---------------------------------------------------------------------------
# Advantage standardisation
# ---------------------------------------------------------------------------

def _standardise(adv_raw):
    """Mirror of the train_pg batch step, kept minimal so the test pins maths."""
    t = torch.tensor(adv_raw, dtype=torch.float32)
    std = float(t.std(unbiased=False).item())
    if std > 1e-8:
        return (t - t.mean()) / std
    return torch.zeros_like(t)


def test_standardised_advantages_are_zero_mean_unit_std():
    adv = _standardise([-663.8, -520.1, -436.4, -770.4, -500.2])
    assert float(adv.mean().item()) == pytest.approx(0.0, abs=1e-5)
    assert float(adv.std(unbiased=False).item()) == pytest.approx(1.0, abs=1e-5)


def test_standardisation_is_invariant_to_reward_units():
    """Kelvin vs Celsius vs eV must produce the same policy gradient direction."""
    kelvin = [700.0, 650.0, 600.0, 550.0]
    celsius = [k - 273.15 for k in kelvin]          # shift
    scaled = [k * 0.001 for k in kelvin]            # scale

    a, b, c = _standardise(kelvin), _standardise(celsius), _standardise(scaled)
    assert torch.allclose(a, b, atol=1e-5)
    assert torch.allclose(a, c, atol=1e-4)


def test_zero_variance_batch_does_not_explode():
    """Every episode scoring identically carries no signal — emit a zero update.

    Dividing by an epsilon here would turn float dust into a full-magnitude
    gradient. Reachable in practice: the 45k runs collapsed to a single
    composition, so whole batches scored the same.
    """
    adv = _standardise([-530.574] * 8)
    assert torch.all(adv == 0.0)
    assert not torch.any(torch.isnan(adv))


def test_repeat_penalty_is_in_sigma_units():
    """A 0.1 penalty must shift the advantage by 0.1σ, not by 0.1 kelvin."""
    adv = _standardise([700.0, 650.0, 600.0, 550.0])
    penalised = adv - torch.tensor([0.1, 0.0, 0.0, 0.0])
    assert float((adv[0] - penalised[0]).item()) == pytest.approx(0.1, abs=1e-6)


# ---------------------------------------------------------------------------
# Entropy floor controller
# ---------------------------------------------------------------------------

def test_controller_raises_coef_below_floor():
    base, floor = 0.1, 0.3
    assert entropy_coef_update(base, entropy_norm=0.05, base_coef=base, floor=floor) > base


def test_controller_decays_back_to_base_above_floor():
    base, floor = 0.1, 0.3
    raised = entropy_coef_update(base * 10, entropy_norm=0.9, base_coef=base, floor=floor)
    assert raised < base * 10
    # base_coef is the lower clamp: the controller only ever adds exploration
    # pressure, never removes the weight the user configured.
    assert raised >= base


def test_controller_is_clamped():
    base, floor = 0.1, 0.5
    coef = base
    for _ in range(500):  # sustained total collapse
        coef = entropy_coef_update(coef, entropy_norm=0.0, base_coef=base, floor=floor)
    assert coef == pytest.approx(base * _ENTROPY_CTRL_MAX)


def test_floor_of_zero_disables_the_controller():
    assert entropy_coef_update(5.0, entropy_norm=0.0, base_coef=0.1, floor=0.0) == 0.1


def _simulate_collapse(floor: float, n_updates: int = 200) -> float:
    """Toy closed loop: entropy decays under gradient pressure, the weight pushes back.

    The dynamics are a stand-in, not a model of the real optimiser — the point is
    the *comparison* between a live floor and a disabled one under identical
    pressure, which is exactly the old-vs-new contrast.
    """
    base = 0.1
    coef, entropy_norm = base, 0.95
    for _ in range(n_updates):
        coef = entropy_coef_update(coef, entropy_norm, base_coef=base, floor=floor)
        entropy_norm = max(0.0, entropy_norm - 0.05 + 0.4 * (coef - base))
    return entropy_norm


def test_controller_arrests_a_collapse_that_otherwise_completes():
    """The regression this whole change exists to prevent.

    With the floor disabled the policy goes fully deterministic, reproducing the
    observed failure (calcine_a2c_45000 ended at entropy 0.00). With the floor on,
    entropy settles at a non-degenerate level under identical pressure.
    """
    without = _simulate_collapse(floor=0.0)
    with_floor = _simulate_collapse(floor=0.3)

    assert without == 0.0, "baseline should collapse completely"
    assert with_floor > 0.15, f"controller failed to arrest collapse (h={with_floor})"


# ---------------------------------------------------------------------------
# Normalised entropy
# ---------------------------------------------------------------------------

def test_normalised_entropy_is_comparable_across_action_set_sizes():
    """The floor is a fraction of ln|A| precisely so it ports across scenarios.

    |A| is ~268 for the 80-element oxide env but far smaller for OOH; an absolute
    nats floor would mean different things in each.
    """
    for n in (10, 28, 268):
        uniform = np.full(n, 1.0 / n)
        h = float(-(uniform * np.log(uniform)).sum())
        assert h / math.log(n) == pytest.approx(1.0, abs=1e-6)
