"""DQN trajectory-permutation augmentation: correctness + dedup + validity rules."""
from __future__ import annotations

import collections
import math
import random
from typing import List, Tuple

import numpy as np
import pytest


# ============================================================================
# Helpers
# ============================================================================

def _oh(idx: int, n: int) -> Tuple[float, ...]:
    v = [0.0] * n
    v[idx] = 1.0
    return tuple(v)


def _build_env_simple(reward_fn=None, phase_filter=None):
    """5-step env over 5 distinct cations, fractions on 0.05 grid summing to 1."""
    from rl_matdesign.env import CompositionEnv

    return CompositionEnv(
        species_set=["Fe", "Co", "Ni", "Mn", "Cr"],
        fraction_set=["0.05", "0.10", "0.15", "0.20", "0.25",
                      "0.30", "0.35", "0.40"],
        anion_formula="",
        n_components=5,
        reward_fn=reward_fn or (lambda _f: 17.0),
        phase_filter=phase_filter,
    )


def _build_env_with_O_last():
    """IntegerRatioEnv with LastStepElementFilter(O, reserve_for_last=True)."""
    from rl_matdesign.env_integer import IntegerRatioEnv
    from rl_matdesign.constraints.last_step_element import LastStepElementFilter

    f = LastStepElementFilter(
        required_elements=["O"],
        nonzero_ratio=True,
        reserve_for_last=True,
    )
    return IntegerRatioEnv(
        species_set=["O", "Fe", "Co", "Ni", "Mn", "Cr"],
        ratio_set=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        n_components=5,
        reward_fn=lambda _f: 42.0,
        phase_filter=f,
    )


def _build_env_fixed_order():
    """CompositionEnv with episode_style='fixed_order_amount'."""
    from rl_matdesign.env import CompositionEnv

    return CompositionEnv(
        species_set=["Fe", "Co", "Ni", "Mn", "Cr"],
        fraction_set=["0.05", "0.10", "0.15", "0.20", "0.25",
                      "0.30", "0.35", "0.40"],
        anion_formula="",
        n_components=5,
        reward_fn=lambda _f: 11.0,
        episode_style="fixed_order_amount",
    )


def _walk(env, action_seq):
    env.initialize()
    species_set = env.species_set
    fraction_set = env.fraction_set
    for elem, frac_str in action_seq:
        elem_oh = _oh(species_set.index(elem), len(species_set))
        comp_oh = _oh(fraction_set.index(frac_str), len(fraction_set))
        env.step((elem_oh, comp_oh))


def _make_elem_feats(env):
    """Tiny stand-in for the Magpie elem_feats_scaled array."""
    return np.eye(len(env.species_set), dtype=float)


# ============================================================================
# Baseline behavior: K=0 is byte-for-byte identical to add_episode_to_buffer
# ============================================================================

def test_k_zero_is_no_op_identical_to_original():
    """K=0 must produce exactly the same rows as the unaugmented path."""
    from rl_matdesign.training import (
        _augment_episode_in_buffer, add_episode_to_buffer,
    )

    env = _build_env_simple()
    seq = [("Fe", "0.30"), ("Ni", "0.30"), ("Co", "0.20"),
           ("Mn", "0.10"), ("Cr", "0.10")]

    _walk(env, seq)
    fraction_set = env.fraction_set
    elem_feats = _make_elem_feats(env)

    buf_a: collections.deque = collections.deque()
    add_episode_to_buffer(env.path, buf_a, elem_feats, fraction_set)

    buf_b: collections.deque = collections.deque()
    _augment_episode_in_buffer(
        env, env.path, buf_b, elem_feats, fraction_set, K=0,
    )

    assert len(buf_a) == len(buf_b) == 5
    for ra, rb in zip(buf_a, buf_b):
        assert ra["a_elem_idx"] == rb["a_elem_idx"]
        assert ra["a_comp_val"] == rb["a_comp_val"]
        assert ra["reward"] == rb["reward"]
        assert ra["done"] == rb["done"]
        np.testing.assert_array_equal(ra["s_mat_raw"], rb["s_mat_raw"])


# ============================================================================
# Buffer row count with augmentation (bounded by (K+1)*N, dedup may drop some)
# ============================================================================

def test_buffer_row_count_within_bound():
    from rl_matdesign.training import _augment_episode_in_buffer

    env = _build_env_simple()
    seq = [("Fe", "0.30"), ("Ni", "0.30"), ("Co", "0.20"),
           ("Mn", "0.10"), ("Cr", "0.10")]

    _walk(env, seq)
    fraction_set = env.fraction_set
    elem_feats = _make_elem_feats(env)

    K = 3
    N = 5
    buf: collections.deque = collections.deque()
    random.seed(0)
    _augment_episode_in_buffer(
        env, env.path, buf, elem_feats, fraction_set, K=K,
    )
    # With no last-position pin and 5 distinct elements, all 5! orderings exist.
    # Step 0 row varies by first element so augmented step-0 rows are NEW;
    # late-step rows may collide. Total is in [N, (K+1)*N].
    assert N <= len(buf) <= (K + 1) * N


# ============================================================================
# Terminal reward + bootstrap target invariance
# ============================================================================

def test_terminal_reward_matches_original_for_all_augmented_rows():
    """Every terminal-row reward across original + augmented copies equals R."""
    from rl_matdesign.training import _augment_episode_in_buffer

    env = _build_env_simple(reward_fn=lambda _f: 5.5)
    seq = [("Fe", "0.30"), ("Ni", "0.30"), ("Co", "0.20"),
           ("Mn", "0.10"), ("Cr", "0.10")]

    _walk(env, seq)
    fraction_set = env.fraction_set
    elem_feats = _make_elem_feats(env)

    buf: collections.deque = collections.deque()
    random.seed(1)
    _augment_episode_in_buffer(
        env, env.path, buf, elem_feats, fraction_set, K=4,
    )

    terminal_rewards = [r["reward"] for r in buf if r["done"]]
    # At least the original's terminal row is there; augmented terminal rows
    # are dedup'd when last position is pinned, but here there's no pin so
    # the augmented terminal rows differ in (s_mat, action) — survive dedup.
    assert terminal_rewards, "expected at least one terminal row"
    for r in terminal_rewards:
        assert r == pytest.approx(5.5)


def test_non_terminal_rows_have_zero_immediate_reward():
    """All non-terminal rows carry reward=0; only the terminal row gets R."""
    from rl_matdesign.training import _augment_episode_in_buffer

    env = _build_env_simple(reward_fn=lambda _f: 7.7)
    seq = [("Fe", "0.30"), ("Ni", "0.30"), ("Co", "0.20"),
           ("Mn", "0.10"), ("Cr", "0.10")]
    _walk(env, seq)

    buf: collections.deque = collections.deque()
    random.seed(2)
    _augment_episode_in_buffer(
        env, env.path, buf, _make_elem_feats(env), env.fraction_set, K=3,
    )
    for r in buf:
        if r["done"]:
            assert r["reward"] == pytest.approx(7.7)
        else:
            assert r["reward"] == 0.0


# ============================================================================
# Validity rule: last_step_element.reserve_for_last keeps O at end
# ============================================================================

def test_last_position_pinned_by_last_step_element_filter():
    """With LastStepElementFilter(O, reserve_for_last=True), all augmented
    paths still end with an O action."""
    from rl_matdesign.training import _augment_episode_in_buffer
    from rl_matdesign.encoding import decode_one_hot

    env = _build_env_with_O_last()
    seq = [("Fe", "3"), ("Co", "2"), ("Ni", "2"), ("Mn", "1"), ("O", "5")]
    _walk(env, seq)

    buf: collections.deque = collections.deque()
    random.seed(3)
    _augment_episode_in_buffer(
        env, env.path, buf, _make_elem_feats(env), env.fraction_set, K=3,
    )

    # Locate every terminal row and verify its action is "add O".
    terminal_rows = [r for r in buf if r["done"]]
    assert terminal_rows
    O_idx = env.species_set.index("O")
    for r in terminal_rows:
        assert r["a_elem_idx"] == O_idx, (
            "last action should remain O when LastStepElementFilter pins it"
        )


def test_last_position_pin_detected_inside_chain_filter():
    """ChainConstraintFilter wrapping LastStepElementFilter is detected."""
    pytest.importorskip("smact")
    from rl_matdesign.training import _detect_last_position_pin
    from rl_matdesign.registry import resolve_constraint
    from rl_matdesign.env_integer import IntegerRatioEnv

    chain = resolve_constraint("chain", {
        "filters": [
            {"constraint_filter": "last_step_element",
             "required_elements": ["O"],
             "reserve_for_last": True},
            {"constraint_filter": "smact_charge",
             "smact_anions": [{"symbol": "O", "charge": -2, "stoich": 1.5}]},
        ],
    }, env=None)
    env = IntegerRatioEnv(
        species_set=["O", "Fe", "Co", "Ni"],
        ratio_set=["0", "1", "2"],
        n_components=4,
        phase_filter=chain,
    )
    pinned = _detect_last_position_pin(env)
    assert pinned == {"O"}


# ============================================================================
# Validity rule: fixed_order_amount short-circuits
# ============================================================================

def test_fixed_order_amount_short_circuits_to_original_only(capsys):
    """episode_style='fixed_order_amount' disables augmentation entirely."""
    from rl_matdesign.training import _augment_episode_in_buffer

    env = _build_env_fixed_order()
    # Fixed-order picks the species_set order forcibly; we just pick amounts.
    seq = [("Fe", "0.30"), ("Co", "0.30"), ("Ni", "0.20"),
           ("Mn", "0.10"), ("Cr", "0.10")]
    _walk(env, seq)

    buf: collections.deque = collections.deque()
    _augment_episode_in_buffer(
        env, env.path, buf, _make_elem_feats(env), env.fraction_set, K=5,
    )
    # Only the original N rows: K augmentations were skipped.
    assert len(buf) == 5
    captured = capsys.readouterr()
    assert "fixed_order_amount" in captured.out.lower()


# ============================================================================
# Dedup: within-episode duplicates are skipped
# ============================================================================

def test_dedup_within_episode_drops_duplicate_terminal_rows():
    """With last position pinned, augmented terminal rows match the original's
    terminal row (same prefix multiset + same last action) and are skipped."""
    from rl_matdesign.training import _augment_episode_in_buffer

    env = _build_env_with_O_last()
    seq = [("Fe", "3"), ("Co", "2"), ("Ni", "2"), ("Mn", "1"), ("O", "5")]
    _walk(env, seq)

    buf: collections.deque = collections.deque()
    random.seed(4)
    K = 3
    _augment_episode_in_buffer(
        env, env.path, buf, _make_elem_feats(env), env.fraction_set, K=K,
    )

    # No two row dicts within this single episode's augmentation pass share
    # the same (s_mat_raw, s_step, a_elem_idx, a_comp_val, done) fingerprint.
    fingerprints = set()
    for r in buf:
        fp = (
            tuple(np.round(r["s_mat_raw"], 6).tolist()),
            tuple(r["s_step"].tolist()),
            int(r["a_elem_idx"]),
            float(r["a_comp_val"]),
            bool(r["done"]),
        )
        assert fp not in fingerprints, (
            "found duplicate fingerprint within a single episode's "
            "augmentation pass — dedup should have skipped it"
        )
        fingerprints.add(fp)

    # The terminal row is pinned + multiset converges → exactly ONE terminal row.
    terminal_rows = [r for r in buf if r["done"]]
    assert len(terminal_rows) == 1, (
        "with last position pinned, all augmented terminal rows are "
        "identical to the original's terminal row and should dedup to one"
    )


def test_dedup_scope_is_per_episode_not_global():
    """Identical-state rows across two separate episode augmentations are kept,
    not deduped. They represent legitimate independent observations."""
    from rl_matdesign.training import _augment_episode_in_buffer

    env_a = _build_env_with_O_last()
    seq = [("Fe", "3"), ("Co", "2"), ("Ni", "2"), ("Mn", "1"), ("O", "5")]
    _walk(env_a, seq)

    env_b = _build_env_with_O_last()
    _walk(env_b, seq)  # second episode reaches the same multiset

    buf: collections.deque = collections.deque()
    random.seed(5)
    K = 0
    _augment_episode_in_buffer(
        env_a, env_a.path, buf, _make_elem_feats(env_a),
        env_a.fraction_set, K=K,
    )
    rows_after_a = len(buf)
    _augment_episode_in_buffer(
        env_b, env_b.path, buf, _make_elem_feats(env_b),
        env_b.fraction_set, K=K,
    )
    # Both episodes produced 5 rows even though they're row-for-row identical.
    assert len(buf) == 2 * rows_after_a == 10


# ============================================================================
# Augmentation truly produces new (state, action) coverage at early steps
# ============================================================================

def test_augmentation_produces_distinct_first_step_action():
    """With no last-position pin and K > 0, at least one augmented step-0 row
    differs from the original's step-0 action (new first-element coverage)."""
    from rl_matdesign.training import _augment_episode_in_buffer

    env = _build_env_simple()
    seq = [("Fe", "0.30"), ("Ni", "0.30"), ("Co", "0.20"),
           ("Mn", "0.10"), ("Cr", "0.10")]
    _walk(env, seq)

    buf: collections.deque = collections.deque()
    random.seed(6)
    _augment_episode_in_buffer(
        env, env.path, buf, _make_elem_feats(env), env.fraction_set, K=3,
    )

    # Original step-0 has Fe (idx 0) as the first action.
    step0_actions = []
    for r in buf:
        # Step-0 rows have s_step == [1, 0, 0, 0, 0] (one-hot of step 1).
        if int(np.argmax(r["s_step"])) == 0 and r["a_comp_val"] > 0.0:
            step0_actions.append(r["a_elem_idx"])

    assert len(set(step0_actions)) >= 2, (
        "expected augmentation to produce at least one new first-step element"
    )


# ============================================================================
# K capping when alternatives_count < K
# ============================================================================

def test_K_capped_when_alternatives_run_out(capsys):
    """With N=2 + last-pinned, alternatives_count=0 → no augmentation
    inserted regardless of K."""
    from rl_matdesign.training import _augment_episode_in_buffer
    from rl_matdesign.env_integer import IntegerRatioEnv
    from rl_matdesign.constraints.last_step_element import LastStepElementFilter

    f = LastStepElementFilter(["O"], reserve_for_last=True)
    env = IntegerRatioEnv(
        species_set=["O", "Fe"],
        ratio_set=["0", "1", "2"],
        n_components=2,
        phase_filter=f,
        reward_fn=lambda _f: 1.0,
    )
    _walk(env, [("Fe", "1"), ("O", "1")])

    buf: collections.deque = collections.deque()
    _augment_episode_in_buffer(
        env, env.path, buf, _make_elem_feats(env), env.fraction_set, K=10,
    )
    # Only the original 2 rows.
    assert len(buf) == 2


# ============================================================================
# reward_fn is restored after augmentation
# ============================================================================

def test_reward_fn_restored_after_augmentation():
    """env.reward_fn is the original callable after augmentation completes."""
    from rl_matdesign.training import _augment_episode_in_buffer

    sentinel = lambda f: 12.5
    env = _build_env_simple(reward_fn=sentinel)
    seq = [("Fe", "0.30"), ("Ni", "0.30"), ("Co", "0.20"),
           ("Mn", "0.10"), ("Cr", "0.10")]
    _walk(env, seq)

    assert env.reward_fn is sentinel

    buf: collections.deque = collections.deque()
    _augment_episode_in_buffer(
        env, env.path, buf, _make_elem_feats(env), env.fraction_set, K=3,
    )
    assert env.reward_fn is sentinel, (
        "augmentation should restore env.reward_fn even on the happy path"
    )
