"""Tests for MultiGroupEnv (N sublattice groups, each summing to 1).

Covers:
* N=1 reproduces CompositionEnv exactly (backward-compat keystone).
* N=2 episodes walk groups in order, each group sums to 1, terminal is structured.
* prior_groups delivers earlier groups' compositions to a later group's filter.
* fraction_set expansion ({min,max,step} regular grid / explicit list) and the
  total_units it derives.
* The "host absorbs the rest" pattern is now plain composition + element_bounds,
  not a dedicated host: knob / host_complement filter.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rl_matdesign.env import CompositionEnv  # noqa: E402
from rl_matdesign.env_multigroup import MultiGroupEnv, normalize_group_spec  # noqa: E402
from rl_matdesign.constraints.base import ConstraintFilter  # noqa: E402
from rl_matdesign.registry import build_constraints, resolve_constraint  # noqa: E402
from rl_matdesign.encoding import decode_one_hot  # noqa: E402


def _pick(env, element, value):
    """Step the allowed action whose (element/slot-name, value-code) match."""
    for a in env.allowed_actions():
        el = decode_one_hot(a[0], env.species_set)
        code = decode_one_hot(a[1], env.fraction_set)
        if el == element and abs(float(code) - float(value)) < 1e-9:
            env.step(a)
            return
    raise AssertionError(f"no allowed action for ({element}, {value})")


def _allowed_values(env, slot_name):
    """Decoded value-codes available for a given slot/element at the current step."""
    return {
        decode_one_hot(a[1], env.fraction_set)
        for a in env.allowed_actions()
        if decode_one_hot(a[0], env.species_set) == slot_name
    }


def _drive_first_allowed(env):
    env.initialize()
    while env.counter < env.n_components:
        env.step(env.allowed_actions()[0])
    return env


def _build(group_specs):
    """Normalize + resolve each group's filters:, exactly like build_env does."""
    built = []
    for g in group_specs:
        gs = normalize_group_spec(g)
        gs["constraint_filter"] = build_constraints(gs, env=None)
        built.append(gs)
    return MultiGroupEnv(groups=built)


def test_n1_reproduces_composition_env():
    kw = dict(species_set=["Fe", "Ni", "Co", "Mn"], fraction_set=["0.25", "0.50", "0.75"],
              n_components=4, total_units=4)
    ce = _drive_first_allowed(CompositionEnv(**kw))
    mg = _drive_first_allowed(MultiGroupEnv(groups=[dict(name="g", **kw)]))

    assert mg.n_components == ce.n_components
    assert mg.species_set == ce.species_set
    assert mg.fraction_set == ce.fraction_set
    assert len(mg.path) == len(ce.path) == 4
    for a, b in zip(ce.path, mg.path):
        assert np.allclose(a.state_material_features, b.state_material_features)
        assert np.array_equal(a.state_step_onehot, b.state_step_onehot)
        assert np.array_equal(a.action_elem_onehot, b.action_elem_onehot)
        assert np.array_equal(a.action_comp_onehot, b.action_comp_onehot)
    # MultiGroup terminal is structured; the single group's comp matches CompositionEnv.
    assert mg.terminal_cation_fractions()["g"] == ce.terminal_cation_fractions()


def test_n2_groups_each_sum_to_one_and_structured_terminal():
    g1 = dict(name="P_site", species_set=["Mn", "Ni", "P"], fraction_set=["0.05", "0.95"],
              n_components=2, total_units=20)
    g2 = dict(name="S_site", species_set=["S", "O", "Cl"], fraction_set=["0.10", "0.20", "0.70"],
              n_components=3, total_units=10)
    mg = _drive_first_allowed(MultiGroupEnv(groups=[g1, g2]))

    assert mg.n_components == 5  # 2 + 3
    term = mg.terminal_cation_fractions()
    assert set(term.keys()) == {"P_site", "S_site"}
    assert abs(sum(term["P_site"].values()) - 1.0) < 1e-9
    assert abs(sum(term["S_site"].values()) - 1.0) < 1e-9
    # Union alphabet spans both groups.
    assert set(mg.species_set) == {"Mn", "Ni", "P", "S", "O", "Cl"}
    # Dedup key is structured + hashable.
    assert hash(mg.terminal_comp_key())


def test_prior_groups_delivers_earlier_group_to_later_filter():
    captured = []

    class RecordPrior(ConstraintFilter):
        def filter_actions(self, *, actions, prior_groups=None, **kw):
            captured.append(prior_groups)
            return actions

    g1 = dict(name="P_site", species_set=["Mn", "Ni", "P"], fraction_set=["0.05", "0.95"],
              n_components=2, total_units=20)
    g2 = dict(name="S_site", species_set=["S", "O", "Cl"], fraction_set=["0.10", "0.20", "0.70"],
              n_components=3, total_units=10, constraint_filter=RecordPrior())
    _drive_first_allowed(MultiGroupEnv(groups=[g1, g2]))

    # The S-site filter must see the completed P-site composition. (Some calls are
    # None — the inner env's own bookkeeping path, which MultiGroupEnv discards —
    # so filters must tolerate prior_groups=None; we assert the real ones are right.)
    real = [p for p in captured if p is not None]
    assert real, "S-site filter never received prior_groups"
    p_site_comp = {"Mn": 0.05, "Ni": 0.95}
    assert all(p == [p_site_comp] for p in real)


def test_fraction_set_regular_grid_expands_and_derives_total_units():
    g = normalize_group_spec(
        {"species_set": ["Ti", "Al"], "fraction_set": {"min": 0.0, "max": 0.04, "step": 0.01},
         "n_components": 2}
    )
    assert g["fraction_set"] == ["0.00", "0.01", "0.02", "0.03", "0.04"]
    assert g["total_units"] == 100
    assert g["sites"] == 1


def test_explicit_host_species_with_element_bounds_absorbs_remainder():
    # New equivalent of the old `host:` friendly-knob + host_complement filter:
    # list the host explicitly in species_set and pin its range via
    # element_bounds. The remaining budget is whatever's left, exactly as
    # before, but derived from composition's native sum-to-1 mechanics rather
    # than a dedicated filter (see configs/lips_sse.yaml's migration).
    g = {"name": "P_site", "kind": "composition", "species_set": ["Mn", "Ni", "Ru", "P"],
         "n_components": 2,
         "fraction_set": ["0.02", "0.03", "0.04", "0.05", "0.06", "0.07", "0.08",
                           "0.92", "0.93", "0.94", "0.95", "0.96", "0.97", "0.98"],
         "element_bounds": {"P": [0.92, 0.98]}}
    env = _build([g])
    levels = {f"{x/100:.2f}" for x in range(2, 9)}
    for _ in range(50):
        env.initialize()
        while env.counter < env.n_components:
            actions = env.allowed_actions()
            env.step(actions[int(np.random.randint(len(actions)))])
        comp = env.terminal_cation_fractions()["P_site"]
        assert "P" in comp
        metals = [k for k in comp if k != "P"]
        assert len(metals) == 1                               # one dopant + host
        assert f"{comp[metals[0]]:.2f}" in levels               # dopant at a level
        assert abs(comp["P"] - (1 - comp[metals[0]])) < 1e-9    # host took the rest


def test_sites_assembles_real_counts_and_formula():
    g1 = {"name": "P", "kind": "composition", "species_set": ["Mn", "P"],
          "n_components": 2, "fraction_set": ["0.05", "0.95"], "sites": 1}
    g2 = {"name": "X", "species_set": ["A", "B"], "fraction_set": ["0.10", "0.20", "0.80", "0.90"],
          "n_components": 2, "sites": 6}
    env = _drive_first_allowed(_build([g1, g2]))
    asm = env.assembled_composition()
    # P-site sites=1 -> fractions; X-site sites=6 -> counts summing to 6
    assert abs((asm.get("Mn", 0) + asm.get("P", 0)) - 1.0) < 1e-9
    assert abs((asm.get("A", 0) + asm.get("B", 0)) - 6.0) < 1e-9
    assert env.terminal_formula  # readable, non-empty


def test_categorical_group_returns_real_values():
    p = {"name": "P_site", "kind": "composition", "species_set": ["Mn", "Ni", "P"],
         "n_components": 2, "fraction_set": ["0.05", "0.95"]}
    s = {"name": "S_site", "kind": "categorical", "sites": 6,
         "choices": [{"element": "O", "values": ["none", "oxide"]},
                     {"element": "Cl", "values": [0.6, 0.8, 1.0, 1.2, 1.4]}]}
    env = _build([p, s])
    assert env.n_components == 4  # P: 2 steps + S: 2 slots
    for _ in range(60):
        env.initialize()
        while env.counter < env.n_components:
            env.step(env.allowed_actions()[int(np.random.randint(len(env.allowed_actions())))])
        t = env.terminal_cation_fractions()["S_site"]
        assert t["O"] in ("none", "oxide")              # ORIGINAL label, not a code
        assert t["Cl"] in (0.6, 0.8, 1.0, 1.2, 1.4)     # ORIGINAL number, not a fraction
    # assembled composition: Cl is a real count; the O label is not an atom
    asm = env.assembled_composition()
    assert asm.get("Cl") in (0.6, 0.8, 1.0, 1.2, 1.4)
    assert "none" not in asm and "oxide" not in asm


def test_sse_doping_two_metals_per_metal_masking():
    # Two dopants Ru (metal_only) + Al (oxide_only), plus the host "P" so the
    # group has somewhere for the un-doped remainder to go (SSEDopingFilter's
    # host_P defaults to "P", matching configs/lips_sse.yaml); S-site has
    # per-metal O slots.
    p = {"name": "P_site", "kind": "composition",
         "species_set": ["P", "Ru", "Al", "Mn"],
         "n_components": 3, "fraction_set": {"min": 0.0, "max": 1.0, "step": 0.01}}
    s = {"name": "S_site", "kind": "categorical", "sites": 6,
         "choices": [{"name": "O_a", "element": "O", "values": [0, 1]},
                     {"name": "O_b", "element": "O", "values": [0, 1]},
                     {"element": "Cl", "values": [0.6, 1.0]}],
         "filters": [{"type": "sse_doping", "o_element": "O",
                      "metal_only": ["Ru"], "oxide_only": ["Al"]}]}
    env = _build([p, s])

    env.initialize()
    _pick(env, "Ru", "0.05")
    _pick(env, "Al", "0.06")
    _pick(env, "P", "0.89")
    # Sorted metals = [Al, Ru] -> O_a ↔ Al (oxide_only: only 1), O_b ↔ Ru (metal_only: only 0).
    assert _allowed_values(env, "O_a") == {"1.00"}
    _pick(env, "O_a", "1.00")
    assert _allowed_values(env, "O_b") == {"0.00"}
    _pick(env, "O_b", "0.00")
    # Cl slot is never masked.
    assert _allowed_values(env, "Cl") == {"0.60", "1.00"}


def test_categorical_filter_sees_prior_groups():
    captured = []

    class Rec(ConstraintFilter):
        def filter_actions(self, *, actions, prior_groups=None, **kw):
            captured.append(prior_groups)
            return actions

    p = {"name": "P_site", "kind": "composition", "species_set": ["Mn", "P"],
         "n_components": 2, "fraction_set": ["0.05", "0.95"]}
    s = {"name": "S_site", "kind": "categorical", "sites": 6,
         "choices": [{"element": "Cl", "values": [0.6, 1.0]}],
         "constraint_filter": Rec()}
    # constraint is already an instance -> pass groups straight to the env
    built = [normalize_group_spec(p), normalize_group_spec(s)]
    built[0]["constraint_filter"] = resolve_constraint(built[0].get("constraint_filter"), built[0], env=None)
    _drive_first_allowed(MultiGroupEnv(groups=built))
    real = [c for c in captured if c is not None]
    assert real and real[0] == [{"Mn": 0.05, "P": 0.95}]


# --------------------------------------------------------------------------- #
# Fraction-grid precision.
#
# Codes ARE the one-hot alphabet: _format_fraction renders an action's amount and
# encode_choice looks it up in fraction_set. Rendering a 0.125 grid at the old
# hard-coded 2 decimals produced "0.12", which is not a member -> ValueError. The
# first test is the back-compat gate; the last is the end-to-end gate.
# --------------------------------------------------------------------------- #

def test_two_decimal_grids_are_unchanged_character_for_character():
    from rl_matdesign.env import expand_fraction_set, derive_total_units, _decimals_of, _format_fraction

    codes = expand_fraction_set({"min": 0.05, "max": 0.20, "step": 0.05})
    assert codes == ["0.05", "0.10", "0.15", "0.20"]
    assert derive_total_units(codes) == 20
    codes = expand_fraction_set({"min": 0.0, "max": 0.03, "step": 0.01})
    assert codes == ["0.00", "0.01", "0.02", "0.03"]
    assert expand_fraction_set([0.0, 0.02, 0.08]) == ["0.00", "0.02", "0.08"]
    # the historical call signature and result
    assert _format_fraction(1, 20) == "0.05"
    assert _format_fraction(1, 100) == "0.01"
    assert _decimals_of(["0.05", "0.45", "1.00"]) == 2
    assert _decimals_of(["0.0", "1.0"]) == 2          # never drops below 2


def test_eighth_grid_keeps_full_precision():
    from rl_matdesign.env import expand_fraction_set, derive_total_units, _decimals_of, _format_fraction

    codes = expand_fraction_set({"min": 0.0, "max": 1.0, "step": 0.125})
    assert codes == ["0.000", "0.125", "0.250", "0.375",
                     "0.500", "0.625", "0.750", "0.875", "1.000"]
    assert derive_total_units(codes) == 8
    assert _decimals_of(codes) == 3
    assert _format_fraction(1, 8, 3) == "0.125"       # was "0.12"
    assert _format_fraction(3, 8, 3) == "0.375"       # was "0.38"


def test_full_multigroup_episode_on_an_eighth_grid():
    """End-to-end gate for the whole precision contract.

    Drives a real 4-group, 13-step episode on the 0.125 grid — the shape the
    Cs2AgBiCl6 configs use. Before the fix this raised
    ``ValueError: Unknown choice '0.12'`` on the very first step, so a partial fix
    (touching expand_fraction_set but not _format_fraction) fails right here.
    """
    import random
    from rl_matdesign.env_multigroup import MultiGroupEnv, normalize_group_spec

    eighths = ["0.000", "0.125", "0.250", "0.375", "0.500",
               "0.625", "0.750", "0.875", "1.000"]
    specs = [
        {"name": "A_site", "kind": "composition", "sites": 16,
         "species_set": ["Cs", "Rb", "K"], "n_components": 3},
        {"name": "B1_site", "kind": "composition", "sites": 8,
         "species_set": ["Ag", "Cu", "Na"], "n_components": 3},
        {"name": "B3_site", "kind": "composition", "sites": 8,
         "species_set": ["Bi", "Sb", "In", "Ga"], "n_components": 4},
        {"name": "X_site", "kind": "composition", "sites": 48,
         "species_set": ["Cl", "Br", "I"], "n_components": 3},
    ]
    for s in specs:
        s.update(episode_style="fixed_order_amount", fraction_set=list(eighths))
    env = MultiGroupEnv(groups=[normalize_group_spec(s) for s in specs],
                        reward_fn=lambda groups: 0.0)

    rng = random.Random(0)
    for _ in range(5):
        env.initialize()
        steps = 0
        while True:
            actions = env.allowed_actions()
            if not actions:
                break
            env.step(rng.choice(actions))
            steps += 1
        assert steps == 13                       # 3 + 3 + 4 + 3
        candidate = env.terminal_cation_fractions()
        assert set(candidate) == {"A_site", "B1_site", "B3_site", "X_site"}
        for group, picks in candidate.items():
            # exact, not approximate: these are the values the builder turns into
            # integer atom counts.
            assert sum(picks.values()) == 1.0, (group, picks)
            assert all(round(v * 8) == v * 8 for v in picks.values())
