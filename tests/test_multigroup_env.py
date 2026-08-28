"""Tests for MultiGroupEnv (N sublattice groups, each summing to 1).

Covers:
* N=1 reproduces CompositionEnv exactly (backward-compat keystone).
* N=2 episodes walk groups in order, each group sums to 1, terminal is structured.
* prior_groups delivers earlier groups' compositions to a later group's filter.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rl_matdesign.env import CompositionEnv  # noqa: E402
from rl_matdesign.env_multigroup import MultiGroupEnv, normalize_group_spec  # noqa: E402
from rl_matdesign.constraints.base import ConstraintFilter  # noqa: E402
from rl_matdesign.registry import resolve_constraint  # noqa: E402
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
    built = []
    for g in group_specs:
        gs = normalize_group_spec(g)
        gs["constraint_filter"] = resolve_constraint(gs.get("constraint_filter"), gs, env=None)
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


def test_amount_range_expands_fraction_set():
    g = normalize_group_spec(
        {"species_set": ["Ti", "Al"], "amount": {"min": 0.0, "max": 0.04, "step": 0.01},
         "n_components": 2}
    )
    assert g["fraction_set"] == ["0.00", "0.01", "0.02", "0.03", "0.04"]
    assert g["total_units"] == 100
    assert g["sites"] == 1


def test_host_knob_dopant_at_level_host_takes_rest():
    g = {"name": "P_site", "species_set": ["Mn", "Ni", "Ru"], "host": "P",
         "amount": {"min": 0.02, "max": 0.08, "step": 0.01}, "sites": 1}
    # normalization wires the host_complement filter + complements automatically
    n = normalize_group_spec(g)
    assert n["species_set"][-1] == "P" and n["constraint_filter"] == "host_complement"
    assert "0.95" in n["fraction_set"] and "0.05" in n["fraction_set"]

    env = _build([g])
    levels = {f"{x/100:.2f}" for x in range(2, 9)}
    for _ in range(50):
        env.initialize()
        while env.counter < env.n_components:
            env.step(env.allowed_actions()[int(np.random.randint(len(env.allowed_actions())))])
        comp = env.terminal_cation_fractions()["P_site"]
        metals = [k for k in comp if k != "P"]
        assert len(metals) == 1
        assert f"{comp[metals[0]]:.2f}" in levels            # dopant at a level
        assert abs(comp["P"] - (1 - comp[metals[0]])) < 1e-9  # host took the rest


def test_sites_assembles_real_counts_and_formula():
    g1 = {"name": "P", "species_set": ["Mn"], "host": "P",
          "amount": {"min": 0.05, "max": 0.05, "step": 0.01}, "sites": 1}
    g2 = {"name": "X", "species_set": ["A", "B"], "fraction_set": ["0.10", "0.20", "0.80", "0.90"],
          "total_units": 10, "n_components": 2, "sites": 6}
    env = _drive_first_allowed(_build([g1, g2]))
    asm = env.assembled_composition()
    # P-site sites=1 -> fractions; X-site sites=6 -> counts summing to 6
    assert abs((asm.get("Mn", 0) + asm.get("P", 0)) - 1.0) < 1e-9
    assert abs((asm.get("A", 0) + asm.get("B", 0)) - 6.0) < 1e-9
    assert env.terminal_formula  # readable, non-empty


def test_categorical_group_returns_real_values():
    p = {"name": "P_site", "species_set": ["Mn", "Ni"], "host": "P",
         "amount": {"min": 0.05, "max": 0.05, "step": 0.01}, "sites": 1}
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


def test_independent_group_two_picks_merge_and_repeat():
    # kind: independent -> two (metal, amount) picks; repeats allowed; host dropped.
    g = {"name": "P_site", "kind": "independent", "species_set": ["Mn", "Ni", "P"],
         "host": "P", "n_dopants": 2, "amount": {"min": 0.02, "max": 0.08, "step": 0.01}}
    n = normalize_group_spec(g)
    assert n["n_components"] == 2
    assert "P" not in n["species_set"]                       # host dropped (builder fills it)
    assert n["fraction_set"] == [f"{x/100:.2f}" for x in range(2, 9)]

    env = _build([g])
    levels = {f"{x/100:.2f}" for x in range(2, 9)}
    for _ in range(40):
        env.initialize()
        while env.counter < env.n_components:
            env.step(env.allowed_actions()[int(np.random.randint(len(env.allowed_actions())))])
        comp = env.terminal_cation_fractions()["P_site"]
        assert 1 <= len(comp) <= 2                          # may merge to one metal
        assert "P" not in comp                              # host not in the dopant map
        total = sum(comp.values())
        assert total <= 0.16 + 1e-9                         # two picks, each <= 0.08


def test_independent_group_allows_same_element_merges():
    g = {"name": "P_site", "kind": "independent", "species_set": ["Mn", "Ni"],
         "n_dopants": 2, "amount": {"min": 0.02, "max": 0.08, "step": 0.01}}
    env = _build([g])
    env.initialize()
    _pick(env, "Mn", "0.06")
    _pick(env, "Mn", "0.04")                                # same element twice
    comp = env.terminal_cation_fractions()["P_site"]
    assert comp == {"Mn": 0.10}                             # merged to combined fraction


def test_sse_doping_two_metals_per_metal_masking():
    # Two dopants Ru (metal_only) + Al (oxide_only); S-site has per-metal O slots.
    p = {"name": "P_site", "kind": "independent", "species_set": ["Ru", "Al", "Mn"],
         "n_dopants": 2, "amount": {"min": 0.05, "max": 0.06, "step": 0.01}}
    s = {"name": "S_site", "kind": "categorical", "sites": 6,
         "choices": [{"name": "O_a", "element": "O", "values": [0, 1]},
                     {"name": "O_b", "element": "O", "values": [0, 1]},
                     {"element": "Cl", "values": [0.6, 1.0]}],
         "constraint_filter": "sse_doping", "o_element": "O",
         "metal_only": ["Ru"], "oxide_only": ["Al"]}
    env = _build([p, s])

    env.initialize()
    _pick(env, "Ru", "0.05")
    _pick(env, "Al", "0.06")
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

    p = {"name": "P_site", "species_set": ["Mn"], "host": "P",
         "amount": {"min": 0.05, "max": 0.05, "step": 0.01}, "sites": 1}
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
    from rl_matdesign.env_multigroup import _amounts_to_strs
    from rl_matdesign.env import _decimals_of, _format_fraction

    codes, step, dec = _amounts_to_strs({"min": 0.05, "max": 0.20, "step": 0.05})
    assert codes == ["0.05", "0.10", "0.15", "0.20"] and dec == 2
    codes, _s, dec = _amounts_to_strs({"min": 0.0, "max": 0.03, "step": 0.01})
    assert codes == ["0.00", "0.01", "0.02", "0.03"] and dec == 2
    assert _amounts_to_strs([0.0, 0.02, 0.08])[0] == ["0.00", "0.02", "0.08"]
    # the historical call signature and result
    assert _format_fraction(1, 20) == "0.05"
    assert _format_fraction(1, 100) == "0.01"
    assert _decimals_of(["0.05", "0.45", "1.00"]) == 2
    assert _decimals_of(["0.0", "1.0"]) == 2          # never drops below 2


def test_eighth_grid_keeps_full_precision():
    from rl_matdesign.env_multigroup import _amounts_to_strs
    from rl_matdesign.env import _decimals_of, _format_fraction

    codes, step, dec = _amounts_to_strs({"min": 0.0, "max": 1.0, "step": 0.125})
    assert dec == 3 and step == 0.125
    assert codes == ["0.000", "0.125", "0.250", "0.375",
                     "0.500", "0.625", "0.750", "0.875", "1.000"]
    assert _decimals_of(codes) == 3
    assert _format_fraction(1, 8, 3) == "0.125"       # was "0.12"
    assert _format_fraction(3, 8, 3) == "0.375"       # was "0.38"


def test_host_complement_pairs_sum_to_exactly_one_on_an_eighth_grid():
    from rl_matdesign.env_multigroup import normalize_group_spec

    g = normalize_group_spec({"name": "g", "kind": "composition",
                              "species_set": ["Rb", "Cs"], "host": "Cs",
                              "amount": {"min": 0.0, "max": 1.0, "step": 0.125}})
    fs = g["fraction_set"]
    half = len(fs) // 2
    for dopant, complement in zip(fs[:half], fs[half:]):
        assert float(dopant) + float(complement) == 1.0


def test_full_multigroup_episode_on_an_eighth_grid():
    """End-to-end gate for the whole precision contract.

    Drives a real 4-group, 13-step episode on the 0.125 grid — the shape the
    Cs2AgBiCl6 configs use. Before the fix this raised
    ``ValueError: Unknown choice '0.12'`` on the very first step, so a partial fix
    (touching _amounts_to_strs but not _format_fraction) fails right here.
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
        s.update(episode_style="fixed_order_amount", total_units=8,
                 fraction_set=list(eighths))
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


def test_shipped_amount_configs_still_use_two_decimal_codes():
    import glob
    import yaml
    from rl_matdesign.env_multigroup import _amounts_to_strs

    root = os.path.join(os.path.dirname(__file__), "..", "configs")
    checked = 0
    for path in sorted(glob.glob(os.path.join(root, "*.yaml"))):
        with open(path) as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            continue
        for group in cfg.get("groups") or []:
            amount = group.get("amount")
            if amount is None:
                continue
            codes, _step, dec = _amounts_to_strs(amount)
            assert dec == 2, (path, group.get("name"), codes)
            checked += 1
    assert checked > 0, "no shipped config exercises the amount: shorthand"
