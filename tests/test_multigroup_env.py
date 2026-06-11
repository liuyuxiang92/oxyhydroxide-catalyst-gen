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
    kw = dict(cation_set=["Fe", "Ni", "Co", "Mn"], fraction_set=["0.25", "0.50", "0.75"],
              n_components=4, total_units=4)
    ce = _drive_first_allowed(CompositionEnv(**kw))
    mg = _drive_first_allowed(MultiGroupEnv(groups=[dict(name="g", **kw)]))

    assert mg.n_components == ce.n_components
    assert mg.cation_set == ce.cation_set
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
    g1 = dict(name="P_site", cation_set=["Mn", "Ni", "P"], fraction_set=["0.05", "0.95"],
              n_components=2, total_units=20)
    g2 = dict(name="S_site", cation_set=["S", "O", "Cl"], fraction_set=["0.10", "0.20", "0.70"],
              n_components=3, total_units=10)
    mg = _drive_first_allowed(MultiGroupEnv(groups=[g1, g2]))

    assert mg.n_components == 5  # 2 + 3
    term = mg.terminal_cation_fractions()
    assert set(term.keys()) == {"P_site", "S_site"}
    assert abs(sum(term["P_site"].values()) - 1.0) < 1e-9
    assert abs(sum(term["S_site"].values()) - 1.0) < 1e-9
    # Union alphabet spans both groups.
    assert set(mg.cation_set) == {"Mn", "Ni", "P", "S", "O", "Cl"}
    # Dedup key is structured + hashable.
    assert hash(mg.terminal_comp_key())


def test_prior_groups_delivers_earlier_group_to_later_filter():
    captured = []

    class RecordPrior(ConstraintFilter):
        def filter_actions(self, *, actions, prior_groups=None, **kw):
            captured.append(prior_groups)
            return actions

    g1 = dict(name="P_site", cation_set=["Mn", "Ni", "P"], fraction_set=["0.05", "0.95"],
              n_components=2, total_units=20)
    g2 = dict(name="S_site", cation_set=["S", "O", "Cl"], fraction_set=["0.10", "0.20", "0.70"],
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
        {"cation_set": ["Ti", "Al"], "amount": {"min": 0.0, "max": 0.04, "step": 0.01},
         "n_components": 2}
    )
    assert g["fraction_set"] == ["0.00", "0.01", "0.02", "0.03", "0.04"]
    assert g["total_units"] == 100
    assert g["sites"] == 1


def test_host_knob_dopant_at_level_host_takes_rest():
    g = {"name": "P_site", "cation_set": ["Mn", "Ni", "Ru"], "host": "P",
         "amount": {"min": 0.02, "max": 0.08, "step": 0.01}, "sites": 1}
    # normalization wires the host_complement filter + complements automatically
    n = normalize_group_spec(g)
    assert n["cation_set"][-1] == "P" and n["constraint_filter"] == "host_complement"
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
    g1 = {"name": "P", "cation_set": ["Mn"], "host": "P",
          "amount": {"min": 0.05, "max": 0.05, "step": 0.01}, "sites": 1}
    g2 = {"name": "X", "cation_set": ["A", "B"], "fraction_set": ["0.10", "0.20", "0.80", "0.90"],
          "total_units": 10, "n_components": 2, "sites": 6}
    env = _drive_first_allowed(_build([g1, g2]))
    asm = env.assembled_composition()
    # P-site sites=1 -> fractions; X-site sites=6 -> counts summing to 6
    assert abs((asm.get("Mn", 0) + asm.get("P", 0)) - 1.0) < 1e-9
    assert abs((asm.get("A", 0) + asm.get("B", 0)) - 6.0) < 1e-9
    assert env.terminal_formula  # readable, non-empty


def test_categorical_group_returns_real_values():
    p = {"name": "P_site", "cation_set": ["Mn", "Ni"], "host": "P",
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


def test_categorical_filter_sees_prior_groups():
    captured = []

    class Rec(ConstraintFilter):
        def filter_actions(self, *, actions, prior_groups=None, **kw):
            captured.append(prior_groups)
            return actions

    p = {"name": "P_site", "cation_set": ["Mn"], "host": "P",
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
