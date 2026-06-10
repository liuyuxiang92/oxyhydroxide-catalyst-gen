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
from rl_matdesign.env_multigroup import MultiGroupEnv  # noqa: E402
from rl_matdesign.constraints.base import ConstraintFilter  # noqa: E402


def _drive_first_allowed(env):
    env.initialize()
    while env.counter < env.n_components:
        env.step(env.allowed_actions()[0])
    return env


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
