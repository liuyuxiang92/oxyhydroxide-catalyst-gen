"""The user-friendly `env_type: lips` scenario expander.

Verifies that a friendly config (counts + ranges) expands into a valid
multi_group config whose episodes are charge-neutral and whose Cl choices match
exactly what the user asked for — without the user writing any fraction grids,
selectors, cl_map, or O flags.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rl_matdesign.scenarios.lips import expand  # noqa: E402
from rl_matdesign.env_multigroup import MultiGroupEnv  # noqa: E402
from rl_matdesign.registry import resolve_constraint  # noqa: E402
from rl_matdesign.predictors.builders.sse import SSESupercellBuilder  # noqa: E402


def _friendly_cfg():
    return {
        "env_type": "lips",
        "base_poscar": "POSCAR_supercell",
        "formula_units": 500,
        "dopant_metals": ["Mn", "Ni", "Ru", "Mg", "W"],
        "metal_level": {"min": 0.02, "max": 0.08, "step": 0.01},
        "cl_per_fu": {"min": 0.6, "max": 1.4, "step": 0.2},
        "halide_total": 1.7,
        "metal_only": ["Ru"],
        "oxide_only": ["Mg"],
        "valences": {
            "Li": 1, "P": 5, "S": -2, "O": -2, "Cl": -1, "Br": -1,
            "Mn": {"sulfide": 2, "oxide": 4}, "Ni": {"sulfide": 2, "oxide": 3},
            "Ru": {"sulfide": 4, "oxide": 2}, "Mg": {"sulfide": 2, "oxide": 2},
            "W": {"sulfide": 4, "oxide": 6},
        },
    }


def test_expand_generates_internal_encoding():
    ex = expand(_friendly_cfg())
    assert ex["env_type"] == "multi_group"
    # friendly-only keys are consumed
    for k in ("dopant_metals", "metal_level", "cl_per_fu", "metal_only", "oxide_only"):
        assert k not in ex
    p, s = ex["groups"]
    # P-site: levels + complements generated; one metal + P
    assert p["levels"] == ["0.02", "0.03", "0.04", "0.05", "0.06", "0.07", "0.08"]
    assert "0.95" in p["fraction_set"] and p["cation_set"][-1] == "P"
    # S-site: 5 Cl selectors, O flags, fixed order, residuals present
    assert len(s["cl_values"]) == 5
    assert s["o_off"] != s["o_on"] and s["episode_style"] == "fixed_order_amount"
    # cl_map maps the selectors to exactly the requested counts
    counts = sorted(ex["cl_map"][round(float(v), 2)] for v in s["cl_values"])
    assert counts == [0.6, 0.8, 1.0, 1.2, 1.4]


def test_expanded_config_is_chargeneutral_and_drives():
    ex = expand(_friendly_cfg())
    built = []
    for g in ex["groups"]:
        gs = dict(g)
        gs["constraint_filter"] = resolve_constraint(g.get("constraint_filter"), g, env=None)
        built.append(gs)
    env = MultiGroupEnv(groups=built)
    builder = SSESupercellBuilder(ex)
    val = ex["valences"]

    def v(el, sc):
        x = val[el]
        return x[sc] if isinstance(x, dict) else x

    rng = np.random.default_rng(0)
    for _ in range(150):
        env.initialize()
        while env.counter < env.n_components:
            a = env.allowed_actions()
            assert a, "expanded env stranded the agent"
            env.step(a[int(rng.integers(len(a)))])
        term = env.terminal_cation_fractions()
        c = builder.counts(term)
        metal = [k for k in term["P_site"] if k != "P"][0]
        o = term["S_site"].get("O", 0.0)
        sc = "oxide" if o > builder.o_off + 1e-9 else "sulfide"
        cation = c["Li"] + c["P"] * 5 + c["metal"] * v(metal, sc)
        anion = c["S"] * 2 + c["O"] * 2 + c["Cl"] + c["Br"]
        assert cation == anion, (metal, sc, c)
        # Ru is metal-only -> can never be the oxide form
        if metal == "Ru":
            assert sc == "sulfide"
