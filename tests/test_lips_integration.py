"""End-to-end (no-GPU) integration for the LiPS config: env + sse_doping + recipe.

Drives many random episodes through the MultiGroupEnv built from configs/lips_sse.yaml
and asserts the constraint masks and the recipe's charge neutrality. The DeepMD
property eval / geo-opt are GPU-gated and not exercised here.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import yaml  # noqa: E402

from rl_matdesign.env_multigroup import MultiGroupEnv  # noqa: E402
from rl_matdesign.registry import resolve_constraint  # noqa: E402
from rl_matdesign.predictors.builders.sse import SSESupercellBuilder  # noqa: E402

CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "lips_sse.yaml")


def _load_cfg():
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


def _build_env(cfg):
    built = []
    for g in cfg["groups"]:
        gspec = dict(g)
        gspec["constraint_filter"] = resolve_constraint(
            g.get("constraint_filter"), g, env=None
        )
        built.append(gspec)
    return MultiGroupEnv(groups=built)


def _val(valences, el, scenario):
    v = valences[el]
    return v[scenario] if isinstance(v, dict) else v


def test_lips_env_constraints_and_charge_neutrality():
    cfg = _load_cfg()
    env = _build_env(cfg)
    builder = SSESupercellBuilder(cfg)  # reads cfg; base_poscar not opened by counts()

    metal_only = set(cfg["groups"][1]["metal_only"])
    oxide_only = set(cfg["groups"][1]["oxide_only"])
    o_off, o_on = cfg["groups"][1]["o_off"], cfg["groups"][1]["o_on"]
    levels = set(cfg["groups"][0]["levels"])
    cl_values = set(cfg["groups"][1]["cl_values"])
    valences = cfg["valences"]

    rng = np.random.default_rng(0)
    seen_metals, seen_forms = set(), set()

    for _ in range(300):
        env.initialize()
        while env.counter < env.n_components:
            allowed = env.allowed_actions()
            assert allowed, "constraint stranded the agent with no legal action"
            env.step(allowed[int(rng.integers(len(allowed)))])

        term = env.terminal_cation_fractions()
        p_site, s_site = term["P_site"], term["S_site"]

        # P-site: exactly one metal at a level fraction + host P.
        metals = [el for el in p_site if el != "P"]
        assert len(metals) == 1
        metal = metals[0]
        assert f"{p_site[metal]:.2f}" in levels
        seen_metals.add(metal)

        # S-site: O is one of the two form flags; the form respects the metal category.
        o_frac = s_site.get("O")
        o_str = f"{o_frac:.2f}"
        assert o_str in (o_off, o_on)
        seen_forms.add((metal in metal_only, metal in oxide_only, o_str))
        if metal in metal_only:
            assert o_str == o_off                    # Ru: metal form only
        elif metal in oxide_only:
            assert o_str == o_on                     # oxide-only: oxide form only
        # Cl is one of the selectors.
        assert f"{s_site['Cl']:.2f}" in cl_values

        # Recipe: charge-neutral integer counts.
        c = builder.counts(term)
        scenario = "oxide" if o_frac > builder.o_off + 1e-9 else "sulfide"
        cation = c["Li"] * 1 + c["P"] * 5 + c["metal"] * _val(valences, metal, scenario)
        anion = (c["S"] * 2 + c["O"] * 2 + c["Cl"] * 1 + c["Br"] * 1)
        assert cation == anion, (metal, scenario, c)
        assert 0 <= c["Li_delete"] <= 3000
        assert c["Br"] == int(round(1.7 * 500)) - c["Cl"]

    # Sanity: the random walk actually exercised variety.
    assert len(seen_metals) > 10


def test_metal_only_ru_forced_metal_form():
    """A metal-only dopant (Ru) must never be offered the oxide flag at the O step."""
    cfg = _load_cfg()
    env = _build_env(cfg)
    o_on = cfg["groups"][1]["o_on"]

    # Force the P-site to pick Ru, then inspect the S-site O step's legal actions.
    env.initialize()
    # Step P-site to Ru at some level: find an allowed action whose elem decodes to Ru.
    from rl_matdesign.encoding import decode_one_hot
    picked = False
    for a in env.allowed_actions():
        if decode_one_hot(a[0], env.cation_set) == "Ru":
            env.step(a)
            picked = True
            break
    assert picked
    # finish the P-site (host P at complement)
    env.step(env.allowed_actions()[0])
    # Now at the S-site O step: o_on must be masked out for Ru.
    o_actions = env.allowed_actions()
    o_fracs = {f"{float(decode_one_hot(a[1], env.fraction_set)):.2f}" for a in o_actions}
    assert o_on not in o_fracs, f"Ru wrongly offered oxide flag: {o_fracs}"
