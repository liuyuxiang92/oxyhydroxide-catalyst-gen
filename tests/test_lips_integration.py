"""End-to-end (no-GPU) integration for the LiPS config on the general multi_group.

Builds the env from the friendly configs/lips_sse.yaml (P-site host+amount,
S-site categorical), drives episodes, and asserts the O-form masking by metal
category and the recipe's charge neutrality. DeepMD eval / geo-opt are GPU-gated.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import yaml  # noqa: E402

from rl_matdesign.env_multigroup import MultiGroupEnv, normalize_group_spec  # noqa: E402
from rl_matdesign.registry import resolve_constraint  # noqa: E402
from rl_matdesign.predictors.builders.sse import SSESupercellBuilder  # noqa: E402

CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "lips_sse.yaml")


def _load_cfg():
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


def _build_env(cfg):
    built = []
    for g in cfg["groups"]:
        gs = normalize_group_spec(g)
        cf = gs.get("constraint_filter")
        if isinstance(cf, str) or cf is None:
            gs["constraint_filter"] = resolve_constraint(cf, gs, env=None)
        built.append(gs)
    return MultiGroupEnv(groups=built)


def _val(valences, el, sc):
    v = valences[el]
    return v[sc] if isinstance(v, dict) else v


def test_lips_config_is_friendly():
    """The config carries no internal-encoding knobs (checks parsed keys)."""
    cfg = _load_cfg()
    assert cfg["env_type"] == "multi_group"
    banned = {"cl_map", "o_off", "o_on", "formula_units", "total_units", "fraction_set"}
    keys = set(cfg) | {k for g in cfg["groups"] for k in g}
    assert not (keys & banned), f"internal-encoding knobs leaked back: {keys & banned}"


def test_lips_masking_and_charge_neutrality():
    cfg = _load_cfg()
    env = _build_env(cfg)
    # counts() with an explicit formula_units (so it needn't read the POSCAR here)
    builder = SSESupercellBuilder({**cfg, "formula_units": 500})

    s_group = next(g for g in cfg["groups"] if g["name"] == "S_site")
    metal_only, oxide_only = set(s_group["metal_only"]), set(s_group["oxide_only"])
    o_form_slots = [str(c.get("name", c["element"])) for c in s_group["choices"]
                    if str(c["element"]) == s_group.get("o_element", "O")]
    valences = cfg["valences"]

    rng = np.random.default_rng(0)
    seen_metals = set()
    for _ in range(300):
        env.initialize()
        while env.counter < env.n_components:
            a = env.allowed_actions()
            assert a, "agent stranded with no legal action"
            env.step(a[int(rng.integers(len(a)))])
        term = env.terminal_cation_fractions()
        p_site, s_site = term["P_site"], term["S_site"]

        metals = sorted(p_site)                                  # 1-2 distinct dopants, no host
        assert "P" not in p_site                                 # host derived by the builder
        assert 1 <= len(metals) <= 2
        seen_metals.update(metals)
        for m in metals:
            assert 0.02 - 1e-9 <= p_site[m] <= 0.16 + 1e-9       # 1-2 merged picks, each <=0.08
        assert sum(p_site.values()) <= 0.16 + 1e-9

        # Per-metal O-form masking: the i-th sorted metal reads the i-th O slot.
        for i, m in enumerate(metals):
            o_form = s_site[o_form_slots[i]]
            assert o_form in (0, 1)
            if m in metal_only:
                assert o_form == 0                               # Ru: metal form only
            elif m in oxide_only:
                assert o_form == 1                               # oxide-only: oxide only
        assert s_site["Cl"] in (0.6, 0.8, 1.0, 1.2, 1.4)         # real Cl count

        c = builder.counts(term)
        cation = c["Li"] + c["P"] * 5
        for m, n_m in c["metals"].items():
            sc = "oxide" if c["o_forms"][m] > 0 else "sulfide"
            cation += n_m * _val(valences, m, sc)
        anion = c["S"] * 2 + c["O"] * 2 + c["Cl"] + c["Br"]
        assert cation == anion, (metals, c)
        assert c["Br"] == int(round(1.7 * 500)) - c["Cl"]
        assert 0 <= c["Li_delete"] <= 3000

    assert len(seen_metals) > 10
