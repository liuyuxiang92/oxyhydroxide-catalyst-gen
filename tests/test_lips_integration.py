"""End-to-end (no-GPU) integration for the LiPS config on the general multi_group.

Builds the env from the friendly configs/lips_sse.yaml (P-site composition
group with the host included explicitly, S-site categorical), drives
episodes, and asserts the O-form masking by metal category and the recipe's
charge neutrality. DeepMD eval / geo-opt are GPU-gated.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import yaml  # noqa: E402

from rl_matdesign.env_multigroup import MultiGroupEnv, normalize_group_spec  # noqa: E402
from rl_matdesign.registry import build_constraints  # noqa: E402
from rl_matdesign.predictors.builders.sse import SSESupercellBuilder  # noqa: E402

CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "lips_sse.yaml")


def _load_cfg():
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


def _build_env(cfg):
    built = []
    for g in cfg["groups"]:
        gs = normalize_group_spec(g)
        gs["constraint_filter"] = build_constraints(gs, env=None)
        built.append(gs)
    return MultiGroupEnv(groups=built)


def _val(valences, el, sc):
    v = valences[el]
    return v[sc] if isinstance(v, dict) else v


def test_lips_config_is_friendly():
    """The config carries no internal-encoding knobs (checks parsed keys)."""
    cfg = _load_cfg()
    assert "env_type" not in cfg                       # removed public key
    assert "constraint_filter" not in cfg
    # total_units is still derived-only (never authored); fraction_set is now
    # the one public grid knob, so it's legitimately present, not banned.
    banned = {"cl_map", "o_off", "o_on", "formula_units", "total_units"}
    keys = set(cfg) | {k for g in cfg["groups"] for k in g}
    assert not (keys & banned), f"internal-encoding knobs leaked back: {keys & banned}"
    p_site = next(g for g in cfg["groups"] if g["name"] == "P_site")
    assert "P" in p_site["species_set"]                 # host listed explicitly


def test_lips_masking_and_charge_neutrality():
    cfg = _load_cfg()
    env = _build_env(cfg)
    # counts() with an explicit formula_units (so it needn't read the POSCAR here)
    builder = SSESupercellBuilder({**cfg, "formula_units": 500})

    s_group = next(g for g in cfg["groups"] if g["name"] == "S_site")
    sse_filter_cfg = next(f for f in s_group["filters"] if f.get("type") == "sse_doping")
    metal_only, oxide_only = set(sse_filter_cfg["metal_only"]), set(sse_filter_cfg["oxide_only"])
    o_form_slots = [str(c.get("name", c["element"])) for c in s_group["choices"]
                    if str(c["element"]) == sse_filter_cfg.get("o_element", "O")]
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

        assert "P" in p_site                                      # host now explicit
        # 1-2 distinct dopants: an unused second slot lands on some OTHER
        # element at 0.0 (a filler pick, not an active dopant) -- exclude it,
        # matching SSESupercellBuilder._decode's own `p_site[el] > 0` filter.
        metals = sorted(k for k in p_site if k != "P" and p_site[k] > 0)
        assert 1 <= len(metals) <= 2
        seen_metals.update(metals)
        dopant_total = sum(p_site[m] for m in metals)
        # element_bounds P:[0.92,0.98] <=> combined dopant fraction in [0.02,0.08]
        assert 0.02 - 1e-9 <= dopant_total <= 0.08 + 1e-9
        assert 0.92 - 1e-9 <= p_site["P"] <= 0.98 + 1e-9
        assert abs(p_site["P"] + dopant_total - 1.0) < 1e-9

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
