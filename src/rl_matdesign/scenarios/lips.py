"""User-friendly LiPS scenario -> full multi_group config expander.

The doped-Li6PS6 scenario needs a lot of internal env encoding (per-group
fraction grids, Cl selectors + ``cl_map``, the O form flags, S residual values,
sum-to-1 bookkeeping). Users should not write any of that. Instead they write a
small config in chemically-meaningful terms::

    env_type: lips
    base_poscar: POSCAR_supercell
    formula_units: 500
    dopant_metals: [Mn, Ni, ..., Lu]
    metal_level:  {min: 0.02, max: 0.08, step: 0.01}   # fraction of P replaced
    cl_per_fu:    {min: 0.6,  max: 1.4,  step: 0.2}     # # of S (of 6) replaced by Cl
    halide_total: 1.7                                   # Cl + Br per formula unit
    metal_only:   [Ru]                                  # never take oxygen
    oxide_only:   [Mg, Al, ...]                         # must take oxygen
    valences:     {...}
    properties:   [...]            # conductivity / stability (structure_pipeline)
    geo_opt:      {...}

:func:`expand` turns that into the equivalent ``env_type: multi_group`` config
(two groups + ``sse_doping`` filters + the ``sse`` builder with a generated
``cl_map``). Everything not LiPS-specific (RL hyperparameters, properties,
geo_opt, valences, …) passes through untouched.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List


def _grid(spec: Dict[str, Any]) -> List[float]:
    """Inclusive numeric grid from {min, max, step} (or pass a list through)."""
    if isinstance(spec, (list, tuple)):
        return [float(x) for x in spec]
    lo, hi, st = float(spec["min"]), float(spec["max"]), float(spec["step"])
    n = int(round((hi - lo) / st))
    return [round(lo + i * st, 6) for i in range(n + 1)]


def _f2(x: float) -> str:
    return f"{x:.2f}"


def expand(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Expand a ``env_type: lips`` config into a full ``multi_group`` config."""
    cfg = copy.deepcopy(cfg)

    for required in ("dopant_metals", "metal_level", "cl_per_fu", "valences"):
        if required not in cfg:
            raise ValueError(f"lips scenario requires '{required}' in the config.")

    fu = int(cfg.get("formula_units", 500))
    s_per_fu = int(cfg.get("s_site_per_fu", 6))
    metals = list(cfg["dopant_metals"])
    host_P = str(cfg.get("host", {}).get("P", "P")) if isinstance(cfg.get("host"), dict) else "P"

    # ---------------- P-site group: metal at a level, P takes the rest -------
    levels = _grid(cfg["metal_level"])
    level_strs = [_f2(x) for x in levels]
    complement_strs = [_f2(round(1.0 - x, 2)) for x in levels]
    step = float(cfg["metal_level"]["step"]) if isinstance(cfg["metal_level"], dict) else 0.01
    total_p = int(round(1.0 / step))

    p_group = {
        "name": "P_site",
        "cation_set": metals + [host_P],
        "fraction_set": level_strs + complement_strs,
        "total_units": total_p,
        "n_components": 2,
        "episode_style": "element_then_amount",
        "constraint_filter": "sse_doping",
        "role": "p_site",
        "host_P": host_P,
        "levels": level_strs,
    }

    # ---------------- S-site group: O form flag, Cl selector, S residual -----
    cl_counts = _grid(cfg["cl_per_fu"])
    # Cl selector fraction = (Cl count) / (S sites per f.u.), rounded to 2 dp; the
    # exact count is carried by cl_map so there is no rounding error downstream.
    cl_selectors = [round(c / s_per_fu, 2) for c in cl_counts]
    cl_sel_strs = [_f2(s) for s in cl_selectors]
    cl_map = {round(c / s_per_fu, 2): float(c) for c in cl_counts}

    o_off, o_on = 0.01, 0.02            # O form flags (metal / metal-oxide)
    o_off_str, o_on_str = _f2(o_off), _f2(o_on)

    # S residual = 1 - O - Cl for every legal (O, Cl) combination.
    residuals = sorted({round(1.0 - o - s, 2) for o in (o_off, o_on) for s in cl_selectors})
    s_fraction = sorted(set([o_off, o_on] + cl_selectors + residuals))
    s_fraction_strs = [_f2(x) for x in s_fraction]

    metal_only = list(cfg.get("metal_only", []))
    oxide_only = list(cfg.get("oxide_only", []))

    s_group = {
        "name": "S_site",
        "cation_set": ["O", "Cl", "S"],
        "fraction_set": s_fraction_strs,
        "total_units": 100,
        "n_components": 3,
        "episode_style": "fixed_order_amount",
        "constraint_filter": "sse_doping",
        "role": "s_site",
        "host_P": host_P,
        "o_off": o_off_str,
        "o_on": o_on_str,
        "cl_values": cl_sel_strs,
        "metal_only": metal_only,
        "oxide_only": oxide_only,
    }

    # ---------------- assemble the multi_group + predictor config ------------
    out = cfg
    # strip the high-level-only keys
    for k in ("dopant_metals", "metal_level", "cl_per_fu", "metal_only", "oxide_only"):
        out.pop(k, None)

    out["env_type"] = "multi_group"
    out["groups"] = [p_group, s_group]
    out.setdefault("predictor", "structure_pipeline")
    out.setdefault("builder", "sse")
    # builder keys the recipe reads (top-level)
    out["cl_map"] = cl_map
    out["o_off"] = o_off
    out.setdefault("p_site_group", "P_site")
    out.setdefault("s_site_group", "S_site")
    out.setdefault("eligible_region", {"symbol": "S", "take": "last", "count": 1000})
    out.setdefault("s_site_per_fu", s_per_fu)
    return out
