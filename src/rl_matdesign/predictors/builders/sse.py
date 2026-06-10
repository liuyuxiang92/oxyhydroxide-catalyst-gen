"""SSESupercellBuilder — the doped-Li6PS6 recipe (Layer 2).

Turns the agent's structured picks ``{P_site: {...}, S_site: {...}}`` into a
doped supercell by:

1. **Decoding** the picks: the P-site dopant metal + its level ``x``; the S-site
   oxygen fraction (``0`` ⇒ sulfide scenario, ``>0`` ⇒ oxide) and chlorine fraction.
2. **Deriving** the rest by chemistry (never agent choices):
   - bromine ``n_Br = halide_total·fu − n_Cl``;
   - the charge-neutral lithium count via a **generic, table-driven charge
     balance** over a user ``valences`` table (the metal uses its ``sulfide`` or
     ``oxide`` valence depending on the scenario);
   - the Li vacancy = host Li − neutral Li.
3. **Emitting** :class:`SublatticeOp` ops and calling the general
   :func:`build_substituted_structure` engine (Layer 1): P→metal, the eligible
   S region → O/Cl/Br, and Li deletions.

All charge logic is table-driven, so a user-supplied ``valences`` map fully
determines the vacancy (the "0.7 base" emerges from the halide budget on its own).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


class SSESupercellBuilder:
    def __init__(self, cfg: Dict[str, Any], *, seed: Optional[int] = None) -> None:
        poscar = cfg.get("base_poscar") or cfg.get("poscar")
        if not poscar:
            raise ValueError("SSESupercellBuilder needs 'base_poscar'.")
        self.base_poscar: str = str(poscar)

        self.valences: Dict[str, Any] = dict(cfg["valences"])  # {el: int | {sulfide,oxide}}
        self.fu: int = int(cfg.get("formula_units", 500))
        self.halide_total: float = float(cfg.get("halide_total", 1.7))  # per f.u.
        self.eligible_region = cfg.get("eligible_region", {"symbol": "S", "take": "last", "count": 1000})

        # Group names + host symbols / per-f.u. site counts.
        self.p_site_group: str = str(cfg.get("p_site_group", "P_site"))
        self.s_site_group: str = str(cfg.get("s_site_group", "S_site"))
        host = cfg.get("host", {})
        self.host_P: str = str(host.get("P", "P"))
        self.host_S: str = str(host.get("S", "S"))
        self.host_Li: str = str(host.get("Li", "Li"))
        self.p_site_per_fu: int = int(cfg.get("p_site_per_fu", 1))
        self.s_site_per_fu: int = int(cfg.get("s_site_per_fu", 6))
        self.li_per_fu: int = int(cfg.get("li_per_fu", 6))

        self._seed = seed

    # ------------------------------------------------------------------

    def _valence(self, el: str, *, scenario: str) -> float:
        v = self.valences.get(el)
        if v is None:
            raise KeyError(
                f"No valence for {el!r} in the 'valences' table — add it (an int, "
                "or a {sulfide:.., oxide:..} map for scenario-dependent metals)."
            )
        if isinstance(v, dict):
            if scenario not in v:
                raise KeyError(f"valences[{el!r}] has no {scenario!r} entry: {v}")
            return float(v[scenario])
        return float(v)

    def _decode(self, candidate: Dict[str, Dict[str, float]]):
        """Return (metal, level, o_frac, cl_frac) from the structured picks."""
        try:
            p_site = candidate[self.p_site_group]
            s_site = candidate[self.s_site_group]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"SSE builder expected groups {self.p_site_group!r} and "
                f"{self.s_site_group!r} in the candidate, got {candidate!r}."
            ) from exc

        metals = [el for el in p_site if el != self.host_P and p_site[el] > 0]
        if len(metals) != 1:
            raise ValueError(
                f"P-site must contain exactly one dopant metal (besides host "
                f"{self.host_P!r}); got {p_site!r}."
            )
        metal = metals[0]
        level = float(p_site[metal])
        o_frac = float(s_site.get("O", 0.0))
        cl_frac = float(s_site.get("Cl", 0.0))
        return metal, level, o_frac, cl_frac

    def counts(self, candidate: Dict[str, Dict[str, float]]) -> Dict[str, int]:
        """Integer supercell counts + the charge-neutral Li vacancy (table-driven)."""
        metal, level, o_frac, cl_frac = self._decode(candidate)
        scenario = "oxide" if o_frac > 0 else "sulfide"

        n_P_total = self.p_site_per_fu * self.fu
        n_S_total = self.s_site_per_fu * self.fu
        n_Li_total = self.li_per_fu * self.fu

        n_metal = int(round(level * n_P_total))
        n_P = n_P_total - n_metal
        n_O = int(round(o_frac * n_S_total))
        n_Cl = int(round(cl_frac * n_S_total))
        n_Br = int(round(self.halide_total * self.fu)) - n_Cl
        n_S = n_S_total - n_O - n_Cl - n_Br
        if n_Br < 0 or n_S < 0:
            raise ValueError(
                f"Infeasible S-site allocation: O={n_O} Cl={n_Cl} Br={n_Br} S={n_S} "
                f"(of {n_S_total}). Check halide_total / fractions."
            )

        # Generic charge balance: solve for neutral Li count.
        anion_charge = (
            n_S * abs(self._valence(self.host_S, scenario=scenario))
            + n_O * abs(self._valence("O", scenario=scenario))
            + n_Cl * abs(self._valence("Cl", scenario=scenario))
            + n_Br * abs(self._valence("Br", scenario=scenario))
        )
        non_li_cation_charge = (
            n_P * self._valence(self.host_P, scenario=scenario)
            + n_metal * self._valence(metal, scenario=scenario)
        )
        v_li = self._valence(self.host_Li, scenario=scenario)
        n_Li = int(round((anion_charge - non_li_cation_charge) / v_li))
        n_Li_delete = n_Li_total - n_Li
        if not (0 <= n_Li_delete <= n_Li_total):
            raise ValueError(
                f"Charge-neutral Li ({n_Li}) out of range [0,{n_Li_total}] for "
                f"metal={metal} level={level} O={o_frac} Cl={cl_frac}."
            )

        return {
            "metal": n_metal, "P": n_P, "O": n_O, "Cl": n_Cl, "Br": n_Br, "S": n_S,
            "Li": n_Li, "Li_delete": n_Li_delete, "metal_symbol": metal,  # type: ignore[dict-item]
        }

    def build(
        self, candidate: Dict[str, Dict[str, float]], *, n_configs: int = 1, rng=None
    ) -> List["ase.Atoms"]:
        from ase.io import read as ase_read
        from ...utils.structure import (
            SublatticeOp,
            build_substituted_structure,
            resolve_region,
        )

        if rng is None:
            rng = np.random.default_rng(self._seed)

        c = self.counts(candidate)
        template = ase_read(self.base_poscar)
        s_region = resolve_region(template, self.eligible_region)

        ops = [
            SublatticeOp(sites=self.host_P, put={c["metal_symbol"]: c["metal"]}),
            SublatticeOp(sites=s_region, put={"O": c["O"], "Cl": c["Cl"], "Br": c["Br"]}),
            SublatticeOp(sites=self.host_Li, put={}, remove=c["Li_delete"]),
        ]
        return build_substituted_structure(template, ops, n_configs=n_configs, rng=rng)
