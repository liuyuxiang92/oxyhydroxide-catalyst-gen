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
        self.halide_total: float = float(cfg.get("halide_total", 1.7))  # per f.u.
        self.eligible_region = cfg.get("eligible_region", {"symbol": "S", "take": "last", "count": 1000})
        # formula_units is normally inferred from the POSCAR (host-P count /
        # p_site_per_fu); an explicit config value overrides.
        self._fu_cfg = cfg.get("formula_units")
        self._fu_cached: Optional[int] = None

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

        # S-site slot wiring. With multiple dopant metals the S-site carries one
        # O-form slot per metal (e.g. names O_a, O_b sharing element O); the i-th
        # O-form slot pairs with the i-th *sorted* dopant metal. Discover the
        # ordered O-form slot names and the Cl slot name from the categorical
        # group's `choices`; fall back to bare element names for legacy configs.
        self.o_element: str = str(cfg.get("o_element", "O"))
        self.cl_element: str = str(cfg.get("cl_element", "Cl"))
        self.o_form_slots, self.cl_slot = self._discover_s_slots(cfg)

        self._seed = seed

    def _discover_s_slots(self, cfg: Dict[str, Any]):
        """Ordered O-form slot names + the Cl slot name, read from the S-site group."""
        o_slots: List[str] = []
        cl_slot = self.cl_element
        for g in cfg.get("groups", []) or []:
            if str(g.get("name", "")) != self.s_site_group:
                continue
            for c in g.get("choices", []) or []:
                name = str(c.get("name", c["element"]))
                el = str(c["element"])
                if el == self.o_element:
                    o_slots.append(name)
                elif el == self.cl_element:
                    cl_slot = name
        return (o_slots or [self.o_element]), cl_slot

    # ------------------------------------------------------------------

    @property
    def fu(self) -> int:
        """Formula units in the supercell (config override, else inferred from POSCAR)."""
        if self._fu_cfg is not None:
            return int(self._fu_cfg)
        if self._fu_cached is None:
            from ase.io import read as ase_read

            syms = list(ase_read(self.base_poscar).get_chemical_symbols())
            self._fu_cached = syms.count(self.host_P) // self.p_site_per_fu
        return self._fu_cached

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
        """Return (metals, levels, o_forms, cl_count) from the structured picks.

        ``metals`` is the **sorted** list of distinct dopant symbols (same-element
        picks already merged upstream into one summed fraction); ``levels`` /
        ``o_forms`` map each metal to its fraction-of-P and its O-form flag (the
        i-th sorted metal reads the i-th O-form slot). ``cl_count`` is per-f.u.
        """
        try:
            p_site = candidate[self.p_site_group]
            s_site = candidate[self.s_site_group]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"SSE builder expected groups {self.p_site_group!r} and "
                f"{self.s_site_group!r} in the candidate, got {candidate!r}."
            ) from exc

        metals = sorted(el for el in p_site if el != self.host_P and p_site[el] > 0)
        if not metals:
            raise ValueError(
                f"P-site has no dopant metal (besides host {self.host_P!r}): {p_site!r}."
            )
        levels = {m: float(p_site[m]) for m in metals}
        # Per-metal O-form: O is a numeric flag (0 = metal/sulfide form, >0 = oxide
        # form). The i-th sorted metal reads the i-th O-form slot; extra metals
        # beyond the available slots default to sulfide.
        o_forms: Dict[str, float] = {}
        for i, m in enumerate(metals):
            slot = self.o_form_slots[i] if i < len(self.o_form_slots) else None
            o_forms[m] = float(s_site.get(slot, 0.0)) if slot is not None else 0.0
        cl_count = float(s_site.get(self.cl_slot, 0.0))
        return metals, levels, o_forms, cl_count

    def counts(self, candidate: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Integer supercell counts + the charge-neutral Li vacancy (table-driven).

        Handles one or more dopant metals, each in its own sulfide/oxide form.
        """
        metals, levels, o_forms, cl_count = self._decode(candidate)

        fu = self.fu
        n_P_total = self.p_site_per_fu * fu
        n_S_total = self.s_site_per_fu * fu
        n_Li_total = self.li_per_fu * fu

        metal_counts = {m: int(round(levels[m] * n_P_total)) for m in metals}
        n_metal_total = sum(metal_counts.values())
        n_P = n_P_total - n_metal_total

        # O is derived per oxide-form metal: n_O = Σ (oxide_valence/2)·n_metal.
        n_O = 0
        for m in metals:
            if o_forms[m] > 0:
                n_O += int(round(0.5 * self._valence(m, scenario="oxide") * metal_counts[m]))

        n_Cl = int(round(cl_count * fu))                       # Cl is a real per-f.u. count
        n_Br = int(round(self.halide_total * fu)) - n_Cl
        n_S = n_S_total - n_O - n_Cl - n_Br
        if n_Br < 0 or n_S < 0 or n_P < 0:
            raise ValueError(
                f"Infeasible allocation: P={n_P} O={n_O} Cl={n_Cl} Br={n_Br} S={n_S} "
                f"(metals={metal_counts}). Check halide_total / amounts."
            )

        # Generic charge balance: solve for the neutral Li count. Anions and the
        # host cations are constant-valence; each metal contributes with the
        # valence of *its* form (oxide or sulfide).
        anion_charge = (
            n_S * abs(self._valence(self.host_S, scenario="sulfide"))
            + n_O * abs(self._valence("O", scenario="sulfide"))
            + n_Cl * abs(self._valence("Cl", scenario="sulfide"))
            + n_Br * abs(self._valence("Br", scenario="sulfide"))
        )
        non_li_cation_charge = n_P * self._valence(self.host_P, scenario="sulfide")
        for m in metals:
            scen = "oxide" if o_forms[m] > 0 else "sulfide"
            non_li_cation_charge += metal_counts[m] * self._valence(m, scenario=scen)
        v_li = self._valence(self.host_Li, scenario="sulfide")
        n_Li = int(round((anion_charge - non_li_cation_charge) / v_li))
        n_Li_delete = n_Li_total - n_Li
        if not (0 <= n_Li_delete <= n_Li_total):
            raise ValueError(
                f"Charge-neutral Li ({n_Li}) out of range [0,{n_Li_total}] for "
                f"metals={levels} O_forms={o_forms} Cl={cl_count}."
            )

        return {
            "metals": metal_counts, "P": n_P, "O": n_O, "Cl": n_Cl, "Br": n_Br,
            "S": n_S, "Li": n_Li, "Li_delete": n_Li_delete, "o_forms": dict(o_forms),
        }

    def composition_formula(self, candidate: Dict[str, Dict[str, float]]) -> str:
        """Full per-formula-unit composition of the *built* structure.

        Unlike the env's `terminal_formula` (which only sees the agent's picks —
        the P-site metal and the S-site O/Cl), this reports every element the
        builder actually places, including the host ``S`` that survives, the
        derived ``Br`` (``halide_total − Cl``), and the charge-balanced ``Li``.
        Used for the ``formula`` column in generated.csv so the label matches the
        structure that was scored.
        """
        c = self.counts(candidate)
        fu = self.fu
        merged: Dict[str, float] = {}

        def _add(el: str, n: float) -> None:
            if n > 0:
                merged[el] = merged.get(el, 0.0) + n / fu

        for sym, cnt in c["metals"].items():
            _add(sym, cnt)
        _add(self.host_P, c["P"])
        _add("O", c["O"])
        _add("Cl", c["Cl"])
        _add("Br", c["Br"])
        _add(self.host_S, c["S"])
        _add(self.host_Li, c["Li"])
        items = sorted(merged.items(), key=lambda t: (-t[1], t[0]))
        return "".join(f"{el}{n:.3g}" for el, n in items)

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

        metal_put = {m: n for m, n in c["metals"].items() if n > 0}
        ops = [
            SublatticeOp(sites=self.host_P, put=metal_put),
            SublatticeOp(sites=s_region, put={"O": c["O"], "Cl": c["Cl"], "Br": c["Br"]}),
            SublatticeOp(sites=self.host_Li, put={}, remove=c["Li_delete"]),
        ]
        return build_substituted_structure(template, ops, n_configs=n_configs, rng=rng)
