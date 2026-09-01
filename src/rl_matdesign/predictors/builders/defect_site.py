"""DefectSiteBuilder — A/B-site substitution doping plus a signed A-site defect axis.

Pure structure construction, no physics: turns the structured ``MultiGroupEnv``
candidate produced by a ``composition``-kind A/B dopant group (host species
included explicitly in ``species_set``) plus one ``categorical`` defect group
into a doped, optionally defected ``ase.Atoms``.

Candidate shape expected (group names are configurable, defaults match the
perovskite Level-2 scenario)::

    {
      "A_dopant": {"Sr": 0.21875, "Ca": 0.5, "Ba": 0.28125},   # sums to 1.0; Sr is the host
      "A_defect": {"defect": "Sr_-0.09375"},                    # "<species>_<signed amount>"
      "B_dopant": {"Fe": 0.5, "Mn": 0.34375, "Co": 0.15625},
    }

The defect pick is a SINGLE step: one categorical slot whose values are
flattened ``"<species>_<signed amount>"`` labels (e.g. ``Ca_0.03125``), so
choosing which species AND how much/which sign happens in one action — the
same "one pick, one step" shape ``A_dopant``/``B_dopant`` have, just via a
different group kind (see "Why categorical, not composition" below).

Decoding
--------
1. A/B dopant fractions -> integer atom counts on the A-/B-site sublattice
   (``round(fraction * n_sites)``). The host species (``a_site_symbol`` /
   ``b_site_symbol``) is excluded from the ``put`` counts even though it is
   now an explicit key in the candidate dict — the template's sublattice
   atoms already ARE that element, so "putting" it back is a no-op that would
   otherwise double-count against the vacancy math below. Because the group
   is a single sum-to-1 ``composition`` pick, the non-host counts can never
   exceed ``n_sites`` combined (a structural guarantee, not a runtime check),
   unlike the old ``independent``-kind group which needed a defensive
   proportional-scale-down clamp for a combined pick that overshot 100%; that
   clamp is kept as a harmless safety net but should no longer trigger.
2. The defect pick is *signed*: negative -> vacancy of ``defect_species``,
   positive -> interstitial of ``defect_species`` (new atoms, not tied to any
   existing site — see :func:`rl_matdesign.utils.structure.build_substituted_structure`'s
   ``insert``). A vacancy is folded into the *same* ``SublatticeOp`` as the A-site
   doping (reducing that species' ``put`` count and adding to ``remove``) because
   :func:`build_substituted_structure` resolves ``sites=`` against the **original**
   template symbols — there is no way to select "the Ca atoms just placed by
   doping" as a separate op, since the template has no Ca atoms at all before
   substitution happens. Folding into one op sidesteps that entirely. A vacancy
   amount is clamped to what's actually present for the chosen species (with a
   warning) so an infeasible pick never crashes a training episode.
3. ``build_substituted_structure`` is called once with the combined ops.

Why categorical, not composition
----------------------------------
``A_dopant``/``B_dopant`` use ``kind: composition``, which *also* has a
built-in joint element+amount pick (its action space is the
``species_set x fraction_set`` cross product, one distinct species per step).
The defect axis does NOT reuse that mechanism, because a composition group
enforces fractions that sum to 1.0 and drops zero-fraction entries from its
dedup key — a NEGATIVE fraction (a vacancy pick) doesn't fit that "amounts of
a whole" model at all. ``CategoricalGroup``'s ``terminal_comp_key()`` has no
such assumption (it just stores ``str(value)``), so encoding the signed pick
as an opaque compound label in a categorical slot sidesteps that mismatch
entirely.

Config keys
-----------
    base_poscar (or legacy ``poscar`` / ``poscar_template``): template path.
    a_site_symbol / b_site_symbol (default ``Sr`` / ``Fe``): placeholder element
        for the A-/B-site sublattice in the template. Must also be the host
        entry in the corresponding group's ``species_set``.
    a_dopant_group / b_dopant_group (default ``A_dopant`` / ``B_dopant``): the
        candidate keys holding the ``composition``-kind dopant+host picks.
    a_defect_group (default ``A_defect``): the candidate key holding the single
        combined ``"<species>_<signed amount>"`` categorical pick.
    interstitial_min_dist (default 1.5 Å): passed through to the new atoms'
        minimum allowed distance from any existing atom.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


class DefectSiteBuilder:
    def __init__(self, cfg: Dict[str, Any], *, seed: Optional[int] = None) -> None:
        poscar = cfg.get("base_poscar") or cfg.get("poscar") or cfg.get("poscar_template")
        if not poscar:
            raise ValueError(
                "DefectSiteBuilder needs a POSCAR template — set 'base_poscar' "
                "(or legacy 'poscar' / 'poscar_template') in the config."
            )
        self.base_poscar: str = str(poscar)
        self.a_site_symbol: str = str(cfg.get("a_site_symbol", "Sr"))
        self.b_site_symbol: str = str(cfg.get("b_site_symbol", "Fe"))
        self.a_group: str = str(cfg.get("a_dopant_group", "A_dopant"))
        self.b_group: str = str(cfg.get("b_dopant_group", "B_dopant"))
        self.a_defect_group: str = str(cfg.get("a_defect_group", "A_defect"))
        self.min_dist: float = float(cfg.get("interstitial_min_dist", 1.5))
        # Random-placement success drops sharply with distance (measured on
        # perovskite.vasp: 0.7% of random points valid at 1.7 A vs 6.1% at
        # 1.5 A), and up to 4 interstitials can be requested in one pick, each
        # shrinking the remaining free volume for the next -- the SublatticeOp
        # default of 200 attempts is NOT enough at realistic min_dist values.
        self.max_attempts: int = int(cfg.get("interstitial_max_attempts", 2000))
        self._seed = seed

    # ------------------------------------------------------------------

    def build(
        self,
        candidate: Dict[str, Dict[str, Any]],
        *,
        n_configs: int = 1,
        rng: Optional[np.random.Generator] = None,
    ) -> List["ase.Atoms"]:
        from ...utils.structure import SublatticeOp, build_substituted_structure
        from ase.io import read as ase_read

        if rng is None:
            rng = np.random.default_rng(self._seed)

        template = ase_read(self.base_poscar)
        symbols = template.get_chemical_symbols()
        n_a = sum(1 for s in symbols if s == self.a_site_symbol)
        n_b = sum(1 for s in symbols if s == self.b_site_symbol)
        if n_a == 0:
            raise ValueError(
                f"No sites with symbol {self.a_site_symbol!r} found in {self.base_poscar}."
            )
        if n_b == 0:
            raise ValueError(
                f"No sites with symbol {self.b_site_symbol!r} found in {self.base_poscar}."
            )

        a_counts, b_counts, a_remove, insert_spec = self._resolve_ops(candidate, n_a, n_b)

        ops = [
            SublatticeOp(sites=self.a_site_symbol, put=a_counts, remove=a_remove),
            SublatticeOp(sites=self.b_site_symbol, put=b_counts),
        ]
        if insert_spec:
            ops.append(SublatticeOp(
                sites=[], insert=insert_spec, min_dist=self.min_dist,
                max_attempts=self.max_attempts,
            ))

        return build_substituted_structure(template, ops, n_configs=n_configs, rng=rng)

    def composition_formula(self, candidate: Dict[str, Dict[str, Any]]) -> str:
        """Display-only label reflecting what was ACTUALLY built, not the raw
        candidate picks -- routes through the same ``_resolve_ops`` decoding
        ``build()`` uses (host excluded from dopant counts, vacancy/interstitial
        folded in), so the label can never drift from the real structure. With
        the old ``independent``-kind dopant group, two picks landing on the same
        element could sum past 1.0 (e.g. Ca=0.65625 + Ca=0.65625 = 1.3125) --
        physically impossible as a site fraction, silently clamped by
        `_resolve_ops` -- so the raw candidate value could print e.g. "Ca1.31"
        while the structure that was actually built, MD'd and scored was
        fully-substituted (Ca1.00), a real bug this repo hit in practice. The
        current ``composition``-kind dopant group makes that overshoot
        structurally impossible (one sum-to-1 pick), but this still routes
        through ``_resolve_ops`` rather than the raw candidate for the same
        reason: the displayed label must reflect what was built, not what was
        requested.
        """
        from ase.io import read as ase_read

        template = ase_read(self.base_poscar)
        symbols = template.get_chemical_symbols()
        n_a = sum(1 for s in symbols if s == self.a_site_symbol)
        n_b = sum(1 for s in symbols if s == self.b_site_symbol)
        a_counts, b_counts, a_remove, insert_spec = self._resolve_ops(candidate, n_a, n_b)
        defect_species, _defect_amount = self._decode_defect(candidate)

        parts = []
        for el, c in sorted(a_counts.items()):
            parts.append(f"{el}{c / n_a:.3g}")
        for el, c in sorted(b_counts.items()):
            parts.append(f"{el}{c / n_b:.3g}")
        if a_remove:
            parts.append(f"{defect_species}_vac{a_remove / n_a:.3g}")
        for el, c in insert_spec.items():
            parts.append(f"{el}_int{c / n_a:.3g}")
        return "".join(parts)

    # ------------------------------------------------------------------

    def _resolve_ops(
        self, candidate: Dict[str, Dict[str, Any]], n_a: int, n_b: int,
    ) -> tuple:
        """Decode *candidate* into ``(a_counts, b_counts, a_remove, insert_spec)``
        — the single source of truth for what gets built, shared by ``build()``
        and ``composition_formula()`` so the displayed label can never drift from
        the real structure (see ``composition_formula``'s docstring for why that
        matters: it already happened once).
        """
        a_counts = self._dopant_counts(
            candidate.get(self.a_group) or {}, n_a, "A-site", host=self.a_site_symbol
        )
        b_counts = self._dopant_counts(
            candidate.get(self.b_group) or {}, n_b, "B-site", host=self.b_site_symbol
        )

        defect_species, defect_amount = self._decode_defect(candidate)
        a_remove = 0
        insert_spec: Dict[str, int] = {}
        if defect_species and defect_amount != 0.0:
            n_defect = int(round(abs(defect_amount) * n_a))
            if defect_amount < 0:
                if defect_species == self.a_site_symbol:
                    present = n_a - sum(a_counts.values())
                else:
                    present = a_counts.get(defect_species, 0)
                n_vac = min(n_defect, present)
                if n_vac < n_defect:
                    print(
                        f"[defect_site] vacancy request for {n_defect} {defect_species} "
                        f"atom(s) exceeds the {present} present after doping — clamped "
                        f"to {n_vac}."
                    )
                if n_vac > 0:
                    if defect_species != self.a_site_symbol:
                        a_counts[defect_species] = a_counts.get(defect_species, 0) - n_vac
                    a_remove = n_vac
            else:
                insert_spec = {defect_species: n_defect}

        return a_counts, b_counts, a_remove, insert_spec

    def _decode_defect(self, candidate: Dict[str, Dict[str, Any]]) -> tuple:
        """Parse the combined ``"<species>_<signed amount>"`` categorical pick."""
        defect_pick = candidate.get(self.a_defect_group) or {}
        label = next(iter(defect_pick.values()), None)
        if label is None:
            return None, 0.0
        species, _, amount_str = str(label).partition("_")
        return species, float(amount_str)

    @staticmethod
    def _dopant_counts(
        fractions: Dict[str, Any], n_sites: int, label: str, *, host: str
    ) -> Dict[str, int]:
        """``{element: fraction}`` (sums to 1 including ``host``'s own fraction)
        -> ``{dopant_element: integer atom count}``, excluding ``host`` -- the
        template's sublattice atoms already ARE the host element, so there is
        nothing to "put" for it; only non-host entries become substitution
        counts. Clamped so the total never exceeds ``n_sites``: this is now a
        structural guarantee for a single sum-to-1 ``composition`` pick (dopant
        fractions alone can never exceed ``1 - host_fraction <= 1``), kept as a
        defensive no-op fallback for any other caller shape.
        """
        counts = {
            str(el): int(round(float(f) * n_sites))
            for el, f in fractions.items()
            if el != host and float(f) > 0
        }
        total = sum(counts.values())
        if total > n_sites:
            scale = n_sites / total
            counts = {el: int(np.floor(c * scale)) for el, c in counts.items()}
            print(
                f"[defect_site] combined {label} dopant fractions {fractions} exceed "
                f"the {n_sites}-site sublattice — scaled down to {counts}."
            )
        return counts
