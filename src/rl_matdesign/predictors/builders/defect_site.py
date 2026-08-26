"""DefectSiteBuilder — A/B-site substitution doping plus a signed A-site defect axis.

Pure structure construction, no physics: turns the structured ``MultiGroupEnv``
candidate produced by an ``independent``-kind A/B dopant group plus one
``categorical`` defect group into a doped, optionally defected ``ase.Atoms``.

Candidate shape expected (group names are configurable, defaults match the
perovskite Level-2 scenario)::

    {
      "A_dopant": {"Ca": 0.09375, "Ba": 0.03125},   # 0-2 keys (repeats merge)
      "A_defect": {"defect": "Sr_-0.09375"},          # "<species>_<signed amount>"
      "B_dopant": {"Mn": 0.0625, "Co": 0.03125},
    }

The defect pick is a SINGLE step: one categorical slot whose values are
flattened ``"<species>_<signed amount>"`` labels (e.g. ``Ca_0.03125``), so
choosing which species AND how much/which sign happens in one action — the
same "one pick, one step" shape ``A_dopant``/``B_dopant`` have, just via a
different group kind (see "Why categorical, not independent" below).

Decoding
--------
1. A/B dopant fractions -> integer atom counts on the A-/B-site sublattice
   (``round(fraction * n_sites)``, clamped so the combined dopant count never
   exceeds the sublattice size — two independent picks can in principle land
   above 100% between them; this is a rare edge case, so the clamp just scales
   counts down proportionally and prints a warning rather than raising).
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

Why categorical, not independent
---------------------------------
``A_dopant``/``B_dopant`` use ``kind: independent`` (``IndependentDopantsGroup``),
which *also* has a built-in joint element+amount pick (its action space is the
full ``species_set x fraction_set`` cross product). The defect axis does NOT
reuse that mechanism, because ``IndependentDopantsGroup.terminal_comp_key()``
computes ``units = round(fraction * total_units)`` and keeps the entry only if
``units > 0`` — a NEGATIVE fraction (a vacancy pick) would be silently dropped
from the dedup key used by ``generated.csv``, letting two structurally
different vacancy candidates collapse into the same key. ``CategoricalGroup``'s
own ``terminal_comp_key()`` has no such filter (it just stores ``str(value)``),
so encoding the signed pick as an opaque compound label in a categorical slot
sidesteps that bug entirely.

Config keys
-----------
    base_poscar (or legacy ``poscar`` / ``poscar_template``): template path.
    a_site_symbol / b_site_symbol (default ``Sr`` / ``Fe``): placeholder element
        for the A-/B-site sublattice in the template.
    a_dopant_group / b_dopant_group (default ``A_dopant`` / ``B_dopant``): the
        candidate keys holding the ``independent``-kind dopant picks.
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

        a_counts = self._dopant_counts(candidate.get(self.a_group) or {}, n_a, "A-site")
        b_counts = self._dopant_counts(candidate.get(self.b_group) or {}, n_b, "B-site")

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

        ops = [
            SublatticeOp(sites=self.a_site_symbol, put=a_counts, remove=a_remove),
            SublatticeOp(sites=self.b_site_symbol, put=b_counts),
        ]
        if insert_spec:
            ops.append(SublatticeOp(sites=[], insert=insert_spec, min_dist=self.min_dist))

        return build_substituted_structure(template, ops, n_configs=n_configs, rng=rng)

    def composition_formula(self, candidate: Dict[str, Dict[str, Any]]) -> str:
        """Display-only label: A/B dopant picks + the signed defect pick, if any."""
        parts = []
        for group in (self.a_group, self.b_group):
            for el, frac in (candidate.get(group) or {}).items():
                try:
                    parts.append(f"{el}{float(frac):.3g}")
                except (TypeError, ValueError):
                    parts.append(f"{el}{frac}")
        species, amount = self._decode_defect(candidate)
        if species and amount:
            tag = "vac" if amount < 0 else "int"
            parts.append(f"{species}_{tag}{abs(amount):.3g}")
        return "".join(parts)

    # ------------------------------------------------------------------

    def _decode_defect(self, candidate: Dict[str, Dict[str, Any]]) -> tuple:
        """Parse the combined ``"<species>_<signed amount>"`` categorical pick."""
        defect_pick = candidate.get(self.a_defect_group) or {}
        label = next(iter(defect_pick.values()), None)
        if label is None:
            return None, 0.0
        species, _, amount_str = str(label).partition("_")
        return species, float(amount_str)

    @staticmethod
    def _dopant_counts(fractions: Dict[str, Any], n_sites: int, label: str) -> Dict[str, int]:
        """``{element: fraction}`` (need not sum to 1 — unpicked remainder is host)
        -> ``{element: integer atom count}``, clamped so the total never exceeds
        ``n_sites``. Two independent dopant picks can in principle land above 100%
        combined; that's scaled down proportionally (with a warning) rather than
        raising, so an unlucky pick never crashes a training episode.
        """
        counts = {
            str(el): int(round(float(f) * n_sites))
            for el, f in fractions.items()
            if float(f) > 0
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
