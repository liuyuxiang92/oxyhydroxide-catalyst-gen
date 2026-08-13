"""SitePickBuilder — one-element-per-site substitution across multiple sites.

Turns a :class:`~rl_matdesign.env_multigroup.MultiGroupEnv` candidate built from
one-slot ``categorical`` groups — e.g. ``{"A_site": {"A": "Sr"}, "B_site": {"B": "Fe"}}``
(what a group with a single ``choices:`` slot picking one element yields) — into a
structure by placing each group's picked element onto that group's placeholder
site symbol in a template POSCAR.

This is *not* a chemistry recipe like :class:`~.sse.SSESupercellBuilder` — no
valences, no vacancies, no derived elements. It is the direct multi-site
generalization of :class:`~.substitute.SubstituteBuilder` (which only knows how
to fill *one* placeholder symbol from a *flat* ``{element: fraction}``
candidate): here, each of N sites gets exactly the one element its group picked,
via the same underlying engine.

Config keys
-----------
    base_poscar (or legacy ``poscar`` / ``poscar_template``): template path.
    site_map: ``{group_name: placeholder_site_symbol}`` — e.g.
        ``{A_site: Sr, B_site: Fe}`` for a perovskite template where the A-site
        atoms are placeholder-tagged ``Sr`` and the B-site atoms ``Fe``. Every
        key must be a group name present in the candidate at build time.

The builder presents the same ``build(candidate, *, n_configs, rng)`` interface
as every other builder, so it resolves through
:func:`rl_matdesign.registry.resolve_builder` like ``substitute`` / ``sse``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


class SitePickBuilder:
    def __init__(self, cfg: Dict[str, Any], *, seed: Optional[int] = None) -> None:
        poscar = cfg.get("base_poscar") or cfg.get("poscar") or cfg.get("poscar_template")
        if not poscar:
            raise ValueError(
                "SitePickBuilder needs a POSCAR template — set 'base_poscar' "
                "(or legacy 'poscar' / 'poscar_template') in the config."
            )
        if not isinstance(poscar, (str, bytes)):
            raise TypeError(
                f"SitePickBuilder's base_poscar must be a single path, got "
                f"{type(poscar).__name__}. (Multi-template mode isn't supported here — "
                "use one call per template if you need that.)"
            )
        self.base_poscar: str = str(poscar)

        site_map = cfg.get("site_map")
        if not site_map or not isinstance(site_map, dict):
            raise ValueError(
                "SitePickBuilder needs a non-empty 'site_map' dict: "
                "{group_name: placeholder_site_symbol, ...}."
            )
        self.site_map: Dict[str, str] = {str(k): str(v) for k, v in site_map.items()}
        self._seed = seed

    # ------------------------------------------------------------------

    def build(
        self,
        candidate: Dict[str, Dict[str, Any]],
        *,
        n_configs: int = 1,
        rng: Optional[np.random.Generator] = None,
    ) -> List["ase.Atoms"]:
        """Place each group's picked element onto that group's placeholder site.

        *candidate* is the structured ``{group_name: {slot_name: element}}``
        mapping a ``MultiGroupEnv`` with one-slot categorical groups produces
        (:class:`~rl_matdesign.env_multigroup.CategoricalGroup.cation_fractions`).
        Each group named in ``site_map`` must be present and hold exactly one
        picked value; that element replaces **every** atom in the template
        carrying the group's mapped ``site_symbol`` (the count is read off the
        template itself, not assumed to be 1).
        """
        from ...utils.structure import SublatticeOp, build_substituted_structure
        from ase.io import read as ase_read

        if rng is None:
            rng = np.random.default_rng(self._seed)

        template = ase_read(self.base_poscar)
        symbols = template.get_chemical_symbols()

        ops: List[SublatticeOp] = []
        for group_name, site_symbol in self.site_map.items():
            if group_name not in candidate:
                raise KeyError(
                    f"SitePickBuilder's site_map names group {group_name!r}, but "
                    f"the candidate only has groups {sorted(candidate)}."
                )
            picks = candidate[group_name]
            if len(picks) != 1:
                raise ValueError(
                    f"SitePickBuilder expects group {group_name!r} to have picked "
                    f"exactly one element (a one-slot categorical group), got "
                    f"{len(picks)}: {picks!r}."
                )
            (element,) = picks.values()
            n_sites = sum(1 for s in symbols if s == site_symbol)
            if n_sites == 0:
                raise ValueError(
                    f"No sites with symbol {site_symbol!r} (site_map[{group_name!r}]) "
                    f"found in {self.base_poscar}."
                )
            ops.append(SublatticeOp(sites=site_symbol, put={str(element): n_sites}))

        return build_substituted_structure(template, ops, n_configs=n_configs, rng=rng)

    def composition_formula(self, candidate: Dict[str, Dict[str, Any]]) -> str:
        """Full composition of the built structure — the agent's picks plus any
        template atoms not covered by ``site_map`` (e.g. the O lattice in a
        perovskite). Used for the ``formula`` column in generated.csv, per the
        ``StructureScorePredictor.composition_formula`` optional-hook contract.
        """
        from ase.io import read as ase_read

        template = ase_read(self.base_poscar)
        symbols = template.get_chemical_symbols()
        covered = set(self.site_map.values())

        counts: Dict[str, float] = {}
        for sym in symbols:
            if sym not in covered:
                counts[sym] = counts.get(sym, 0.0) + 1
        for group_name, site_symbol in self.site_map.items():
            picks = candidate.get(group_name, {})
            if len(picks) != 1:
                continue  # let build() raise the real error; this is display-only
            (element,) = picks.values()
            n_sites = sum(1 for s in symbols if s == site_symbol)
            counts[str(element)] = counts.get(str(element), 0.0) + n_sites

        items = sorted(counts.items(), key=lambda t: (-t[1], t[0]))
        return "".join(f"{el}{n:.3g}" for el, n in items)
