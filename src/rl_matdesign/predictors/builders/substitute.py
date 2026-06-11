"""SubstituteBuilder — the lightweight fixed-lattice element-swap builder.

This is the builder behind the old ``dp_structure`` / ``dp_property`` inline
``substitute_sites`` call, promoted to a first-class, registry-resolvable
builder so it can be reused and swapped exactly like the heavier ``sse``
builder.

What it does: take a template POSCAR with placeholder sites marked by
``site_symbol`` (e.g. ``X`` for an FCC alloy, ``Fe`` for a perovskite B-site),
fill those sites per the candidate's fractional composition, and return
``n_configs`` random site assignments. One sublattice, no vacancies, integer
counts allocated from fractions (largest-remainder). The lattice itself is
never changed — for vacancies / multi-sublattice supercells use ``sse`` or a
custom builder.

Config keys
-----------
    base_poscar (or legacy ``poscar`` / ``poscar_template``): template path.
    site_symbol (default ``"X"``): the placeholder element to fill.

The builder presents the same ``build(candidate, *, n_configs, rng)`` interface
as every other builder, so ``structure_score`` / any predictor can resolve it
through :func:`rl_matdesign.registry.resolve_builder`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


class SubstituteBuilder:
    def __init__(self, cfg: Dict[str, Any], *, seed: Optional[int] = None) -> None:
        poscar = cfg.get("base_poscar") or cfg.get("poscar") or cfg.get("poscar_template")
        if not poscar:
            raise ValueError(
                "SubstituteBuilder needs a POSCAR template — set 'base_poscar' "
                "(or legacy 'poscar' / 'poscar_template') in the config."
            )
        self.poscar_template: str = str(poscar)
        self.site_symbol: str = str(cfg.get("site_symbol", "X"))
        self._seed = seed

    def build(
        self,
        candidate: Dict[str, float],
        *,
        n_configs: int = 1,
        rng: Optional[np.random.Generator] = None,
    ) -> List["ase.Atoms"]:
        """Fill the template's ``site_symbol`` sites per *candidate* composition.

        *candidate* is a flat ``{element: fraction}`` mapping (what the
        single-group composition env emits). For a structured ``{group: {...}}``
        candidate use a multi-sublattice builder instead.
        """
        from ...utils.structure import substitute_sites

        if rng is None:
            rng = np.random.default_rng(self._seed)
        return substitute_sites(
            template_poscar=self.poscar_template,
            composition=candidate,
            site_symbol=self.site_symbol,
            n_configs=n_configs,
            rng=rng,
        )
