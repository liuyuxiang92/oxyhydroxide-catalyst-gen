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

Single vs multi-template is **auto-detected** from ``base_poscar`` (mirrors how
``properties`` and ``filters`` auto-detect single vs list):

* ``base_poscar: POSCAR`` — *single mode*: fill one template, return
  ``n_configs`` cells.
* ``base_poscar: [POSCAR_A, POSCAR_B]`` or a list of
  ``{path, n_configs, site_symbol}`` dicts — *multi mode*: fill **each**
  template independently and **concatenate** all the cells into one ensemble.
  Each entry may set its own ``n_configs`` (falls back to the caller's
  ``n_random_configs``) and its own ``site_symbol`` (falls back to the
  top-level one). The combined list is scored as a single ensemble by
  ``structure_score`` — i.e. the property is averaged across all parent
  lattices (e.g. several polymorphs of the same composition).

Config keys
-----------
    base_poscar (or legacy ``poscar`` / ``poscar_template``): template path, or
        a list of paths / ``{path, n_configs?, site_symbol?}`` dicts.
    site_symbol (default ``"X"``): the placeholder element to fill (per-entry
        ``site_symbol`` overrides this in multi mode).

The builder presents the same ``build(candidate, *, n_configs, rng)`` interface
as every other builder, so ``structure_score`` / any predictor can resolve it
through :func:`rl_matdesign.registry.resolve_builder`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class _TemplateSpec:
    """One POSCAR template + how many cells to build from it."""

    path: str
    site_symbol: str
    n_configs: Optional[int] = None  # None -> use the caller's n_configs


class SubstituteBuilder:
    def __init__(self, cfg: Dict[str, Any], *, seed: Optional[int] = None) -> None:
        poscar = cfg.get("base_poscar") or cfg.get("poscar") or cfg.get("poscar_template")
        if not poscar:
            raise ValueError(
                "SubstituteBuilder needs a POSCAR template — set 'base_poscar' "
                "(or legacy 'poscar' / 'poscar_template') in the config. It may be "
                "a single path or a list of paths / {path, n_configs} dicts."
            )
        self.site_symbol: str = str(cfg.get("site_symbol", "X"))
        self.specs: List[_TemplateSpec] = self._parse_templates(poscar)
        # True when more than one template was given (list form). Single-string
        # form keeps the original single-template behavior bit-for-bit.
        self.multi: bool = len(self.specs) > 1
        self._seed = seed

    # ------------------------------------------------------------------

    def _parse_templates(self, poscar: Any) -> List[_TemplateSpec]:
        """Normalize the ``base_poscar`` value into a list of template specs.

        Accepts a scalar path (single mode) or a list of paths / dicts (multi
        mode). Each dict entry may carry ``path`` (or legacy ``base_poscar`` /
        ``poscar``), an optional ``n_configs``, and an optional ``site_symbol``.
        """
        if isinstance(poscar, (str, bytes)):
            return [_TemplateSpec(path=str(poscar), site_symbol=self.site_symbol)]
        if not isinstance(poscar, (list, tuple)):
            raise TypeError(
                f"base_poscar must be a path or a list of paths/dicts, got "
                f"{type(poscar).__name__}."
            )
        specs: List[_TemplateSpec] = []
        for i, entry in enumerate(poscar):
            if isinstance(entry, (str, bytes)):
                specs.append(_TemplateSpec(path=str(entry), site_symbol=self.site_symbol))
                continue
            if not isinstance(entry, dict):
                raise TypeError(
                    f"base_poscar[{i}] must be a path string or a dict, got "
                    f"{type(entry).__name__}."
                )
            path = entry.get("path") or entry.get("base_poscar") or entry.get("poscar")
            if not path:
                raise ValueError(
                    f"base_poscar[{i}] dict needs a 'path' (or legacy 'base_poscar' / "
                    f"'poscar') key: {entry!r}."
                )
            n_cfg = entry.get("n_configs")
            specs.append(
                _TemplateSpec(
                    path=str(path),
                    site_symbol=str(entry.get("site_symbol", self.site_symbol)),
                    n_configs=None if n_cfg is None else int(n_cfg),
                )
            )
        if not specs:
            raise ValueError("base_poscar list is empty — give at least one template.")
        return specs

    # ------------------------------------------------------------------

    def build(
        self,
        candidate: Dict[str, float],
        *,
        n_configs: int = 1,
        rng: Optional[np.random.Generator] = None,
    ) -> List["ase.Atoms"]:
        """Fill each template's ``site_symbol`` sites per *candidate* composition.

        *candidate* is a flat ``{element: fraction}`` mapping (what the
        single-group composition env emits). For a structured ``{group: {...}}``
        candidate use a multi-sublattice builder instead.

        In multi mode the cells from every template are concatenated into one
        list, so ``structure_score`` folds them into a single ensemble (the
        property is averaged across all parent lattices). A template's own
        ``n_configs`` takes precedence; otherwise the caller's ``n_configs`` is
        used for that template.
        """
        from ...utils.structure import substitute_sites

        if rng is None:
            rng = np.random.default_rng(self._seed)

        structures: List["ase.Atoms"] = []
        for spec in self.specs:
            k = spec.n_configs if spec.n_configs is not None else n_configs
            structures.extend(
                substitute_sites(
                    template_poscar=spec.path,
                    composition=candidate,
                    site_symbol=spec.site_symbol,
                    n_configs=k,
                    rng=rng,
                )
            )
        return structures
