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

        # Optional multi-sublattice mode: {group_name: placeholder_symbol}. Same key
        # and meaning as SitePickBuilder's, so the concept transfers; the difference
        # is that each group here carries FRACTIONS rather than a single element.
        site_map = cfg.get("site_map")
        self.site_map: Optional[Dict[str, str]] = None
        if site_map:
            if not isinstance(site_map, dict):
                raise TypeError(
                    f"SubstituteBuilder's site_map must be a dict "
                    f"{{group_name: site_symbol}}, got {type(site_map).__name__}."
                )
            self.site_map = {str(k): str(v) for k, v in site_map.items()}
            symbols = list(self.site_map.values())
            dupes = sorted({x for x in symbols if symbols.count(x) > 1})
            if dupes:
                raise ValueError(
                    f"SubstituteBuilder's site_map maps more than one group onto "
                    f"symbol(s) {dupes}. Site selection is resolved against the "
                    "ORIGINAL template symbols, so two groups sharing a symbol would "
                    "silently overwrite each other. Give each sublattice its own "
                    "placeholder symbol in the template."
                )

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

        if self.site_map is not None:
            return self._build_multi(candidate, n_configs=n_configs, rng=rng)

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

    # ------------------------------------------------------------------
    # Multi-sublattice mode (site_map)
    # ------------------------------------------------------------------

    def _build_multi(
        self,
        candidate: Dict[str, Dict[str, float]],
        *,
        n_configs: int,
        rng: np.random.Generator,
    ) -> List["ase.Atoms"]:
        """Fill several sublattices at once from a structured multi-group candidate.

        *candidate* is ``{group_name: {element: fraction}}`` — what a
        :class:`~rl_matdesign.env_multigroup.MultiGroupEnv` of ``composition``
        groups emits. Each group named in ``site_map`` fills that group's
        placeholder symbol; every other atom in the template is left alone.

        One ``SublatticeOp`` per group, all applied in a single
        ``build_substituted_structure`` call — which resolves each op against the
        ORIGINAL symbols, so the ops are order-independent and cannot interfere.
        """
        from ...utils.structure import (
            SublatticeOp, build_substituted_structure, _fractions_to_counts,
        )
        from ase.io import read as ase_read

        if len(self.specs) != 1:
            raise ValueError(
                "SubstituteBuilder's site_map mode takes a single base_poscar, got "
                f"{len(self.specs)} templates. Multi-template concatenation and "
                "multi-sublattice filling are separate features; use one at a time."
            )
        template = ase_read(self.specs[0].path)
        symbols = template.get_chemical_symbols()

        ops: List[SublatticeOp] = []
        for group, site_symbol in self.site_map.items():
            if group not in candidate:
                raise KeyError(
                    f"SubstituteBuilder's site_map names group {group!r}, but the "
                    f"candidate only has groups {sorted(candidate)}."
                )
            fracs = self._numeric_fractions(candidate[group], group)
            n_sites = sum(1 for sym in symbols if sym == site_symbol)
            if n_sites == 0:
                raise ValueError(
                    f"No sites with symbol {site_symbol!r} (site_map[{group!r}]) found "
                    f"in {self.specs[0].path}."
                )
            total = sum(fracs.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(
                    f"SubstituteBuilder: group {group!r} fractions sum to {total:.6f}, "
                    "not 1.0. A composition group must fill its whole sublattice."
                )
            counts = {el: c for el, c in _fractions_to_counts(fracs, n_sites).items() if c > 0}
            ops.append(SublatticeOp(sites=site_symbol, put=counts))

        return build_substituted_structure(template, ops, n_configs=n_configs, rng=rng)

    @staticmethod
    def _numeric_fractions(picks: Any, group: str) -> Dict[str, float]:
        """Validate one group's ``{element: fraction}`` mapping."""
        if not isinstance(picks, dict):
            raise TypeError(
                f"SubstituteBuilder expects group {group!r} to be a "
                f"{{element: fraction}} mapping, got {type(picks).__name__}."
            )
        out: Dict[str, float] = {}
        for el, frac in picks.items():
            try:
                out[str(el)] = float(frac)
            except (TypeError, ValueError):
                raise TypeError(
                    f"SubstituteBuilder: group {group!r} value {el!r}={frac!r} is not "
                    "numeric. A group that picks element LABELS (a categorical group) "
                    "needs the 'site_pick' builder instead."
                ) from None
        return out

    def composition_formula(self, candidate: Dict[str, Dict[str, float]]) -> Optional[str]:
        """Full composition of the built cell — agent picks plus untouched template atoms.

        Used for the ``formula`` column in generated.csv, per the
        ``StructureScorePredictor.composition_formula`` optional-hook contract.
        Returns ``None`` in flat single-sublattice mode, where the env's own
        terminal formula is already the right label.
        """
        if self.site_map is None:
            return None
        from ase.io import read as ase_read

        template = ase_read(self.specs[0].path)
        symbols = template.get_chemical_symbols()
        covered = set(self.site_map.values())

        counts: Dict[str, float] = {}
        for sym in symbols:
            if sym not in covered:
                counts[sym] = counts.get(sym, 0.0) + 1
        for group, site_symbol in self.site_map.items():
            picks = candidate.get(group)
            if not isinstance(picks, dict):
                continue  # let build() raise the real error; this is display-only
            n_sites = sum(1 for sym in symbols if sym == site_symbol)
            for el, c in _safe_counts(picks, n_sites).items():
                if c > 0:
                    counts[str(el)] = counts.get(str(el), 0.0) + c

        items = sorted(counts.items(), key=lambda t: (-t[1], t[0]))
        return "".join(f"{el}{n:.3g}" for el, n in items)


def _safe_counts(picks: Dict[str, Any], n_sites: int) -> Dict[str, int]:
    """Integer counts for display; never raises (formula labelling is not the gate)."""
    from ...utils.structure import _fractions_to_counts

    try:
        fracs = {str(k): float(v) for k, v in picks.items()}
        return _fractions_to_counts(fracs, n_sites)
    except Exception:  # noqa: BLE001 - display-only
        return {}
