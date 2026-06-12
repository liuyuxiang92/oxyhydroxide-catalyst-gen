"""Structure substitution + relaxation utilities.

Two layers live here:

* **Layer 1 — substitution engine.** :func:`build_substituted_structure` edits a
  base structure to match target *integer atom counts* per sublattice: it
  substitutes species onto, and optionally deletes (vacancies) from, selected
  sites. A "sublattice operation" (:class:`SublatticeOp`) selects sites either by
  chemical symbol or by an explicit index region, places ``{species: count}``
  replacements, and removes ``remove`` sites. The remaining selected sites keep
  their original species. This is the general "count-diff vs the base POSCAR"
  builder used by multi-sublattice scenarios (e.g. doped Li6PS6).

  :func:`substitute_sites` is a thin one-operation wrapper over the engine,
  preserving the original Random-Solid-Solution behavior (fill all placeholder
  sites by fractional composition); it backs the ``substitute`` builder used by
  the HEA / perovskite / ti_alloy ``structure_score`` configs.

* **Relaxation.** :func:`relax_structure` is the shared geometry-optimization
  capability (LBFGS + cell filter + a DeepMD calculator with a named head),
  lifted out of the OOH predictor so any structure-based predictor can reuse it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np


# ---------------------------------------------------------------------------
# Layer 1: substitution engine
# ---------------------------------------------------------------------------

def resolve_region(
    atoms_or_symbols: Union["ase.Atoms", Sequence[str]],
    spec: Union[str, Dict[str, Any]],
) -> List[int]:
    """Resolve a declarative site-selection *spec* to concrete atom indices.

    A general, scenario-agnostic selector so eligible regions (e.g. "the last
    1000 S sites") live in config rather than hardcoded Python.

    Supported specs::

        "S"                                       # all S sites
        {"symbol": "S"}                           # all S sites
        {"symbol": "S", "take": "last", "count": 1000}   # last 1000 S (by index)
        {"symbol": "S", "take": "first", "count": 2000}
        {"symbol": "S", "index_range": [3500, 4500]}     # S sites with global index in [a, b)
        {"indices": [...]}                        # explicit global indices

    ``take``/``index_range`` operate on the symbol-matched indices in ascending
    global-index order, so they are robust to where the element block sits in
    the POSCAR.
    """
    symbols = (
        list(atoms_or_symbols.get_chemical_symbols())
        if hasattr(atoms_or_symbols, "get_chemical_symbols")
        else list(atoms_or_symbols)
    )

    if isinstance(spec, str):
        spec = {"symbol": spec}
    if not isinstance(spec, dict):
        raise TypeError(f"region spec must be a str or dict, got {type(spec).__name__}")

    if "indices" in spec:
        return [int(i) for i in spec["indices"]]

    symbol = spec.get("symbol")
    if symbol is None:
        raise ValueError(f"region spec needs 'symbol' or 'indices': {spec!r}")
    matched = [i for i, s in enumerate(symbols) if s == symbol]

    if "index_range" in spec:
        lo, hi = spec["index_range"]
        return [i for i in matched if lo <= i < hi]

    take = spec.get("take")
    if take is None:
        return matched
    count = int(spec["count"])
    if take == "first":
        return matched[:count]
    if take == "last":
        return matched[-count:]
    raise ValueError(f"region spec 'take' must be 'first' or 'last', got {take!r}")


@dataclass
class SublatticeOp:
    """One substitution/deletion operation on a base structure.

    Parameters
    ----------
    sites:
        Either a chemical symbol (``str``) — selects *all* atoms of that symbol —
        or an explicit sequence of integer site indices (an eligible region, e.g.
        "the last 1000 S atoms").
    put:
        ``{species: count}`` — integer **atom counts** of replacement species to
        place on the selected sites (random assignment among them).
    remove:
        Number of selected sites to delete (vacancies).

    The remaining ``n_selected - sum(put.values()) - remove`` selected sites keep
    their original species. Requires ``sum(put.values()) + remove <= n_selected``.
    """

    sites: Union[str, Sequence[int]]
    put: Dict[str, int] = field(default_factory=dict)
    remove: int = 0


def _resolve_sites(symbols: Sequence[str], sites: Union[str, Sequence[int]]) -> List[int]:
    if isinstance(sites, str):
        return [i for i, s in enumerate(symbols) if s == sites]
    return [int(i) for i in sites]


def build_substituted_structure(
    base: Union[str, "ase.Atoms"],
    ops: Sequence[SublatticeOp],
    n_configs: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> List["ase.Atoms"]:
    """Build *n_configs* random structures by applying *ops* to *base*.

    Parameters
    ----------
    base:
        A POSCAR/CONTCAR path *or* an ASE ``Atoms`` object (used as a template;
        copied per config).
    ops:
        Sublattice operations to apply. All site selections are resolved against
        the **original** symbols, so the order of ``ops`` does not matter and ops
        on different sublattices never interfere.
    n_configs:
        Number of random realizations to generate.
    rng:
        NumPy generator for reproducibility.

    Returns
    -------
    List of ``n_configs`` ASE ``Atoms`` objects (with deletions applied).
    """
    try:
        from ase.io import read as ase_read
    except ImportError as exc:  # pragma: no cover - exercised only without ASE
        raise ImportError("build_substituted_structure requires ASE: pip install ase") from exc

    if rng is None:
        rng = np.random.default_rng()

    template = ase_read(base) if isinstance(base, str) else base

    configs: List["ase.Atoms"] = []
    for _ in range(n_configs):
        atoms = template.copy()
        orig_symbols = list(atoms.get_chemical_symbols())
        symbols = list(orig_symbols)
        delete_indices: List[int] = []

        for op in ops:
            sel = _resolve_sites(orig_symbols, op.sites)
            n_sel = len(sel)
            n_put = sum(int(c) for c in op.put.values())
            n_rm = int(op.remove)
            if n_put + n_rm > n_sel:
                sel_repr = op.sites if not isinstance(op.sites, (list, tuple)) else (
                    f"<{n_sel} sites: {list(op.sites[:6])}…>" if n_sel > 6 else list(op.sites)
                )
                raise ValueError(
                    f"SublatticeOp wants to place {n_put} + delete {n_rm} on only "
                    f"{n_sel} selected sites (selector={sel_repr}). The eligible "
                    "region is too small for this composition — enlarge it (for the "
                    "SSE builder, set a bigger `eligible_region` or an "
                    "`eligible_region_fallback` that holds all of O+Cl+Br)."
                )
            order = [int(i) for i in rng.permutation(sel)]
            cursor = 0
            for species, count in op.put.items():
                for _ in range(int(count)):
                    symbols[order[cursor]] = species
                    cursor += 1
            for _ in range(n_rm):
                delete_indices.append(order[cursor])
                cursor += 1

        atoms.set_chemical_symbols(symbols)
        if delete_indices:
            drop = set(delete_indices)
            keep = [i for i in range(len(atoms)) if i not in drop]
            atoms = atoms[keep]
        configs.append(atoms)

    return configs


# ---------------------------------------------------------------------------
# Backward-compatible one-operation wrapper (Random Solid Solution)
# ---------------------------------------------------------------------------

def substitute_sites(
    template_poscar: str,
    composition: Dict[str, float],
    site_symbol: str = "X",
    n_configs: int = 5,
    rng: Optional[np.random.Generator] = None,
) -> List["ase.Atoms"]:
    """Fill all ``site_symbol`` sites of a template per fractional *composition*.

    Thin wrapper over :func:`build_substituted_structure` preserving the original
    HEA/perovskite/ti_alloy behavior: one sublattice, no vacancies, integer counts
    allocated from fractions (largest-remainder).
    """
    try:
        from ase.io import read as ase_read
    except ImportError as exc:  # pragma: no cover
        raise ImportError("substitute_sites requires ASE: pip install ase") from exc

    if rng is None:
        rng = np.random.default_rng()

    template = ase_read(template_poscar)
    site_indices = [
        i for i, s in enumerate(template.get_chemical_symbols()) if s == site_symbol
    ]
    if not site_indices:
        raise ValueError(
            f"No sites with symbol '{site_symbol}' found in {template_poscar}. "
            "Check that site_symbol matches the placeholder element in the template."
        )

    counts = _fractions_to_counts(composition, len(site_indices))
    op = SublatticeOp(sites=site_symbol, put=counts, remove=0)
    return build_substituted_structure(template, [op], n_configs=n_configs, rng=rng)


def _fractions_to_counts(composition: Dict[str, float], n_sites: int) -> Dict[str, int]:
    """Allocate integer atom counts from fractional composition.

    Uses a largest-remainder method to guarantee sum == n_sites exactly.
    """
    elements = list(composition.keys())
    fracs = [composition[e] for e in elements]

    raw = [f * n_sites for f in fracs]
    floors = [int(x) for x in raw]
    remainders = [(raw[i] - floors[i], i) for i in range(len(floors))]

    deficit = n_sites - sum(floors)
    if deficit < 0 or deficit > len(elements):
        raise ValueError(f"Cannot allocate {n_sites} sites from fractions {composition}.")

    remainders.sort(reverse=True)
    counts_list = floors[:]
    for k in range(deficit):
        counts_list[remainders[k][1]] += 1

    return {elements[i]: counts_list[i] for i in range(len(elements))}


# ---------------------------------------------------------------------------
# Shared relaxation capability
# ---------------------------------------------------------------------------

def relax_structure(
    atoms: "ase.Atoms",
    *,
    model: str = "models/DPA-3.1-3M.pt",
    head: Optional[str] = None,
    calc: Optional[Any] = None,
    fmax: float = 0.001,
    steps: int = 1000,
    relax_cell: bool = True,
    mask_indices: Optional[Sequence[int]] = None,
) -> "ase.Atoms":
    """Geometry-optimize *atoms* with a DeepMD calculator.

    A first-class, shared capability (formerly buried in the OOH predictor). Any
    structure-based predictor can call it.

    Parameters
    ----------
    model:
        DeepMD checkpoint path. Defaults to ``DPA-3.1-3M.pt``.
    head:
        Output head of a multi-task checkpoint (e.g. ``"SSE_ABACUS"``). The model
        has a default; **the head is the caller's responsibility**.
    calc:
        A pre-built ASE calculator to reuse (avoids reloading the model every
        call). If given, ``model``/``head`` are ignored.
    fmax, steps, relax_cell:
        LBFGS force threshold, max steps, and whether to relax the cell
        (``UnitCellFilter``).
    mask_indices:
        Optional site indices to *exclude* from optimization (e.g. OOH adsorbate
        placeholders); their positions are restored unchanged in the result.
    """
    from ase.optimize import LBFGS

    try:
        from ase.filters import UnitCellFilter  # ASE >= 3.23
    except ImportError:  # pragma: no cover
        from ase.constraints import UnitCellFilter

    keep_indices: Optional[List[int]] = None
    if mask_indices:
        drop = {int(i) for i in mask_indices}
        keep_indices = [i for i in range(len(atoms)) if i not in drop]
        work = atoms[keep_indices].copy()
    else:
        work = atoms.copy()

    if calc is None:
        from deepmd.calculator import DP as DPCalculator
        kwargs = {"head": head} if head else {}
        calc = DPCalculator(model=model, **kwargs)
    work.calc = calc

    target = UnitCellFilter(work, scalar_pressure=0.0) if relax_cell else work
    opt = LBFGS(target)
    try:
        opt.run(fmax=fmax, steps=steps)
    except Exception as exc:  # noqa: BLE001 - relaxation failures shouldn't crash the run
        print(f"relax_structure: optimization did not converge: {exc}")

    if keep_indices is None:
        return work

    # Rebuild the full structure: relaxed cell + relaxed positions for kept atoms,
    # original positions for masked atoms.
    result = atoms.copy()
    result.set_cell(work.get_cell())
    positions = result.get_positions().copy()
    work_positions = work.get_positions()
    for new_i, orig_i in enumerate(keep_indices):
        positions[orig_i] = work_positions[new_i]
    result.set_positions(positions)
    return result
