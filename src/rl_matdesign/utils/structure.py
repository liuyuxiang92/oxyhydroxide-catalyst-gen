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
# Module-level (not lazy like the ASE imports elsewhere in this file): scipy is
# already a guaranteed transitive dependency (pymatgen requires it), and tests
# need to monkeypatch `rl_matdesign.utils.structure.Voronoi` directly to
# exercise the QhullError failure path -- a function-local `from scipy.spatial
# import ...` would not create a patchable module attribute.
from scipy.spatial import QhullError, Voronoi, cKDTree


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
    """One substitution/deletion/insertion operation on a base structure.

    Parameters
    ----------
    sites:
        Either a chemical symbol (``str``) — selects *all* atoms of that symbol —
        or an explicit sequence of integer site indices (an eligible region, e.g.
        "the last 1000 S atoms"). Unused (may be ``[]``) for an insert-only op.
    put:
        ``{species: count}`` — integer **atom counts** of replacement species to
        place on the selected sites (random assignment among them).
    remove:
        Number of selected sites to delete (vacancies).
    insert:
        ``{species: count}`` — integer atom counts of *new* atoms to add at
        random positions **not** tied to any existing site (interstitials).
        Independent of ``sites``/``put``/``remove`` — applied after them, against
        the post-put/remove structure, so it never collides with a substitution
        or vacancy from the same op list.
    min_dist:
        Minimum allowed distance (Å, minimum-image) between an inserted atom and
        any existing or previously-inserted atom in the same call. Candidate
        positions closer than this are resampled.
    max_attempts:
        Resample attempts per inserted atom before giving up.

    The remaining ``n_selected - sum(put.values()) - remove`` selected sites keep
    their original species. Requires ``sum(put.values()) + remove <= n_selected``.
    """

    sites: Union[str, Sequence[int]]
    put: Dict[str, int] = field(default_factory=dict)
    remove: int = 0
    insert: Dict[str, int] = field(default_factory=dict)
    min_dist: float = 1.5
    max_attempts: int = 200


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

        insert_specs = [
            (species, int(count)) for op in ops for species, count in op.insert.items()
        ]
        if insert_specs:
            atoms = _insert_interstitials(
                atoms, insert_specs, rng,
                min_dist=max((op.min_dist for op in ops if op.insert), default=1.5),
                max_attempts=max((op.max_attempts for op in ops if op.insert), default=200),
            )

        configs.append(atoms)

    return configs


def _atomic_radius(symbol: str) -> Optional[float]:
    """ASE/Cordero covalent radius (Å) for *symbol*, or ``None`` if unavailable.

    A cheap geometric size descriptor -- not an ionic radius, not a thermodynamic
    stability model -- used only to RANK interstitial void candidates that have
    already passed the hard ``interstitial_min_dist`` distance check (see
    :func:`_insert_interstitials`); atomic size never rejects a candidate on its
    own. Works automatically for any element ASE knows about via
    ``ase.data.atomic_numbers``/``covalent_radii`` -- there is no per-pair table
    to maintain, and this never raises for an exotic/unknown symbol: it returns
    ``None`` so callers can fall back to a species-agnostic ranking cleanly.
    """
    from ase.data import atomic_numbers, covalent_radii

    z = atomic_numbers.get(symbol)
    if z is None or z >= len(covalent_radii):
        return None
    r = float(covalent_radii[z])
    if not np.isfinite(r) or r <= 0:
        return None
    return r


def _candidate_free_radius(candidate: np.ndarray, atoms: "ase.Atoms") -> float:
    """``min_j`` of the PBC (minimum-image) distance from *candidate* to every
    atom currently in *atoms*. Species-agnostic; used as the tie-breaker under
    :func:`_species_clearance_score`, and as the sole ranking metric when no
    species radius data is available at all.
    """
    from ase.geometry import get_distances

    _, dists = get_distances(
        [candidate], atoms.get_positions(), cell=atoms.get_cell(), pbc=True
    )
    return float(dists.min())


def _species_clearance_score(
    insert_species: str, candidate: np.ndarray, atoms: "ase.Atoms",
) -> Optional[float]:
    """Normalized void-clearance score for inserting *insert_species* at
    *candidate*: ``min_j [ d_PBC(candidate, atom_j) / (r_insert + r_j) ]`` over
    neighbors ``j`` with a known radius. Larger is better -- more room relative
    to how big the inserted atom and its neighbor actually are, so e.g. a large
    species like Ba automatically needs proportionally more space than a small
    one, with zero species-pair-specific code.

    RANKING ONLY -- see :func:`_insert_interstitials`: the hard accept/reject
    rule is ``interstitial_min_dist`` alone, this score never rejects a
    candidate, only orders the ones that already passed that check.

    Returns ``None`` (signalling "no species-aware signal available", not "this
    candidate is invalid") in two cases: the inserted species' own radius is
    unknown (this disables species-aware ranking for every candidate in this
    insertion request, since ``r_insert`` is the same for all of them), or this
    specific candidate has no neighbor at all with a known radius. Callers fall
    back to :func:`_candidate_free_radius` in both cases.
    """
    from ase.geometry import get_distances

    r_insert = _atomic_radius(insert_species)
    if r_insert is None:
        return None

    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    _, dists = get_distances([candidate], positions, cell=atoms.get_cell(), pbc=True)

    ratios = []
    for d, sym in zip(dists[0], symbols):
        r_j = _atomic_radius(sym)
        if r_j is None:
            continue
        ratios.append(float(d) / (r_insert + r_j))
    if not ratios:
        return None
    return min(ratios)


def _voronoi_void_candidates(atoms: "ase.Atoms") -> List[np.ndarray]:
    """Candidate interstitial positions: local void centers of the periodic
    structure, as Cartesian coordinates wrapped into the central cell.

    Approach -- a "reasonable periodic-image construction", not a rigorous
    infinite-periodic Voronoi tessellation. Adequate for a roughly-cubic cell
    like ``perovskite.vasp``'s; not a universal guarantee for arbitrarily
    skewed cells (the ±1 image shell is not made configurable):

    1. Replicate the current atoms across the ±1 periodic-image shell (27
       images including the identity).
    2. Tessellate the expanded point cloud with ``scipy.spatial.Voronoi`` and
       take its vertices -- points locally equidistant from ≥4 neighbors, i.e.
       local void centers.
    3. Discard vertices near the outer hull of this *finite* replica: convert
       to fractional coordinates of the central cell and drop anything outside
       roughly ``[-0.5, 1.5)`` on any axis. A true infinite tessellation would
       push these into the next shell; keeping them would fold in spurious
       candidates once wrapped.
    4. Wrap the survivors into ``[0, 1)`` and back to Cartesian.
    5. Deduplicate, **periodically**: plain Euclidean distance on the wrapped
       positions alone is not enough, since two genuinely-close candidates can
       land near opposite cell faces (fractional 0.01 vs. 0.99) and read as far
       apart in ordinary Cartesian distance. Any candidate within a small
       margin of a cell face gets temporary "ghost" copies (shifted ±1 along
       the near axes) added before building a ``scipy.spatial.cKDTree``, so
       cross-boundary duplicates collide in the same search neighborhood;
       cluster on the padded set and keep one representative per cluster.
    6. Exclude any candidate numerically coincident with an existing atom.
    """
    from ase.geometry import get_distances

    positions = atoms.get_positions()
    if len(positions) < 4:
        raise ValueError(
            f"Cannot construct Voronoi void candidates: only {len(positions)} "
            "atom(s) present, need >= 4 non-coplanar points for a 3D tessellation."
        )
    cell = atoms.get_cell()

    shifts = [
        np.array([i, j, k], dtype=float)
        for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)
    ]
    expanded = np.concatenate(
        [positions + shift @ cell for shift in shifts], axis=0
    )

    try:
        vor = Voronoi(expanded)
    except QhullError as exc:
        raise ValueError(
            f"Could not construct Voronoi void candidates: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    verts = vor.vertices
    if len(verts) == 0:
        return []

    frac = cell.scaled_positions(verts)
    inside = np.all((frac >= -0.5) & (frac < 1.5), axis=1)
    frac = frac[inside]
    if len(frac) == 0:
        return []

    wrapped_frac = frac % 1.0
    wrapped_cart = cell.cartesian_positions(wrapped_frac)

    margin = 0.05
    ghost_cart: List[np.ndarray] = []
    ghost_owner: List[int] = []
    for idx, f in enumerate(wrapped_frac):
        axis_options = []
        for a in range(3):
            opts = [0]
            if f[a] < margin:
                opts.append(-1)
            elif f[a] > 1 - margin:
                opts.append(1)
            axis_options.append(opts)
        for dx in axis_options[0]:
            for dy in axis_options[1]:
                for dz in axis_options[2]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    ghost_frac = f + np.array([dx, dy, dz], dtype=float)
                    ghost_cart.append(cell.cartesian_positions(ghost_frac))
                    ghost_owner.append(idx)

    if ghost_cart:
        pool_cart = np.concatenate([wrapped_cart, np.array(ghost_cart)], axis=0)
        pool_owner = list(range(len(wrapped_cart))) + ghost_owner
    else:
        pool_cart = wrapped_cart
        pool_owner = list(range(len(wrapped_cart)))

    tol = 0.05  # Å -- "numerically identical" for both dedup and atom-coincidence checks
    tree = cKDTree(pool_cart)
    pairs = tree.query_pairs(r=tol)

    parent = list(range(len(wrapped_cart)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i, j in pairs:
        union(pool_owner[i], pool_owner[j])

    clusters: Dict[int, List[int]] = {}
    for idx in range(len(wrapped_cart)):
        clusters.setdefault(find(idx), []).append(idx)
    deduped = [wrapped_cart[members[0]] for members in clusters.values()]

    if deduped:
        cand_arr = np.array(deduped)
        _, dists_to_atoms = get_distances(cand_arr, positions, cell=cell, pbc=True)
        keep = dists_to_atoms.min(axis=1) > tol
        deduped = [c for c, k in zip(deduped, keep) if k]

    return deduped


def _insert_interstitials(
    atoms: "ase.Atoms",
    specs: Sequence[Any],
    rng: np.random.Generator,
    *,
    min_dist: float = 1.5,
    max_attempts: int = 200,
) -> "ase.Atoms":
    """Add new atoms at deterministically-chosen void positions, not tied to any
    existing site.

    Applied *after* any ``put``/``remove`` in the same :func:`build_substituted_structure`
    call, so an inserted atom never collides with a substitution/deletion from the same
    op list.

    Placement pipeline, per requested species (in the order given by *specs*;
    multiple atoms of the same species share one Voronoi candidate pool,
    generated once and consumed as atoms are placed):

    1. Generate periodic Voronoi void candidates (:func:`_voronoi_void_candidates`).
    2. HARD reject any candidate within ``min_dist`` (minimum-image PBC) of an
       existing atom -- this is the ONLY rejection rule. There is deliberately
       no second, species-aware distance threshold: atomic radii RANK the
       survivors (see :func:`_species_clearance_score`), they never reject one.
    3. Rank survivors by normalized species clearance, falling back to
       species-agnostic :func:`_candidate_free_radius` when radius data isn't
       available, then a deterministic rounded-fractional-coordinate tie-break
       -- so placement is fully reproducible for a given structure. No RNG is
       involved in the choice; ``rng``/``max_attempts`` are accepted for
       call-signature compatibility with :func:`build_substituted_structure`
       but unused here -- there is no more per-point resampling loop to bound.
    4. Place the single best (rank #1) candidate, remove it from the pool, and
       repeat for the next atom of this species against the *updated*
       structure -- so a second interstitial of the same species is screened
       and ranked against the first (its radius participates automatically,
       via its real chemical symbol, with no species-pair-specific code).
    """
    from ase import Atom

    for species, count in specs:
        pool = _voronoi_void_candidates(atoms)
        for _ in range(count):
            n_candidates = len(pool)
            if not pool:
                raise ValueError(
                    f"Could not place {species} interstitial: no Voronoi void "
                    "candidates remain for the current structure."
                )

            cell = atoms.get_cell()
            positions = atoms.get_positions()
            valid = []
            for cand in pool:
                free_r = _candidate_free_radius_from(cand, positions, cell)
                if free_r >= min_dist:
                    valid.append(cand)

            if not valid:
                raise ValueError(
                    f"Could not place {species} interstitial: {n_candidates} "
                    f"Voronoi candidates generated, 0 satisfy "
                    f"interstitial_min_dist={min_dist} Å."
                )

            scored = []
            for cand in valid:
                score = _species_clearance_score(species, cand, atoms)
                free_r = _candidate_free_radius(cand, atoms)
                tiebreak = tuple(np.round(cell.scaled_positions(cand), 6))
                scored.append((score is not None, score, free_r, tiebreak, cand))
            scored.sort(
                key=lambda t: (
                    t[0], t[1] if t[1] is not None else float("-inf"), t[2], t[3],
                ),
                reverse=True,
            )

            has_score, best_score, best_free_r, _tb, best_cand = scored[0]
            score_str = f"{best_score:.3f}" if has_score else "n/a"
            print(
                f"[interstitial] species={species} candidates={n_candidates} "
                f"valid={len(valid)} selected_rank=1 "
                f"free_radius={best_free_r:.2f} Å species_score={score_str}"
            )

            atoms.append(Atom(species, position=best_cand))
            pool = [c for c in pool if not np.allclose(c, best_cand)]
    return atoms


def _candidate_free_radius_from(
    candidate: np.ndarray, positions: np.ndarray, cell: "ase.cell.Cell",
) -> float:
    """Same computation as :func:`_candidate_free_radius`, taking already-fetched
    ``positions``/``cell`` -- used by :func:`_insert_interstitials`'s hard-filter
    pass, which evaluates every pool candidate once per placed atom and would
    otherwise re-fetch ``atoms.get_positions()`` from scratch each time.
    """
    from ase.geometry import get_distances

    _, dists = get_distances([candidate], positions, cell=cell, pbc=True)
    return float(dists.min())


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
        # LBFGS.run() returns a bool (converged or not) -- running out of `steps`
        # without reaching fmax is NOT an exception, so the except branch alone
        # was silently swallowing that case (the caller got an unconverged
        # structure back with no signal at all that it hadn't converged).
        converged = bool(opt.run(fmax=fmax, steps=steps))
        if not converged:
            final_fmax = float((work.get_forces() ** 2).sum(axis=1).max() ** 0.5)
            print(
                f"relax_structure: did NOT converge to fmax={fmax} within {steps} "
                f"steps (reached fmax={final_fmax:.4f})."
            )
    except Exception as exc:  # noqa: BLE001 - relaxation failures shouldn't crash the run
        print(f"relax_structure: optimization raised and did not complete: {exc}")

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
