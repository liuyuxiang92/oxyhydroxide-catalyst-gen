"""Tests for the periodic-Voronoi interstitial placement in ``utils/structure.py``.

``_insert_interstitials`` picks *where* to put a new (not-site-tied) atom:
generate periodic Voronoi void candidates, hard-reject any candidate closer
than ``interstitial_min_dist`` to an existing atom (the ONLY accept/reject
rule), then rank the survivors by a dimensionless normalized species-size
clearance and deterministically place the single best one. Atomic radii are
looked up generically via ``ase.data`` for any element -- there is no
Ba/Fe/O-specific table anywhere.

Uses a genuinely 3D, non-collinear toy lattice (unlike the other builder
tests' ``positions=[(i, 0, 0) for i in ...]`` templates, which are degenerate
for Voronoi/Delaunay construction).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ase import Atoms  # noqa: E402
from ase.geometry import get_distances  # noqa: E402

import rl_matdesign.utils.structure as structure_mod  # noqa: E402
from rl_matdesign.utils.structure import (  # noqa: E402
    _atomic_radius,
    _candidate_free_radius,
    _candidate_free_radius_from,
    _insert_interstitials,
    _species_clearance_score,
    _voronoi_void_candidates,
)


def _rocksalt_lattice(n: int = 3, a: float = 2.8) -> Atoms:
    """A small, genuinely-3D toy lattice: alternating Na/Cl on a simple cubic
    grid. Non-degenerate for Voronoi/Delaunay construction, unlike the
    collinear ``positions=[(i, 0, 0) for i in ...]`` templates the other
    builder test files use (fine for their purposes -- pure substitution
    counting -- but unusable here)."""
    positions, symbols = [], []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                positions.append([i * a, j * a, k * a])
                symbols.append("Na" if (i + j + k) % 2 == 0 else "Cl")
    return Atoms(symbols, positions=positions, cell=[n * a] * 3, pbc=True)


def _tetrahedron_vertices(circumradius: float) -> list:
    """4 points of a regular tetrahedron, each exactly *circumradius* from the
    origin (the base unit tetrahedron [(1,1,1),(1,-1,-1),(-1,1,-1),(-1,-1,1)]
    has circumradius sqrt(3))."""
    base = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
    return (base * (circumradius / np.sqrt(3.0))).tolist()


# --------------------------------------------------------------------------- #
# A. Automatic radius lookup
# --------------------------------------------------------------------------- #

def test_atomic_radius_generic_for_many_elements():
    from ase.data import atomic_numbers, covalent_radii

    for el in ["Ba", "Ca", "Sr", "Fe", "Mn", "O", "Al", "Ti"]:
        expected = float(covalent_radii[atomic_numbers[el]])
        assert _atomic_radius(el) == pytest.approx(expected)


def test_atomic_radius_none_for_unknown_symbol():
    assert _atomic_radius("Xx") is None
    assert _atomic_radius("NotAnElement") is None


# --------------------------------------------------------------------------- #
# B. Species-dependent ranking
# --------------------------------------------------------------------------- #

def test_species_score_depends_on_inserted_species():
    """Same candidate, same neighbors -- only the inserted species' radius
    changes -- must give a different (and correctly-ordered) score."""
    atoms = _rocksalt_lattice()
    cand = _voronoi_void_candidates(atoms)[0]

    score_small = _species_clearance_score("O", cand, atoms)   # r_O = 0.66 A
    score_large = _species_clearance_score("Ba", cand, atoms)  # r_Ba = 2.15 A
    assert score_small is not None and score_large is not None
    # d_ij / (r_insert + r_j): larger r_insert -> larger denominator -> smaller score
    assert score_small > score_large


# --------------------------------------------------------------------------- #
# C. Hard global floor
# --------------------------------------------------------------------------- #

def test_hard_min_dist_rejects_regardless_of_species():
    atoms = _rocksalt_lattice()
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    # A point almost on top of an existing atom -- must read as clearly closer
    # than any realistic interstitial_min_dist, independent of what would be
    # inserted there.
    close_candidate = positions[0] + np.array([0.3, 0.0, 0.0])
    free_r = _candidate_free_radius_from(close_candidate, positions, cell)
    assert free_r < 1.7


def test_insert_interstitials_result_always_respects_min_dist():
    atoms = _rocksalt_lattice()
    min_dist = 1.7
    out = _insert_interstitials(
        atoms.copy(), [("Ba", 1)], np.random.default_rng(0),
        min_dist=min_dist, max_attempts=2000,
    )
    new_pos = out.get_positions()[-1]
    _, dists = get_distances(
        [new_pos], atoms.get_positions(), cell=atoms.get_cell(), pbc=True
    )
    assert dists.min() >= min_dist - 1e-9


# --------------------------------------------------------------------------- #
# D. Radius is ranking-only, NOT a second hard filter (v2 regression guard)
# --------------------------------------------------------------------------- #

def test_radius_ranking_does_not_hard_reject():
    """A void closer than r_insert + r_neighbor (would have failed the old
    max(min_dist, r_i + r_j) hard rule from an earlier design) must still be
    scored and placeable, as long as it satisfies the flat interstitial_min_dist
    floor -- atomic size only ever RANKS candidates, never rejects one."""
    d = 1.9
    positions = _tetrahedron_vertices(circumradius=d)
    atoms = Atoms(["O"] * 4, positions=positions, cell=[20, 20, 20], pbc=True)
    atoms.translate([10, 10, 10])
    void_center = np.array([10.0, 10.0, 10.0])

    free_r = _candidate_free_radius(void_center, atoms)
    assert free_r == pytest.approx(d, abs=1e-6)

    score = _species_clearance_score("Ba", void_center, atoms)  # r_Ba+r_O = 2.81 A
    assert score is not None
    assert score < 1.0  # d(1.9) < r_i+r_j(2.81) -- the old hard rule would reject this

    # The real pipeline, with a loose min_dist well below d, must still place it.
    out = _insert_interstitials(
        atoms.copy(), [("Ba", 1)], np.random.default_rng(0),
        min_dist=1.5, max_attempts=2000,
    )
    assert len(out) == 5


# --------------------------------------------------------------------------- #
# E. Radius fallback when data is unavailable
# --------------------------------------------------------------------------- #

def test_radius_fallback_when_unavailable(monkeypatch):
    atoms = _rocksalt_lattice()
    cand = _voronoi_void_candidates(atoms)[0]

    monkeypatch.setattr(structure_mod, "_atomic_radius", lambda symbol: None)
    assert structure_mod._species_clearance_score("Ba", cand, atoms) is None

    # End-to-end: placement still succeeds, falling back to free-radius ranking.
    out = structure_mod._insert_interstitials(
        atoms.copy(), [("Ba", 1)], np.random.default_rng(0),
        min_dist=1.5, max_attempts=2000,
    )
    assert len(out) == len(atoms) + 1


def test_radius_fallback_for_unknown_neighbor_only(monkeypatch):
    """Inserted species has a valid radius, but every neighbor's radius is
    unavailable -- this specific candidate falls back to None (no crash).
    ASE's own covalent_radii table is fully populated for every real element
    it can construct an Atoms object with, so simulate "unavailable" the same
    way test E does: monkeypatch `_atomic_radius` for the neighbor symbol only."""
    atoms = Atoms(["H"] * 4, positions=_tetrahedron_vertices(2.0),
                   cell=[20, 20, 20], pbc=True)
    atoms.translate([10, 10, 10])
    cand = np.array([10.0, 10.0, 10.0])

    real_radius = structure_mod._atomic_radius

    def _patched(symbol):
        return None if symbol == "H" else real_radius(symbol)

    monkeypatch.setattr(structure_mod, "_atomic_radius", _patched)
    assert structure_mod._atomic_radius("Ba") is not None
    assert structure_mod._atomic_radius("H") is None
    assert structure_mod._species_clearance_score("Ba", cand, atoms) is None
    assert _candidate_free_radius(cand, atoms) == pytest.approx(2.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# F. Deterministic best-site selection
# --------------------------------------------------------------------------- #

def test_deterministic_selection_same_input_same_output():
    atoms = _rocksalt_lattice()
    out1 = _insert_interstitials(
        atoms.copy(), [("Ba", 1)], np.random.default_rng(1),
        min_dist=1.7, max_attempts=2000,
    )
    out2 = _insert_interstitials(
        atoms.copy(), [("Ba", 1)], np.random.default_rng(999),
        min_dist=1.7, max_attempts=2000,
    )
    assert np.allclose(out1.get_positions()[-1], out2.get_positions()[-1])


# --------------------------------------------------------------------------- #
# G. Multiple interstitials
# --------------------------------------------------------------------------- #

def test_multiple_interstitials_sequential_screening():
    atoms = _rocksalt_lattice()
    min_dist = 1.7
    out = _insert_interstitials(
        atoms.copy(), [("Ba", 2)], np.random.default_rng(0),
        min_dist=min_dist, max_attempts=2000,
    )
    assert len(out) == len(atoms) + 2
    pos1, pos2 = out.get_positions()[-2], out.get_positions()[-1]
    assert not np.allclose(pos1, pos2)
    _, d = get_distances([pos1], [pos2], cell=out.get_cell(), pbc=True)
    assert d[0, 0] >= min_dist - 1e-9  # 2nd Ba screened against the 1st


# --------------------------------------------------------------------------- #
# H. PBC-aware duplicate handling
# --------------------------------------------------------------------------- #

def test_pbc_aware_dedup_no_near_duplicates_in_output():
    """A plain (non-periodic) dedup would miss two candidates that are close
    through a periodic boundary (e.g. fractional 0.01 vs 0.99); the real
    minimum-image distance between any two returned candidates must exceed
    the internal dedup tolerance."""
    atoms = _rocksalt_lattice()
    cands = np.array(_voronoi_void_candidates(atoms))
    assert len(cands) > 1
    cell = atoms.get_cell()
    _, dmat = get_distances(cands, cands, cell=cell, pbc=True)
    np.fill_diagonal(dmat, np.inf)
    assert dmat.min() > 0.05


# --------------------------------------------------------------------------- #
# I. PBC distance filtering (min_dist check itself is periodic)
# --------------------------------------------------------------------------- #

def test_min_dist_filter_is_pbc_aware():
    """A candidate that reads as far away in ordinary (non-periodic) Cartesian
    distance but is close through the periodic boundary must be caught by the
    minimum-image distance check."""
    cell = [10.0, 10.0, 10.0]
    positions = [[0.2, 5, 5], [5, 0.2, 5], [5, 9.7, 5], [9.7, 5, 5]]
    atoms = Atoms(["Fe"] * 4, positions=positions, cell=cell, pbc=True)
    candidate = np.array([9.9, 5.0, 5.0])

    naive_dist = float(np.linalg.norm(candidate - np.array(positions[0])))
    assert naive_dist > 9.0  # looks far in plain Cartesian terms

    free_r = _candidate_free_radius_from(candidate, atoms.get_positions(), atoms.get_cell())
    assert free_r < 0.5  # true PBC distance is actually small


# --------------------------------------------------------------------------- #
# J. Failure paths
# --------------------------------------------------------------------------- #

def test_failure_too_few_atoms():
    atoms = Atoms(["Fe", "Fe"], positions=[[0, 0, 0], [2, 0, 0]],
                   cell=[20, 20, 20], pbc=True)
    with pytest.raises(ValueError, match="need >= 4"):
        _voronoi_void_candidates(atoms)


def test_failure_qhull_error(monkeypatch):
    def _raise(*_args, **_kwargs):
        raise structure_mod.QhullError("degenerate point set")

    monkeypatch.setattr(structure_mod, "Voronoi", _raise)
    atoms = _rocksalt_lattice()
    with pytest.raises(ValueError, match="Could not construct Voronoi"):
        structure_mod._voronoi_void_candidates(atoms)


def test_failure_no_candidates_satisfy_min_dist():
    atoms = _rocksalt_lattice()
    with pytest.raises(ValueError, match="0 satisfy"):
        _insert_interstitials(
            atoms.copy(), [("Ba", 1)], np.random.default_rng(0),
            min_dist=100.0, max_attempts=2000,
        )


def test_failure_no_generated_voronoi_sites(monkeypatch):
    """Voronoi construction succeeds but yields no usable vertices at all."""
    class _EmptyVoronoi:
        def __init__(self, *_args, **_kwargs):
            self.vertices = np.zeros((0, 3))

    monkeypatch.setattr(structure_mod, "Voronoi", _EmptyVoronoi)
    atoms = _rocksalt_lattice()
    assert structure_mod._voronoi_void_candidates(atoms) == []
    with pytest.raises(ValueError, match="no Voronoi void candidates"):
        structure_mod._insert_interstitials(
            atoms.copy(), [("Ba", 1)], np.random.default_rng(0),
            min_dist=1.7, max_attempts=2000,
        )
