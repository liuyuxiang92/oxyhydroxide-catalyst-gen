"""SubstituteBuilder tests — the fixed-lattice element-swap builder.

Verifies the builder is registry-resolvable and produces the same structures as
the underlying ``substitute_sites`` (the behavior dp_structure/dp_property used
inline before it was promoted to a reusable builder).
"""
import os
import sys
from collections import Counter

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rl_matdesign.predictors.builders.substitute import SubstituteBuilder  # noqa: E402


def _template(tmp_path, n_sites=10):
    from ase import Atoms
    from ase.io import write

    # n_sites placeholder "X" atoms on a line + a couple spectators.
    syms = ["X"] * n_sites + ["O", "O"]
    atoms = Atoms(syms, positions=[(i, 0, 0) for i in range(len(syms))],
                  cell=[40, 40, 40], pbc=True)
    path = str(tmp_path / "tmpl.vasp")
    write(path, atoms, format="vasp")
    return path


def test_registry_resolves_substitute_builder(tmp_path):
    from rl_matdesign.registry import resolve_builder, BUILDERS
    assert "substitute" in BUILDERS
    b = resolve_builder("substitute", {"base_poscar": _template(tmp_path), "site_symbol": "X"})
    assert hasattr(b, "build")


def test_missing_poscar_raises():
    from rl_matdesign.predictors.builders.substitute import SubstituteBuilder
    with pytest.raises(ValueError) as info:
        SubstituteBuilder({"site_symbol": "X"})
    assert "POSCAR" in str(info.value) or "base_poscar" in str(info.value)


def test_build_matches_substitute_sites(tmp_path):
    """Builder output is identical to calling substitute_sites directly."""
    from rl_matdesign.predictors.builders.substitute import SubstituteBuilder
    from rl_matdesign.utils.structure import substitute_sites

    poscar = _template(tmp_path, n_sites=10)
    comp = {"Fe": 0.6, "Ni": 0.4}

    b = SubstituteBuilder({"base_poscar": poscar, "site_symbol": "X"})
    built = b.build(comp, n_configs=3, rng=np.random.default_rng(0))
    direct = substitute_sites(template_poscar=poscar, composition=comp,
                              site_symbol="X", n_configs=3, rng=np.random.default_rng(0))

    assert len(built) == len(direct) == 3
    for s_b, s_d in zip(built, direct):
        assert s_b.get_chemical_symbols() == s_d.get_chemical_symbols()


def test_build_places_correct_counts(tmp_path):
    """A 0.6/0.4 split over 10 sites → 6 Fe / 4 Ni, spectators untouched."""
    from rl_matdesign.predictors.builders.substitute import SubstituteBuilder

    b = SubstituteBuilder({"base_poscar": _template(tmp_path, n_sites=10), "site_symbol": "X"})
    for st in b.build({"Fe": 0.6, "Ni": 0.4}, n_configs=3, rng=np.random.default_rng(1)):
        cnt = Counter(st.get_chemical_symbols())
        assert cnt["Fe"] == 6 and cnt["Ni"] == 4
        assert cnt["O"] == 2 and "X" not in cnt


# --------------------------------------------------------------------------- #
# site_map mode: fractional occupancy across SEVERAL sublattices.
#
# `substitute` was fractional-but-one-sublattice and `site_pick` was
# multi-sublattice-but-one-element-per-site; this fills the missing quadrant
# without adding a fifth builder.
# --------------------------------------------------------------------------- #

def _four_sublattice_template(tmp_path):
    """A tiny stand-in for a double perovskite: 8 A, 4 B1, 4 B3, 8 X sites."""
    from ase import Atoms

    symbols = ["Cs"] * 8 + ["Ag"] * 4 + ["Bi"] * 4 + ["Cl"] * 8
    atoms = Atoms(symbols,
                  positions=[(i, 0, 0) for i in range(len(symbols))],
                  cell=[len(symbols) + 5, 10, 10], pbc=True)
    path = tmp_path / "template.vasp"
    atoms.write(str(path), format="vasp")
    return str(path)


def _site_map_cfg(tmp_path):
    return {"base_poscar": _four_sublattice_template(tmp_path),
            "site_map": {"A": "Cs", "B1": "Ag", "B3": "Bi", "X": "Cl"}}


def test_no_site_map_keeps_flat_behaviour(tmp_path):
    # Back-compat gate: without site_map nothing about the old path changes.
    from ase import Atoms
    atoms = Atoms(["X"] * 4, positions=[(i, 0, 0) for i in range(4)],
                  cell=[10, 10, 10], pbc=True)
    path = tmp_path / "flat.vasp"
    atoms.write(str(path), format="vasp")
    b = SubstituteBuilder({"base_poscar": str(path), "site_symbol": "X"}, seed=0)
    assert b.site_map is None
    cells = b.build({"Fe": 0.5, "Ni": 0.5}, n_configs=2)
    assert len(cells) == 2
    assert Counter(cells[0].get_chemical_symbols()) == Counter({"Fe": 2, "Ni": 2})
    assert b.composition_formula({"Fe": 0.5}) is None   # env's own label is used


def test_site_map_fills_every_sublattice_with_integer_counts(tmp_path):
    b = SubstituteBuilder(_site_map_cfg(tmp_path), seed=0)
    cand = {"A":  {"Cs": 0.75, "Rb": 0.25},
            "B1": {"Ag": 0.50, "Cu": 0.25, "Na": 0.25},
            "B3": {"Bi": 0.75, "Sb": 0.25},
            "X":  {"Cl": 0.50, "Br": 0.25, "I": 0.25}}
    cells = b.build(cand, n_configs=3, rng=np.random.default_rng(0))
    assert len(cells) == 3
    expected = Counter({"Cs": 6, "Rb": 2, "Ag": 2, "Cu": 1, "Na": 1,
                        "Bi": 3, "Sb": 1, "Cl": 4, "Br": 2, "I": 2})
    for cell in cells:
        assert Counter(cell.get_chemical_symbols()) == expected
        assert len(cell) == 24


def test_site_map_n_configs_same_counts_different_assignment(tmp_path):
    b = SubstituteBuilder(_site_map_cfg(tmp_path), seed=0)
    cand = {"A": {"Cs": 0.5, "Rb": 0.5}, "B1": {"Ag": 1.0},
            "B3": {"Bi": 1.0}, "X": {"Cl": 0.5, "Br": 0.5}}
    cells = b.build(cand, n_configs=4, rng=np.random.default_rng(1))
    counts = {tuple(sorted(Counter(c.get_chemical_symbols()).items())) for c in cells}
    assert len(counts) == 1                       # composition is fixed
    seqs = {tuple(c.get_chemical_symbols()) for c in cells}
    assert len(seqs) > 1                          # decorations genuinely differ


def test_site_map_missing_group_raises(tmp_path):
    b = SubstituteBuilder(_site_map_cfg(tmp_path), seed=0)
    with pytest.raises(KeyError) as info:
        b.build({"A": {"Cs": 1.0}}, n_configs=1)
    assert "B1" in str(info.value)


def test_site_map_fractions_must_sum_to_one(tmp_path):
    b = SubstituteBuilder(_site_map_cfg(tmp_path), seed=0)
    cand = {"A": {"Cs": 0.5}, "B1": {"Ag": 1.0}, "B3": {"Bi": 1.0}, "X": {"Cl": 1.0}}
    with pytest.raises(ValueError) as info:
        b.build(cand, n_configs=1)
    assert "sum to" in str(info.value)


def test_duplicate_mapped_symbols_raise(tmp_path):
    # Two groups on one symbol would silently clobber: ops resolve against the
    # ORIGINAL symbols, so both would target the same sites.
    with pytest.raises(ValueError) as info:
        SubstituteBuilder({"base_poscar": _four_sublattice_template(tmp_path),
                           "site_map": {"A": "Cs", "B1": "Cs"}}, seed=0)
    assert "Cs" in str(info.value)


def test_categorical_values_point_at_site_pick(tmp_path):
    b = SubstituteBuilder(_site_map_cfg(tmp_path), seed=0)
    cand = {"A": {"A": "Cs"}, "B1": {"Ag": 1.0}, "B3": {"Bi": 1.0}, "X": {"Cl": 1.0}}
    with pytest.raises(TypeError) as info:
        b.build(cand, n_configs=1)
    assert "site_pick" in str(info.value)


def test_site_map_composition_formula_counts_untouched_atoms(tmp_path):
    b = SubstituteBuilder(_site_map_cfg(tmp_path), seed=0)
    cand = {"A": {"Cs": 1.0}, "B1": {"Ag": 1.0}, "B3": {"Bi": 1.0},
            "X": {"Cl": 0.5, "Br": 0.5}}
    formula = b.composition_formula(cand)
    for frag in ("Cs8", "Ag4", "Bi4", "Cl4", "Br4"):
        assert frag in formula
