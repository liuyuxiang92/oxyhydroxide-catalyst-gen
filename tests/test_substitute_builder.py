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
