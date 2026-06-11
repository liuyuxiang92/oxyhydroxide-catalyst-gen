"""Tests for the SSE (doped Li6PS6) builder recipe — chemistry + structure assembly.

The S-site is now a categorical group: O is a form flag (0 = metal form, 1 = oxide)
and Cl is a real per-formula-unit count. No cl_map / selectors. DeepMD eval /
geo-opt are GPU-gated and not exercised here.
"""
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rl_matdesign.predictors.builders.sse import SSESupercellBuilder  # noqa: E402

VALENCES = {
    "Li": 1, "P": 5, "S": -2, "O": -2, "Cl": -1, "Br": -1,
    "Mn": {"sulfide": 2, "oxide": 4},
    "Mg": {"sulfide": 2, "oxide": 2},
    "W": {"sulfide": 4, "oxide": 6},
}


def _builder(**over):
    cfg = dict(base_poscar="/dev/null", valences=VALENCES, formula_units=500,
               halide_total=1.7, p_site_group="P_site", s_site_group="S_site")
    cfg.update(over)
    return SSESupercellBuilder(cfg)


def test_metal_form_charge_neutral_matches_derivation():
    b = _builder()
    # Mn metal form (O=0), level 0.05, Cl = 1.0 atom per f.u.
    cand = {"P_site": {"P": 0.95, "Mn": 0.05}, "S_site": {"O": 0, "Cl": 1.0}}
    c = b.counts(cand)
    assert c["Li_delete"] == 275                      # v = 0.7 - 0.05*3 = 0.55 -> 275/3000
    assert c["Br"] == int(round(1.7 * 500)) - c["Cl"]  # Br = 1.7*fu - Cl
    cation = c["Li"] * 1 + c["P"] * 5 + c["metal"] * 2  # Mn sulfide = +2
    anion = c["S"] * 2 + c["O"] * 2 + c["Cl"] * 1 + c["Br"] * 1
    assert cation == anion


def test_oxide_form_uses_oxide_valence_and_derives_O():
    b = _builder()
    # W oxide form (O=1), level 0.06 -> v = 0.7 + 0.06 = 0.76; n_O = (6/2)*30 = 90.
    cand = {"P_site": {"P": 0.94, "W": 0.06}, "S_site": {"O": 1, "Cl": 1.0}}
    c = b.counts(cand)
    assert c["O"] == 90
    assert c["Li_delete"] == int(round(0.76 * 500))
    cation = c["Li"] * 1 + c["P"] * 5 + c["metal"] * 6  # W oxide = +6
    anion = c["S"] * 2 + c["O"] * 2 + c["Cl"] * 1 + c["Br"] * 1
    assert cation == anion


def test_formula_units_inferred_from_poscar(tmp_path):
    from ase import Atoms
    from ase.io import write

    fu = 4  # Li24 P4 S24
    syms = ["Li"] * (6 * fu) + ["P"] * fu + ["S"] * (6 * fu)
    write(str(tmp_path / "base.vasp"),
          Atoms(syms, positions=[(i, 0, 0) for i in range(len(syms))], cell=[60, 60, 60], pbc=True),
          format="vasp")
    # no formula_units in cfg -> inferred from POSCAR (4 P / 1 per f.u. = 4)
    b = SSESupercellBuilder(dict(base_poscar=str(tmp_path / "base.vasp"), valences=VALENCES,
                                 halide_total=1.7, eligible_region={"symbol": "S", "take": "last", "count": 8}))
    assert b.fu == 4


def test_build_produces_correct_atom_counts(tmp_path):
    from ase import Atoms
    from ase.io import write

    fu = 4
    syms = ["Li"] * (6 * fu) + ["P"] * fu + ["S"] * (6 * fu)
    write(str(tmp_path / "base.vasp"),
          Atoms(syms, positions=[(i, 0, 0) for i in range(len(syms))], cell=[60, 60, 60], pbc=True),
          format="vasp")
    b = _builder(formula_units=fu, base_poscar=str(tmp_path / "base.vasp"),
                 eligible_region={"symbol": "S", "take": "last", "count": 8})
    cand = {"P_site": {"P": 0.75, "Mn": 0.25}, "S_site": {"O": 0, "Cl": 1.0}}
    c = b.counts(cand)
    for st in b.build(cand, n_configs=3, rng=np.random.default_rng(0)):
        cnt = Counter(st.get_chemical_symbols())
        assert cnt["Mn"] == c["metal"] and cnt["P"] == c["P"]
        assert cnt["O"] == c["O"] and cnt["Cl"] == c["Cl"] and cnt["Br"] == c["Br"]
        assert cnt["S"] == c["S"] and cnt["Li"] == c["Li"]
        assert len(st) == len(syms) - c["Li_delete"]
