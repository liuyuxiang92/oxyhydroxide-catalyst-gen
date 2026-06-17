"""Tests for the SSE (doped Li6PS6) builder recipe — chemistry + structure assembly.

The S-site is a categorical group: each O slot is a form flag (0 = metal/sulfide
form, 1 = oxide) and Cl is a real per-formula-unit count. The P-site picks one or
more dopant metals (``kind: independent``); with two metals the S-site carries one
O-form slot per metal (``O_a``, ``O_b``), paired to the sorted metals. DeepMD eval
/ geo-opt are GPU-gated and not exercised here.
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
    "Sn": {"sulfide": 4, "oxide": 4},
}

# Two-O-slot S-site (per-metal forms), passed via `groups` so the builder
# discovers the ordered O-form slot names O_a, O_b and the Cl slot.
TWO_O_GROUPS = [
    {"name": "S_site", "kind": "categorical", "choices": [
        {"name": "O_a", "element": "O", "values": [0, 1]},
        {"name": "O_b", "element": "O", "values": [0, 1]},
        {"element": "Cl", "values": [0.6, 0.8, 1.0, 1.2, 1.4]},
    ]},
]


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
    cation = c["Li"] * 1 + c["P"] * 5 + c["metals"]["Mn"] * 2  # Mn sulfide = +2
    anion = c["S"] * 2 + c["O"] * 2 + c["Cl"] * 1 + c["Br"] * 1
    assert cation == anion


def test_oxide_form_uses_oxide_valence_and_derives_O():
    b = _builder()
    # W oxide form (O=1), level 0.06 -> v = 0.7 + 0.06 = 0.76; n_O = (6/2)*30 = 90.
    cand = {"P_site": {"P": 0.94, "W": 0.06}, "S_site": {"O": 1, "Cl": 1.0}}
    c = b.counts(cand)
    assert c["O"] == 90
    assert c["Li_delete"] == int(round(0.76 * 500))
    cation = c["Li"] * 1 + c["P"] * 5 + c["metals"]["W"] * 6  # W oxide = +6
    anion = c["S"] * 2 + c["O"] * 2 + c["Cl"] * 1 + c["Br"] * 1
    assert cation == anion


def test_two_distinct_metals_mixed_forms_charge_neutral():
    # Mn sulfide (O_a=0) + W oxide (O_b=1); sorted metals = [Mn, W] -> O_a↔Mn, O_b↔W.
    b = _builder(groups=TWO_O_GROUPS)
    cand = {"P_site": {"Mn": 0.05, "W": 0.06},
            "S_site": {"O_a": 0, "O_b": 1, "Cl": 1.0}}
    c = b.counts(cand)
    assert c["metals"] == {"Mn": 25, "W": 30}
    assert c["O"] == int(round(0.5 * 6 * 30))         # only W (oxide) contributes O
    cation = (c["Li"] * 1 + c["P"] * 5
              + c["metals"]["Mn"] * 2                 # Mn sulfide = +2
              + c["metals"]["W"] * 6)                 # W oxide   = +6
    anion = c["S"] * 2 + c["O"] * 2 + c["Cl"] * 1 + c["Br"] * 1
    assert cation == anion


def test_o_form_slot_pairs_with_sorted_metal():
    # Swap which metal is oxide: Mn oxide (O_a=1), W sulfide (O_b=0).
    b = _builder(groups=TWO_O_GROUPS)
    cand = {"P_site": {"Mn": 0.05, "W": 0.06},
            "S_site": {"O_a": 1, "O_b": 0, "Cl": 1.0}}
    c = b.counts(cand)
    assert c["O"] == int(round(0.5 * 4 * 25))         # only Mn (oxide, valence 4) -> O
    cation = (c["Li"] + c["P"] * 5
              + c["metals"]["Mn"] * 4                 # Mn oxide = +4
              + c["metals"]["W"] * 4)                 # W sulfide = +4
    anion = c["S"] * 2 + c["O"] * 2 + c["Cl"] + c["Br"]
    assert cation == anion


def test_same_element_merges_and_uses_first_slot():
    # Same metal picked twice merges to one fraction; only the first O slot is read.
    b = _builder(groups=TWO_O_GROUPS)
    cand = {"P_site": {"Mn": 0.10}, "S_site": {"O_a": 0, "O_b": 1, "Cl": 1.0}}
    c = b.counts(cand)
    assert c["metals"] == {"Mn": 50}                  # 0.10 * 500
    assert c["O"] == 0                                 # O_a=0 (sulfide); O_b ignored


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
        assert cnt["Mn"] == c["metals"]["Mn"] and cnt["P"] == c["P"]
        assert cnt["O"] == c["O"] and cnt["Cl"] == c["Cl"] and cnt["Br"] == c["Br"]
        assert cnt["S"] == c["S"] and cnt["Li"] == c["Li"]
        assert len(st) == len(syms) - c["Li_delete"]


def _lips_supercell(tmp_path, fu):
    from ase import Atoms
    from ase.io import write

    syms = ["Li"] * (6 * fu) + ["P"] * fu + ["S"] * (6 * fu)
    path = str(tmp_path / "base.vasp")
    write(path, Atoms(syms, positions=[(i, 0, 0) for i in range(len(syms))],
                      cell=[5000, 5000, 5000], pbc=True), format="vasp")
    return path


def test_eligible_region_fallback_used_only_on_overflow(tmp_path):
    # fu=20 -> 120 S. O+Cl+Br for two oxide dopants overflows a small primary
    # region; the larger fallback is used only then (and the build succeeds).
    base = _lips_supercell(tmp_path, fu=20)
    cand = {"P_site": {"Mn": 0.1, "W": 0.1}, "S_site": {"O_a": 1, "O_b": 1, "Cl": 1.0}}

    b = _builder(formula_units=20, base_poscar=base, groups=TWO_O_GROUPS,
                 eligible_region={"symbol": "S", "take": "last", "count": 20},
                 eligible_region_fallback={"symbol": "S", "take": "first", "count": 80})
    c = b.counts(cand)
    n_sub = c["O"] + c["Cl"] + c["Br"]
    assert n_sub > 20                                    # overflows the primary region
    st = b.build(cand, n_configs=1, rng=np.random.default_rng(0))[0]
    cnt = Counter(st.get_chemical_symbols())
    assert cnt["O"] == c["O"] and cnt["Cl"] == c["Cl"]
    assert cnt["Br"] == c["Br"] and cnt["S"] == c["S"]


def test_eligible_region_no_fallback_overflow_raises(tmp_path):
    import pytest

    base = _lips_supercell(tmp_path, fu=20)
    cand = {"P_site": {"Mn": 0.1, "W": 0.1}, "S_site": {"O_a": 1, "O_b": 1, "Cl": 1.0}}
    b = _builder(formula_units=20, base_poscar=base, groups=TWO_O_GROUPS,
                 eligible_region={"symbol": "S", "take": "last", "count": 20})  # no fallback
    with pytest.raises(ValueError, match="too small"):
        b.build(cand, n_configs=1, rng=np.random.default_rng(0))


# ---------------------------------------------------------------------------
# Multi-template mode (auto-detected from a list-valued base_poscar)
# ---------------------------------------------------------------------------

def test_single_mode_str_unchanged(tmp_path):
    base = _lips_supercell(tmp_path, fu=4)
    b = _builder(base_poscar=base, formula_units=4)
    assert b.multi is False and len(b.specs) == 1 and b.specs[0].path == base
    assert b.base_poscar == base and b.fu == 4


def test_multi_per_template_n_configs_concatenated(tmp_path):
    # Two templates, per-template n_configs -> cells concatenated into one ensemble.
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    a = _lips_supercell(tmp_path / "a", fu=4)
    b_path = _lips_supercell(tmp_path / "b", fu=4)
    cand = {"P_site": {"P": 0.75, "Mn": 0.25}, "S_site": {"O": 0, "Cl": 1.0}}

    bld = _builder(base_poscar=[{"path": a, "n_configs": 3}, {"path": b_path, "n_configs": 2}],
                   formula_units=4,
                   eligible_region={"symbol": "S", "take": "last", "count": 8})
    assert bld.multi is True
    structs = bld.build(cand, n_configs=99, rng=np.random.default_rng(0))
    assert len(structs) == 5                       # 3 + 2, caller n_configs ignored

    c = bld.counts(cand, fu=4)
    assert all(len(s) == 13 * 4 - c["Li_delete"] for s in structs)  # 6 Li +1 P +6 S per f.u.


def test_multi_infers_fu_per_template(tmp_path):
    # formula_units omitted -> each template infers its own fu from its POSCAR.
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    a = _lips_supercell(tmp_path / "a", fu=4)
    b_path = _lips_supercell(tmp_path / "b", fu=6)
    bld = _builder(base_poscar=[a, b_path], formula_units=None,
                   eligible_region={"symbol": "S", "take": "last", "count": 8})
    assert bld._fu_for(bld.specs[0]) == 4 and bld._fu_for(bld.specs[1]) == 6
    assert bld.fu == 4                             # primary-template back-compat accessor


def test_multi_bare_strings_fall_back_to_caller_n_configs(tmp_path):
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    a = _lips_supercell(tmp_path / "a", fu=4)
    b_path = _lips_supercell(tmp_path / "b", fu=4)
    cand = {"P_site": {"P": 0.75, "Mn": 0.25}, "S_site": {"O": 0, "Cl": 1.0}}
    bld = _builder(base_poscar=[a, b_path], formula_units=4,
                   eligible_region={"symbol": "S", "take": "last", "count": 8})
    structs = bld.build(cand, n_configs=2, rng=np.random.default_rng(0))
    assert len(structs) == 4                       # 2 templates * caller n_configs(2)


def test_multi_bad_entry_and_empty_list_raise(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="path"):
        _builder(base_poscar=[{"n_configs": 2}], formula_units=4)
    with pytest.raises(ValueError):
        _builder(base_poscar=[], formula_units=4)


def test_small_composition_stays_in_primary_region(tmp_path):
    # When O+Cl+Br fits the primary region, the fallback is NOT used: the
    # substituted sites must all be within the last-`count` S (the primary).
    base = _lips_supercell(tmp_path, fu=20)
    cand = {"P_site": {"Mn": 0.05}, "S_site": {"O_a": 0, "Cl": 0.6}}   # sulfide, low Cl
    primary_count = 60
    b = _builder(formula_units=20, base_poscar=base, groups=TWO_O_GROUPS,
                 eligible_region={"symbol": "S", "take": "last", "count": primary_count},
                 eligible_region_fallback={"symbol": "S", "take": "first", "count": 120})
    c = b.counts(cand)
    assert c["O"] + c["Cl"] + c["Br"] <= primary_count
    st = b.build(cand, n_configs=1, rng=np.random.default_rng(0))[0]
    syms = st.get_chemical_symbols()
    n_S_total = 6 * 20
    primary_idx = set(range(len(syms) - primary_count, len(syms)))   # last `count` indices
    subbed = [i for i, s in enumerate(syms) if s in ("O", "Cl", "Br")]
    assert subbed and all(i in primary_idx for i in subbed)          # stayed in primary


def test_build_places_two_metals(tmp_path):
    from ase import Atoms
    from ase.io import write

    fu = 8
    syms = ["Li"] * (6 * fu) + ["P"] * fu + ["S"] * (6 * fu)
    write(str(tmp_path / "base.vasp"),
          Atoms(syms, positions=[(i, 0, 0) for i in range(len(syms))], cell=[99, 99, 99], pbc=True),
          format="vasp")
    b = _builder(formula_units=fu, base_poscar=str(tmp_path / "base.vasp"),
                 groups=TWO_O_GROUPS,
                 eligible_region={"symbol": "S", "take": "last", "count": 48})
    cand = {"P_site": {"Mn": 0.25, "W": 0.125},
            "S_site": {"O_a": 0, "O_b": 1, "Cl": 1.0}}
    c = b.counts(cand)
    for st in b.build(cand, n_configs=2, rng=np.random.default_rng(0)):
        cnt = Counter(st.get_chemical_symbols())
        assert cnt["Mn"] == c["metals"]["Mn"]
        assert cnt["W"] == c["metals"]["W"]
        assert cnt["P"] == c["P"]
