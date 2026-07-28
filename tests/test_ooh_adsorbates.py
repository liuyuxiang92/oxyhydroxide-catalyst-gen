"""Config-selectable adsorbates for the OOH predictor.

First coverage for ``abcde_ooh/dp_predictor.py`` — the adsorbate builder, the
frame packing and the ``-1`` masking were all previously untested.

``ase`` is available but ``deepmd`` is not, so ``_lazy_import_ase_deepmd`` is
monkeypatched to hand back the real ASE objects plus a fake ``DeepProperty`` that
records the batch it was given. Structure building therefore stays real; only the
model call is faked. Mirrors the stubbing style of ``tests/test_structure_score.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

from abcde_ooh import dp_predictor as dpp
from abcde_ooh.dp_predictor import ADSORBATE_MODES, DPConfig, normalize_adsorbates


# ---------------------------------------------------------------------------
# normalize_adsorbates
# ---------------------------------------------------------------------------

def test_default_names_map_to_starred_modes():
    assert normalize_adsorbates(["O", "OH", "OOH"]) == ("O*", "OH*", "OOH*")


def test_starred_and_mixed_case_forms_accepted():
    assert normalize_adsorbates(["o*", "Oh", "OOH*"]) == ("O*", "OH*", "OOH*")


@pytest.mark.parametrize("spec", [None, [], "none", "NONE", ""])
def test_bare_selections_normalize_to_empty(spec):
    assert normalize_adsorbates(spec) == ()


def test_order_is_preserved_and_duplicates_dropped():
    # Order matters: it fixes frame order, which output_index selects from.
    assert normalize_adsorbates(["OOH", "O", "OOH"]) == ("OOH*", "O*")


def test_unknown_adsorbate_raises_with_valid_names_listed():
    with pytest.raises(ValueError) as info:
        normalize_adsorbates(["OOOH"])
    msg = str(info.value)
    assert "OOOH" in msg and "OOH" in msg


def test_none_cannot_be_mixed_with_real_adsorbates():
    # A bare frame has nat_slab atoms and an adsorbate frame nat_slab+3; mixing
    # them in one batch would break the equal-natoms invariant.
    with pytest.raises(ValueError) as info:
        normalize_adsorbates(["O", "none"])
    assert "empty list" in str(info.value)


# ---------------------------------------------------------------------------
# Fixtures: real ASE, fake DeepProperty
# ---------------------------------------------------------------------------

class _FakeDeepProperty:
    """Stand-in for deepmd's DeepProperty; records every batch it evaluates."""

    calls: list = []

    def __init__(self, model_file=None, auto_batch_size=False, head=None):
        self.model_file = model_file

    def eval(self, coords, cells, atom_types, mixed_type=False):
        _FakeDeepProperty.calls.append({
            "nframes": coords.shape[0],
            "natoms": coords.shape[1],
            "atom_types": np.array(atom_types, copy=True),
        })
        # One scalar per frame, distinguishable per frame index.
        return np.arange(coords.shape[0], dtype=float).reshape(-1, 1)


@pytest.fixture(autouse=True)
def _clear_calls():
    _FakeDeepProperty.calls = []
    yield
    _FakeDeepProperty.calls = []


@pytest.fixture
def slab_poscar(tmp_path):
    """A tiny Co-oxyhydroxide-ish slab written as a VASP POSCAR."""
    from ase import Atoms
    from ase.io import write

    # 6 metal sites + 2 O + 1 H, non-cubic c so the surface normal is well defined.
    positions = [
        (0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0),
        (2.0, 2.0, 0.0), (1.0, 1.0, 2.0), (3.0, 1.0, 2.0),
        (1.0, 3.0, 4.0), (3.0, 3.0, 4.0),
        (1.0, 1.0, 5.0),
    ]
    atoms = Atoms("Co6O2H", positions=positions, cell=[4.0, 4.0, 12.0], pbc=True)
    path = tmp_path / "POSCAR"
    write(str(path), atoms, format="vasp")
    return str(path)


@pytest.fixture
def make_predictor(monkeypatch, slab_poscar):
    from ase import Atoms
    from ase.data import chemical_symbols
    from ase.io import read as ase_read
    from ase.io import write as ase_write

    monkeypatch.setattr(
        dpp, "_lazy_import_ase_deepmd",
        lambda: (Atoms, ase_read, ase_write, chemical_symbols, _FakeDeepProperty),
    )

    def _make(**kwargs):
        cfg = DPConfig(
            base_poscar=slab_poscar,
            model_files=("fake_model.pt",),
            n_random_configs=kwargs.pop("n_random_configs", 1),
            seed=7,
            **kwargs,
        )
        return dpp.DeepMDOverpotentialPredictor(cfg)

    return _make


_COMP = {"Ni": 0.5, "Fe": 0.5}


# ---------------------------------------------------------------------------
# Frame construction
# ---------------------------------------------------------------------------

def test_default_builds_three_frames_with_three_adsorbate_atoms(make_predictor):
    p = make_predictor()
    assert p.cfg.adsorbates == ADSORBATE_MODES
    p.predict_overpotential(_COMP)

    call = _FakeDeepProperty.calls[0]
    assert call["nframes"] == 3
    assert call["natoms"] == p.nat_slab + 3

    # O* hides O2 and H, OH* hides O2, OOH* hides nothing.
    masked_per_frame = [int((row == -1).sum()) for row in call["atom_types"]]
    assert masked_per_frame == [2, 1, 0]


def test_empty_selection_builds_one_bare_frame(make_predictor):
    p = make_predictor(adsorbates=())
    p.predict_overpotential(_COMP)

    call = _FakeDeepProperty.calls[0]
    assert call["nframes"] == 1
    # No adsorbate atoms appended at all, and nothing masked.
    assert call["natoms"] == p.nat_slab
    assert int((call["atom_types"] == -1).sum()) == 0


def test_single_adsorbate_builds_one_frame_carrying_its_mask(make_predictor):
    p = make_predictor(adsorbates=("O*",))
    p.predict_overpotential(_COMP)

    call = _FakeDeepProperty.calls[0]
    assert call["nframes"] == 1
    assert call["natoms"] == p.nat_slab + 3
    assert int((call["atom_types"] == -1).sum()) == 2   # O* masks O2 and H


@pytest.mark.parametrize("ads,expected", [
    ((), 1), (("O*",), 1), (("O*", "OH*"), 2), (ADSORBATE_MODES, 3),
])
def test_nframes_follows_the_configured_list(make_predictor, ads, expected):
    # Guards the removed `nframes = 3` literal.
    p = make_predictor(adsorbates=ads)
    p.predict_overpotential(_COMP)
    assert _FakeDeepProperty.calls[0]["nframes"] == expected


def test_bare_selection_cuts_frame_work_threefold(make_predictor):
    p3 = make_predictor(n_random_configs=4)
    p3.predict_overpotential(_COMP)
    frames_3 = sum(c["nframes"] for c in _FakeDeepProperty.calls)

    _FakeDeepProperty.calls = []
    p0 = make_predictor(n_random_configs=4, adsorbates=())
    p0.predict_overpotential(_COMP)
    frames_bare = sum(c["nframes"] for c in _FakeDeepProperty.calls)

    assert frames_3 == 3 * frames_bare


def test_unknown_mode_still_rejected_by_the_builder(make_predictor):
    p = make_predictor()
    with pytest.raises(ValueError):
        p._add_adsorbates_equalized(p.base_slab, "OOOH*", height=1.9, dz_chain=1.0)


# ---------------------------------------------------------------------------
# Cache-key fingerprint
# ---------------------------------------------------------------------------

def _ooh_wrapper(monkeypatch, slab_poscar, **kwargs):
    from ase import Atoms
    from ase.data import chemical_symbols
    from ase.io import read as ase_read
    from ase.io import write as ase_write

    monkeypatch.setattr(
        dpp, "_lazy_import_ase_deepmd",
        lambda: (Atoms, ase_read, ase_write, chemical_symbols, _FakeDeepProperty),
    )
    from rl_matdesign.predictors.ooh import OOHCatalystPredictor
    return OOHCatalystPredictor(
        base_poscar=slab_poscar, dp_models=["fake_model.pt"],
        n_random_configs=1, **kwargs,
    )


def test_cache_key_differs_between_adsorbate_settings(monkeypatch, slab_poscar):
    # A dp_cache restored from a checkpoint must not serve 3-adsorbate values to a
    # bare run — the number describes a different structure.
    three = _ooh_wrapper(monkeypatch, slab_poscar)
    bare = _ooh_wrapper(monkeypatch, slab_poscar, adsorbates=[])
    assert three._comp_key(_COMP) != bare._comp_key(_COMP)


def test_cache_key_differs_between_output_indices(monkeypatch, slab_poscar):
    a = _ooh_wrapper(monkeypatch, slab_poscar, output_index=0)
    b = _ooh_wrapper(monkeypatch, slab_poscar, output_index=1)
    assert a._comp_key(_COMP) != b._comp_key(_COMP)


def test_cache_key_still_order_invariant_in_composition(monkeypatch, slab_poscar):
    p = _ooh_wrapper(monkeypatch, slab_poscar)
    assert p._comp_key({"Ni": 0.5, "Fe": 0.5}) == p._comp_key({"Fe": 0.5, "Ni": 0.5})


def test_cache_prevents_a_second_model_call(monkeypatch, slab_poscar):
    p = _ooh_wrapper(monkeypatch, slab_poscar, adsorbates=[])
    p.predict_raw(_COMP)
    n_after_first = len(_FakeDeepProperty.calls)
    p.predict_raw(_COMP)
    assert len(_FakeDeepProperty.calls) == n_after_first


# ---------------------------------------------------------------------------
# debug_dir dumps the frames that were actually evaluated
# ---------------------------------------------------------------------------

def test_debug_dump_names_follow_the_adsorbate_list(make_predictor, tmp_path):
    out = tmp_path / "dump3"
    p = make_predictor(debug_dir=str(out))
    p.predict_overpotential(_COMP)
    names = sorted(f.name for f in out.glob("*.vasp"))
    assert len(names) == 3
    assert [n.rsplit("_", 1)[-1] for n in sorted(names)] == sorted(
        ["O.vasp", "OH.vasp", "OOH.vasp"])


def test_bare_debug_dump_is_tagged_clean(make_predictor, tmp_path):
    out = tmp_path / "dump0"
    p = make_predictor(adsorbates=(), debug_dir=str(out))
    p.predict_overpotential(_COMP)
    files = list(out.glob("*.vasp"))
    assert len(files) == 1
    assert files[0].name.endswith("_clean.vasp")


def test_debug_dir_does_not_perturb_the_result(make_predictor, tmp_path):
    # The dump used to rebuild the frames from the same rng, which both desynced
    # the random stream and made the dumped POSCARs differ from what was scored.
    without = make_predictor().predict_overpotential(_COMP)
    _FakeDeepProperty.calls = []
    with_dump = make_predictor(debug_dir=str(tmp_path / "d")).predict_overpotential(_COMP)
    assert without == with_dump


def test_dumped_structure_matches_the_evaluated_frame(make_predictor, tmp_path):
    from ase.io import read as ase_read

    out = tmp_path / "dump"
    p = make_predictor(adsorbates=(), debug_dir=str(out))
    p.predict_overpotential(_COMP)

    dumped = ase_read(str(next(iter(out.glob("*.vasp")))), format="vasp")
    call = _FakeDeepProperty.calls[0]
    assert len(dumped) == call["natoms"]
