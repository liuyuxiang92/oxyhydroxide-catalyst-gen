"""pick_scalar helper tests."""
from __future__ import annotations

import numpy as np
import pytest


def test_pick_scalar_default_index_0():
    from rl_matdesign.utils.dp_eval import pick_scalar
    assert pick_scalar(np.array([1.0, 2.0, 3.0])) == 1.0


def test_pick_scalar_explicit_index():
    from rl_matdesign.utils.dp_eval import pick_scalar
    assert pick_scalar(np.array([1.0, 2.0, 3.0]), output_index=2) == 3.0


def test_pick_scalar_mean_aggregator():
    from rl_matdesign.utils.dp_eval import pick_scalar
    assert pick_scalar(np.array([1.0, 2.0, 3.0]), output_aggregator="mean") == 2.0


def test_pick_scalar_max_aggregator():
    from rl_matdesign.utils.dp_eval import pick_scalar
    assert pick_scalar(np.array([1.0, 2.0, 3.0]), output_aggregator="max") == 3.0


def test_pick_scalar_handles_2d_input():
    """Multi-output models often return shape (1, K) — must be flattened."""
    from rl_matdesign.utils.dp_eval import pick_scalar
    assert pick_scalar(np.array([[10.0, 20.0]]), output_index=1) == 20.0


def test_pick_scalar_index_out_of_range_raises():
    from rl_matdesign.utils.dp_eval import pick_scalar
    with pytest.raises(IndexError) as info:
        pick_scalar(np.array([1.0]), output_index=5)
    msg = str(info.value)
    assert "5" in msg and "1" in msg          # surfaces both the index and the vector length


def test_pick_scalar_unknown_aggregator_raises():
    from rl_matdesign.utils.dp_eval import pick_scalar
    with pytest.raises(ValueError):
        pick_scalar(np.array([1.0, 2.0]), output_aggregator="median")


def test_pick_scalar_empty_vector_raises():
    from rl_matdesign.utils.dp_eval import pick_scalar
    with pytest.raises(ValueError):
        pick_scalar(np.array([]))


# --------------------------------------------------------------------------- #
# Fold orientation: both helpers must return (n_models, n_structures).
#
# They used to iterate in OPPOSITE orders and return flat lists — eval_energy_ase
# structure-major, eval_property_ensemble model-major — so a caller reshaping either
# one blindly would average the wrong axis and silently report configurational
# scatter as model uncertainty.
# --------------------------------------------------------------------------- #

class _FakeCalc:
    """Minimal ASE-like calculator: energy = offset * natoms."""

    def __init__(self, offset):
        self.offset = offset

    def get_potential_energy(self, atoms=None):
        return self.offset * len(self._atoms)


class _FakeAtoms:
    def __init__(self, n, tag):
        self.n, self.tag, self.calc = n, tag, None

    def __len__(self):
        return self.n

    def copy(self):
        return _FakeAtoms(self.n, self.tag)

    def get_potential_energy(self):
        return float(self.calc.offset * 100 + self.tag)


def test_eval_energy_ase_returns_models_by_structures():
    from rl_matdesign.utils.dp_eval import eval_energy_ase

    structures = [_FakeAtoms(1, 1), _FakeAtoms(1, 2), _FakeAtoms(1, 3)]
    calcs = [_FakeCalc(1), _FakeCalc(2)]
    out = eval_energy_ase(structures, calcs, energy_per_atom=False)
    assert out.shape == (2, 3)                     # (n_models, n_structures)
    # row = model, column = structure
    assert list(out[0]) == [101.0, 102.0, 103.0]
    assert list(out[1]) == [201.0, 202.0, 203.0]


def test_fold_averages_structures_then_spreads_over_models():
    from rl_matdesign.predictors.structure_score import _fold_models_structures

    # 2 models x 3 structures. Per-model means are 10 and 20 regardless of how
    # scattered the structures are within each model.
    arr = np.array([[0.0, 10.0, 20.0],
                    [19.0, 20.0, 21.0]])
    mean, std = _fold_models_structures(arr)
    assert mean == pytest.approx(15.0)             # mean(10, 20)
    assert std == pytest.approx(5.0)               # population std of (10, 20)


def test_fold_with_one_model_has_zero_std_however_scattered():
    from rl_matdesign.predictors.structure_score import _fold_models_structures

    mean, std = _fold_models_structures(np.array([[-100.0, 0.0, 100.0]]))
    assert mean == pytest.approx(0.0)
    assert std == 0.0
