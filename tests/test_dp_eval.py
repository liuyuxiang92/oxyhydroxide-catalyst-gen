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
