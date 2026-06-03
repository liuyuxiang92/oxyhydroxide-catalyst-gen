"""ChainConstraintFilter — registry dispatch, composition order, safety fallback."""
from __future__ import annotations

import pytest


def _oh(idx: int, n: int) -> tuple:
    v = [0.0] * n
    v[idx] = 1.0
    return tuple(v)


def test_chain_registered_under_short_name():
    from rl_matdesign.registry import CONSTRAINTS, resolve_constraint

    assert "chain" in CONSTRAINTS

    cfg = {
        "filters": [
            {"constraint_filter": "last_step_element",
             "required_elements": ["O"],
             "reserve_for_last": True},
        ],
    }
    chain = resolve_constraint("chain", cfg, env=None)
    assert type(chain).__name__ == "ChainConstraintFilter"
    assert len(chain.children) == 1
    assert type(chain.children[0]).__name__ == "LastStepElementFilter"


def test_chain_composes_real_children_at_last_step():
    """Two-element chain on a real LastStepElementFilter pair prunes the
    final-step action list down to (O, digit in {1, 2})."""
    from rl_matdesign.registry import resolve_constraint

    cation_set = ["Fe", "Co", "Ni", "O"]
    fraction_set = ["0", "1", "2"]
    cfg = {
        "filters": [
            {"constraint_filter": "last_step_element",
             "required_elements": ["O"],
             "nonzero_ratio_at_last": True,
             "reserve_for_last": True},
            {"constraint_filter": "last_step_element",
             "required_elements": ["O"],
             "reserve_for_last": False},
        ],
    }
    chain = resolve_constraint("chain", cfg, env=None)

    actions = [(_oh(i, 4), _oh(j, 3))
               for i in range(4) for j in range(3)]
    out = chain.filter_actions(
        actions=actions, units_map={"Fe": 1, "Co": 1, "Ni": 1},
        steps_left=0, allowed_units=[0, 1, 2], possible_sums_by_k=[],
        cation_set=cation_set, fraction_set=fraction_set,
    )
    assert len(out) == 2
    for elem_oh, comp_oh in out:
        assert elem_oh == _oh(3, 4)                  # only O
        assert comp_oh in (_oh(1, 3), _oh(2, 3))      # only nonzero digits


def test_chain_runs_children_in_order():
    """Second child sees the first child's *output*, not the original input."""
    from rl_matdesign.constraints.chain import ChainConstraintFilter

    seen_by_second: list = []

    class _DropFirst:
        def filter_actions(self, *, actions, **_):
            return actions[1:]

    class _RecordAndPassThrough:
        def filter_actions(self, *, actions, **_):
            seen_by_second.append(list(actions))
            return actions

    chain = ChainConstraintFilter.__new__(ChainConstraintFilter)
    chain.children = [_DropFirst(), _RecordAndPassThrough()]

    chain.filter_actions(actions=["a", "b", "c"])
    assert seen_by_second == [["b", "c"]]


def test_chain_safety_fallback_when_child_empties():
    """When any child empties the action list, the chain returns the pre-chain
    input rather than propagating an empty list (which would strand the agent)."""
    from rl_matdesign.constraints.chain import ChainConstraintFilter

    class _AlwaysEmpty:
        def filter_actions(self, *, actions, **_):
            return []

    chain = ChainConstraintFilter.__new__(ChainConstraintFilter)
    chain.children = [_AlwaysEmpty()]

    original = [("a", "b"), ("c", "d")]
    out = chain.filter_actions(actions=original)
    assert out == original


def test_chain_safety_fallback_when_intermediate_step_empties():
    """Even if the *intermediate* (not final) child empties, fall back to original."""
    from rl_matdesign.constraints.chain import ChainConstraintFilter

    class _DropAll:
        def filter_actions(self, *, actions, **_):
            return []

    class _Identity:
        def filter_actions(self, *, actions, **_):
            return actions

    chain = ChainConstraintFilter.__new__(ChainConstraintFilter)
    chain.children = [_DropAll(), _Identity()]

    original = [("x", "y")]
    out = chain.filter_actions(actions=original)
    assert out == original


def test_chain_empty_filters_raises():
    from rl_matdesign.constraints.chain import ChainConstraintFilter

    with pytest.raises(ValueError, match="non-empty"):
        ChainConstraintFilter({"filters": []}, env=None)


def test_chain_missing_filters_key_raises():
    from rl_matdesign.constraints.chain import ChainConstraintFilter

    with pytest.raises(ValueError, match="non-empty"):
        ChainConstraintFilter({}, env=None)


def test_chain_non_list_filters_raises():
    from rl_matdesign.constraints.chain import ChainConstraintFilter

    with pytest.raises(ValueError, match="must be a list"):
        ChainConstraintFilter({"filters": {"k": "v"}}, env=None)


def test_chain_non_dict_child_raises():
    from rl_matdesign.constraints.chain import ChainConstraintFilter

    with pytest.raises(ValueError, match="must be a dict"):
        ChainConstraintFilter({"filters": ["not_a_dict"]}, env=None)


def test_chain_child_missing_constraint_filter_key_raises():
    from rl_matdesign.constraints.chain import ChainConstraintFilter

    with pytest.raises(ValueError, match=r"missing 'constraint_filter'"):
        ChainConstraintFilter(
            {"filters": [{"required_elements": ["O"]}]}, env=None,
        )


def test_oxide_yaml_round_trip_resolves_to_chain():
    """The shipped oxide configs (sinter + calcine) should both resolve
    to a ChainConstraintFilter wrapping LastStepElementFilter + SMACTChargeFilter."""
    pytest.importorskip("smact")  # SMACT-dependent test

    import yaml
    from pathlib import Path
    from rl_matdesign.registry import resolve_constraint

    repo = Path(__file__).resolve().parent.parent
    for name in ("oxides_sinter.yaml", "oxides_calcine.yaml"):
        cfg = yaml.safe_load((repo / "configs" / name).read_text())
        assert cfg["constraint_filter"] == "chain", name
        chain = resolve_constraint(cfg["constraint_filter"], cfg, env=None)
        child_types = [type(c).__name__ for c in chain.children]
        assert child_types == ["LastStepElementFilter", "SMACTChargeFilter"], (
            f"{name}: unexpected child types {child_types}"
        )
