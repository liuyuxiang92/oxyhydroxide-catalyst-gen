"""PhasePatternFilter rule + action-filter tests."""
from __future__ import annotations

import pytest


def _nife_pattern():
    from rl_matdesign.constraints.phase_pattern import ALL_OTHERS
    return {
        "name": "NiFe",
        "required": {"Ni": [0.40, 0.80], "Fe": [0.20, 0.40]},
        "primary_sum": {"elements": ["Ni", "Fe"], "min": 0.75},
        "dopant_sum": {"elements": ALL_OTHERS, "max": 0.25},
        "ratios": [{"num": "Ni", "den": ["Ni", "Fe"], "min": 0.6, "max": 0.8}],
        "forbidden": ["Cr"],
    }


def test_single_pattern_match():
    from rl_matdesign.constraints.phase_pattern import PhasePatternFilter
    f = PhasePatternFilter(patterns=[_nife_pattern()])
    ok, name = f.check_composition({"Ni": 0.50, "Fe": 0.30, "Mn": 0.10, "Co": 0.10})
    assert ok and name == "NiFe"


def test_forbidden_element_blocks_match():
    from rl_matdesign.constraints.phase_pattern import PhasePatternFilter
    f = PhasePatternFilter(patterns=[_nife_pattern()])
    assert f.check_composition({"Ni": 0.50, "Fe": 0.30, "Cr": 0.20}) == (False, "")


def test_required_range_violation_blocks_match():
    from rl_matdesign.constraints.phase_pattern import PhasePatternFilter
    f = PhasePatternFilter(patterns=[_nife_pattern()])
    # Fe out of [0.20, 0.40]:
    assert f.check_composition({"Ni": 0.70, "Fe": 0.10, "Mn": 0.20}) == (False, "")


def test_ratio_violation_blocks_match():
    from rl_matdesign.constraints.phase_pattern import PhasePatternFilter
    f = PhasePatternFilter(patterns=[_nife_pattern()])
    # Ni/(Ni+Fe) = 0.40/0.80 = 0.5, outside [0.6, 0.8]
    assert f.check_composition({"Ni": 0.40, "Fe": 0.40, "Mn": 0.20}) == (False, "")


def test_primary_sum_floor_violation_blocks_match():
    from rl_matdesign.constraints.phase_pattern import PhasePatternFilter
    f = PhasePatternFilter(patterns=[_nife_pattern()])
    # Ni+Fe = 0.50, below primary_sum.min = 0.75
    assert f.check_composition({"Ni": 0.30, "Fe": 0.20, "Mn": 0.50}) == (False, "")


def test_multi_pattern_or_returns_first_match():
    from rl_matdesign.constraints.phase_pattern import PhasePatternFilter
    f = PhasePatternFilter(patterns=[
        {"name": "NiFe",  "required": {"Ni": [0.40, 0.80], "Fe": [0.20, 0.40]},
         "ratios": [{"num": "Ni", "den": ["Ni", "Fe"], "min": 0.55, "max": 0.8}]},
        {"name": "Ni",    "required": {"Ni": [0.50, 0.99]}},
    ])
    # Matches NiFe (listed first); the bare Ni pattern would also match but isn't reached.
    assert f.check_composition({"Ni": 0.50, "Fe": 0.30, "Mn": 0.20}) == (True, "NiFe")
    # Only the Ni pattern matches here.
    assert f.check_composition({"Ni": 0.60, "Mn": 0.40}) == (True, "Ni")
    # Nothing matches.
    assert f.check_composition({"Mn": 1.0}) == (False, "")


def test_empty_patterns_raises():
    from rl_matdesign.constraints.phase_pattern import PhasePatternFilter
    with pytest.raises(ValueError):
        PhasePatternFilter(patterns=[])


def test_required_lo_greater_than_hi_raises():
    from rl_matdesign.constraints.phase_pattern import PhasePatternFilter
    with pytest.raises(ValueError):
        PhasePatternFilter(patterns=[{"required": {"A": [0.5, 0.1]}}])


def test_terminal_action_filter_admits_matching_composition():
    """At the last step (steps_left == 0) the filter computes the would-be
    terminal composition and admits actions whose terminal matches a pattern."""
    from rl_matdesign.constraints.phase_pattern import PhasePatternFilter

    f = PhasePatternFilter(patterns=[
        {"name": "NiFe",  "required": {"Ni": [0.40, 0.80], "Fe": [0.20, 0.40]},
         "ratios": [{"num": "Ni", "den": ["Ni", "Fe"], "min": 0.55, "max": 0.8}]},
    ])

    species_set = ["Ni", "Fe", "Mn"]
    fraction_set = ["0.20", "0.30", "0.50"]

    def _oh(idx, n):
        v = [0.0] * n
        v[idx] = 1.0
        return tuple(v)

    # Existing: Ni=10/20, Fe=6/20 (i.e. 0.50, 0.30 once the env normalises).
    # Last action puts Mn at 0.20 → terminal {Ni:0.5, Fe:0.3, Mn:0.2}.
    good = [(_oh(2, 3), _oh(0, 3))]
    out = f.filter_actions(
        actions=good, units_map={"Ni": 10, "Fe": 6}, steps_left=0,
        allowed_units=[4, 6, 10], possible_sums_by_k=[],
        species_set=species_set, fraction_set=fraction_set,
    )
    assert len(out) == 1


def test_terminal_action_filter_blocks_nonmatching_composition():
    from rl_matdesign.constraints.phase_pattern import PhasePatternFilter
    f = PhasePatternFilter(patterns=[
        {"name": "NiFe",  "required": {"Ni": [0.40, 0.80], "Fe": [0.20, 0.40]},
         "ratios": [{"num": "Ni", "den": ["Ni", "Fe"], "min": 0.55, "max": 0.8}]},
    ])
    species_set = ["Ni", "Fe", "Mn"]
    fraction_set = ["0.20", "0.30", "0.50"]

    def _oh(idx, n):
        v = [0.0] * n
        v[idx] = 1.0
        return tuple(v)

    # Last action puts Mn at 0.50 → terminal {Ni:0.25, Fe:0.15, Mn:0.5} doesn't match.
    bad = [(_oh(2, 3), _oh(2, 3))]
    out = f.filter_actions(
        actions=bad, units_map={"Ni": 10, "Fe": 6}, steps_left=0,
        allowed_units=[4, 6, 10], possible_sums_by_k=[],
        species_set=species_set, fraction_set=fraction_set,
    )
    assert out == []
