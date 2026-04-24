"""LastStepElementFilter — force the final action to a required element.

Matches the reference DQN's "step 5 = O" rule from
``RL_materials_generation/env_constrained.py:299-344``.  The required
element is reserved for the last slot: earlier-step actions that pick it
are pruned, and at the last step only ``(required_element, digit)`` pairs
(optionally with nonzero digit) survive.

Plugs into the ``phase_filter`` slot of either :class:`CompositionEnv` or
:class:`IntegerRatioEnv`; the ``filter_actions`` signature matches
:class:`SMACTChargeFilter`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from .base import ConstraintFilter


class LastStepElementFilter(ConstraintFilter):
    """Restrict the final-step action to one of ``required_elements``.

    Parameters
    ----------
    required_elements:
        Element symbols allowed at the final step (e.g. ``["O"]``).
    nonzero_ratio:
        If True (default), disallow ratio/fraction values that decode to
        ``"0"`` at the final step — matches the reference rule where
        oxygen always has a digit in 1–9.
    reserve_for_last:
        If True (default), prune earlier-step actions that pick any
        element in ``required_elements`` so it is reserved for the last
        slot.  Set False to allow the required element anywhere.
    """

    def __init__(
        self,
        required_elements: List[str],
        *,
        nonzero_ratio: bool = True,
        reserve_for_last: bool = True,
    ) -> None:
        if not required_elements:
            raise ValueError("required_elements must be a non-empty list.")
        self.required_elements = set(required_elements)
        self.nonzero_ratio = nonzero_ratio
        self.reserve_for_last = reserve_for_last

    def filter_actions(
        self,
        *,
        actions: List[Tuple[Tuple[float, ...], Tuple[float, ...]]],
        units_map: Dict[str, int],
        steps_left: int,
        allowed_units: Sequence[int],
        possible_sums_by_k: List[Any],
        cation_set: List[str],
        fraction_set: List[str],
    ) -> List[Tuple[Tuple[float, ...], Tuple[float, ...]]]:
        from ..encoding import decode_one_hot

        is_last_step = steps_left == 0

        filtered = []
        for elem_oh, comp_oh in actions:
            elem = decode_one_hot(elem_oh, cation_set)
            digit_str = decode_one_hot(comp_oh, fraction_set)

            if is_last_step:
                if elem not in self.required_elements:
                    continue
                if self.nonzero_ratio and digit_str == "0":
                    continue
            else:
                if self.reserve_for_last and elem in self.required_elements:
                    continue

            filtered.append((elem_oh, comp_oh))

        return filtered if filtered else actions
