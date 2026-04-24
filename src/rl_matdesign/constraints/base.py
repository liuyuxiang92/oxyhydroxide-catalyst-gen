"""ConstraintFilter base class for pluggable composition constraints.

The :class:`CompositionEnv` accepts an optional ``phase_filter`` argument.
Any object that implements ``filter_actions()`` can serve as a filter.
The base class here is a no-op pass-through; subclass it to add constraints.

Two built-in subclasses are provided:
- :class:`~rl_matdesign.constraints.smact_filter.SMACTChargeFilter`
  for ionic/oxide systems requiring charge neutrality.

For OOH catalyst phase constraints see the ``main`` branch
(``src/abcde_ooh/constraints/``).
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple


class ConstraintFilter:
    """Pass-through constraint filter (no constraints applied).

    Override ``filter_actions`` to implement domain-specific rules.
    The interface mirrors the existing ``phase_filter`` contract in
    :class:`~rl_matdesign.env.CompositionEnv` so existing phase-filter
    implementations work without modification.
    """

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
        """Return the subset of *actions* consistent with this constraint.

        The default implementation returns *actions* unchanged.
        """
        return actions
