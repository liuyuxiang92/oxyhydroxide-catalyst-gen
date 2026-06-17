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

from typing import Any, Dict, List, Optional, Sequence, Tuple


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
        species_set: List[str],
        fraction_set: List[str],
        prior_groups: Optional[List[Dict[str, float]]] = None,
        **_: Any,
    ) -> List[Tuple[Tuple[float, ...], Tuple[float, ...]]]:
        """Return the subset of *actions* consistent with this constraint.

        The default implementation returns *actions* unchanged.

        ``prior_groups`` carries the already-completed group compositions (in
        group order) when this filter runs inside a :class:`MultiGroupEnv`, so a
        later sublattice can depend on an earlier pick (e.g. the S-site O range
        bounded by the P-site metal). It is ``None`` for single-group
        (:class:`CompositionEnv`) runs. Filters that ignore cross-group coupling
        can simply not read it. The trailing ``**_`` keeps subclasses forward
        compatible with future context kwargs.
        """
        return actions
