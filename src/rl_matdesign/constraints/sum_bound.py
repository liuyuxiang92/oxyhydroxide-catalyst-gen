"""IndependentSumBoundFilter — cap the *combined* amount of an independent group.

An :class:`~rl_matdesign.env_multigroup.IndependentDopantsGroup` makes ``K``
independent ``(element, amount)`` picks whose amounts add up (same element merges).
By default each pick draws from the same grid, so the combined doping spans
``[K·g_min, K·g_max]`` — usually far wider than the chemistry allows.

This filter masks each pick so the **running and final sum** of the group's picks
stays in ``[sum_min, sum_max]``. It only looks at the *amount* of each candidate
action (never the element), so same-element merging is unaffected.

Config (read off the group spec, like every other group filter)::

    constraint_filter: sum_bound
    sum_amount: {min: 0.02, max: 0.08}

Feasibility test for a candidate amount ``a`` with ``steps_left`` picks remaining
*after* it, grid min/max ``g_min``/``g_max`` and running sum ``r``: keep ``a`` iff a
valid completion exists ::

    (r + a) + steps_left * g_min <= sum_max     # can finish without overshooting
    (r + a) + steps_left * g_max >= sum_min     # can still reach the floor
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .base import ConstraintFilter


class IndependentSumBoundFilter(ConstraintFilter):
    def __init__(self, cfg: Dict[str, Any], *, env=None) -> None:
        bound = cfg.get("sum_amount") or {}
        try:
            self.sum_min = float(bound["min"])
            self.sum_max = float(bound["max"])
        except (KeyError, TypeError) as e:
            raise ValueError(
                "sum_bound filter needs 'sum_amount: {min: ..., max: ...}' on the group."
            ) from e

    def filter_actions(
        self,
        *,
        actions: List[Tuple[Tuple[float, ...], Tuple[float, ...]]],
        steps_left: int,
        fraction_set: List[str],
        running_amount: float = 0.0,
        **_: Any,
    ) -> List[Tuple[Tuple[float, ...], Tuple[float, ...]]]:
        from ..encoding import decode_one_hot

        grid = [float(f) for f in fraction_set]
        g_min, g_max = min(grid), max(grid)
        # tiny epsilon so float round-off (0.02 + 0.06 = 0.07999...) doesn't drop a
        # legitimately on-grid completion.
        eps = 1e-9

        out = []
        for elem_oh, comp_oh in actions:
            a = float(decode_one_hot(comp_oh, fraction_set))
            new_sum = running_amount + a
            if new_sum + steps_left * g_min > self.sum_max + eps:
                continue
            if new_sum + steps_left * g_max < self.sum_min - eps:
                continue
            out.append((elem_oh, comp_oh))
        return out if out else actions
