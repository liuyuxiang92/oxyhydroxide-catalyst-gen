"""HostComplementFilter — "dopants at a level, host takes the rest".

A general composition-group pattern: a group lists its **dopant** elements plus a
designated **host**, and an `amount` range for the dopant(s). This filter forces:

* non-final steps -> a **non-host** element at an amount in ``levels``;
* the final step  -> the **host** (which absorbs the remaining fraction to 1.0).

It generalizes the hand-written "metal at level + P at complement" pattern (the old
``sse_doping`` p_site role), so a friendly group config needs only ``species_set``
(dopants), ``host:``, and ``amount: {min,max,step}`` — the env's ``host`` knob wires
this filter in automatically.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .base import ConstraintFilter


class HostComplementFilter(ConstraintFilter):
    def __init__(self, cfg: Dict[str, Any], *, env=None) -> None:
        self.host = str(cfg["host_element"])
        self.levels = {str(x) for x in cfg["levels"]}

    def filter_actions(
        self,
        *,
        actions: List[Tuple[Tuple[float, ...], Tuple[float, ...]]],
        steps_left: int,
        species_set: List[str],
        fraction_set: List[str],
        **_: Any,
    ) -> List[Tuple[Tuple[float, ...], Tuple[float, ...]]]:
        from ..encoding import decode_one_hot

        is_last = steps_left == 0
        out = []
        for elem_oh, comp_oh in actions:
            el = decode_one_hot(elem_oh, species_set)
            if is_last:
                if el != self.host:
                    continue
            else:
                if el == self.host:
                    continue
                if decode_one_hot(comp_oh, fraction_set) not in self.levels:
                    continue
            out.append((elem_oh, comp_oh))
        return out if out else actions
