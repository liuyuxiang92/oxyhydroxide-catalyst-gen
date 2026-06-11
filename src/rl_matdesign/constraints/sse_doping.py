"""SSEDopingFilter — mask the doped-Li₆PS₆ S-site oxygen *form* by the metal category.

On the categorical S-site, the ``O`` slot is a form flag: ``0`` = metal form (no
oxygen), ``> 0`` = metal-oxide form. Which forms are legal depends on the P-site
dopant metal (read from ``prior_groups``):

* ``metal_only`` metals (e.g. Ru) -> only ``O = 0`` (metal form);
* ``oxide_only`` metals -> only ``O > 0`` (oxide form);
* "both" (any metal in neither list) -> either.

Only the O slot is touched; the Cl slot's values are already the real allowed
counts (no masking needed). The old p_site role is gone — the env's ``host`` knob
handles "metal at a level + P takes the rest".
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .base import ConstraintFilter


class SSEDopingFilter(ConstraintFilter):
    def __init__(self, cfg: Dict[str, Any], *, env=None) -> None:
        self.host_P = str(cfg.get("host_P", "P"))
        self.o_element = str(cfg.get("o_element", "O"))
        self.metal_only = set(cfg.get("metal_only", []))
        self.oxide_only = set(cfg.get("oxide_only", []))

    def filter_actions(
        self,
        *,
        actions: List[Tuple[Tuple[float, ...], Tuple[float, ...]]],
        cation_set: List[str],
        fraction_set: List[str],
        prior_groups: Optional[List[Dict[str, float]]] = None,
        **_: Any,
    ) -> List[Tuple[Tuple[float, ...], Tuple[float, ...]]]:
        from ..encoding import decode_one_hot

        if not actions:
            return actions
        # Categorical slots are single-element; only mask the O-form slot.
        if decode_one_hot(actions[0][0], cation_set) != self.o_element:
            return actions

        metal = self._metal(prior_groups)
        out = []
        for elem_oh, comp_oh in actions:
            o = float(decode_one_hot(comp_oh, fraction_set))
            if metal in self.metal_only and o > 0:
                continue
            if metal in self.oxide_only and o == 0:
                continue
            out.append((elem_oh, comp_oh))
        return out if out else actions

    def _metal(self, prior_groups: Optional[List[Dict[str, float]]]) -> Optional[str]:
        if not prior_groups:
            return None
        p_site = prior_groups[0]  # P-site is the first completed group
        metals = [k for k in p_site if k != self.host_P and p_site.get(k, 0) > 0]
        return metals[0] if len(metals) == 1 else None
