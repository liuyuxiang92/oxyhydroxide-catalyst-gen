"""SSEDopingFilter — mask the doped-Li₆PS₆ S-site oxygen *form* by the metal category.

On the categorical S-site, each ``O``-form slot is a flag: ``0`` = metal form (no
oxygen), ``> 0`` = metal-oxide form. Which forms are legal depends on the P-site
dopant metal it belongs to (read from ``prior_groups``):

* ``metal_only`` metals (e.g. Ru) -> only ``O = 0`` (metal form);
* ``oxide_only`` metals -> only ``O > 0`` (oxide form);
* "both" (any metal in neither list) -> either.

**Multi-dopant.** With several dopant metals the S-site carries one O-form slot
per metal (e.g. slots ``O_a``, ``O_b``). The slots are paired to the dopant
metals in **sorted** order — the i-th O-form slot (in config order) governs the
i-th alphabetically-sorted P-site metal — which is the only order-invariant
handle once same-element picks have merged into a ``{metal: fraction}`` map.
Slots with no corresponding metal (fewer distinct metals than slots) are left
unconstrained. The Cl slot is never touched.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .base import ConstraintFilter


class SSEDopingFilter(ConstraintFilter):
    def __init__(self, cfg: Dict[str, Any], *, env=None) -> None:
        self.host_P = str(cfg.get("host_P", cfg.get("host", {}).get("P", "P")))
        self.o_element = str(cfg.get("o_element", "O"))
        self.metal_only = set(cfg.get("metal_only", []))
        self.oxide_only = set(cfg.get("oxide_only", []))
        self.o_form_slots = self._discover_o_slots(cfg)

    def _discover_o_slots(self, cfg: Dict[str, Any]) -> List[str]:
        """Ordered O-form slot names, read from the S-site categorical group.

        ``cfg`` may be the S-site **group spec** itself (carries ``choices`` — the
        runtime path, since each group's filter is resolved with its own spec) or
        the **full config** (carries ``groups``). Both are supported.
        """
        choices = cfg.get("choices")
        if choices is None:
            s_group = str(cfg.get("s_site_group", "S_site"))
            categoricals = [g for g in (cfg.get("groups") or []) if g.get("kind") == "categorical"]
            named = [g for g in categoricals if str(g.get("name", "")) == s_group]
            target = named[0] if named else (categoricals[0] if len(categoricals) == 1 else None)
            choices = (target or {}).get("choices", []) if target else []
        slots = [
            str(c.get("name", c["element"]))
            for c in (choices or [])
            if str(c["element"]) == self.o_element
        ]
        return slots or [self.o_element]

    def filter_actions(
        self,
        *,
        actions: List[Tuple[Tuple[float, ...], Tuple[float, ...]]],
        species_set: List[str],
        fraction_set: List[str],
        prior_groups: Optional[List[Dict[str, float]]] = None,
        **_: Any,
    ) -> List[Tuple[Tuple[float, ...], Tuple[float, ...]]]:
        from ..encoding import decode_one_hot

        if not actions:
            return actions
        # Each categorical slot is single-named; only mask O-form slots.
        slot_name = decode_one_hot(actions[0][0], species_set)
        if slot_name not in self.o_form_slots:
            return actions

        metal = self._metal_for_slot(slot_name, prior_groups)
        if metal is None:
            return actions  # inert slot (no corresponding dopant metal)
        out = []
        for elem_oh, comp_oh in actions:
            o = float(decode_one_hot(comp_oh, fraction_set))
            if metal in self.metal_only and o > 0:
                continue
            if metal in self.oxide_only and o == 0:
                continue
            out.append((elem_oh, comp_oh))
        return out if out else actions

    def _metal_for_slot(
        self, slot_name: str, prior_groups: Optional[List[Dict[str, float]]]
    ) -> Optional[str]:
        if not prior_groups:
            return None
        p_site = prior_groups[0]  # P-site is the first completed group
        metals = sorted(k for k in p_site if k != self.host_P and p_site.get(k, 0) > 0)
        idx = self.o_form_slots.index(slot_name)
        return metals[idx] if idx < len(metals) else None
