"""SSEDopingFilter — action masks for the doped-Li6PS6 multi-group env.

One filter class, two roles (configured per group via ``role:``):

* ``role: p_site`` — the P-site group ``[<metals>, P]``, ``element_then_amount``,
  ``n_components: 2``. First step: only a dopant **metal** at a **level** fraction
  (e.g. 0.02–0.08). Last step: only the host ``P`` (takes the complement).

* ``role: s_site`` — the S-site group ``[O, Cl, S]``, ``fixed_order_amount`` (so
  the element at each step is pinned, and we mask by step):
    - **O step**: the O fraction encodes the *form* — ``o_off`` (metal form, no
      oxygen) or ``o_on`` (metal-oxide form). Which is allowed depends on the
      P-site metal's category (read from ``prior_groups``): metal-only → only
      ``o_off``; oxide-only → only ``o_on``; both → either.
    - **Cl step**: only the configured halide ``cl_values``.
    - **S step**: residual, unconstrained.

Filters tolerate ``prior_groups=None`` (the inner env's own bookkeeping calls);
in that case the O mask is permissive (the result is unused, and real masking
happens when MultiGroupEnv supplies the completed P-site composition).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .base import ConstraintFilter


class SSEDopingFilter(ConstraintFilter):
    def __init__(self, cfg: Dict[str, Any], *, env=None) -> None:
        self.role = str(cfg.get("role", "")).lower()
        if self.role == "p_site":
            self.host_P = str(cfg.get("host_P", "P"))
            self.levels = {str(x) for x in cfg["levels"]}
        elif self.role == "s_site":
            self.o_off = str(cfg["o_off"])
            self.o_on = str(cfg["o_on"])
            self.cl_values = {str(x) for x in cfg["cl_values"]}
            self.host_P = str(cfg.get("host_P", "P"))
            self.metal_only = set(cfg.get("metal_only", []))
            self.oxide_only = set(cfg.get("oxide_only", []))
            # "both" = any metal in neither set.
        else:
            raise ValueError(
                f"sse_doping needs role: 'p_site' or 's_site', got {self.role!r}."
            )

    # ------------------------------------------------------------------

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
        prior_groups: Optional[List[Dict[str, float]]] = None,
        **_: Any,
    ) -> List[Tuple[Tuple[float, ...], Tuple[float, ...]]]:
        from ..encoding import decode_one_hot

        if not actions:
            return actions

        if self.role == "p_site":
            is_last = steps_left == 0
            out = []
            for elem_oh, comp_oh in actions:
                el = decode_one_hot(elem_oh, cation_set)
                fr = decode_one_hot(comp_oh, fraction_set)
                if is_last:
                    if el != self.host_P:
                        continue
                else:
                    if el == self.host_P or fr not in self.levels:
                        continue
                out.append((elem_oh, comp_oh))
            return out if out else actions

        # role == s_site: fixed element order, so all actions share one element.
        el = decode_one_hot(actions[0][0], cation_set)
        if el == "O":
            allowed_o = self._allowed_o(self._metal_from_prior(prior_groups))
            keep = allowed_o
        elif el == "Cl":
            keep = self.cl_values
        else:  # S residual — no constraint
            return actions

        out = []
        for elem_oh, comp_oh in actions:
            if decode_one_hot(comp_oh, fraction_set) in keep:
                out.append((elem_oh, comp_oh))
        return out if out else actions

    # ------------------------------------------------------------------

    def _allowed_o(self, metal: Optional[str]) -> set:
        if metal is None:  # inner-bookkeeping call; be permissive (unused result)
            return {self.o_off, self.o_on}
        if metal in self.metal_only:
            return {self.o_off}
        if metal in self.oxide_only:
            return {self.o_on}
        return {self.o_off, self.o_on}  # both

    def _metal_from_prior(self, prior_groups: Optional[List[Dict[str, float]]]) -> Optional[str]:
        if not prior_groups:
            return None
        p_site = prior_groups[0]  # P-site is the first completed group
        metals = [k for k in p_site if k != self.host_P and p_site.get(k, 0) > 0]
        return metals[0] if len(metals) == 1 else None
