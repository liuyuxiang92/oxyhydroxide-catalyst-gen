"""ChainConstraintFilter — apply multiple constraint filters in sequence.

Composes N child constraint filters so each step's action list is pruned by
every filter in turn. Useful when a system has stacked rules — e.g. an oxide
that needs both charge neutrality (``SMACTChargeFilter``) **and** "step N = O"
(``LastStepElementFilter``).

YAML configuration
------------------
.. code-block:: yaml

    constraint_filter: chain
    filters:
      - constraint_filter: last_step_element        # cheap rule first
        required_elements: ["O"]
        nonzero_ratio_at_last: true
        reserve_for_last: true
      - constraint_filter: smact_charge             # expensive rule second
        smact_anions:
          - {symbol: O, charge: -2, stoich: 3.0}

Each entry under ``filters:`` is a self-contained sub-config carrying its
own ``constraint_filter:`` key plus whatever kwargs that child filter
needs. The chain runs filters in list order; each filter's output feeds the
next filter's input.

Children are resolved through the same :func:`resolve_constraint` used at
the top level, so any combination of built-in short names and user FQN
plug-ins (``pkg.module:ClassName``) works inside a chain — including nesting
another ``chain`` if you really want to.

Safety
------
Every built-in filter already honours the project-wide convention "if
filtering empties the list, return the original input." The chain adds a
top-level fallback: if any intermediate step or the final result is empty,
the *pre-chain* action list is returned. Combined, the agent is never
stranded with zero legal actions.

Performance note: put cheap, identity-based filters (e.g.
``last_step_element``) before expensive ones (e.g. ``smact_charge`` whose
SMACT screen iterates over oxidation-state combinations). Ordering does
not affect the final set, only the work done.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .base import ConstraintFilter


class ChainConstraintFilter(ConstraintFilter):
    """Apply a sequence of :class:`ConstraintFilter` instances in order."""

    def __init__(self, cfg: Dict[str, Any], *, env=None) -> None:
        from ..registry import resolve_constraint

        children_cfgs = cfg.get("filters")
        if not children_cfgs:
            raise ValueError(
                "ChainConstraintFilter requires a non-empty 'filters' list. "
                "Example: filters: [{constraint_filter: smact_charge, ...}, "
                "{constraint_filter: last_step_element, ...}]"
            )
        if not isinstance(children_cfgs, list):
            raise ValueError(
                f"'filters' must be a list of dicts, got "
                f"{type(children_cfgs).__name__}."
            )

        self.children: List[ConstraintFilter] = []
        for i, sub in enumerate(children_cfgs):
            if not isinstance(sub, dict):
                raise ValueError(
                    f"filters[{i}] must be a dict, got {type(sub).__name__}."
                )
            sub_cfg = dict(sub)
            sub_kind = sub_cfg.pop("constraint_filter", None)
            if not sub_kind:
                raise ValueError(
                    f"filters[{i}] is missing 'constraint_filter' "
                    "(short name or 'pkg.module:ClassName' FQN)."
                )
            child = resolve_constraint(sub_kind, sub_cfg, env=env)
            if child is None:
                raise ValueError(
                    f"filters[{i}].constraint_filter={sub_kind!r} resolved "
                    "to None — expected a ConstraintFilter instance."
                )
            self.children.append(child)

    def filter_actions(self, *, actions, **kw):
        original = actions
        for child in self.children:
            actions = child.filter_actions(actions=actions, **kw)
            if not actions:
                return original
        return actions if actions else original
