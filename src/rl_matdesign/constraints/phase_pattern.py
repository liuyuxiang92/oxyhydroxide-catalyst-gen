"""Declarative phase-pattern constraint filter.

A YAML-driven alternative to writing a custom :class:`ConstraintFilter`
subclass. Supports the most common composition rule shapes:

- **required**: ``{element: [min_frac, max_frac]}`` — each named element must
  end up in this fractional range.
- **forbidden**: ``[element, ...]`` — these elements must be 0 in the terminal.
- **primary_sum**: ``{elements: [...], min: ..., max: ...}`` — sum of listed
  elements must be in [min, max].
- **dopant_sum**: ``{elements: ALL_OTHERS, max: ...}`` — sum of all elements
  NOT named in ``required`` must be at most this value.
- **ratios**: ``[{num: el, den: [el, ...], min: ..., max: ...}]`` — ratio
  ``frac(num) / sum(frac(el) for el in den)`` bounded.

A composition satisfies a *pattern* if all of the pattern's rules hold. A
composition satisfies the **filter** if it satisfies at least one pattern
(OR across the pattern list).

Action filtering is currently exact at the **terminal step only** (when one
action fully determines the composition). At intermediate steps it passes
actions through without pruning — invalid intermediate trajectories simply
fail at the terminal. For complex cross-element inequalities (e.g. OOH's
``Fe <= max(Ni, Co)``) write a dedicated Python ConstraintFilter; this
declarative filter is for the simpler-rule majority.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple


ALL_OTHERS = "ALL_OTHERS"


def _normalize_pattern(p: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and return a normalized copy of a single pattern dict."""
    out: Dict[str, Any] = {
        "name": p.get("name", ""),
        "required": {},
        "forbidden": [],
        "primary_sum": None,
        "dopant_sum": None,
        "ratios": [],
    }
    if "required" in p:
        for el, rng in p["required"].items():
            lo, hi = float(rng[0]), float(rng[1])
            if lo > hi:
                raise ValueError(f"pattern {out['name']!r}: required[{el!r}] lo>hi: {lo}>{hi}")
            out["required"][el] = (lo, hi)
    if "forbidden" in p:
        out["forbidden"] = list(p["forbidden"])
    if "primary_sum" in p:
        ps = p["primary_sum"]
        out["primary_sum"] = {
            "elements": list(ps["elements"]),
            "min": float(ps.get("min", 0.0)),
            "max": float(ps.get("max", 1.0)),
        }
    if "dopant_sum" in p:
        ds = p["dopant_sum"]
        out["dopant_sum"] = {
            "elements": ds["elements"],  # may be ALL_OTHERS sentinel string
            "min": float(ds.get("min", 0.0)),
            "max": float(ds.get("max", 1.0)),
        }
    if "ratios" in p:
        for r in p["ratios"]:
            out["ratios"].append({
                "num": r["num"],
                "den": list(r["den"]),
                "min": float(r.get("min", 0.0)),
                "max": float(r.get("max", 1.0)),
            })
    return out


def _matches_pattern(comp: Dict[str, float], pattern: Dict[str, Any]) -> bool:
    """Return True if *comp* satisfies every rule in *pattern*."""
    for el, (lo, hi) in pattern["required"].items():
        if not (lo - 1e-9 <= comp.get(el, 0.0) <= hi + 1e-9):
            return False

    for el in pattern["forbidden"]:
        if comp.get(el, 0.0) > 1e-9:
            return False

    if pattern["primary_sum"] is not None:
        ps = pattern["primary_sum"]
        s = sum(comp.get(e, 0.0) for e in ps["elements"])
        if not (ps["min"] - 1e-9 <= s <= ps["max"] + 1e-9):
            return False

    if pattern["dopant_sum"] is not None:
        ds = pattern["dopant_sum"]
        elements = ds["elements"]
        if elements == ALL_OTHERS:
            primary = set(pattern["required"])
            elements = [e for e in comp if e not in primary]
        s = sum(comp.get(e, 0.0) for e in elements)
        if not (ds["min"] - 1e-9 <= s <= ds["max"] + 1e-9):
            return False

    for r in pattern["ratios"]:
        num = comp.get(r["num"], 0.0)
        den = sum(comp.get(e, 0.0) for e in r["den"])
        if den < 1e-12:
            return False  # Avoid division by zero — pattern not applicable.
        ratio = num / den
        if not (r["min"] - 1e-9 <= ratio <= r["max"] + 1e-9):
            return False

    return True


class PhasePatternFilter:
    """Filter actions / validate terminal compositions against a YAML rule set."""

    def __init__(self, patterns: List[Dict[str, Any]]) -> None:
        if not patterns:
            raise ValueError("PhasePatternFilter needs at least one pattern.")
        self.patterns: List[Dict[str, Any]] = [_normalize_pattern(p) for p in patterns]

    # ------------------------------------------------------------------
    # Post-hoc validation
    # ------------------------------------------------------------------

    def check_composition(self, composition: Dict[str, float]) -> Tuple[bool, str]:
        """Return ``(matches, label)``.

        ``label`` is the name of the first matched pattern, or ``""`` if none.
        """
        for p in self.patterns:
            if _matches_pattern(composition, p):
                return True, str(p.get("name", ""))
        return False, ""

    # ------------------------------------------------------------------
    # Action filtering (called by CompositionEnv during rollout)
    # ------------------------------------------------------------------

    def filter_actions(
        self,
        *,
        actions: List[Tuple[Tuple[float, ...], Tuple[float, ...]]],
        units_map: Dict[str, int],
        steps_left: int,
        allowed_units: Sequence[int],
        possible_sums_by_k,  # noqa: ARG002
        species_set: List[str],
        fraction_set: List[str],
        **_: Any,
    ) -> List[Tuple[Tuple[float, ...], Tuple[float, ...]]]:
        # Only do a thorough check when this action is the LAST one in the
        # episode (steps_left counts post-action remaining steps). Earlier
        # actions pass through — failed trajectories simply yield an
        # unmatched terminal that gets filtered post-hoc.
        if steps_left > 0:
            return actions

        # Reconstruct total_units from full grid: sum of any complete episode = total.
        # We get it from the partial units_map + the last action's amount.
        # We don't know total_units directly, so derive from the maximum reachable
        # via remaining units; but easier: assume the env normalizes to sum=1.0.
        out: List[Tuple[Tuple[float, ...], Tuple[float, ...]]] = []
        # Used units across all picks so far.
        used = sum(units_map.values())
        # Need total_units. The simplest valid recovery: any action being the
        # LAST means the action's amount = total_units - used.
        # We get total_units by looking at the first action's amount + used + 0
        # (since steps_left == 0, this action completes the episode).
        for elem_oh, comp_oh in actions:
            try:
                elem = species_set[list(elem_oh).index(1.0)]
                comp_str = fraction_set[list(comp_oh).index(1.0)]
            except ValueError:
                continue
            amount = float(comp_str)
            # Build the would-be terminal composition (fractions).
            # Convert units_map to fractions: each unit is amount / (used + amount_int) — no,
            # we need to know total_units in units. Easiest: total_units in fractional space = 1.0,
            # so each unit = 1.0 / total_units. We don't have total_units here directly, but the
            # sum of (units / total_units) over all elements must equal 1.0.
            # Derive total_units from this action: amount + sum_of_existing_fractions = 1.0
            # → total_units = round((used + amount_int) / 1.0). But amount_int isn't known
            # without total_units. So instead: build comp using the action's amount float
            # (already in [0, 1]) and the existing units_map scaled by (1 - amount) / used
            # if used > 0, else just {elem: amount}.
            comp: Dict[str, float] = {}
            if used > 0:
                # Existing entries: their fractions are units / total_units. We know
                # sum(existing fractions) = 1.0 - amount (since this action completes
                # the episode). So scale: frac[el] = units_map[el] / used * (1 - amount).
                scale = (1.0 - amount) / used
                for el, u in units_map.items():
                    comp[el] = u * scale
            comp[elem] = comp.get(elem, 0.0) + amount

            ok, _label = self.check_composition(comp)
            if ok:
                out.append((elem_oh, comp_oh))
        return out
