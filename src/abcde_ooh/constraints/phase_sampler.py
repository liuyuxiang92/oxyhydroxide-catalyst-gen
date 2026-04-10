"""Forward feasibility pruning for phase-constrained action spaces.

PhaseActionFilter wraps around ABCDEOOHEnv.allowed_actions() and removes
any action whose partial-composition cannot be completed into a valid
terminal composition of the requested phase(s). This guarantees 100% pass
rate without post-generation filtering.

Phase rules mirror check_primary_phase in primary_phase.py:
  - Ni / Co   : single primary ≥ 0.75; all others are dopants (< 0.25 total)
  - NiFe      : ni+fe ≥ 0.75, ni/(ni+fe) in [2/3, 3/4], fe ≤ ni; rest dopants
  - CoFe      : co+fe ≥ 0.75, fe ≤ co; rest dopants
  - NiFeCo    : ni+fe+co ≥ 0.75, fe ≤ max(ni, co); rest dopants
  - any       : passes if any of the five checks passes

Multiple target phases may be passed; an action is kept if its partial
composition can be completed into *any* of the listed phases.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set, Tuple, Union


class PhaseActionFilter:
    PHASES: Set[str] = {"Ni", "Co", "NiFe", "CoFe", "NiFeCo", "any"}

    def __init__(
        self,
        target_phase: Union[str, Sequence[str]],
        allowed_units: List[int],
        possible_sums_by_k: List[set],
        total_units: int = 20,
    ) -> None:
        if isinstance(target_phase, str):
            phases: List[str] = [target_phase]
        else:
            phases = list(target_phase)
        if not phases:
            raise ValueError("target_phase must not be empty.")
        for p in phases:
            if p not in self.PHASES:
                raise ValueError(f"Unknown target_phase '{p}'. Valid: {self.PHASES}")
        # If "any" is present, it subsumes everything else.
        if "any" in phases:
            phases = ["any"]
        self.target_phases: List[str] = phases
        # Backwards-compatible alias when a single phase is configured.
        self.target_phase: str = phases[0] if len(phases) == 1 else ",".join(phases)
        self._allowed_units: List[int] = list(allowed_units)
        self._allowed_units_set: Set[int] = set(allowed_units)
        self._possible_sums_by_k = possible_sums_by_k
        self._total_units = total_units

        # Dopant budget: total non-primary units must be < 0.25*total = 5 units
        # i.e. ≤ 4 units. With 5 elements each ≥1 unit on a 0.05 grid the
        # primary group must use ≥ 16 units and the dopant group ≤ 4 units.
        self._max_dopant_units: int = 4  # strictly < 5 units

        # Pre-compute valid terminal (primary, dopant) unit allocations
        self._valid_nife_pairs: List[Tuple[int, int]] = self._compute_binary_pairs(
            lambda ni, fe: (ni + fe >= 15) and ((2 / 3) <= ni / (ni + fe) <= (3 / 4)) and (fe <= ni)
        )
        self._valid_cofe_pairs: List[Tuple[int, int]] = self._compute_binary_pairs(
            lambda co, fe: (co + fe >= 15) and (fe <= co)
        )
        self._valid_nifeco_triples: List[Tuple[int, int, int]] = self._compute_ternary_triples(
            lambda ni, fe, co: (ni + fe + co >= 15)
            and (fe <= max(ni, co))
        )

    # ------------------------------------------------------------------
    # Pre-computation helpers
    # ------------------------------------------------------------------

    def _compute_binary_pairs(self, pred) -> List[Tuple[int, int]]:
        """Valid (p1_units, p2_units) terminal allocations, with 3 dopants each ≥1."""
        pairs: List[Tuple[int, int]] = []
        for p1 in self._allowed_units:
            for p2 in self._allowed_units:
                if p1 + p2 > self._total_units:
                    continue
                dopant_total = self._total_units - p1 - p2
                if dopant_total > self._max_dopant_units:
                    continue
                if dopant_total < 0:
                    continue
                # Must accommodate 3 more distinct elements each ≥ 1 unit
                if dopant_total not in self._possible_sums_by_k[3]:
                    continue
                if pred(p1, p2):
                    pairs.append((p1, p2))
        return pairs

    def _compute_ternary_triples(self, pred) -> List[Tuple[int, int, int]]:
        """Valid (p1, p2, p3) terminal allocations, with 2 dopants each ≥1."""
        triples: List[Tuple[int, int, int]] = []
        for p1 in self._allowed_units:
            for p2 in self._allowed_units:
                for p3 in self._allowed_units:
                    total_primary = p1 + p2 + p3
                    if total_primary > self._total_units:
                        continue
                    dopant_total = self._total_units - total_primary
                    if dopant_total > self._max_dopant_units:
                        continue
                    if dopant_total < 0:
                        continue
                    # Must accommodate 2 more distinct elements each ≥ 1 unit
                    if dopant_total not in self._possible_sums_by_k[2]:
                        continue
                    if pred(p1, p2, p3):
                        triples.append((p1, p2, p3))
        return triples

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def filter_actions(
        self,
        actions: List[Tuple],
        units_map: Dict[str, int],
        steps_left: int,
        allowed_units: List[int],
        possible_sums_by_k: List[set],
        cation_set: List[str],
        fraction_set: List[str],
    ) -> List[Tuple]:
        """Return subset of *actions* that can lead to a valid terminal phase."""
        from abcde_ooh.encoding import decode_one_hot

        filtered: List[Tuple] = []
        for action in actions:
            elem_oh, comp_oh = action
            elem = decode_one_hot(elem_oh, cation_set)
            comp_str = decode_one_hot(comp_oh, fraction_set)
            cand_units = int(round(float(comp_str) * 20))

            # Build hypothetical units_map after this action
            new_map = dict(units_map)
            new_map[elem] = cand_units
            budget_after = self._total_units - sum(new_map.values())

            if self._is_feasible(new_map, steps_left, budget_after):
                filtered.append(action)

        return filtered

    # ------------------------------------------------------------------
    # Per-phase feasibility
    # ------------------------------------------------------------------

    def _is_feasible(self, units_map: Dict[str, int], steps_left: int, budget_left: int) -> bool:
        for phase in self.target_phases:
            if self._is_feasible_single_phase(phase, units_map, steps_left, budget_left):
                return True
        return False

    def _is_feasible_single_phase(
        self,
        phase: str,
        units_map: Dict[str, int],
        steps_left: int,
        budget_left: int,
    ) -> bool:
        if phase == "any":
            return (
                self._is_completable_single_primary("Ni", units_map, steps_left, budget_left)
                or self._is_completable_single_primary("Co", units_map, steps_left, budget_left)
                or self._is_completable_binary("NiFe", units_map, steps_left, budget_left)
                or self._is_completable_binary("CoFe", units_map, steps_left, budget_left)
                or self._is_completable_nifeco(units_map, steps_left, budget_left)
            )
        if phase in {"Ni", "Co"}:
            return self._is_completable_single_primary(phase, units_map, steps_left, budget_left)
        if phase == "NiFe":
            return self._is_completable_binary("NiFe", units_map, steps_left, budget_left)
        if phase == "CoFe":
            return self._is_completable_binary("CoFe", units_map, steps_left, budget_left)
        if phase == "NiFeCo":
            return self._is_completable_nifeco(units_map, steps_left, budget_left)
        return False

    def _is_completable_single_primary(
        self,
        primary: str,
        units_map: Dict[str, int],
        steps_left: int,
        budget_left: int,
    ) -> bool:
        """Can the partial composition be completed into a valid single-primary phase?

        Rule: primary ≥ 15 units (≥0.75), all 4 others are dopants (≤4 total).
        With 5 elements total, primary=16 and each of 4 dopants=1 is the only solution.
        More precisely: primary ≥ 15, sum(others) ≤ 4.
        """
        primary_units = units_map.get(primary, 0)
        other_units = sum(v for k, v in units_map.items() if k != primary)

        if other_units > self._max_dopant_units:
            return False

        primary_chosen = primary in units_map
        n_others_placed = sum(1 for k in units_map if k != primary)
        # steps_left = steps that haven't been taken yet (after the action being evaluated)
        # total elements placed so far = len(units_map) = n_others_placed + (1 if primary_chosen)
        # total elements that will be placed = 5
        n_to_add = steps_left  # number of future steps remaining

        # We need to end with exactly 5 elements total placed.
        # n_to_add future steps add n_to_add elements.
        # Currently we have len(units_map) elements. Target = 5.
        # So n_to_add must equal 5 - len(units_map). That's guaranteed by the env.
        # We just need to check if a valid allocation exists.

        other_budget_remaining = self._max_dopant_units - other_units

        if primary_chosen:
            if primary_units < 15:
                return False
            # All remaining steps go to dopants
            n_dopants_to_add = n_to_add
            if n_dopants_to_add == 0:
                return budget_left == 0
            if budget_left < 0 or budget_left > other_budget_remaining:
                return False
            return budget_left in self._possible_sums_by_k[n_dopants_to_add]
        else:
            # Primary not yet placed; need to place it in one of the remaining steps
            # and the rest go to dopants.
            n_dopants_to_add = n_to_add - 1  # one step for primary, rest for dopants
            if n_dopants_to_add < 0:
                return False
            for primary_add in self._allowed_units:
                if primary_add < 15:
                    continue
                dopant_budget = budget_left - primary_add
                if dopant_budget < 0 or dopant_budget > other_budget_remaining:
                    continue
                if n_dopants_to_add == 0:
                    if dopant_budget == 0:
                        return True
                else:
                    if dopant_budget in self._possible_sums_by_k[n_dopants_to_add]:
                        return True
            return False

    def _is_completable_binary(
        self,
        phase: str,
        units_map: Dict[str, int],
        steps_left: int,
        budget_left: int,
    ) -> bool:
        """Can the partial composition be completed into a valid binary-primary phase?"""
        if phase == "NiFe":
            p1, p2 = "Ni", "Fe"
            valid_pairs = self._valid_nife_pairs
        else:  # CoFe
            p1, p2 = "Co", "Fe"
            valid_pairs = self._valid_cofe_pairs

        p1_units = units_map.get(p1, 0)
        p2_units = units_map.get(p2, 0)
        other_units = sum(v for k, v in units_map.items() if k not in {p1, p2})

        if other_units > self._max_dopant_units:
            return False

        p1_chosen = p1 in units_map
        p2_chosen = p2 in units_map
        n_primary_to_add = (0 if p1_chosen else 1) + (0 if p2_chosen else 1)
        n_dopants_to_add = steps_left - n_primary_to_add
        if n_dopants_to_add < 0:
            return False

        other_budget_remaining = self._max_dopant_units - other_units

        for (p1_final, p2_final) in valid_pairs:
            p1_add = p1_final - p1_units
            p2_add = p2_final - p2_units

            if p1_chosen and p1_add != 0:
                continue
            if p2_chosen and p2_add != 0:
                continue
            if not p1_chosen and p1_add not in self._allowed_units_set:
                continue
            if not p2_chosen and p2_add not in self._allowed_units_set:
                continue

            primary_budget_needed = (p1_add if not p1_chosen else 0) + (p2_add if not p2_chosen else 0)
            dopant_budget = budget_left - primary_budget_needed

            if dopant_budget < 0 or dopant_budget > other_budget_remaining:
                continue
            if n_dopants_to_add == 0:
                if dopant_budget == 0:
                    return True
            else:
                if dopant_budget in self._possible_sums_by_k[n_dopants_to_add]:
                    return True

        return False

    def _is_completable_nifeco(
        self,
        units_map: Dict[str, int],
        steps_left: int,
        budget_left: int,
    ) -> bool:
        """Can the partial composition be completed into a valid NiFeCo phase?"""
        ni_units = units_map.get("Ni", 0)
        fe_units = units_map.get("Fe", 0)
        co_units = units_map.get("Co", 0)
        other_units = sum(v for k, v in units_map.items() if k not in {"Ni", "Fe", "Co"})

        if other_units > self._max_dopant_units:
            return False

        ni_chosen = "Ni" in units_map
        fe_chosen = "Fe" in units_map
        co_chosen = "Co" in units_map

        n_primary_to_add = (0 if ni_chosen else 1) + (0 if fe_chosen else 1) + (0 if co_chosen else 1)
        n_dopants_to_add = steps_left - n_primary_to_add
        if n_dopants_to_add < 0:
            return False

        other_budget_remaining = self._max_dopant_units - other_units

        for (ni_final, fe_final, co_final) in self._valid_nifeco_triples:
            ni_add = ni_final - ni_units
            fe_add = fe_final - fe_units
            co_add = co_final - co_units

            if ni_chosen and ni_add != 0:
                continue
            if fe_chosen and fe_add != 0:
                continue
            if co_chosen and co_add != 0:
                continue
            if not ni_chosen and ni_add not in self._allowed_units_set:
                continue
            if not fe_chosen and fe_add not in self._allowed_units_set:
                continue
            if not co_chosen and co_add not in self._allowed_units_set:
                continue

            primary_budget_needed = (
                (ni_add if not ni_chosen else 0)
                + (fe_add if not fe_chosen else 0)
                + (co_add if not co_chosen else 0)
            )
            dopant_budget = budget_left - primary_budget_needed

            if dopant_budget < 0 or dopant_budget > other_budget_remaining:
                continue
            if n_dopants_to_add == 0:
                if dopant_budget == 0:
                    return True
            else:
                if dopant_budget in self._possible_sums_by_k[n_dopants_to_add]:
                    return True

        return False
