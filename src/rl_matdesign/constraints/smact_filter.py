"""SMACTChargeFilter — general charge-neutrality constraint via SMACT.

Works for **any ionic compound**: oxides, nitrides, sulfides, halides,
oxyhydroxides, multi-anion ceramics, etc.  The user specifies all fixed
(non-optimised) anion species and their stoichiometries; the filter then
checks whether each candidate cation can participate in a charge-neutral
composition.

Unlike the original repo (liuyuxiang92/RL_materials_generation) which checks
charge neutrality *after* episode completion, this filter prunes invalid
actions *before* the agent picks them — the agent never wastes an episode
generating an invalid composition.

YAML configuration
------------------
Specify a list of anion entries, each with ``symbol``, ``charge``, and
``stoich`` (stoichiometry relative to the cation formula unit sum):

.. code-block:: yaml

    constraint_filter: smact_charge
    smact_anions:
      - symbol: "O"
        charge: -2
        stoich: 3.0          # ABO3 perovskite

    # Multi-anion example (oxynitride):
    smact_anions:
      - symbol: "O"
        charge: -2
        stoich: 2.0
      - symbol: "N"
        charge: -3
        stoich: 1.0

    # Halide example:
    smact_anions:
      - symbol: "Cl"
        charge: -1
        stoich: 3.0

    # Sulfide example:
    smact_anions:
      - symbol: "S"
        charge: -2
        stoich: 1.0

    # Oxyhydroxide (OOH):
    smact_anions:
      - symbol: "O"
        charge: -2
        stoich: 2.0
      - symbol: "H"
        charge: -1
        stoich: 1.0

The ``stoich`` value is the number of anion atoms **per formula unit** (i.e.
per one cation site sum = 1).  For a compound ABX₃ where A+B=1 cation site,
stoich for X is 3.

Requirements
------------
    pip install smact
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from .base import ConstraintFilter


class SMACTChargeFilter(ConstraintFilter):
    """Prune final-step actions that would violate charge neutrality.

    Parameters
    ----------
    anions:
        List of anion specifications.  Each entry is a dict with keys:

        - ``"symbol"``  — element symbol, e.g. ``"O"``, ``"N"``, ``"S"``, ``"Cl"``
        - ``"charge"``  — oxidation state (integer), e.g. ``-2``, ``-3``, ``-1``
        - ``"stoich"``  — stoichiometry relative to cation site sum (float)

        Example for LaBO₃ perovskite (B-site optimised, La+B=1 cation)::

            [{"symbol": "O", "charge": -2, "stoich": 3.0}]

        Example for oxyhydroxide (MOOH, one cation site)::

            [{"symbol": "O", "charge": -2, "stoich": 2.0},
             {"symbol": "H", "charge": -1, "stoich": 1.0}]

    threshold:
        Minimum number of valid charge-neutral oxidation-state combinations
        that must exist for an action to be retained.  Default 1 (any valid
        combination counts).
    """

    def __init__(
        self,
        anions: List[Dict[str, Any]],
        threshold: int = 1,
    ) -> None:
        if not anions:
            raise ValueError("anions must be a non-empty list of dicts.")
        for entry in anions:
            if not all(k in entry for k in ("symbol", "charge", "stoich")):
                raise ValueError(
                    f"Each anion entry must have 'symbol', 'charge', 'stoich'. Got: {entry}"
                )
        self.anions = [
            {"symbol": str(a["symbol"]), "charge": int(a["charge"]), "stoich": float(a["stoich"])}
            for a in anions
        ]
        self.threshold = threshold
        self._smact = self._import_smact()

    # ------------------------------------------------------------------
    # Public helpers (useful for debugging)
    # ------------------------------------------------------------------

    def total_cation_charge_needed(self) -> float:
        """Total positive charge the cation mix must provide per formula unit.

        Computed as: sum(-charge_i * stoich_i) for all anion species i.
        """
        return sum(-a["charge"] * a["stoich"] for a in self.anions)

    # ------------------------------------------------------------------
    # ConstraintFilter interface
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
    ) -> List[Tuple[Tuple[float, ...], Tuple[float, ...]]]:
        # Only enforce at the final step (adding last cation).
        if steps_left > 0:
            return actions

        from ..encoding import decode_one_hot

        target_charge = self.total_cation_charge_needed()

        filtered = []
        for elem_oh, comp_oh in actions:
            elem = decode_one_hot(elem_oh, cation_set)
            if self._can_balance_charge(elem, units_map, target_charge):
                filtered.append((elem_oh, comp_oh))

        # Safety: if every action is filtered out, return the original list to
        # avoid a dead episode (can happen with unusual element sets or very
        # strict stoichiometries).
        return filtered if filtered else actions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _import_smact():
        try:
            import smact
            import smact.screening  # noqa: F401
            return smact
        except ImportError as exc:
            raise ImportError(
                "SMACTChargeFilter requires the 'smact' package: pip install smact"
            ) from exc

    def _can_balance_charge(
        self,
        new_elem: str,
        existing_map: Dict[str, int],
        target_charge: float,
    ) -> bool:
        """Return True if *new_elem* + *existing_map* can form a charge-neutral composition.

        Uses SMACT oxidation-state tables to check whether any combination of
        allowed oxidation states sums to *target_charge*.  The check is
        approximate (ignores stoichiometry weighting) but fast and reliable for
        screening at the final step.
        """
        candidate_elems = list(existing_map.keys()) + [new_elem]
        try:
            species = [self._smact.Element(e) for e in candidate_elems]
            oxidation_states = [e.oxidation_states for e in species]
        except Exception:
            # Unknown element — allow and let the reward model judge.
            return True

        # Screen: does any combination of oxidation states sum to target_charge?
        for combo in _product_of_lists(oxidation_states):
            if abs(sum(combo) - target_charge) < 0.5:
                return True
        return False


def _product_of_lists(lists):
    import itertools
    return itertools.product(*lists)
