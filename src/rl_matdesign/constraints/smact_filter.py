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

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .base import ConstraintFilter


class SMACTChargeFilter(ConstraintFilter):
    """Prune final-step actions that would make the *whole formula* non-neutral.

    At the final step this builds the **complete** composition — every element
    the agent picked (including any anion it picks itself, e.g. O in the oxide
    configs) merged with a fixed *scaffold* (host + anions that are NOT part of
    the optimised sites, e.g. perovskite's ``La`` + ``O`` from the template, or
    an oxyhydroxide's ``O2H1``) — and tests amount-weighted charge neutrality via
    :func:`rl_matdesign.constraints.charge.charge_neutral`. Unlike the original
    unweighted cation-set screen, this respects the real picked amounts.

    Parameters
    ----------
    scaffold_fixed:
        ``{element: count}`` of fixed (non-working-site) elements taken from the
        "whole formula before substitution" — e.g. ``{La: 1, O: 3}`` parsed from a
        perovskite template, or ``{O: 2, H: 1}`` from an ``anion_formula``. Counts
        are normalized to a per-working-formula-unit basis using ``scaffold_n_sites``.
    scaffold_n_sites:
        Number of working (placeholder) sites the fixed counts were measured
        against — used to normalize ``scaffold_fixed`` to one working formula unit.
        Default 1 (e.g. ``anion_formula`` is already per formula unit).
    anions:
        **Deprecated** ``[{symbol, charge, stoich}, ...]`` list (old oxide configs).
        Retained for backward compatibility: any anion symbol that is *not* in the
        env's ``species_set`` (i.e. a genuinely external anion) is folded into the
        scaffold at its ``stoich``; anions that the agent actually picks (e.g. O in
        the oxide species_set) are ignored here and taken from the real pick.

    When ``smact_charge`` is configured, generation also drops any candidate whose
    whole formula is not charge-neutral (post-episode), so ``generated.csv`` holds
    only neutral compositions.
    """

    def __init__(
        self,
        anions: Optional[List[Dict[str, Any]]] = None,
        *,
        scaffold_fixed: Optional[Dict[str, float]] = None,
        scaffold_n_sites: int = 1,
        allow_alloys: bool = True,
        tol: float = 0.5,
        threshold: int = 1,
        use_pauling: bool = False,
    ) -> None:
        self.allow_alloys = bool(allow_alloys)
        self.tol = float(tol)
        self.threshold = threshold
        self.use_pauling = bool(use_pauling)

        # Explicit scaffold (from scaffold_formula / scaffold_poscar / anion_formula),
        # normalized to one working formula unit.
        n = max(int(scaffold_n_sites), 1)
        self.scaffold_per_fu: Dict[str, float] = (
            {str(e): float(c) / n for e, c in scaffold_fixed.items()} if scaffold_fixed else {}
        )

        # Deprecated anion list -> resolved lazily against species_set in
        # filter_actions (we only know which anions are agent-picked there).
        self._deprecated_anions: List[Dict[str, Any]] = []
        if anions:
            warnings.warn(
                "SMACTChargeFilter: 'smact_anions' is deprecated. The check now uses "
                "the real picked amounts plus a 'scaffold_formula'/'scaffold_poscar'. "
                "External anions in smact_anions are still honored for compatibility.",
                DeprecationWarning,
            )
            for entry in anions:
                if not all(k in entry for k in ("symbol", "charge", "stoich")):
                    raise ValueError(
                        f"Each anion entry must have 'symbol', 'charge', 'stoich'. Got: {entry}"
                    )
                self._deprecated_anions.append(
                    {"symbol": str(entry["symbol"]), "stoich": float(entry["stoich"])}
                )

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
        species_set: List[str],
        fraction_set: List[str],
        **_: Any,
    ) -> List[Tuple[Tuple[float, ...], Tuple[float, ...]]]:
        # Only enforce at the final step (the composition is complete only then).
        if steps_left > 0:
            return actions

        import numpy as np

        from ..encoding import decode_one_hot
        from .charge import charge_neutral

        # External anions (not agent-picked) contribute to the scaffold; anions the
        # agent picks itself (in species_set, e.g. O for oxides) come from the pick.
        species = set(species_set)
        scaffold = dict(self.scaffold_per_fu)
        for a in self._deprecated_anions:
            if a["symbol"] not in species:
                scaffold[a["symbol"]] = scaffold.get(a["symbol"], 0.0) + a["stoich"]

        filtered = []
        for elem_oh, comp_oh in actions:
            elem = decode_one_hot(elem_oh, species_set)
            cand_units = float(allowed_units[int(np.asarray(comp_oh).argmax())])
            full = self._full_composition(elem, cand_units, units_map, scaffold)
            if charge_neutral(
                full,
                tol=self.tol,
                allow_alloys=self.allow_alloys,
                use_pauling=self.use_pauling,
            ):
                filtered.append((elem_oh, comp_oh))

        # Safety: never strand the agent with zero legal actions.
        return filtered if filtered else actions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _full_composition(
        self,
        elem: str,
        cand_units: float,
        units_map: Dict[str, int],
        scaffold: Dict[str, float],
    ) -> Dict[str, float]:
        """Assemble the complete composition for the candidate final pick.

        Working-site picks = ``units_map`` plus the candidate. With no scaffold
        (oxides, where the agent also picks O) the working picks *are* the whole
        formula. With a scaffold, the working picks are normalized to one formula
        unit and the fixed scaffold is added on top.
        """
        working: Dict[str, float] = {k: float(v) for k, v in units_map.items()}
        working[elem] = working.get(elem, 0.0) + cand_units

        if not scaffold:
            return working

        total = sum(working.values())
        if total <= 0:
            return working
        full = {e: v / total for e, v in working.items()}
        for e, c in scaffold.items():
            full[e] = full.get(e, 0.0) + c
        return full


class ElectronegativityFilter(SMACTChargeFilter):
    """Charge-neutrality **plus** Pauling electronegativity (smact ``pauling_test``).

    Identical whole-formula machinery as :class:`SMACTChargeFilter`, but defaults
    ``use_pauling=True`` so a final pick survives only if the complete composition
    admits an assignment that is *both* charge-neutral *and* electronegativity-
    sensible. This is the ``pauling_en`` constraint — a separate on/off flag that
    can be chained after (or used instead of) ``smact_charge``. Because it is
    strictly stronger than neutrality, using it alone already guarantees neutral
    output.
    """

    def __init__(self, *args: Any, use_pauling: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, use_pauling=use_pauling, **kwargs)
