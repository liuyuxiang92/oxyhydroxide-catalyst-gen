"""Stoichiometry-aware charge-neutrality on the *whole* composition.

This is the single source of truth shared by the in-episode
:class:`~rl_matdesign.constraints.smact_filter.SMACTChargeFilter` and the
post-episode validation in :func:`rl_matdesign.training.generate_candidates`.

The check answers: *given these exact element amounts, does some assignment of
one oxidation state per element make the **amount-weighted** charge sum zero?*
(and, with ``use_pauling``, is that same assignment electronegativity-sensible?).
This is fundamentally different from ``smact.screening.smact_validity``, which
asks whether *some* integer stoichiometry could be neutral — we already have a
fixed stoichiometry (the agent picked it), so we must weight by the real amounts.

Oxidation states and Pauling electronegativities come from **smact**
(``smact.Element(sym).oxidation_states`` / ``.pauling_eneg``), matching the
reference benchmark ``RL_materials_generation/constraints/checkers.py``. We use
smact's *tables* with our own amount-weighted search rather than
``smact.neutral_ratios`` (whose return type drifts across smact versions). smact
is therefore required whenever a ``smact_charge`` / ``pauling_en`` constraint is
configured (the imports stay lazy so other scenarios don't need it).

Why weighted (and why this fixes the old filter)
------------------------------------------------
The previous filter screened only the *cation element identities* against a
hardcoded anion stoichiometry, with an **unweighted** oxidation-state sum that
ignored the picked amounts entirely (and, for oxides, double-counted O). Here we
build the complete formula — cations + every anion, at their actual amounts —
integerize it, and test ``Σ amount_i · ox_i == 0`` exactly.

All-metal compositions are treated as valid alloys (``allow_alloys``), and
unknown / oxidation-state-less elements make the check lenient (return ``True``),
matching the original "allow and let the reward model judge" behavior.
"""
from __future__ import annotations

import itertools
import warnings
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

# Guard against pathological state-space blow-ups (distinct elements each with
# many oxidation states). Typical compositions are 5-9 elements x 2-6 states,
# i.e. well under this cap; SSE-like 9-element formulas stay in the low millions.
_MAX_COMBOS = 5_000_000


def _to_amounts(comp: Union[Dict[str, float], str, "Composition"]) -> Dict[str, float]:
    """Normalize a dict / formula string / pymatgen Composition to ``{el: amount}``."""
    if isinstance(comp, dict):
        return {str(k): float(v) for k, v in comp.items() if float(v) > 0.0}
    from pymatgen.core import Composition

    c = comp if isinstance(comp, Composition) else Composition(comp)
    return {str(el): float(n) for el, n in c.get_el_amt_dict().items() if n > 0.0}


def _integerize(amounts: Dict[str, float]) -> Dict[str, int]:
    """Reduce real amounts to the smallest integer stoichiometry.

    Integer amounts + integer oxidation states make the neutrality test exact
    (the weighted sum is an integer, so the tolerance collapses to "== 0").
    """
    from pymatgen.core import Composition

    comp = Composition(amounts)
    int_formula, _ = comp.get_integer_formula_and_factor()
    int_comp = Composition(int_formula)
    return {str(el): int(round(n)) for el, n in int_comp.get_el_amt_dict().items() if n > 0}


@lru_cache(maxsize=None)
def _oxidation_states(symbol: str) -> Tuple[int, ...]:
    """Oxidation states for *symbol* from **smact** (the broad table).

    Uses ``smact.Element(symbol).oxidation_states`` — the same table the reference
    benchmark (``RL_materials_generation/constraints/checkers.py``) validates
    against, so our in-episode/post-episode neutrality matches it. Cached because
    constructing a smact ``Element`` reads data files and the in-episode filter
    queries this per candidate action. Returns ``()`` for unknown symbols / no
    data (caller -> be lenient).
    """
    try:
        import smact

        states = smact.Element(symbol).oxidation_states
        return tuple(int(s) for s in states) if states else ()
    except Exception:
        return ()


@lru_cache(maxsize=None)
def _pauling_eneg(symbol: str) -> Optional[float]:
    """Pauling electronegativity for *symbol* from smact, or ``None`` if missing."""
    try:
        import smact

        eneg = smact.Element(symbol).pauling_eneg
        return float(eneg) if eneg is not None else None
    except Exception:
        return None


def charge_neutral(
    comp: Union[Dict[str, float], str, "Composition"],
    *,
    tol: float = 0.5,
    allow_alloys: bool = True,
    use_pauling: bool = False,
) -> bool:
    """Return True if *comp* admits a charge-neutral oxidation-state assignment.

    Parameters
    ----------
    comp:
        The **full** composition (cations + all anions) as a ``{element: amount}``
        mapping, a formula string, or a pymatgen ``Composition``. Amounts may be
        fractional; they are reduced to integers internally.
    tol:
        Absolute tolerance on the weighted charge sum. With integerized amounts
        and integer oxidation states this is effectively an exact "== 0" test.
    allow_alloys:
        When True, an all-metal composition is considered valid (an alloy), as in
        ``smact.screening.smact_validity(include_alloys=True)``.
    use_pauling:
        When True, a charge-neutral assignment must **also** pass smact's Pauling
        electronegativity test (``smact.screening.pauling_test``) to count as
        valid — mirrors the reference ``check_electronegativity`` (neutral AND
        electronegativity-sensible). Missing electronegativity data -> lenient.
    """
    amounts = _to_amounts(comp)
    if not amounts:
        return True

    elems = list(amounts)

    # All-metal -> valid alloy (no ionic charge balance required).
    if allow_alloys:
        try:
            from pymatgen.core import Element

            if all(Element(e).is_metal for e in elems):
                return True
        except Exception:
            pass

    try:
        int_amounts = _integerize(amounts)
    except Exception:
        int_amounts = {e: max(int(round(a)), 1) for e, a in amounts.items()}

    elems = list(int_amounts)
    counts = [int_amounts[e] for e in elems]
    state_lists = [_oxidation_states(e) for e in elems]

    # Unknown element (no oxidation states) -> can't judge -> be lenient.
    if any(not s for s in state_lists):
        return True

    # Pauling test needs electronegativities; missing data -> skip the EN check.
    enegs = [_pauling_eneg(e) for e in elems] if use_pauling else None
    do_pauling = use_pauling and enegs is not None and None not in enegs

    n_combos = 1
    for s in state_lists:
        n_combos *= len(s)
    if n_combos > _MAX_COMBOS:
        warnings.warn(
            f"charge_neutral: {n_combos} oxidation-state combinations for "
            f"{elems} exceeds the cap ({_MAX_COMBOS}); skipping check (allow).",
            RuntimeWarning,
        )
        return True

    for combo in itertools.product(*state_lists):
        if abs(sum(c * o for c, o in zip(counts, combo))) <= tol:
            if not do_pauling:
                return True
            try:
                from smact.screening import pauling_test

                if pauling_test(combo, enegs, elems):
                    return True
            except Exception:
                # Can't evaluate EN -> fall back to neutrality-only acceptance.
                return True
    return False


def parse_formula(formula: Optional[str]) -> Dict[str, float]:
    """Parse a fixed-anion / scaffold formula string (e.g. ``"O2H1"``) to counts.

    Returns ``{}`` for an empty / falsy formula.
    """
    if not formula:
        return {}
    from pymatgen.core import Composition

    return {str(el): float(n) for el, n in Composition(formula).get_el_amt_dict().items()}


def template_scaffold(poscar_path: str, site_symbol: str) -> Tuple[Dict[str, float], int]:
    """Parse a template POSCAR into ``(fixed_counts, n_site_atoms)``.

    This is "the whole formula before substitution": every non-``site_symbol``
    species (host + anions, e.g. ``{La: 1, O: 3}`` for ``LaBO3``) becomes a fixed
    count, and the number of ``site_symbol`` placeholder atoms (the working sites
    the RL agent fills) is returned so callers can normalize the fixed part to a
    per-working-formula-unit basis. Reuses ASE just like
    :func:`rl_matdesign.utils.structure.substitute_sites`.
    """
    from ase.io import read as ase_read

    atoms = ase_read(poscar_path)
    symbols = list(atoms.get_chemical_symbols())
    n_sites = sum(1 for s in symbols if s == site_symbol)
    if n_sites == 0:
        raise ValueError(
            f"No '{site_symbol}' placeholder sites in {poscar_path}; "
            "scaffold_site_symbol must match the working-site element."
        )
    fixed: Dict[str, float] = {}
    for s in symbols:
        if s == site_symbol:
            continue
        fixed[s] = fixed.get(s, 0.0) + 1.0
    return fixed, n_sites
