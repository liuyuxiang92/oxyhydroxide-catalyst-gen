from __future__ import annotations

import random
import re
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from .encoding import decode_one_hot, encode_choice
from .featurization import featurize_formula


EpisodeStyle = Literal["element_then_amount", "fixed_order_amount"]


@dataclass
class EpisodeStep:
    state_material_features: Sequence[float]
    state_step_onehot: Sequence[float]
    action_elem_onehot: Sequence[float]
    action_comp_onehot: Sequence[float]
    reward: float
    allowed_actions: List = field(default_factory=list)


def _decimals_of(fraction_set: Sequence[str], minimum: int = 2) -> int:
    """Decimal places this grid's codes use — at least *minimum* (2).

    The precision is read off the ``fraction_set`` tokens themselves rather than
    configured, because those tokens are exactly what ``encode_choice`` must match.
    Returns 2 for every grid whose entries have <= 2 decimals (0.05, 0.01, …), so
    existing scenarios keep byte-identical codes; a 0.125 grid returns 3.
    """
    d = minimum
    for token in fraction_set:
        text = str(token)
        if "." in text:
            d = max(d, len(text.split(".", 1)[1]))
    return d


def _format_fraction(units: int, total_units: int = 20, decimals: int = 2) -> str:
    """Render ``units/total_units`` as a fraction code.

    *decimals* must match the precision of the env's ``fraction_set``: the result is
    fed straight to ``encode_choice`` against that alphabet, so a 0.125 grid rendered
    at 2 decimals yields "0.12", which is not a member and raises. Defaults to 2, the
    historical behaviour and the right answer for every 20/50/100-unit grid.
    """
    return f"{units / total_units:.{decimals}f}"


def _fractions_to_units(fractions: Sequence[str], total_units: int = 20) -> List[int]:
    out: List[int] = []
    for f in fractions:
        val = float(f)
        out.append(int(round(val * total_units)))
    return out


def _possible_sums(units: Sequence[int], k: int, max_total: int) -> set[int]:
    sums = {0}
    for _ in range(k):
        next_sums: set[int] = set()
        for s in sums:
            for u in units:
                t = s + u
                if t <= max_total:
                    next_sums.add(t)
        sums = next_sums
    return sums


def _step_one_hot(step: int, max_steps: int) -> np.ndarray:
    if not (1 <= step <= max_steps):
        raise ValueError("Step out of range")
    v = np.zeros(max_steps, dtype=float)
    v[step - 1] = 1.0
    return v


DEFAULT_FRACTION_SET: List[str] = [
    "0.05", "0.10", "0.15", "0.20", "0.25", "0.30", "0.35",
    "0.40", "0.45", "0.50", "0.55", "0.60", "0.65", "0.70", "0.75", "0.80",
]


class CompositionEnv:
    """General N-step environment for constrained composition design.

    At each step the agent picks one element from ``species_set`` (no repeats)
    and one fraction from ``fraction_set``.  All fractions must sum to exactly
    1.0 at the terminal step.  The ``_possible_sums_by_k`` table is precomputed
    so every action is guaranteed to leave a feasible path to a valid terminal
    composition — no invalid episodes are ever generated.

    Parameters
    ----------
    species_set:
        Ordered list of candidate element symbols.
    fraction_set:
        Ordered list of fraction strings on a regular grid (e.g. "0.05" … "0.80").
        Must be multiples of 1/total_units (default total_units=20 → 0.05 grid).
    anion_formula:
        Appended to the terminal formula string.  Pass ``""`` for metallic alloys,
        ``"O2H1"`` for oxyhydroxides, ``"O3"`` for perovskite oxides, etc.
    n_components:
        Number of distinct cations in the final composition (episode length).
    reward_fn:
        Callable ``(formula: str) -> float`` evaluated only at the terminal step.
        Defaults to a zero reward (useful for testing).
    state_featurizer:
        Callable ``(partial_formula: str) -> np.ndarray``.
        Defaults to Magpie composite features via matminer.
    phase_filter:
        Optional :class:`~rl_matdesign.constraints.base.ConstraintFilter` instance
        that prunes ``allowed_actions`` at each step.  Pass ``None`` (default) for
        no domain constraint beyond fraction feasibility.
    total_units:
        Internal resolution of the fraction grid.  Default 20 means the grid runs
        in steps of 1/20 = 0.05.
    element_bounds:
        Optional ``{element: (min_fraction, max_fraction)}`` overriding the
        global ``fraction_set`` range on a per-element basis.  Bounds are in
        the same fractional units as ``fraction_set`` (e.g. ``(0.45, 0.90)``).
        Elements absent from the dict inherit ``[0, 1]``.  Supported by BOTH
        episode styles. Under ``"element_then_amount"`` a lower bound > 0 makes
        that element **mandatory** (an absent element has fraction 0, which would
        violate ``lo>0``), and the legal amounts at each step are pruned so the
        remaining steps can always complete to a valid bounded composition.
    episode_style:
        ``"element_then_amount"`` (default, current behavior) — the agent picks
        both element and amount at each step.  ``"fixed_order_amount"`` — every
        element in ``species_set`` appears in that fixed order, and the agent
        only chooses an amount per step.  ``n_components`` is then forced to
        ``len(species_set)`` and the last step's amount is determined by the
        residual budget.
    """

    def __init__(
        self,
        *,
        species_set: Sequence[str],
        fraction_set: Sequence[str] = DEFAULT_FRACTION_SET,
        anion_formula: str = "",
        n_components: int = 5,
        reward_fn: Callable[[str], float] | None = None,
        state_featurizer: Callable[[str], np.ndarray] = featurize_formula,
        phase_filter=None,
        total_units: int = 20,
        element_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        episode_style: EpisodeStyle = "element_then_amount",
    ) -> None:
        self.species_set = list(species_set)
        self.fraction_set = list(fraction_set)
        self.anion_formula = anion_formula
        self.reward_fn = reward_fn or (lambda _formula: 0.0)
        self.state_featurizer = state_featurizer
        self.phase_filter = phase_filter
        # See IntegerRatioEnv: skip phase_filter when disabled (constrain_training).
        self.constraints_enabled = True
        self.episode_style = episode_style

        if episode_style not in ("element_then_amount", "fixed_order_amount"):
            raise ValueError(
                f"episode_style must be 'element_then_amount' or "
                f"'fixed_order_amount', got {episode_style!r}."
            )
        if episode_style == "fixed_order_amount":
            self.n_components = len(self.species_set)
            if n_components != len(self.species_set):
                warnings.warn(
                    f"episode_style='fixed_order_amount' forces n_components="
                    f"len(species_set)={len(self.species_set)}; ignoring requested "
                    f"n_components={n_components}.",
                    UserWarning, stacklevel=2,
                )
        else:
            self.n_components = n_components
        self.max_steps = self.n_components  # alias used in existing training code

        self._total_units = total_units
        # Precision of this grid's codes; see _format_fraction.
        self._frac_decimals = _decimals_of(self.fraction_set)
        self._allowed_units = _fractions_to_units(self.fraction_set, self._total_units)
        self._possible_sums_by_k: List[set[int]] = [
            _possible_sums(self._allowed_units, k, self._total_units)
            for k in range(self.n_components + 1)
        ]

        # Per-element unit bounds. `_element_unit_bounds` is indexed by species_set
        # position (used by fixed_order_amount); `_element_unit_bounds_by_sym`
        # maps symbol -> (lo_u, hi_u) (used by element_then_amount, where the
        # element at each step is the agent's free choice, not its position).
        self._element_unit_bounds: Optional[List[Tuple[int, int]]] = None
        self._element_unit_bounds_by_sym: Optional[Dict[str, Tuple[int, int]]] = None
        if element_bounds is not None:
            unknown = sorted(set(element_bounds) - set(self.species_set))
            if unknown:
                warnings.warn(
                    f"element_bounds keys not in species_set are ignored: {unknown}",
                    UserWarning, stacklevel=2,
                )
            bounds: List[Tuple[int, int]] = []
            for el in self.species_set:
                lo, hi = element_bounds.get(el, (0.0, 1.0))
                lo_u = int(round(float(lo) * self._total_units))
                hi_u = int(round(float(hi) * self._total_units))
                if lo_u > hi_u:
                    raise ValueError(
                        f"element_bounds[{el!r}] has lower > upper: {lo_u} > {hi_u}."
                    )
                bounds.append((lo_u, hi_u))
            self._element_unit_bounds = bounds
            self._element_unit_bounds_by_sym = dict(zip(self.species_set, bounds))

            if self.episode_style == "fixed_order_amount":
                # Every cation appears exactly once, so feasibility is the simple
                # sum-of-bounds window over the full set.
                sum_min = sum(lo for lo, _ in bounds)
                sum_max = sum(hi for _, hi in bounds)
                if not (sum_min <= self._total_units <= sum_max):
                    raise ValueError(
                        f"element_bounds is infeasible: sum(mins)={sum_min}, "
                        f"sum(maxes)={sum_max}, total_units={self._total_units}. "
                        "Need sum(mins) <= total_units <= sum(maxes)."
                    )
            else:
                # element_then_amount: the agent picks n_components distinct
                # cations. Any element with lo_u > 0 is MANDATORY (an absent
                # element has fraction 0, which would violate its lower bound), so
                # there must be room for all mandatory cations and a completion
                # that hits total_units must exist.
                n_mandatory = sum(1 for lo, _ in bounds if lo > 0)
                if n_mandatory > self.n_components:
                    raise ValueError(
                        f"element_bounds requires {n_mandatory} mandatory cations "
                        f"(lower bound > 0) but n_components={self.n_components}. "
                        "Raise n_components or relax some lower bounds to 0."
                    )
                if not self._completable(self._total_units, self.n_components, bounds):
                    raise ValueError(
                        "element_bounds is infeasible for element_then_amount: no "
                        f"choice of {self.n_components} cations can sum to "
                        f"total_units={self._total_units} within bounds."
                    )

        self.state: str = ""
        self.counter: int = 0
        self.path: List[EpisodeStep] = []
        self._selected: set[str] = set()
        self._used_units: int = 0
        self._units_map: Dict[str, int] = {}

    def initialize(self) -> None:
        """Reset the environment to the start of a new episode."""
        self.state = ""
        self.counter = 0
        self.path = []
        self._selected = set()
        self._used_units = 0
        self._units_map = {}

    @property
    def remaining_units(self) -> int:
        return self._total_units - self._used_units

    @property
    def terminal_formula(self) -> str:
        """Canonical formula string at the terminal step (empty if not terminal)."""
        if self.counter != self.n_components:
            return ""

        comp = self.cation_fractions()
        items: List[Tuple[str, int]] = []
        for el, frac in comp.items():
            units = int(round(float(frac) * self._total_units))
            if units <= 0:
                continue
            items.append((el, units))

        # Major cations first; tie-break alphabetically for determinism.
        items.sort(key=lambda t: (-t[1], t[0]))
        state = "".join(
            f"{el}{_format_fraction(units, self._total_units, self._frac_decimals)}"
            for el, units in items
        )
        return f"{state}{self.anion_formula}"

    def current_state_features(self) -> np.ndarray:
        """Material features of the current (partial) state, for action selection.

        Uniform accessor shared with :class:`MultiGroupEnv` so training/generation
        code never reaches into env-specific state representation.
        """
        return np.asarray(self.state_featurizer(self.state), dtype=float)

    def cation_fractions(self) -> Dict[str, float]:
        if not self.state:
            return {}
        parts = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]+)", self.state)
        out: Dict[str, float] = {}
        for el, frac in parts:
            out[el] = out.get(el, 0.0) + float(frac)
        return out

    def terminal_cation_fractions(self) -> Dict[str, float]:
        if self.counter != self.n_components:
            raise RuntimeError("terminal_cation_fractions called before terminal step")
        return self.cation_fractions()

    def terminal_comp_key(self) -> tuple:
        """Canonical hashable key for dedup/visit-tracking.

        Quantises fractions to ``self._total_units`` and sorts alphabetically.
        Used by ``training.py`` as the cross-env dedup interface.
        """
        items = []
        for el, frac in self.terminal_cation_fractions().items():
            units = int(round(float(frac) * self._total_units))
            if units > 0:
                items.append((str(el), units))
        return tuple(sorted(items))

    @staticmethod
    def _completable(target: int, k_slots: int, pool: Sequence[Tuple[int, int]]) -> bool:
        """Can exactly ``k_slots`` distinct elements from ``pool`` sum to ``target``?

        ``pool`` is the list of selectable elements' ``(lo_u, hi_u)`` unit bounds.
        On a step-1 unit grid the achievable sums of any chosen k-subset form the
        full integer interval ``[Σlo, Σhi]`` (each element varies independently by
        1), so the subset-sum-with-bounds feasibility reduces to a window check:

        * Every element with ``lo_u > 0`` is **mandatory** (it cannot be left out
          without violating its own lower bound), so all of them must be among the
          ``k_slots`` chosen — hence ``#mandatory <= k_slots`` and they all count
          toward ``Σlo``/``Σhi``.
        * The remaining ``k_slots - #mandatory`` slots are filled from the optional
          elements (``lo_u == 0``). They never raise the floor (their ``lo`` is 0);
          to maximise the reachable ceiling we take the optionals with the largest
          ``hi``. ``target`` is reachable iff ``Σlo <= target <= Σhi_max``.
        """
        mandatory = [(lo, hi) for lo, hi in pool if lo > 0]
        optional_his = sorted((hi for lo, hi in pool if lo == 0), reverse=True)
        if len(mandatory) > k_slots:
            return False
        if len(pool) < k_slots:                 # not enough distinct elements
            return False
        k_opt = k_slots - len(mandatory)
        lo_total = sum(lo for lo, _ in mandatory)
        hi_total = sum(hi for _, hi in mandatory) + sum(optional_his[:k_opt])
        return lo_total <= target <= hi_total

    def _allowed_units_for_symbol(self, elem: str) -> List[int]:
        """Allowed fraction-units for picking *elem* now under element_then_amount.

        Enforces ``elem``'s own ``[lo, hi]`` bound and only keeps amounts that
        leave a budget the remaining steps can still complete within every
        unselected element's bounds (via :meth:`_completable`).
        """
        steps_left = self.n_components - self.counter
        remaining = self.remaining_units
        bsym = self._element_unit_bounds_by_sym
        lo_u, hi_u = (bsym.get(elem, (0, self._total_units)) if bsym is not None
                      else (0, self._total_units))
        # Pool the remaining steps draw from: unselected cations other than `elem`.
        pool = [
            (bsym.get(e, (0, self._total_units)) if bsym is not None else (0, self._total_units))
            for e in self.species_set
            if e not in self._selected and e != elem
        ]
        allowed: List[int] = []
        for u in self._allowed_units:
            if u < lo_u or u > hi_u or u > remaining:
                continue
            if self._completable(remaining - u, steps_left - 1, pool):
                allowed.append(u)
        return allowed

    def _allowed_fraction_units_now(self, *, for_element_idx: Optional[int] = None) -> List[int]:
        steps_left = self.n_components - self.counter
        remaining = self.remaining_units

        # Per-element bounds clamp (only used in fixed_order_amount mode).
        if self._element_unit_bounds is not None and for_element_idx is not None:
            lo_u, hi_u = self._element_unit_bounds[for_element_idx]
            # Range of remaining elements' (sum_min, sum_max) determines feasibility.
            rest_min = sum(
                self._element_unit_bounds[j][0]
                for j in range(for_element_idx + 1, self.n_components)
            )
            rest_max = sum(
                self._element_unit_bounds[j][1]
                for j in range(for_element_idx + 1, self.n_components)
            )
            allowed: List[int] = []
            for u in self._allowed_units:
                if u < lo_u or u > hi_u:
                    continue
                if u > remaining:
                    continue
                rem_after = remaining - u
                if rem_after < rest_min or rem_after > rest_max:
                    continue
                # On a step=1 unit grid every integer in [rest_min, rest_max] is
                # reachable; the global possible_sums_by_k check is a safety net.
                if rem_after in self._possible_sums_by_k[steps_left - 1]:
                    allowed.append(u)
            return allowed

        # Default global feasibility (current behavior, byte-identical when
        # element_bounds is None).
        allowed = []
        for u in self._allowed_units:
            if u > remaining:
                continue
            rem_after = remaining - u
            if rem_after in self._possible_sums_by_k[steps_left - 1]:
                allowed.append(u)
        return allowed

    def allowed_actions(
        self, *, prior_groups: Optional[List[Dict[str, float]]] = None
    ) -> List[Tuple[Tuple[float, ...], Tuple[float, ...]]]:
        """Return all feasibility-guaranteed (elem_onehot, comp_onehot) pairs.

        ``prior_groups`` is forwarded to the constraint filter for cross-group
        coupling under :class:`MultiGroupEnv`; it is ``None`` (and ignored) for
        standalone single-group use, so existing callers are unaffected.
        """
        if self.counter >= self.n_components:
            return []

        actions: List[Tuple[Tuple[float, ...], Tuple[float, ...]]] = []
        if self.episode_style == "fixed_order_amount":
            elem = self.species_set[self.counter]
            elem_units = {elem: self._allowed_fraction_units_now(for_element_idx=self.counter)}
        elif self._element_unit_bounds_by_sym is not None:
            # element_then_amount with per-element bounds: allowed amounts depend
            # on which element is being picked, so compute units per candidate.
            elem_units = {
                e: self._allowed_units_for_symbol(e)
                for e in self.species_set if e not in self._selected
            }
        else:
            shared = self._allowed_fraction_units_now()
            elem_units = {e: shared for e in self.species_set if e not in self._selected}

        for elem, units in elem_units.items():
            elem_oh = tuple(encode_choice(elem, self.species_set).tolist())
            for u in units:
                comp = _format_fraction(u, self._total_units, self._frac_decimals)
                comp_oh = tuple(encode_choice(comp, self.fraction_set).tolist())
                actions.append((elem_oh, comp_oh))

        if self.phase_filter is not None and self.constraints_enabled:
            kw = dict(
                actions=actions,
                units_map=self._units_map,
                steps_left=self.n_components - self.counter - 1,
                allowed_units=self._allowed_units,
                possible_sums_by_k=self._possible_sums_by_k,
                species_set=self.species_set,
                fraction_set=self.fraction_set,
            )
            # Only forward prior_groups when it is actually set (i.e. from
            # MultiGroupEnv). Standalone single-group use passes None, so filters
            # are called exactly as before — external/user filters that predate
            # the cross-group hook keep working unchanged.
            if prior_groups is not None:
                kw["prior_groups"] = prior_groups
            actions = self.phase_filter.filter_actions(**kw)

        return actions

    def sample_random_action(self) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        actions = self.allowed_actions()
        if not actions:
            raise RuntimeError("No valid actions available.")
        return random.choice(actions)

    def step(self, action: Tuple[Tuple[float, ...], Tuple[float, ...]]) -> None:
        elem_oh, comp_oh = action
        elem = decode_one_hot(elem_oh, self.species_set)
        comp_str = decode_one_hot(comp_oh, self.fraction_set)

        if self.episode_style == "fixed_order_amount":
            expected = self.species_set[self.counter]
            if elem != expected:
                raise ValueError(
                    f"episode_style='fixed_order_amount' expects element "
                    f"{expected!r} at step {self.counter}, got {elem!r}."
                )
        elif elem in self._selected:
            raise ValueError(f"Repeated element '{elem}' is not allowed.")

        comp_units = int(round(float(comp_str) * self._total_units))
        if comp_units not in self._allowed_units:
            raise ValueError("Composition not in allowed fraction set.")

        if self._element_unit_bounds is not None:
            if self.episode_style == "fixed_order_amount":
                lo_u, hi_u = self._element_unit_bounds[self.counter]
            else:
                lo_u, hi_u = self._element_unit_bounds_by_sym.get(elem, (0, self._total_units))
            if not (lo_u <= comp_units <= hi_u):
                raise ValueError(
                    f"Amount {comp_units} for element {elem!r} violates bounds "
                    f"[{lo_u}, {hi_u}] in units of 1/{self._total_units}."
                )

        steps_left = self.n_components - self.counter
        remaining = self.remaining_units
        if comp_units > remaining:
            raise ValueError("Action exceeds remaining fraction budget.")
        if (remaining - comp_units) not in self._possible_sums_by_k[steps_left - 1]:
            raise ValueError("Action makes valid terminal composition impossible.")

        current_allowed = self.allowed_actions()
        old_state = self.state

        if self.counter == 0:
            self.state = f"{elem}{comp_str}"
        else:
            self.state = f"{self.state}{elem}{comp_str}"

        self._selected.add(elem)
        self._units_map[elem] = comp_units
        self._used_units += comp_units
        self.counter += 1

        reward = 0.0
        if self.counter == self.n_components:
            reward = float(self.reward_fn(self.terminal_formula))

        s_material = self.state_featurizer(old_state)
        s_step = _step_one_hot(self.counter, self.n_components)

        self.path.append(
            EpisodeStep(
                state_material_features=s_material,
                state_step_onehot=s_step,
                action_elem_onehot=elem_oh,
                action_comp_onehot=comp_oh,
                reward=reward,
                allowed_actions=current_allowed,
            )
        )
