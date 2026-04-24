from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from .encoding import decode_one_hot, encode_choice
from .featurization import featurize_formula


@dataclass
class EpisodeStep:
    state_material_features: Sequence[float]
    state_step_onehot: Sequence[float]
    action_elem_onehot: Sequence[float]
    action_comp_onehot: Sequence[float]
    reward: float
    allowed_actions: List = field(default_factory=list)


def _format_fraction(units: int) -> str:
    return f"{units / 20:.2f}"


def _fractions_to_units(fractions: Sequence[str]) -> List[int]:
    out: List[int] = []
    for f in fractions:
        val = float(f)
        out.append(int(round(val * 20)))
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

    At each step the agent picks one element from ``cation_set`` (no repeats)
    and one fraction from ``fraction_set``.  All fractions must sum to exactly
    1.0 at the terminal step.  The ``_possible_sums_by_k`` table is precomputed
    so every action is guaranteed to leave a feasible path to a valid terminal
    composition — no invalid episodes are ever generated.

    Parameters
    ----------
    cation_set:
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
    """

    def __init__(
        self,
        *,
        cation_set: Sequence[str],
        fraction_set: Sequence[str] = DEFAULT_FRACTION_SET,
        anion_formula: str = "",
        n_components: int = 5,
        reward_fn: Callable[[str], float] | None = None,
        state_featurizer: Callable[[str], np.ndarray] = featurize_formula,
        phase_filter=None,
        total_units: int = 20,
    ) -> None:
        self.cation_set = list(cation_set)
        self.fraction_set = list(fraction_set)
        self.anion_formula = anion_formula
        self.n_components = n_components
        self.max_steps = n_components  # alias used in existing training code
        self.reward_fn = reward_fn or (lambda _formula: 0.0)
        self.state_featurizer = state_featurizer
        self.phase_filter = phase_filter

        self._total_units = total_units
        self._allowed_units = _fractions_to_units(self.fraction_set)
        self._possible_sums_by_k: List[set[int]] = [
            _possible_sums(self._allowed_units, k, self._total_units)
            for k in range(self.n_components + 1)
        ]

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
        state = "".join(f"{el}{_format_fraction(units)}" for el, units in items)
        return f"{state}{self.anion_formula}"

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

    def _allowed_fraction_units_now(self) -> List[int]:
        steps_left = self.n_components - self.counter
        remaining = self.remaining_units

        allowed: List[int] = []
        for u in self._allowed_units:
            if u > remaining:
                continue
            rem_after = remaining - u
            if rem_after in self._possible_sums_by_k[steps_left - 1]:
                allowed.append(u)
        return allowed

    def allowed_actions(self) -> List[Tuple[Tuple[float, ...], Tuple[float, ...]]]:
        """Return all feasibility-guaranteed (elem_onehot, comp_onehot) pairs."""
        if self.counter >= self.n_components:
            return []

        elems = [e for e in self.cation_set if e not in self._selected]
        units = self._allowed_fraction_units_now()

        actions: List[Tuple[Tuple[float, ...], Tuple[float, ...]]] = []
        for elem in elems:
            elem_oh = tuple(encode_choice(elem, self.cation_set).tolist())
            for u in units:
                comp = _format_fraction(u)
                comp_oh = tuple(encode_choice(comp, self.fraction_set).tolist())
                actions.append((elem_oh, comp_oh))

        if self.phase_filter is not None:
            actions = self.phase_filter.filter_actions(
                actions=actions,
                units_map=self._units_map,
                steps_left=self.n_components - self.counter - 1,
                allowed_units=self._allowed_units,
                possible_sums_by_k=self._possible_sums_by_k,
                cation_set=self.cation_set,
                fraction_set=self.fraction_set,
            )

        return actions

    def sample_random_action(self) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        actions = self.allowed_actions()
        if not actions:
            raise RuntimeError("No valid actions available.")
        return random.choice(actions)

    def step(self, action: Tuple[Tuple[float, ...], Tuple[float, ...]]) -> None:
        elem_oh, comp_oh = action
        elem = decode_one_hot(elem_oh, self.cation_set)
        comp_str = decode_one_hot(comp_oh, self.fraction_set)

        if elem in self._selected:
            raise ValueError(f"Repeated element '{elem}' is not allowed.")

        comp_units = int(round(float(comp_str) * self._total_units))
        if comp_units not in self._allowed_units:
            raise ValueError("Composition not in allowed fraction set.")

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
