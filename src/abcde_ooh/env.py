from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from .encoding import decode_one_hot, encode_choice
from .featurization import featurize_formula


DEFAULT_CATION_SET: List[str] = [
    "Mg",
    "Ca",
    "Sc",
    "Ti",
    "Cu",
    "Sr",
    "Y",
    "Zr",
    "Hf",
    "Bi",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Co",
    "Ni",
    "Fe",
    "Mn",
]

DEFAULT_FRACTIONS: List[str] = [
    "0.05",
    "0.10",
    "0.15",
    "0.20",
    "0.25",
    "0.30",
    "0.35",
    "0.40",
    "0.45",
    "0.50",
    "0.55",
    "0.60",
    "0.65",
    "0.70",
    "0.75",
    "0.80",
]


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


@dataclass
class EpisodeStep:
    state_material_features: Sequence[float]
    state_step_onehot: Sequence[float]
    action_elem_onehot: Sequence[float]
    action_comp_onehot: Sequence[float]
    reward: float


class ABCDEOOHEnv:
    """Constrained 5-step environment for ABCDEOOH-like compositions."""

    def __init__(
        self,
        *,
        cation_set: Sequence[str] = DEFAULT_CATION_SET,
        fraction_set: Sequence[str] = DEFAULT_FRACTIONS,
        anion_formula: str = "O2H1",
        max_steps: int = 5,
        reward_fn: Callable[[str], float] | None = None,
        state_featurizer: Callable[[str], np.ndarray] = featurize_formula,
    ) -> None:
        if max_steps != 5:
            raise ValueError("This environment is designed for exactly 5 steps (A..E).")

        self.cation_set = list(cation_set)
        self.fraction_set = list(fraction_set)
        self.anion_formula = anion_formula
        self.max_steps = max_steps
        self.reward_fn = reward_fn or (lambda _formula: 0.0)
        self.state_featurizer = state_featurizer

        self._allowed_units = _fractions_to_units(self.fraction_set)
        self._total_units = 20
        self._possible_sums_by_k: List[set[int]] = [
            _possible_sums(self._allowed_units, k, self._total_units) for k in range(self.max_steps + 1)
        ]

        self.state: str = ""
        self.counter: int = 0
        self.path: List[EpisodeStep] = []
        self._selected: set[str] = set()
        self._used_units: int = 0

    def initialize(self) -> None:
        self.state = ""
        self.counter = 0
        self.path = []
        self._selected = set()
        self._used_units = 0

    @property
    def remaining_units(self) -> int:
        return self._total_units - self._used_units

    @property
    def terminal_formula(self) -> str:
        return f"{self.state}{self.anion_formula}" if self.counter == self.max_steps else ""

    def cation_fractions(self) -> Dict[str, float]:
        if not self.state:
            return {}
        parts = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]+)", self.state)
        out: Dict[str, float] = {}
        for el, frac in parts:
            out[el] = out.get(el, 0.0) + float(frac)
        return out

    def terminal_cation_fractions(self) -> Dict[str, float]:
        if self.counter != self.max_steps:
            raise RuntimeError("terminal_cation_fractions called before terminal")
        return self.cation_fractions()

    def _allowed_fraction_units_now(self) -> List[int]:
        steps_left = self.max_steps - self.counter
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
        if self.counter >= self.max_steps:
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

        comp_units = int(round(float(comp_str) * 20))
        if comp_units not in self._allowed_units:
            raise ValueError("Composition not allowed")

        steps_left = self.max_steps - self.counter
        remaining = self.remaining_units
        if comp_units > remaining:
            raise ValueError("Action exceeds remaining budget")
        if (remaining - comp_units) not in self._possible_sums_by_k[steps_left - 1]:
            raise ValueError("Action makes terminal sum impossible")

        old_state = self.state

        if self.counter == 0:
            self.state = f"{elem}{comp_str}"
        else:
            self.state = f"{self.state}{elem}{comp_str}"

        self._selected.add(elem)
        self._used_units += comp_units
        self.counter += 1

        reward = 0.0
        if self.counter == self.max_steps:
            reward = float(self.reward_fn(self.terminal_formula))

        s_material = self.state_featurizer(old_state)
        s_step = _step_one_hot(self.counter, self.max_steps)

        self.path.append(
            EpisodeStep(
                state_material_features=s_material,
                state_step_onehot=s_step,
                action_elem_onehot=elem_oh,
                action_comp_onehot=comp_oh,
                reward=reward,
            )
        )
