"""IntegerRatioEnv — reference-compatible RL env for oxide composition design.

Mirrors the rules from the reference DQN/PGN repos:
- 80-element cation set (including ``O``).
- Ratio set = integer digits ``["0", ..., "9"]``; no sum-to-1 constraint.
- Fixed 5-step episode; each step picks one ``(element, digit)`` pair.
- No uniqueness requirement — an element may be picked multiple times
  (pymatgen ``Composition`` merges repeated keys).
- Oxide rule (enforced via ``LastStepElementFilter``): step 5 selects ``O``
  with a nonzero digit.

The terminal formula is built by raw concatenation — e.g. ``"Fe3Ti2Co1Ni4O5"`` —
and passed to the reward function / predictor, which parses it through
``pymatgen.Composition`` so the features are normalised automatically.

This class is API-compatible with :class:`~rl_matdesign.env.CompositionEnv`
(same ``EpisodeStep`` schema, same ``allowed_actions`` signature, same
``step`` return contract) so the training loops in ``training.py`` work
without modification.
"""
from __future__ import annotations

import random
import re
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from .encoding import decode_one_hot, encode_choice
from .env import EpisodeStep
from .featurization import featurize_formula


DEFAULT_RATIO_SET: List[str] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


def _step_one_hot(step: int, max_steps: int) -> np.ndarray:
    if not (1 <= step <= max_steps):
        raise ValueError("Step out of range")
    v = np.zeros(max_steps, dtype=float)
    v[step - 1] = 1.0
    return v


class IntegerRatioEnv:
    """Reference-style oxide env with integer digit ratios and no sum-to-1.

    Parameters
    ----------
    cation_set:
        Ordered list of candidate element symbols.  Must include any element
        required by the constraint filter (e.g. ``"O"`` for oxides).
    ratio_set:
        Ordered list of integer-digit strings, default ``["0", ..., "9"]``.
    n_components:
        Number of episode steps (= number of ``(element, digit)`` tokens).
    reward_fn:
        Callable ``(formula: str) -> float`` evaluated at the terminal step.
        The *formula* is a raw empirical string such as ``"Ba1Ti2O3"`` — pass
        it through ``pymatgen.Composition`` to get normalised fractions.
    state_featurizer:
        Callable ``(partial_formula: str) -> np.ndarray``.  Defaults to the
        matminer Magpie feature pipeline.
    phase_filter:
        Optional :class:`~rl_matdesign.constraints.base.ConstraintFilter`
        to prune ``allowed_actions`` at each step (e.g.
        :class:`LastStepElementFilter` for the "step 5 = O" rule).
    """

    def __init__(
        self,
        *,
        cation_set: Sequence[str],
        ratio_set: Sequence[str] = DEFAULT_RATIO_SET,
        n_components: int = 5,
        reward_fn: Callable[[str], float] | None = None,
        state_featurizer: Callable[[str], np.ndarray] = featurize_formula,
        phase_filter=None,
    ) -> None:
        self.cation_set = list(cation_set)
        # ``fraction_set`` alias so downstream code (scripts, filters) that
        # looks at ``env.fraction_set`` keeps working.
        self.fraction_set = list(ratio_set)
        self.ratio_set = self.fraction_set
        self.anion_formula = ""
        self.n_components = n_components
        self.max_steps = n_components
        self.reward_fn = reward_fn or (lambda _formula: 0.0)
        self.state_featurizer = state_featurizer
        self.phase_filter = phase_filter

        # Sentinel: ``_total_units = None`` tells ``_comp_key`` to fall back to
        # a formula-string key (patched in ``training.py``).
        self._total_units = None

        self.state: str = ""
        self.counter: int = 0
        self.path: List[EpisodeStep] = []
        self._digits_map: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Episode control
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        self.state = ""
        self.counter = 0
        self.path = []
        self._digits_map = {}

    # ------------------------------------------------------------------
    # Composition / formula helpers
    # ------------------------------------------------------------------

    @property
    def terminal_formula(self) -> str:
        """Raw empirical formula string built so far (non-empty only at the terminal step)."""
        if self.counter != self.n_components:
            return ""
        return self.state

    def cation_digits(self) -> Dict[str, int]:
        """Return running sum of integer digits per element in the current state."""
        if not self.state:
            return {}
        parts = re.findall(r"([A-Z][a-z]?)([0-9]+)", self.state)
        out: Dict[str, int] = {}
        for el, d in parts:
            out[el] = out.get(el, 0) + int(d)
        return out

    def terminal_cation_fractions(self) -> Dict[str, float]:
        """Normalised fractional composition at the terminal step.

        Uses the raw integer digits directly: element → digit_sum / total_digits.
        Equivalent to what ``pymatgen.Composition`` would produce, but we avoid
        the import cost inside the env.  Elements with digit 0 are dropped.
        """
        if self.counter != self.n_components:
            raise RuntimeError("terminal_cation_fractions called before terminal step")

        digits = self.cation_digits()
        total = sum(digits.values())
        if total <= 0:
            return {}
        return {el: n / total for el, n in digits.items() if n > 0}

    def terminal_comp_key(self) -> tuple:
        """Canonical hashable key for deduplication (raw integer digits).

        Uses reduced integer digits (divided by gcd) so ``"Fe2Ti4O6"`` and
        ``"Fe1Ti2O3"`` collapse to the same key — matches the normalised
        fraction that the predictor sees.
        """
        digits = self.cation_digits()
        items = [(el, n) for el, n in digits.items() if n > 0]
        if not items:
            return ()
        from math import gcd
        from functools import reduce
        g = reduce(gcd, (n for _, n in items))
        items = [(el, n // g) for el, n in items]
        return tuple(sorted(items))

    # ------------------------------------------------------------------
    # Action space
    # ------------------------------------------------------------------

    def allowed_actions(self) -> List[Tuple[Tuple[float, ...], Tuple[float, ...]]]:
        if self.counter >= self.n_components:
            return []

        actions: List[Tuple[Tuple[float, ...], Tuple[float, ...]]] = []
        for elem in self.cation_set:
            elem_oh = tuple(encode_choice(elem, self.cation_set).tolist())
            for d in self.ratio_set:
                comp_oh = tuple(encode_choice(d, self.ratio_set).tolist())
                actions.append((elem_oh, comp_oh))

        if self.phase_filter is not None:
            actions = self.phase_filter.filter_actions(
                actions=actions,
                units_map=self._digits_map,
                steps_left=self.n_components - self.counter - 1,
                allowed_units=[int(d) for d in self.ratio_set],
                possible_sums_by_k=[],
                cation_set=self.cation_set,
                fraction_set=self.ratio_set,
            )
        return actions

    def sample_random_action(self) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        actions = self.allowed_actions()
        if not actions:
            raise RuntimeError("No valid actions available.")
        return random.choice(actions)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action: Tuple[Tuple[float, ...], Tuple[float, ...]]) -> None:
        elem_oh, comp_oh = action
        elem = decode_one_hot(elem_oh, self.cation_set)
        digit_str = decode_one_hot(comp_oh, self.ratio_set)

        current_allowed = self.allowed_actions()
        old_state = self.state

        self.state = f"{self.state}{elem}{digit_str}"
        self._digits_map[elem] = self._digits_map.get(elem, 0) + int(digit_str)
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
