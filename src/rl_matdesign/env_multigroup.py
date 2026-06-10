"""MultiGroupEnv — a fixed-order sequence of N composition groups.

Generalizes :class:`~rl_matdesign.env.CompositionEnv` from "one composition that
sums to 1" to "N **sublattice groups**, each summing to 1, filled in a fixed
order". The single-group case (N=1) reproduces ``CompositionEnv`` exactly, so
existing scenarios are the degenerate instance of this env.

Why this exists
---------------
Multi-sublattice substitutional design (e.g. doped Li6PS6: a P-site group and an
S-site group) cannot be expressed as one sum-to-1 composition — the agent fills
several independent sublattices, each of which sums to 1 on its own. Each group
is a full ``CompositionEnv`` sub-problem (its own ``cation_set``,
``fraction_set``, ``total_units``, ``episode_style``, ``element_bounds`` and
constraint filter); this env chains them.

Design
------
* **Delegation.** One inner ``CompositionEnv`` per group owns that group's
  feasibility / bounds / episode-style logic. This env drives them in order.
* **Union alphabet for network I/O.** Actions are exposed to the training code as
  one-hots over the *union* of all groups' ``cation_set`` / ``fraction_set`` so
  the action dimensions are fixed across groups. ``self.cation_set`` /
  ``self.fraction_set`` are those unions, which is exactly what ``training.py``
  reads to size the networks and decode ``a_elem_idx`` / ``a_comp_val``. Each
  step's allowed actions are re-encoded from the active group's alphabet into the
  union before they leave this env.
* **State features.** Per-step state = concatenation of every group's partial
  Magpie features (so the policy sees each sublattice's chemistry).
* **Cross-group coupling.** When the active group's constraint runs, the
  already-completed groups' compositions are passed as ``prior_groups`` (e.g. the
  S-site O range bounded by the P-site metal).
* **Terminal.** ``terminal_cation_fractions()`` returns the **structured**
  ``{group_name: {element: fraction}}`` mapping, which is what the predictor
  receives (``training.py`` passes it straight to ``predictor.predict``).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .encoding import decode_one_hot, encode_choice
from .env import CompositionEnv, EpisodeStep, _step_one_hot
from .featurization import featurize_formula


class MultiGroupEnv:
    """Fixed-order sequence of ``CompositionEnv`` groups (each sums to 1)."""

    def __init__(
        self,
        *,
        groups: Sequence[Dict[str, Any]],
        reward_fn: Callable[[Dict[str, Dict[str, float]]], float] | None = None,
        state_featurizer: Callable[[str], np.ndarray] = featurize_formula,
    ) -> None:
        if not groups:
            raise ValueError("MultiGroupEnv requires a non-empty 'groups' list.")

        self.reward_fn = reward_fn or (lambda _groups: 0.0)
        self.state_featurizer = state_featurizer
        self.episode_style = "multi_group"

        # Build one inner CompositionEnv per group. Inner reward is a no-op — this
        # env computes the single terminal reward over all groups.
        self.group_names: List[str] = []
        self._inners: List[CompositionEnv] = []
        for i, g in enumerate(groups):
            name = str(g.get("name", f"group{i}"))
            self.group_names.append(name)
            inner = CompositionEnv(
                cation_set=g["cation_set"],
                fraction_set=g.get("fraction_set") or None,
                anion_formula="",  # the predictor/recipe assembles the real structure
                n_components=int(g.get("n_components", 5)),
                reward_fn=lambda _f: 0.0,
                state_featurizer=state_featurizer,
                phase_filter=g.get("constraint_filter"),
                total_units=int(g.get("total_units", 20)),
                element_bounds=g.get("element_bounds"),
                episode_style=g.get("episode_style", "element_then_amount"),
            )
            self._inners.append(inner)

        # Union alphabets (first-seen order, deduplicated). Exact strings/symbols
        # are preserved so a union one-hot decodes back to a group-valid token.
        self.cation_set: List[str] = _ordered_union(
            sym for inner in self._inners for sym in inner.cation_set
        )
        self.fraction_set: List[str] = _ordered_union(
            f for inner in self._inners for f in inner.fraction_set
        )

        # Total episode length across all groups (what training.py reads).
        self._group_lengths = [inner.n_components for inner in self._inners]
        self.n_components = int(sum(self._group_lengths))
        self.max_steps = self.n_components

        self.counter: int = 0
        self.path: List[EpisodeStep] = []
        self._gi: int = 0  # active group index
        self._completed: List[Dict[str, float]] = []  # finished groups' comps

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        for inner in self._inners:
            inner.initialize()
        self.counter = 0
        self.path = []
        self._gi = 0
        self._completed = []

    def _active(self) -> CompositionEnv:
        return self._inners[self._gi]

    def _concat_features(self) -> np.ndarray:
        """Per-step state = concat of each group's partial-composition features."""
        return np.concatenate(
            [np.asarray(self.state_featurizer(inner.state), dtype=float) for inner in self._inners]
        )

    def current_state_features(self) -> np.ndarray:
        """Material features of the current state (uniform accessor; see CompositionEnv)."""
        return self._concat_features()

    def _to_union(
        self, group_action: Tuple[Tuple[float, ...], Tuple[float, ...]], inner: CompositionEnv
    ) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        """Re-encode a group-alphabet (elem_oh, comp_oh) into the union alphabet."""
        elem = decode_one_hot(group_action[0], inner.cation_set)
        comp = decode_one_hot(group_action[1], inner.fraction_set)
        elem_oh = tuple(encode_choice(elem, self.cation_set).tolist())
        comp_oh = tuple(encode_choice(comp, self.fraction_set).tolist())
        return elem_oh, comp_oh

    def allowed_actions(self) -> List[Tuple[Tuple[float, ...], Tuple[float, ...]]]:
        """Union-encoded allowed actions for the active group's current step."""
        if self.counter >= self.n_components:
            return []
        inner = self._active()
        raw = inner.allowed_actions(prior_groups=self._completed)
        return [self._to_union(a, inner) for a in raw]

    def step(self, action: Tuple[Tuple[float, ...], Tuple[float, ...]]) -> None:
        if self.counter >= self.n_components:
            raise RuntimeError("step() called after the episode terminated.")

        inner = self._active()
        # State features (all groups, BEFORE applying this action) and the
        # union-encoded allowed actions are captured pre-mutation, mirroring
        # CompositionEnv.step's ordering.
        s_material = self._concat_features()
        current_allowed = self.allowed_actions()

        # Decode the union action and re-encode into the active group's alphabet.
        elem = decode_one_hot(action[0], self.cation_set)
        comp = decode_one_hot(action[1], self.fraction_set)
        group_action = (
            tuple(encode_choice(elem, inner.cation_set).tolist()),
            tuple(encode_choice(comp, inner.fraction_set).tolist()),
        )
        inner.step(group_action)

        self.counter += 1
        s_step = _step_one_hot(self.counter, self.n_components)

        # If the active group just finished, record its composition for
        # downstream groups' prior_groups, and advance to the next group.
        if inner.counter >= inner.n_components:
            self._completed = self._completed + [inner.cation_fractions()]
            self._gi = min(self._gi + 1, len(self._inners) - 1)

        reward = 0.0
        if self.counter == self.n_components:
            reward = float(self.reward_fn(self.terminal_cation_fractions()))

        self.path.append(
            EpisodeStep(
                state_material_features=s_material,
                state_step_onehot=s_step,
                action_elem_onehot=action[0],
                action_comp_onehot=action[1],
                reward=reward,
                allowed_actions=current_allowed,
            )
        )

    def sample_random_action(self) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        import random

        actions = self.allowed_actions()
        if not actions:
            raise RuntimeError("No valid actions available.")
        return random.choice(actions)

    # ------------------------------------------------------------------
    # Terminal accessors (the predictor/dedup interface)
    # ------------------------------------------------------------------

    def _is_terminal(self) -> bool:
        return self.counter == self.n_components

    def terminal_cation_fractions(self) -> Dict[str, Dict[str, float]]:
        """Structured ``{group_name: {element: fraction}}`` for the predictor."""
        if not self._is_terminal():
            raise RuntimeError("terminal_cation_fractions called before terminal step")
        return {
            name: inner.cation_fractions()
            for name, inner in zip(self.group_names, self._inners)
        }

    def terminal_comp_key(self) -> tuple:
        """Canonical hashable key over all groups for dedup/visit-tracking."""
        return tuple(
            (name, inner.terminal_comp_key())
            for name, inner in zip(self.group_names, self._inners)
        )

    @property
    def terminal_formula(self) -> str:
        """Human-readable combined label (per-group formulas), for generated.csv."""
        if not self._is_terminal():
            return ""
        return " | ".join(
            f"{name}:{inner.terminal_formula}"
            for name, inner in zip(self.group_names, self._inners)
        )


def _ordered_union(items) -> List[str]:
    """Deduplicate while preserving first-seen order."""
    seen: Dict[str, None] = {}
    for it in items:
        if it not in seen:
            seen[it] = None
    return list(seen.keys())
