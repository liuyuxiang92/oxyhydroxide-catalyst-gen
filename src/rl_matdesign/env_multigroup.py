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


def _amounts_to_strs(amount: Any) -> Tuple[List[str], Optional[float]]:
    """Expand an ``amount`` spec into 2-decimal fraction codes + the grid step.

    ``amount`` is either an explicit list of fractions, or a ``{min, max, step}``
    range. Returns ``(["0.02", ...], step)`` (``step`` is ``None`` for a list).
    """
    if isinstance(amount, (list, tuple)):
        amounts = [float(x) for x in amount]
        step: Optional[float] = None
    else:
        lo, hi, step = float(amount["min"]), float(amount["max"]), float(amount["step"])
        n = int(round((hi - lo) / step))
        amounts = [round(lo + i * step, 6) for i in range(n + 1)]
    return [f"{a:.2f}" for a in amounts], step


def normalize_group_spec(g: Dict[str, Any]) -> Dict[str, Any]:
    """Expand a *friendly* composition-group spec into full CompositionEnv params.

    Friendly knobs handled here so the env config stays terse:

    * ``amount: {min, max, step}`` (or a list) -> generates ``fraction_set``.
    * ``host: <el>`` -> lists only the dopants in ``cation_set``; the host is added
      and auto-takes the complement via a ``host_complement`` filter (so no
      hand-written complements and no per-scenario "host-takes-rest" constraint).
    * ``sites: N`` -> kept (the sublattice size; used for formula assembly and by
      the builder to place real atom counts).

    Returns the spec unchanged if it has no ``amount`` (an explicit ``fraction_set``
    is left as-is). The ``constraint_filter`` it may set is a *name* (built later by
    the caller via the registry), matching how groups are otherwise resolved.
    """
    g = dict(g)
    g.setdefault("sites", 1)
    kind = g.get("kind", "composition")

    if kind == "independent":
        # K independent (element, amount) picks; repeats allowed; no host pick —
        # the un-picked remainder is a host filled in downstream by the builder.
        amount = g.pop("amount", None)
        if amount is not None:
            g["fraction_set"] = _amounts_to_strs(amount)[0]
            _, step = _amounts_to_strs(amount)
            g.setdefault("total_units", int(round(1.0 / step)) if step else 100)
        n_dopants = int(g.get("n_dopants", g.get("n_components", g.get("sites", 1))))
        g["n_components"] = n_dopants
        host = g.get("host")
        if host:
            g["cation_set"] = [e for e in g["cation_set"] if e != host]
        return g

    if kind != "composition":
        return g

    amount = g.pop("amount", None)
    if amount is None:
        return g

    amt_strs, step = _amounts_to_strs(amount)
    amounts = [float(s) for s in amt_strs]

    host = g.get("host")
    if host:
        comp_strs = [f"{round(1.0 - a, 2):.2f}" for a in amounts]
        dopants = [e for e in g["cation_set"] if e != host]
        g["cation_set"] = dopants + [host]
        g["fraction_set"] = amt_strs + comp_strs
        g.setdefault("n_components", 2)
        if not g.get("constraint_filter"):
            g["constraint_filter"] = "host_complement"
            g["host_element"] = host
            g["levels"] = amt_strs
    else:
        g["fraction_set"] = amt_strs

    g.setdefault("total_units", int(round(1.0 / step)) if step else 100)
    return g


class CategoricalGroup:
    """A pick-from-list group: fixed-order slots, each ``(element, values)``; no sum-to-1.

    Used for sublattices that are discrete choices rather than a composition (e.g. the
    doped-Li₆PS₆ S-site: an oxygen *form* and a chlorine *count*). The agent picks one
    value per slot in order. Values are encoded to a numeric **code** for the network's
    scalar-amount input; the terminal returns the **original** values (so a builder reads
    the real ``Cl = 1.0`` / ``O = "oxide"`` directly — no selector/cl_map).

    Each slot has a unique **name** (the key the terminal/builder reads) and an
    underlying **element** (the real chemistry). They default to the same string,
    so a plain ``{element: O, values: [...]}`` slot is named ``"O"``. Giving two
    slots the *same* element but *different* names (``{name: O_a, element: O}`` and
    ``{name: O_b, element: O}``) lets one categorical group carry **per-metal**
    O-forms without the names colliding in the one-hot alphabet — the builder maps
    each name back to its element via :attr:`slot_element`.

    Presents the minimal inner-group interface ``MultiGroupEnv`` drives.
    """

    def __init__(self, choices: Sequence[Dict[str, Any]], *, phase_filter=None) -> None:
        # (name, element, values) — name is the unique slot id used for encoding
        # and as the terminal key; element is the real chemistry for the builder.
        self.slots: List[Tuple[str, str, List[Any]]] = [
            (str(c.get("name", c["element"])), str(c["element"]), list(c["values"]))
            for c in choices
        ]
        if not self.slots:
            raise ValueError("categorical group needs a non-empty 'choices' list.")
        self.cation_set: List[str] = [name for name, _el, _v in self.slots]
        if len(set(self.cation_set)) != len(self.cation_set):
            raise ValueError(
                f"categorical slot names must be unique; got {self.cation_set}. "
                "Give same-element slots distinct 'name' keys (e.g. O_a, O_b)."
            )
        self.slot_element: Dict[str, str] = {name: el for name, el, _v in self.slots}
        self.n_components: int = len(self.slots)
        self.phase_filter = phase_filter

        # Per-slot value <-> numeric-code maps; codes go into fraction_set.
        self._slot_codes: List[List[Tuple[Any, str]]] = []
        self._code_to_value: List[Dict[str, Any]] = []
        codes: set[str] = set()
        for _name, _el, values in self.slots:
            sc, c2v = [], {}
            for i, v in enumerate(values):
                code = self._encode_value(v, i)
                sc.append((v, code))
                c2v[code] = v
                codes.add(code)
            self._slot_codes.append(sc)
            self._code_to_value.append(c2v)
        self.fraction_set: List[str] = sorted(codes, key=float)
        self.initialize()

    @staticmethod
    def _encode_value(v: Any, i: int) -> str:
        try:
            return f"{float(v):.2f}"          # numeric value -> its own code
        except (ValueError, TypeError):
            return f"{float(i):.2f}"           # label -> slot-local index code

    def initialize(self) -> None:
        self.counter = 0
        self.state = ""                        # no composition string; featurizer falls back
        self._picked: Dict[str, Any] = {}

    def allowed_actions(self, *, prior_groups=None):
        if self.counter >= self.n_components:
            return []
        name = self.slots[self.counter][0]
        actions = [
            (
                tuple(encode_choice(name, self.cation_set).tolist()),
                tuple(encode_choice(code, self.fraction_set).tolist()),
            )
            for _v, code in self._slot_codes[self.counter]
        ]
        if self.phase_filter is not None:
            kw = dict(
                actions=actions, units_map={},
                steps_left=self.n_components - self.counter - 1,
                allowed_units=[], possible_sums_by_k=[],
                cation_set=self.cation_set, fraction_set=self.fraction_set,
            )
            if prior_groups is not None:
                kw["prior_groups"] = prior_groups
            actions = self.phase_filter.filter_actions(**kw)
        return actions

    def step(self, action) -> None:
        elem_oh, comp_oh = action
        name = decode_one_hot(elem_oh, self.cation_set)
        expected = self.slots[self.counter][0]
        if name != expected:
            raise ValueError(
                f"categorical slot {self.counter} expects {expected!r}, got {name!r}."
            )
        code = decode_one_hot(comp_oh, self.fraction_set)
        self._picked[name] = self._code_to_value[self.counter][code]
        self.counter += 1

    def cation_fractions(self) -> Dict[str, Any]:
        return dict(self._picked)

    def terminal_comp_key(self) -> tuple:
        return tuple(sorted((k, str(v)) for k, v in self._picked.items()))


class IndependentDopantsGroup:
    """``K`` independent ``(element, amount)`` picks on one sublattice; repeats allowed.

    Unlike a composition group, the picks need **not** be distinct and do **not**
    sum to 1 — the un-picked remainder is a *host* that the downstream builder
    fills in (e.g. the doped-Li₆PS₆ P-site: pick two dopant metals from the same
    menu at the same amount range; host P takes the rest). Two picks of the *same*
    element **merge** (their amounts add), so ``cation_fractions`` returns a
    ``{element: summed_fraction}`` map with at most ``K`` distinct keys.

    This is deliberately a separate, isolated inner-group type rather than a knob
    on :class:`~rl_matdesign.env.CompositionEnv`, which hard-enforces *distinct*
    cations and a sum-to-1 budget; bending those would risk every composition
    scenario and the order-invariance guarantees. Presents the same minimal
    inner-group interface ``MultiGroupEnv`` drives.
    """

    def __init__(
        self,
        *,
        cation_set: Sequence[str],
        fraction_set: Sequence[str],
        n_components: int,
        total_units: int = 100,
        state_featurizer: Callable[[str], np.ndarray] = featurize_formula,
        phase_filter=None,
    ) -> None:
        self.cation_set: List[str] = list(cation_set)
        self.fraction_set: List[str] = list(fraction_set)
        self.n_components: int = int(n_components)
        self._total_units = int(total_units)
        self.state_featurizer = state_featurizer
        self.phase_filter = phase_filter
        if not self.cation_set:
            raise ValueError("independent group needs a non-empty 'cation_set'.")
        if not self.fraction_set:
            raise ValueError("independent group needs a non-empty 'fraction_set'.")
        self.initialize()

    def initialize(self) -> None:
        self.counter = 0
        self.state = ""                          # merged partial formula (for featurizer)
        self._picks: List[Tuple[str, str]] = []  # ordered (element, comp_str) picks

    def allowed_actions(self, *, prior_groups=None):
        if self.counter >= self.n_components:
            return []
        actions = [
            (
                tuple(encode_choice(el, self.cation_set).tolist()),
                tuple(encode_choice(comp, self.fraction_set).tolist()),
            )
            for el in self.cation_set
            for comp in self.fraction_set
        ]
        if self.phase_filter is not None:
            kw = dict(
                actions=actions, units_map={},
                steps_left=self.n_components - self.counter - 1,
                allowed_units=[], possible_sums_by_k=[],
                cation_set=self.cation_set, fraction_set=self.fraction_set,
            )
            if prior_groups is not None:
                kw["prior_groups"] = prior_groups
            actions = self.phase_filter.filter_actions(**kw)
        return actions

    def step(self, action) -> None:
        elem_oh, comp_oh = action
        el = decode_one_hot(elem_oh, self.cation_set)
        comp = decode_one_hot(comp_oh, self.fraction_set)
        self._picks.append((el, comp))
        self.state = self._formula(self._picks)
        self.counter += 1

    def _formula(self, picks: List[Tuple[str, str]]) -> str:
        merged: Dict[str, float] = {}
        for el, comp in picks:
            merged[el] = merged.get(el, 0.0) + float(comp)
        return "".join(f"{el}{v:.2f}" for el, v in sorted(merged.items()))

    def cation_fractions(self) -> Dict[str, float]:
        merged: Dict[str, float] = {}
        for el, comp in self._picks:
            merged[el] = merged.get(el, 0.0) + float(comp)
        return merged

    def terminal_comp_key(self) -> tuple:
        items = []
        for el, frac in self.cation_fractions().items():
            units = int(round(frac * self._total_units))
            if units > 0:
                items.append((str(el), units))
        return tuple(sorted(items))


class MultiGroupEnv:
    """Fixed-order sequence of groups (composition: sum-to-1, or categorical: pick-list)."""

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
        self._sites: List[int] = []
        self._kinds: List[str] = []
        self._inners: List[Any] = []
        for i, g in enumerate(groups):
            name = str(g.get("name", f"group{i}"))
            kind = g.get("kind", "composition")
            self.group_names.append(name)
            self._sites.append(int(g.get("sites", 1)))
            self._kinds.append(kind)
            if kind == "categorical":
                inner = CategoricalGroup(
                    g["choices"], phase_filter=g.get("constraint_filter"),
                )
            elif kind == "independent":
                inner = IndependentDopantsGroup(
                    cation_set=g["cation_set"],
                    fraction_set=g["fraction_set"],
                    n_components=int(g.get("n_components", g.get("n_dopants", 1))),
                    total_units=int(g.get("total_units", 100)),
                    state_featurizer=state_featurizer,
                    phase_filter=g.get("constraint_filter"),
                )
            else:
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

    def assembled_composition(self) -> Dict[str, float]:
        """Per-formula-unit composition ``{element: count}`` across all groups.

        Each group contributes ``fraction × sites`` per element, so the result is
        the real chemical composition (counts per formula unit). For a single group
        with ``sites=1`` this is just the fractions — consistent with a plain
        ``fraction`` env.
        """
        out: Dict[str, float] = {}
        for inner, sites, kind in zip(self._inners, self._sites, self._kinds):
            for el, val in inner.cation_fractions().items():
                if kind == "categorical":
                    # categorical values are already per-f.u. counts; non-numeric
                    # labels (e.g. an O "form") are not atoms — the builder resolves them.
                    if isinstance(val, (int, float)):
                        out[el] = out.get(el, 0.0) + float(val)
                else:
                    out[el] = out.get(el, 0.0) + val * sites
        return out

    @property
    def terminal_formula(self) -> str:
        """Readable assembled chemical formula (per formula unit), for generated.csv."""
        if not self._is_terminal():
            return ""
        items = sorted(
            ((el, n) for el, n in self.assembled_composition().items() if n > 1e-9),
            key=lambda t: (-t[1], t[0]),
        )
        return "".join(f"{el}{n:.3g}" for el, n in items)


def _ordered_union(items) -> List[str]:
    """Deduplicate while preserving first-seen order."""
    seen: Dict[str, None] = {}
    for it in items:
        if it not in seen:
            seen[it] = None
    return list(seen.keys())
