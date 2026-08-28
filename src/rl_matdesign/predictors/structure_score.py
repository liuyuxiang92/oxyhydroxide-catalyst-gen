"""The reward engine — a multi-objective combiner over a ``properties:`` list.

One predictor per objective, folded into a single reward:

    for each property:
      predictor: dp_energy | dp_property      -> build (+ optional relax) a
                                                 structure, score with DeepMD
      predictor: rf_magpie | <fqn> | ...      -> score the composition directly
    reward = Σ weight·objective_from_mean_std(direction·mean, std, obj) / scale

Single vs multi-objective is **auto-detected** from the list length — one entry
is the single-objective case, the same fold with one term. There is no separate
mode and no ``backend``/``mode`` key.

Structure objectives (``dp_energy``/``dp_property``) build a cell; the
``share_structure`` boolean controls reuse:

* ``share_structure: true`` (default) — build **one** structure with the
  top-level builder, relax it **once**, score every structure objective on that
  shared cell (e.g. conductivity + stability of the *same* relaxed Li6PS6 cell).
* ``share_structure: false`` — each structure objective builds (and optionally
  relaxes) its **own** cell independently (objectives may use different builders).

Composition objectives (``rf_magpie`` and most FQN leaves) never build a
structure; they receive the candidate composition. A config with no structure
objective needs no builder / ``base_poscar`` at all.

Predictor contract (registry leaves and FQN leaves alike)
--------------------------------------------------------
**A leaf represents exactly ONE model**, and the engine owns the folding. ``model:``
may be one path or a list of paths; each path becomes its own leaf instance.

=========================================  ======================  ==========
method a leaf may expose                   returns                 who folds
=========================================  ======================  ==========
``score_structures(atoms_list)``           one float per structure engine
``score(composition)``                     one float               engine
``predict_structures(atoms_list)``         ``(mean, std)``         leaf
``predict(composition)``                   ``(mean, std)``         leaf
=========================================  ======================  ==========

Everything is auto-detected, with no YAML flag:

* **structure vs composition** — a leaf exposing either ``*_structures`` method is
  a structure objective and receives the built/relaxed cells; otherwise it is a
  composition objective and receives the candidate composition.
* **engine-folded vs self-folding** — the ``score*`` form wins when present. The
  self-folding form stays supported for a leaf with a genuine *internal* ensemble
  (e.g. ``rf_magpie``'s random-forest tree variance), whose ``(mean, std)`` is
  passed through untouched.

Prefer the ``score*`` form for anything wrapping a single checkpoint: the engine
then averages the structures axis and takes mean/std across models, so a single
``model:`` yields ``std == 0`` automatically — no flag, no special case. See
:func:`_fold_models_structures`.

Consumer contract (``training.py``): exposes ``predict`` (reward, std),
``predict_raw`` (Σ weight·direction·mean/scale, for the ``dp_mean`` CSV column),
and ``per_objective_stats`` ({name: (mean, std)} for per-objective CSV columns).

Config keys
-----------
    builder:          builder registry name or ``pkg.mod:Class`` FQN (default
                      ``substitute``); only consulted when a structure objective
                      is present.
    share_structure:  bool, default ``true`` (see above).
    n_random_configs: random realizations per candidate (default 1).
    geo_opt:          optional ``{model, head, fmax, steps, relax_cell, enabled}``;
                      omit / ``enabled: false`` to score the unrelaxed structure.
    k:                uncertainty coefficient for the objective folding.
    sweep:            optional ``{name, values}`` — an inner optimization over a
                      shared operating condition (e.g. temperature). For each
                      candidate, every property is scored at each value and the
                      single value maximizing the combined reward is kept; it is
                      substituted into each property's fparam ``null`` slot and
                      reported as a ``<name>`` stat (→ ``obj_<name>_mean`` CSV
                      column). Structures are built+relaxed once and reused.
    properties:       non-empty list; each entry:
        name, predictor (dp_energy|dp_property|rf_magpie|<fqn>), direction
        (max|min, REQUIRED), weight, scale, objective. Structure predictors also
        take models (or legacy dp_models), head, output_index, output_aggregator,
        energy_per_atom, fparam, aparam; a ``null`` in ``fparam`` marks the slot
        filled by the sweep value (requires a top-level ``sweep``). Composition
        predictors take their own leaf args (e.g. ``model:`` for rf_magpie). In
        ``share_structure: false`` a structure entry may also carry its own
        builder / base_poscar / site_symbol / n_random_configs.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


_DIRECTION_ALIASES = {"min": -1.0, "minimize": -1.0, "max": +1.0, "maximize": +1.0}
# A third direction: hit a specific value rather than push it up or down. Needs
# `target_value`; `target_tolerance` is a deadband inside which the term is at its
# maximum (0), so the objective stops competing once the requirement is met.
_TARGET_DIRECTION = "target"
_TARGET_KEYS = ("target_value", "target_tolerance")
_OBJECTIVES = ("mean", "mean_minus_kstd", "mean_plus_kstd")
# Per-property output transforms, applied to each ensemble member before the
# mean/std fold (so heads that regress log(value) report the real value).
_TRANSFORMS = ("none", "exp")
# Builder knobs inherited by per-objective builders (share_structure: false)
# when the objective entry doesn't set them. Mirrors the old composite behavior.
_SHARED_BUILDER_KEYS = (
    "builder", "base_poscar", "poscar", "poscar_template", "site_symbol",
    "n_random_configs",
)


def _fold_models_structures(arr: "np.ndarray") -> Tuple[float, float]:
    """Fold a ``(n_models, n_structures)`` array into ``(mean, std)``.

    The two axes mean different things and must not be pooled:

    * **axis 1, structures** — the ``n_random_configs`` random decorations of one
      composition. Their scatter is *configurational*: it is how the property is
      defined for a disordered composition, so it is averaged away.
    * **axis 0, models** — independent checkpoints of the same property. Their
      scatter is *epistemic uncertainty*, and is what ``mean_minus_kstd`` penalises.

    Pooling both (the previous behaviour) let decoration scatter masquerade as model
    uncertainty — so a single model with ``n_random_configs > 1`` reported a nonzero
    ``std`` and ``mean_minus_kstd`` silently penalised disorder. With this fold a
    single model gives ``std == 0`` exactly, and ``mean`` is unchanged either way.
    """
    arr = np.atleast_2d(np.asarray(arr, dtype=float))
    per_model = arr.mean(axis=1)
    return float(per_model.mean()), float(per_model.std())


class StructureScorePredictor:
    def __init__(self, cfg: Dict[str, Any], *, seed: Optional[int] = None) -> None:
        from ..registry import resolve_builder

        self.cfg = cfg
        self.share_structure: bool = bool(cfg.get("share_structure", True))
        self.k: float = float(cfg.get("k", 1.0))
        self.n_random_configs: int = int(cfg.get("n_random_configs", 1))
        self._seed = seed

        go = cfg.get("geo_opt") or {}
        self.geo_opt: Dict[str, Any] = dict(go)
        self.geo_opt_enabled: bool = bool(go) and bool(go.get("enabled", True))

        # Optional inner optimization over a shared operating condition (e.g.
        # temperature). For each candidate we score every property at each sweep
        # value and keep the single value maximizing the combined reward. The
        # value is substituted into each property's fparam `null` placeholder(s).
        sweep = cfg.get("sweep")
        self.sweep: Optional[Dict[str, Any]] = None
        if sweep is not None:
            values = sweep.get("values")
            if not values:
                raise ValueError(
                    "sweep needs a non-empty 'values' list (e.g. temperatures to "
                    "try for each composition)."
                )
            self.sweep = {
                "name": str(sweep.get("name", "sweep")),
                "values": [float(v) for v in values],
            }

        raw_props = cfg.get("properties")
        if not raw_props:
            raise ValueError(
                "reward needs a non-empty 'properties' list. Each entry is one "
                "objective: {name, predictor: dp_energy|dp_property|rf_magpie|<fqn>, "
                "direction: max|min, ...}."
            )

        top_builder = str(cfg.get("builder", "substitute"))
        shared_builder_cfg = {k: cfg[k] for k in _SHARED_BUILDER_KEYS if k in cfg}

        self.properties: List[Dict[str, Any]] = []
        # Per-property structure builder (share_structure: false), keyed by name;
        # only structure predictors (dp_energy/dp_property) get one.
        self._builders: Dict[str, Any] = {}
        seen: set[str] = set()
        for i, p in enumerate(raw_props):
            name = str(p.get("name", f"prop{i}"))
            if name in seen:
                raise ValueError(
                    f"properties[{i}].name = {name!r} duplicates an earlier "
                    "objective (used as the CSV column prefix); names must be unique."
                )
            seen.add(name)

            predictor_name = str(p.get("predictor", "")).strip()
            if not predictor_name:
                raise ValueError(
                    f"properties[{i}] ({name!r}) needs a 'predictor' — one of "
                    "'dp_energy', 'dp_property', a registry leaf (e.g. 'rf_magpie', "
                    "'ooh'), or an FQN 'pkg.mod:Class'."
                )
            is_dp_backend = predictor_name in ("dp_energy", "dp_property")

            # direction is REQUIRED — the sign (lower vs higher is better) must be
            # explicit, never implied.
            direction_raw = p.get("direction")
            if direction_raw is None:
                raise ValueError(
                    f"properties[{i}] ({name!r}) needs an explicit 'direction' "
                    "(max or min)."
                )
            direction = str(direction_raw).lower()
            is_target = direction == _TARGET_DIRECTION
            if not is_target and direction not in _DIRECTION_ALIASES:
                raise ValueError(
                    f"properties[{i}].direction = {direction!r}; expected one of "
                    f"{sorted(_DIRECTION_ALIASES) + [_TARGET_DIRECTION]}."
                )

            # `direction: target` -> hit a value. Kept as its own spec rather than a
            # sign, because it is not a sign: see _property_value.
            target_spec = None
            if is_target:
                if p.get("target_value") is None:
                    raise ValueError(
                        f"properties[{i}] ({name!r}) has direction: target but no "
                        "'target_value' -- the value to hit, in the property's real "
                        "units (e.g. target_value: 1.34 for a 1.34 eV bandgap). Note "
                        "this is NOT the leaf's own 'target'/'model' key, which "
                        "selects the checkpoint."
                    )
                try:
                    target_value = float(p["target_value"])
                except (TypeError, ValueError):
                    raise ValueError(
                        f"properties[{i}].target_value = {p['target_value']!r} is not "
                        "a number."
                    ) from None
                target_tol = float(p.get("target_tolerance", 0.0))
                if target_tol < 0.0:
                    raise ValueError(
                        f"properties[{i}].target_tolerance must be >= 0 (got {target_tol})."
                    )
                target_spec = {"value": target_value, "tolerance": target_tol}
            else:
                stray = [k for k in _TARGET_KEYS if k in p]
                if stray:
                    raise ValueError(
                        f"properties[{i}] ({name!r}) sets {stray} but direction="
                        f"{direction!r}. Set 'direction: target' to hit a specific "
                        "value, or remove those keys -- they would otherwise be "
                        "silently ignored."
                    )

            scale = float(p.get("scale", 1.0))
            if scale == 0.0:
                raise ValueError(f"properties[{i}].scale must be non-zero (got 0).")

            objective = str(p.get("objective", "mean_minus_kstd"))
            if objective not in _OBJECTIVES:
                raise ValueError(
                    f"properties[{i}].objective = {objective!r}; expected one of "
                    f"{list(_OBJECTIVES)}."
                )

            # `null` entries in fparam mark the slot filled by the sweep value.
            fparam = p.get("fparam")
            sweep_slots = (
                [j for j, x in enumerate(fparam) if x is None]
                if isinstance(fparam, (list, tuple)) else []
            )
            if sweep_slots and self.sweep is None:
                raise ValueError(
                    f"properties[{i}] ({name!r}) has a null placeholder in fparam "
                    f"but no top-level 'sweep' is configured. Add a 'sweep' block "
                    "(e.g. {name: temperature, values: [...]}) or fill the fparam."
                )

            transform = str(p.get("transform", "none")).lower()
            if transform not in _TRANSFORMS:
                raise ValueError(
                    f"properties[{i}].transform = {transform!r}; expected one of "
                    f"{sorted(_TRANSFORMS)}. Use 'exp' for heads that emit log(value)."
                )

            if is_dp_backend:
                models = p.get("models") or p.get("dp_models")
                if not models:
                    raise ValueError(
                        f"properties[{i}] ({name!r}) with predictor={predictor_name!r} "
                        "needs 'models' (list of DeepMD checkpoint paths)."
                    )
                backend = "energy" if predictor_name == "dp_energy" else "property"
                instance = None
                is_structure = True
            else:
                # Registry leaf or FQN. `model:` may be ONE path or a LIST of paths;
                # each path becomes its own leaf instance, because a leaf represents
                # exactly one model. That is what makes "one model -> std == 0" fall
                # out of the arithmetic instead of needing a flag (see _score).
                from ..registry import resolve_predictor

                models = None
                model_spec = p.get("model", p.get("models"))
                if isinstance(model_spec, (list, tuple)):
                    model_paths = [str(m) for m in model_spec]
                    if not model_paths:
                        raise ValueError(
                            f"properties[{i}] ({name!r}) has an empty 'model' list."
                        )
                elif model_spec is None:
                    model_paths = [None]          # leaf needs no model path
                else:
                    model_paths = [str(model_spec)]

                instance = []
                for j, path in enumerate(model_paths):
                    leaf_cfg = dict(p)
                    if path is not None:
                        leaf_cfg["model"] = path
                    # Decorrelate seeds across properties AND across models.
                    child_seed = None if seed is None else int(seed) + 100 * i + j
                    instance.append(resolve_predictor(predictor_name, leaf_cfg, seed=child_seed))

                # A leaf that can score structures is a structure objective (gets the
                # built cells); otherwise it is a composition objective (gets the
                # candidate). Auto-detected from the leaf, no YAML flag.
                probe = instance[0]
                is_structure = hasattr(probe, "score_structures") or hasattr(
                    probe, "predict_structures"
                )
                backend = "structure_fqn" if is_structure else "composition"

            self.properties.append({
                "name": name,
                "predictor": predictor_name,
                "backend": backend,
                "instance": instance,
                "models": list(models) if models else None,
                "head": p.get("head"),
                "output_index": int(p.get("output_index", 0)),
                "output_aggregator": str(p.get("output_aggregator", "index")),
                "energy_per_atom": bool(p.get("energy_per_atom", True)),
                "fparam": fparam,
                "sweep_slots": sweep_slots,
                "aparam": p.get("aparam"),
                "direction": None if is_target else _DIRECTION_ALIASES[direction],
                "target": target_spec,
                "weight": float(p.get("weight", 1.0)),
                "scale": scale,
                "objective": objective,
                "transform": transform,
                "n_random_configs": int(p.get("n_random_configs", self.n_random_configs)),
            })

            if is_structure and not self.share_structure:
                # Each structure objective builds independently: inherit top-level
                # builder knobs, let the entry override. Decorrelate the RNG.
                sub_cfg = {**shared_builder_cfg, **p}
                child_seed = None if seed is None else int(seed) + i
                self._builders[name] = resolve_builder(
                    str(sub_cfg.get("builder", top_builder)), sub_cfg, seed=child_seed
                )

        # Any structure predictor present? Only then do we need a builder / base
        # POSCAR — a pure composition-predictor config (e.g. rf_magpie) builds
        # nothing.
        self._has_structure = any(p["backend"] != "composition" for p in self.properties)

        # Shared builder (share_structure: true) — build once, score all structure
        # objectives on it. Skipped entirely when no structure objective exists.
        self._shared_builder = (
            resolve_builder(top_builder, cfg, seed=seed)
            if (self.share_structure and self._has_structure) else None
        )

        self._rng = np.random.default_rng(seed)
        self._calc_cache: Dict[str, Any] = {}     # name -> ase calculators (energy backend)
        self._prop_cache: Dict[str, Tuple[List[Any], Dict[str, int]]] = {}  # name -> property models
        self._geo_calc: Optional[Any] = None
        # Persistent per-composition stats cache (LRU). Keyed by the canonical
        # candidate key, it makes each unique composition build+relax+score
        # exactly ONCE for the predictor's lifetime — so training repeats and
        # duplicate generation rollouts of the same composition are free (the
        # single biggest saving when geo_opt is expensive). `predict_cache_size`
        # caps entries (<=0 = unbounded); each entry is a few floats, so a large
        # cap is cheap. Set to 0/1 to effectively disable.
        from collections import OrderedDict

        self._stats_cache: "OrderedDict[tuple, Dict[str, Tuple[float, float]]]" = OrderedDict()
        self._stats_cache_size: int = int(cfg.get("predict_cache_size", 200000))

    # ------------------------------------------------------------------
    # PropertyPredictor protocol
    # ------------------------------------------------------------------

    def _combine(self, stats: Dict[str, Tuple[float, float]]) -> float:
        """Weighted/signed/scaled fold of per-property stats into one reward.

        Shared by ``predict`` and the sweep argmax so both rank temperatures by
        the identical objective.
        """
        reward = 0.0
        for prop in self.properties:
            m, s = stats[prop["name"]]
            reward += prop["weight"] * self._property_value(prop, m, s) / prop["scale"]
        return float(reward)

    def _property_value(self, prop: Dict[str, Any], mean: float, std: float) -> float:
        """Signed 'higher is better' value for ONE objective, before weight/scale.

        max/min: the existing signed objective, unchanged. target: the deadbanded
        distance to ``target_value``, negated so closer is better, then pushed
        through the SAME objective helper -- so uncertainty handling is identical
        and consistent (``mean_minus_kstd`` becomes ``-e - k*std``, i.e. farther
        from target when less certain, which is the pessimistic reading).
        """
        from ..training import objective_from_mean_std

        tgt = prop["target"]
        if tgt is None:
            return objective_from_mean_std(
                prop["direction"] * float(mean), float(std), prop["objective"], self.k
            )
        e = max(0.0, abs(float(mean) - tgt["value"]) - tgt["tolerance"])
        return objective_from_mean_std(-e, float(std), prop["objective"], self.k)

    def predict(self, candidate: Any) -> Tuple[float, float]:
        stats = self._raw_stats(candidate)
        reward = self._combine(stats)
        stds = [stats[prop["name"]][1] for prop in self.properties]
        return reward, float(np.mean(stds)) if stds else 0.0

    def predict_raw(self, candidate: Any) -> Tuple[float, float]:
        """Combined value from raw means (no std penalty) — for the dp_mean column."""
        return float(self.raw_combine(self._raw_stats(candidate))), 0.0

    def per_objective_stats(self, candidate: Any) -> Dict[str, Tuple[float, float]]:
        """Return ``{name: (mean, std)}`` per objective, for CSV logging."""
        return dict(self._raw_stats(candidate))

    def batch_predict(self, candidates: List[Any]) -> List[Tuple[float, float]]:
        return [self.predict(c) for c in candidates]

    def close(self) -> None:
        """Release leaf resources — safe to call more than once.

        A leaf may own an external process (e.g. one ``serve.py`` per
        MGTransformer checkpoint, so a 3-property config holds three). Without
        this they are only reaped in the leaf's ``__del__``, i.e. whenever the GC
        gets around to it. Best-effort: one leaf failing to close must not stop
        the others.
        """
        for prop in self.properties:
            for leaf in prop.get("instance") or []:
                fn = getattr(leaf, "close", None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:  # noqa: BLE001 - teardown is best-effort
                        pass

    def composition_formula(self, candidate: Any) -> Optional[str]:
        """Full composition label of the built structure, if the builder exposes it.

        Delegates to the (shared, or first per-objective) builder's
        ``composition_formula``. Builders that don't implement it (most) return
        ``None``, so callers fall back to the env's ``terminal_formula``.
        """
        builder = self._shared_builder or (self._builders[0] if self._builders else None)
        fn = getattr(builder, "composition_formula", None)
        return fn(candidate) if callable(fn) else None

    def score_breakdown(self, candidate: Any) -> Dict[str, Any]:
        """Diagnostic: per-sweep-value property stats + combined reward.

        Builds (and relaxes) the structure **once**, then scores every property
        at each sweep value (or just once if no sweep). Returns::

            {"rows": [{"value": v, "stats": {name: (mean, std)}, "reward": r}, ...],
             "best": {"value": v*, "stats": {...}, "reward": r*},
             "structures": {prop_name: [relaxed Atoms, ...]}}

        ``value`` is ``None`` when no sweep is configured. ``structures`` are the
        exact (relaxed) cells that were scored, so a caller can save/inspect them
        without re-building or re-relaxing. Used by ``scripts/evaluate_lips.py``.
        """
        prop_structures = self._materialize_structures(candidate)
        values = self.sweep["values"] if self.sweep is not None else [None]
        rows: List[Dict[str, Any]] = []
        for v in values:
            if v is None:                       # no sweep — score each property once
                stats = {
                    p["name"]: self._score(p, prop_structures[p["name"]])
                    for p in self.properties
                }
            else:
                stats = {
                    p["name"]: self._score(
                        p, prop_structures[p["name"]], fparam=self._fparam_at(p, v)
                    )
                    for p in self.properties
                }
            rows.append({"value": v, "stats": stats, "reward": self._combine(stats)})
        best = max(rows, key=lambda r: r["reward"])
        return {"rows": rows, "best": best, "structures": prop_structures}

    def raw_combine(self, stats: Dict[str, Tuple[float, float]]) -> float:
        """Σ weight·direction·mean/scale (no std penalty) — the dp_mean fold.

        Mirrors :meth:`predict_raw` but operates on already-computed stats, so a
        caller holding a ``score_breakdown`` result need not re-score.
        """
        raw = 0.0
        for prop in self.properties:
            m, _s = stats[prop["name"]]
            # std = 0 collapses every objective to the plain value, so for a
            # max/min property this is exactly `direction * mean` as before.
            raw += prop["weight"] * self._property_value(prop, m, 0.0) / prop["scale"]
        return float(raw)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _materialize_structures(self, candidate: Any) -> Dict[str, Any]:
        """Resolve the per-property scoring input once, independent of any sweep.

        For a **structure** objective (dp_energy/dp_property) the input is a list
        of built (and optionally relaxed) cells — shared across objectives when
        ``share_structure``, else built per objective. For a **composition**
        objective (rf_magpie / FQN leaf) the input is just the candidate
        composition — no structure is built. When no structure objective exists,
        the builder is never touched (so ``base_poscar`` is not required).
        """
        prop_inputs: Dict[str, Any] = {}
        structure_props = [p for p in self.properties if p["backend"] != "composition"]
        if structure_props:
            if self.share_structure:
                structures = self._shared_builder.build(
                    candidate, n_configs=self.n_random_configs, rng=self._rng
                )
                if self.geo_opt_enabled:
                    structures = [self._relax(s) for s in structures]
                for prop in structure_props:
                    prop_inputs[prop["name"]] = structures
            else:
                for prop in structure_props:
                    builder = self._builders[prop["name"]]
                    structures = builder.build(
                        candidate, n_configs=prop["n_random_configs"], rng=self._rng
                    )
                    if self.geo_opt_enabled:
                        structures = [self._relax(s) for s in structures]
                    prop_inputs[prop["name"]] = structures
        # Composition objectives: scored directly on the candidate composition.
        for prop in self.properties:
            if prop["backend"] == "composition":
                prop_inputs[prop["name"]] = candidate
        return prop_inputs

    def _raw_stats(self, candidate: Any) -> Dict[str, Tuple[float, float]]:
        key = self._key(candidate)
        cached = self._stats_cache.get(key)
        if cached is not None:
            self._stats_cache.move_to_end(key)   # LRU: mark most-recently used
            return cached

        # 1. Materialize structures once (build + optional relax) — independent
        #    of any sweep value.
        prop_structures = self._materialize_structures(candidate)

        # 2. Score. No sweep: score each property once. Sweep: try each value,
        #    keep the single (shared) value maximizing the combined reward — the
        #    expensive build+relax above is reused across all sweep values.
        if self.sweep is None:
            stats = {
                p["name"]: self._score(p, prop_structures[p["name"]])
                for p in self.properties
            }
        else:
            best_stats: Optional[Dict[str, Tuple[float, float]]] = None
            best_reward: Optional[float] = None
            best_value: Optional[float] = None
            for v in self.sweep["values"]:
                trial = {
                    p["name"]: self._score(
                        p, prop_structures[p["name"]], fparam=self._fparam_at(p, v)
                    )
                    for p in self.properties
                }
                r = self._combine(trial)
                if best_reward is None or r > best_reward:
                    best_reward, best_stats, best_value = r, trial, v
            stats = dict(best_stats)  # type: ignore[arg-type]
            stats[self.sweep["name"]] = (float(best_value), 0.0)

        # Persistent LRU cache: each unique composition is built+relaxed+scored
        # exactly once for the predictor's lifetime; all later occurrences (across
        # training episodes / duplicate generation rollouts) are hits.
        self._stats_cache[key] = stats
        cap = self._stats_cache_size
        if cap and cap > 0:
            while len(self._stats_cache) > cap:
                self._stats_cache.popitem(last=False)   # evict least-recently used
        return stats

    @staticmethod
    def _fparam_at(prop: Dict[str, Any], value: float) -> Any:
        """Return *prop*'s fparam with its ``null`` slots filled by *value*."""
        if not prop["sweep_slots"]:
            return prop["fparam"]      # T-independent property (or no fparam)
        out = list(prop["fparam"])
        for j in prop["sweep_slots"]:
            out[j] = value
        return out

    def _score(
        self, prop: Dict[str, Any], structures: Any, *, fparam: Any = None,
    ) -> Tuple[float, float]:
        if prop["backend"] in ("composition", "structure_fqn"):
            return self._score_leaves(prop, structures)
        if prop["backend"] == "energy":
            from ..utils.dp_eval import eval_energy_ase
            values = eval_energy_ase(
                structures, self._get_calculators(prop),
                energy_per_atom=prop["energy_per_atom"],
            )
        else:
            from ..utils.dp_eval import eval_property_ensemble
            models, elem_to_type = self._get_prop_models(prop)
            values = eval_property_ensemble(
                structures, models, elem_to_type,
                output_index=prop["output_index"],
                output_aggregator=prop["output_aggregator"],
                fparam=fparam if fparam is not None else prop["fparam"],
                aparam=prop["aparam"],
            )
        arr = np.asarray(values, dtype=float)
        # Map each member back to real units before folding, so a head that emits
        # log(value) reports the real mean/std (and the std penalty is taken on the
        # real distribution).
        if prop.get("transform") == "exp":
            arr = np.exp(arr)
        return _fold_models_structures(arr)

    def _score_leaves(self, prop: Dict[str, Any], payload: Any) -> Tuple[float, float]:
        """Score one property through its leaf instance(s) — one instance per model.

        Two leaf styles are supported, detected per instance (see the module
        docstring's "Predictor contract"):

        * ``score_structures(atoms_list) -> [float per structure]`` /
          ``score(composition) -> float`` — the leaf is ONE model and the engine
          does all folding. This is the preferred form: it is the one that makes
          a single ``model:`` yield ``std == 0`` automatically.
        * ``predict_structures(atoms_list) -> (mean, std)`` /
          ``predict(composition) -> (mean, std)`` — the leaf owns its own fold.
          Correct when the leaf really does hold an internal ensemble (e.g.
          ``rf_magpie``'s random-forest tree variance); its ``(mean, std)`` is
          passed through untouched.
        """
        is_struct = prop["backend"] == "structure_fqn"
        per_model: List[float] = []
        for leaf in prop["instance"]:
            engine_folded = getattr(
                leaf, "score_structures" if is_struct else "score", None
            )
            if callable(engine_folded):
                if is_struct:
                    vals = np.asarray(list(engine_folded(payload)), dtype=float)
                    if vals.size == 0:
                        raise ValueError(
                            f"{type(leaf).__name__}.score_structures returned no values "
                            f"for property {prop['name']!r}."
                        )
                    # Average over the structures axis: that IS the property of a
                    # disordered composition, not an uncertainty.
                    per_model.append(float(vals.mean()))
                else:
                    per_model.append(float(engine_folded(payload)))
                continue
            # Self-folding leaf: it reports its own (mean, std). With several such
            # leaves we can only combine their means; a single one (the usual case)
            # passes straight through, preserving its uncertainty exactly.
            self_folded = getattr(leaf, "predict_structures" if is_struct else "predict")
            mean, std = self_folded(payload)
            if len(prop["instance"]) == 1:
                return float(mean), float(std)
            per_model.append(float(mean))

        arr = np.asarray(per_model, dtype=float)
        # Spread ACROSS models is the epistemic uncertainty; one model => 0.0.
        return float(arr.mean()), float(arr.std())

    def _get_calculators(self, prop: Dict[str, Any]) -> List[Any]:
        name = prop["name"]
        if name not in self._calc_cache:
            from ..utils.dp_eval import load_ase_calculators
            self._calc_cache[name] = load_ase_calculators(prop["models"], head=prop["head"])
        return self._calc_cache[name]

    def _get_prop_models(self, prop: Dict[str, Any]) -> Tuple[List[Any], Dict[str, int]]:
        name = prop["name"]
        if name not in self._prop_cache:
            from deepmd.pt.infer.deep_eval import DeepProperty
            models = [
                DeepProperty(model_file=p, auto_batch_size=False, head=prop["head"])
                for p in prop["models"]
            ]
            type_map = models[0].get_type_map()
            self._prop_cache[name] = (models, {el: i for i, el in enumerate(type_map)})
        return self._prop_cache[name]

    def _relax(self, atoms: Any) -> Any:
        from ..utils.structure import relax_structure

        if self._geo_calc is None:
            from deepmd.calculator import DP as DPCalculator
            head = self.geo_opt.get("head")
            model = self.geo_opt.get("model", "models/DPA-3.1-3M.pt")
            self._geo_calc = DPCalculator(model=model, **({"head": head} if head else {}))
        return relax_structure(
            atoms,
            calc=self._geo_calc,
            fmax=float(self.geo_opt.get("fmax", 0.001)),
            steps=int(self.geo_opt.get("steps", 1000)),
            relax_cell=bool(self.geo_opt.get("relax_cell", True)),
        )

    @staticmethod
    def _key(candidate: Any) -> tuple:
        """A hashable key for the candidate (flat comp or {group: {el: frac}}).

        A ``categorical`` group's picked value need not be numeric — it can be
        an element symbol or an O-form label stored verbatim (see
        ``env_multigroup.py:230`` / ``CategoricalGroup``) — so a leaf that
        doesn't parse as a float falls back to its string form.
        """
        def _leaf(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return str(v)
        def _flatten(d):
            items = []
            for el, v in d.items():
                if isinstance(v, dict):
                    items.append((str(el), tuple(sorted((str(e), _leaf(f)) for e, f in v.items()))))
                else:
                    items.append((str(el), _leaf(v)))
            return tuple(sorted(items, key=lambda t: t[0]))
        return _flatten(candidate)
