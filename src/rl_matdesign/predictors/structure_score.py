"""StructureScorePredictor — the one structure-based predictor.

Replaces the four near-twin predictors (``dp_structure``, ``dp_property``,
``composite``, ``structure_pipeline``) with a single pipeline you steer with
config dials:

    builder.build(candidate)            # how to make the structure (default: substitute)
      -> [relax once, optional]         # geo_opt: present  -> relax, absent -> score as-is
      -> score each property            # backend: energy | property
      -> reward = Σ weight·objective_from_mean_std(direction·mean, std, obj) / scale

The two structural regimes are one boolean:

* ``share_structure: true`` (default) — build **one** structure with the
  top-level builder, relax it **once**, score every property on that shared
  cell. This is the old ``structure_pipeline`` (e.g. conductivity + stability of
  the *same* relaxed Li6PS6 supercell). A single property with
  ``backend: energy`` is the old ``dp_structure``; ``backend: property`` is the
  old ``dp_property``.
* ``share_structure: false`` — each objective builds (and optionally relaxes)
  its **own** structure independently. This is the old ``composite`` (objectives
  may use different builders / templates / site_symbols).

Consumer contract (``training.py``): exposes ``predict`` (reward, std),
``predict_raw`` (Σ weight·direction·mean/scale, for the ``dp_mean`` CSV column),
and ``per_objective_stats`` ({name: (mean, std)} for per-objective CSV columns).

Config keys
-----------
    builder:          builder registry name or ``pkg.mod:Class`` FQN (default
                      ``substitute``). Gets the whole cfg.
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
        name, backend (energy|property), models (or legacy dp_models), head,
        output_index, output_aggregator, energy_per_atom, direction (max|min),
        weight, scale, objective, fparam, aparam. A ``null`` in ``fparam`` marks
        the slot filled by the sweep value (requires a top-level ``sweep``). In
        ``share_structure: false`` an entry may also carry its own builder /
        base_poscar / site_symbol / n_random_configs.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


_DIRECTION_ALIASES = {"min": -1.0, "minimize": -1.0, "max": +1.0, "maximize": +1.0}
_OBJECTIVES = ("mean", "mean_minus_kstd", "mean_plus_kstd")
# Builder knobs inherited by per-objective builders (share_structure: false)
# when the objective entry doesn't set them. Mirrors the old composite behavior.
_SHARED_BUILDER_KEYS = (
    "builder", "base_poscar", "poscar", "poscar_template", "site_symbol",
    "n_random_configs",
)


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
                "structure_score needs a non-empty 'properties' list. Each entry "
                "is one objective: {backend: energy|property, models: [...], "
                "direction: max|min, ...}."
            )

        top_builder = str(cfg.get("builder", "substitute"))
        shared_builder_cfg = {k: cfg[k] for k in _SHARED_BUILDER_KEYS if k in cfg}

        self.properties: List[Dict[str, Any]] = []
        self._builders: List[Any] = []      # per-property builder (share_structure: false)
        seen: set[str] = set()
        for i, p in enumerate(raw_props):
            name = str(p.get("name", f"prop{i}"))
            if name in seen:
                raise ValueError(
                    f"properties[{i}].name = {name!r} duplicates an earlier "
                    "objective (used as the CSV column prefix); names must be unique."
                )
            seen.add(name)

            backend = str(p.get("backend", "property")).lower()
            if backend not in ("energy", "property"):
                raise ValueError(
                    f"properties[{i}].backend = {backend!r}; expected 'energy' "
                    "(DP potential energy) or 'property' (DP property head)."
                )

            models = p.get("models") or p.get("dp_models")
            if not models:
                raise ValueError(
                    f"properties[{i}] ({name!r}) needs 'models' (list of DeepMD "
                    "checkpoint paths)."
                )

            direction = str(p.get("direction", "max")).lower()
            if direction not in _DIRECTION_ALIASES:
                raise ValueError(
                    f"properties[{i}].direction = {direction!r}; expected one of "
                    f"{sorted(_DIRECTION_ALIASES)}."
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

            self.properties.append({
                "name": name,
                "backend": backend,
                "models": list(models),
                "head": p.get("head"),
                "output_index": int(p.get("output_index", 0)),
                "output_aggregator": str(p.get("output_aggregator", "index")),
                "energy_per_atom": bool(p.get("energy_per_atom", True)),
                "fparam": fparam,
                "sweep_slots": sweep_slots,
                "aparam": p.get("aparam"),
                "direction": _DIRECTION_ALIASES[direction],
                "weight": float(p.get("weight", 1.0)),
                "scale": scale,
                "objective": objective,
                "n_random_configs": int(p.get("n_random_configs", self.n_random_configs)),
            })

            if not self.share_structure:
                # Each objective builds independently: inherit top-level builder
                # knobs, let the entry override. Decorrelate the per-objective RNG.
                sub_cfg = {**shared_builder_cfg, **p}
                child_seed = None if seed is None else int(seed) + i
                self._builders.append(
                    resolve_builder(str(sub_cfg.get("builder", top_builder)), sub_cfg, seed=child_seed)
                )

        # Shared builder (share_structure: true) — build once, score all on it.
        self._shared_builder = (
            resolve_builder(top_builder, cfg, seed=seed) if self.share_structure else None
        )

        self._rng = np.random.default_rng(seed)
        self._calc_cache: Dict[str, Any] = {}     # name -> ase calculators (energy backend)
        self._prop_cache: Dict[str, Tuple[List[Any], Dict[str, int]]] = {}  # name -> property models
        self._geo_calc: Optional[Any] = None
        self._stats_cache: Dict[tuple, Dict[str, Tuple[float, float]]] = {}

    # ------------------------------------------------------------------
    # PropertyPredictor protocol
    # ------------------------------------------------------------------

    def _combine(self, stats: Dict[str, Tuple[float, float]]) -> float:
        """Weighted/signed/scaled fold of per-property stats into one reward.

        Shared by ``predict`` and the sweep argmax so both rank temperatures by
        the identical objective.
        """
        from ..training import objective_from_mean_std

        reward = 0.0
        for prop in self.properties:
            m, s = stats[prop["name"]]
            v = objective_from_mean_std(prop["direction"] * m, s, prop["objective"], self.k)
            reward += prop["weight"] * v / prop["scale"]
        return float(reward)

    def predict(self, candidate: Any) -> Tuple[float, float]:
        stats = self._raw_stats(candidate)
        reward = self._combine(stats)
        stds = [stats[prop["name"]][1] for prop in self.properties]
        return reward, float(np.mean(stds)) if stds else 0.0

    def predict_raw(self, candidate: Any) -> Tuple[float, float]:
        """Combined value from raw means (no std penalty) — for the dp_mean column."""
        stats = self._raw_stats(candidate)
        raw = 0.0
        for prop in self.properties:
            m, _s = stats[prop["name"]]
            raw += prop["weight"] * prop["direction"] * m / prop["scale"]
        return float(raw), 0.0

    def per_objective_stats(self, candidate: Any) -> Dict[str, Tuple[float, float]]:
        """Return ``{name: (mean, std)}`` per objective, for CSV logging."""
        return dict(self._raw_stats(candidate))

    def batch_predict(self, candidates: List[Any]) -> List[Tuple[float, float]]:
        return [self.predict(c) for c in candidates]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _raw_stats(self, candidate: Any) -> Dict[str, Tuple[float, float]]:
        key = self._key(candidate)
        cached = self._stats_cache.get(key)
        if cached is not None:
            return cached

        # 1. Materialize structures once (build + optional relax) — independent
        #    of any sweep value. Shared: one relaxed cell scored by every
        #    property; else: each property builds (and relaxes) its own.
        prop_structures: Dict[str, List[Any]] = {}
        if self.share_structure:
            structures = self._shared_builder.build(
                candidate, n_configs=self.n_random_configs, rng=self._rng
            )
            if self.geo_opt_enabled:
                structures = [self._relax(s) for s in structures]
            for prop in self.properties:
                prop_structures[prop["name"]] = structures
        else:
            for prop, builder in zip(self.properties, self._builders):
                structures = builder.build(
                    candidate, n_configs=prop["n_random_configs"], rng=self._rng
                )
                if self.geo_opt_enabled:
                    structures = [self._relax(s) for s in structures]
                prop_structures[prop["name"]] = structures

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

        # Single-entry cache: the generation loop already memoizes by comp key
        # at the outer level; this guards accidental double-eval within one call.
        self._stats_cache = {key: stats}
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
        self, prop: Dict[str, Any], structures: List[Any], *, fparam: Any = None,
    ) -> Tuple[float, float]:
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
        return float(np.mean(arr)), float(np.std(arr))

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
        """A hashable key for the candidate (flat comp or {group: {el: frac}})."""
        def _flatten(d):
            items = []
            for el, v in d.items():
                if isinstance(v, dict):
                    items.append((str(el), tuple(sorted((str(e), float(f)) for e, f in v.items()))))
                else:
                    items.append((str(el), float(v)))
            return tuple(sorted(items, key=lambda t: t[0]))
        return _flatten(candidate)
