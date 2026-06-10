"""StructurePipelinePredictor — build one structure, relax once, score N properties.

The "structure-sharing composite": unlike :class:`CompositePredictor` (which hands
each child the *composition* so every child rebuilds its own structure), this
predictor builds **one** structure, optionally **relaxes it once**, and evaluates
**several** DeepMD property ensembles on that single relaxed structure, then
combines them with per-property weights. Right for multi-objective scoring where
the objectives must describe the *same* expensive relaxed cell (e.g. conductivity
+ stability of a doped Li6PS6 supercell).

Pipeline::

    composition/groups
        -> builder.build(...)              # composition/groups -> ASE Atoms (Layer 1+2)
        -> relax_structure(...) [optional] # once, shared
        -> for each property: DeepProperty ensemble -> (mean, std)
        -> reward = Σ weight * objective_from_mean_std(direction*mean, std, obj) / scale

Config keys
-----------
    builder:         builder registry name or ``pkg.mod:Class`` FQN (composition/
                     groups -> ``List[ase.Atoms]``; gets the whole cfg).
    n_random_configs: random realizations per candidate (default 1).
    geo_opt:         optional ``{model, head, fmax, steps, relax_cell, enabled}``.
                     ``model`` defaults to ``models/DPA-3.1-3M.pt``; ``head`` is
                     user-defined (e.g. ``SSE_ABACUS``). Omit / ``enabled: false``
                     to score the unrelaxed structure.
    properties:      list of ``{name, models, head, direction(max|min), weight,
                     scale, objective, output_index, output_aggregator}``.
    k:               uncertainty coefficient for the objective folding.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class StructurePipelinePredictor:
    def __init__(self, cfg: Dict[str, Any], *, seed: Optional[int] = None) -> None:
        from ..registry import resolve_builder

        builder_kind = cfg.get("builder")
        if not builder_kind:
            raise ValueError(
                "structure_pipeline needs 'builder' (registry name or FQN) — the "
                "composition/groups -> ASE Atoms factory."
            )
        self.builder = resolve_builder(builder_kind, cfg, seed=seed)

        self.n_random_configs: int = int(cfg.get("n_random_configs", 1))
        self.k: float = float(cfg.get("k", 1.0))

        go = cfg.get("geo_opt") or {}
        self.geo_opt: Dict[str, Any] = dict(go)
        self.geo_opt_enabled: bool = bool(go) and bool(go.get("enabled", True))

        props = cfg.get("properties")
        if not props:
            raise ValueError("structure_pipeline needs a non-empty 'properties' list.")
        self.properties: List[Dict[str, Any]] = []
        for p in props:
            if "models" not in p:
                raise ValueError(f"property {p.get('name')!r} needs 'models'.")
            self.properties.append(
                {
                    "name": str(p.get("name", f"prop{len(self.properties)}")),
                    "models": list(p["models"]),
                    "head": p.get("head"),
                    "output_index": int(p.get("output_index", 0)),
                    "output_aggregator": str(p.get("output_aggregator", "index")),
                    "direction": 1.0 if str(p.get("direction", "max")).lower() == "max" else -1.0,
                    "weight": float(p.get("weight", 1.0)),
                    "scale": float(p.get("scale", 1.0)),
                    "objective": str(p.get("objective", "mean_minus_kstd")),
                }
            )

        self._rng = np.random.default_rng(seed)
        self._prop_models: Dict[str, Tuple[List[Any], Dict[str, int]]] = {}
        self._geo_calc: Optional[Any] = None

    # ------------------------------------------------------------------

    def predict(self, candidate: Any) -> Tuple[float, float]:
        """*candidate* is whatever the env emits (a flat comp dict or a structured
        ``{group: {el: frac}}`` mapping); the builder knows how to read it."""
        from ..training import objective_from_mean_std

        structures = self.builder.build(
            candidate, n_configs=self.n_random_configs, rng=self._rng
        )
        if self.geo_opt_enabled:
            structures = [self._relax(s) for s in structures]

        reward = 0.0
        stds: List[float] = []
        for prop in self.properties:
            mean_v, std_v = self._eval_property(prop, structures)
            v = objective_from_mean_std(
                prop["direction"] * mean_v, std_v, prop["objective"], self.k
            )
            reward += prop["weight"] * v / prop["scale"]
            stds.append(std_v)
        return float(reward), float(np.mean(stds)) if stds else 0.0

    def batch_predict(self, candidates: List[Any]) -> List[Tuple[float, float]]:
        return [self.predict(c) for c in candidates]

    # ------------------------------------------------------------------

    def _relax(self, atoms: "ase.Atoms") -> "ase.Atoms":
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

    def _eval_property(
        self, prop: Dict[str, Any], structures: List["ase.Atoms"]
    ) -> Tuple[float, float]:
        from ..utils.dp_eval import eval_property_ensemble

        models, elem_to_type = self._get_prop_models(prop)
        values = eval_property_ensemble(
            structures,
            models,
            elem_to_type,
            output_index=prop["output_index"],
            output_aggregator=prop["output_aggregator"],
        )
        arr = np.asarray(values, dtype=float)
        return float(np.mean(arr)), float(np.std(arr))

    def _get_prop_models(
        self, prop: Dict[str, Any]
    ) -> Tuple[List[Any], Dict[str, int]]:
        name = prop["name"]
        if name not in self._prop_models:
            from deepmd.pt.infer.deep_eval import DeepProperty

            models = [
                DeepProperty(model_file=p, auto_batch_size=False, head=prop["head"])
                for p in prop["models"]
            ]
            type_map = models[0].get_type_map()
            self._prop_models[name] = (models, {el: i for i, el in enumerate(type_map)})
        return self._prop_models[name]
