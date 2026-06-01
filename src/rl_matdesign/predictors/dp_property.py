"""DPPropertyPredictor — DeepMD ensemble predictor for vector-valued heads.

Sibling of :class:`DPStructurePredictor`. Both substitute placeholder sites in
a template POSCAR with the target composition; the difference is the DeepMD
backend:

* ``DPStructurePredictor`` — ASE ``DPCalculator`` → scalar potential energy.
  Use for energy-based reward.
* ``DPPropertyPredictor`` (this) — ``deepmd.pt.infer.deep_eval.DeepProperty``
  → property vector → one scalar via ``output_index`` / ``output_aggregator``.
  Use when the DP head you want to read returns a multi-element property
  vector (e.g. a "property" head exposing four physical properties per
  structure).

Configuration keys
------------------
Required:
    - ``dp_models``: list of DeepMD ``.pt`` checkpoint paths.
    - ``base_poscar`` *(or legacy ``poscar`` / ``poscar_template``)*: template
      POSCAR with placeholder sites marked by ``site_symbol``.
Optional:
    - ``site_symbol`` (default ``"X"``).
    - ``dp_head`` (default ``"property"``): which head to evaluate. Required
      for multi-task checkpoints.
    - ``output_index`` (default ``0``): which component of the property vector
      to use as the scalar reward driver.
    - ``output_aggregator`` (default ``"index"``): ``"index"``, ``"mean"``, or
      ``"max"`` — how to collapse the vector to a scalar.
    - ``maximize`` (default ``False``): set ``True`` if higher property value
      → better. Default ``False`` matches formation-energy-style use cases
      where lower is better.
    - ``objective`` (default ``"mean_minus_kstd"``).
    - ``k`` (default ``1.0``).
    - ``n_random_configs`` (default ``5``).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class DPPropertyPredictor:
    """Substitute-and-evaluate DP predictor for vector-output property heads."""

    DEFAULT_SITE_SYMBOL: str = "X"
    DEFAULT_HEAD: str = "property"

    def __init__(self, cfg: Dict[str, Any], *, seed: Optional[int] = None) -> None:
        self.cfg = cfg
        poscar = cfg.get("base_poscar") or cfg.get("poscar") or cfg.get("poscar_template")
        if not poscar:
            raise ValueError(
                "DPPropertyPredictor needs a POSCAR template — set 'base_poscar' "
                "(or legacy 'poscar' / 'poscar_template') in the config."
            )
        if "dp_models" not in cfg:
            raise ValueError("DPPropertyPredictor needs 'dp_models' in the config.")

        self.poscar_template: str = str(poscar)
        self.dp_models_paths: List[str] = list(cfg["dp_models"])
        self.site_symbol: str = cfg.get("site_symbol", self.DEFAULT_SITE_SYMBOL)
        head_val = cfg.get("dp_head", self.DEFAULT_HEAD)
        self.dp_head: str = str(head_val) if head_val else self.DEFAULT_HEAD
        self.output_index: int = int(cfg.get("output_index", 0))
        self.output_aggregator: str = str(cfg.get("output_aggregator", "index"))
        self.maximize: bool = bool(cfg.get("maximize", False))
        self.objective: str = cfg.get("objective", "mean_minus_kstd")
        self.k: float = float(cfg.get("k", 1.0))
        self.n_random_configs: int = int(cfg.get("n_random_configs", 5))
        self._rng = np.random.default_rng(seed)
        self._dp_models: Optional[List[Any]] = None
        self._elem_to_type: Optional[Dict[str, int]] = None

    def predict(self, composition: Dict[str, float]) -> Tuple[float, float]:
        """Return ``(reward, std)`` for *composition*."""
        from ..training import objective_from_mean_std

        mean_v, std_v = self.raw_mean_std(composition)
        reward = objective_from_mean_std(
            self._value_sign() * mean_v, std_v, self.objective, self.k,
        )
        return reward, std_v

    def raw_mean_std(self, composition: Dict[str, float]) -> Tuple[float, float]:
        """Return raw ``(mean, std)`` over the (model × random-config) population.

        No sign flip, no objective folding. Used by :class:`CompositePredictor`
        to combine multiple physical properties at their native scales.
        """
        from ..utils.structure import substitute_sites
        from ..utils.dp_eval import pick_scalar

        models = self._get_models()
        elem_to_type = self._elem_to_type
        assert elem_to_type is not None  # _get_models populates this

        structures = substitute_sites(
            template_poscar=self.poscar_template,
            composition=composition,
            site_symbol=self.site_symbol,
            n_configs=self.n_random_configs,
            rng=self._rng,
        )

        nframes = len(structures)
        natoms = len(structures[0])
        coords = np.zeros((nframes, natoms, 3), dtype=np.float64)
        cells = np.zeros((nframes, 3, 3), dtype=np.float64)
        atom_types = np.zeros((nframes, natoms), dtype=np.int32)
        for i, atoms in enumerate(structures):
            coords[i] = atoms.get_positions()
            cells[i] = atoms.get_cell().array
            syms = atoms.get_chemical_symbols()
            unknown = sorted({s for s in syms if s not in elem_to_type})
            if unknown:
                raise RuntimeError(
                    f"Elements {unknown} are not in the DP model type_map "
                    f"({sorted(elem_to_type)}). Check that the DP checkpoint "
                    "covers every cation in your config."
                )
            atom_types[i] = np.fromiter(
                (elem_to_type[s] for s in syms), dtype=np.int32, count=natoms,
            )

        # Each model returns shape (nframes, output_dim); reduce per-frame to scalar.
        all_values: List[float] = []
        for dp in models:
            res = np.asarray(
                dp.eval(coords=coords, cells=cells, atom_types=atom_types, mixed_type=True)
            )
            if res.ndim == 1:
                res = res.reshape(1, -1) if nframes == 1 else res.reshape(nframes, -1)
            for frame_vec in res:
                all_values.append(
                    pick_scalar(
                        frame_vec,
                        output_index=self.output_index,
                        output_aggregator=self.output_aggregator,
                    )
                )

        arr = np.asarray(all_values, dtype=float)
        return float(np.mean(arr)), float(np.std(arr))

    def batch_predict(
        self, compositions: List[Dict[str, float]]
    ) -> List[Tuple[float, float]]:
        return [self.predict(c) for c in compositions]

    def _value_sign(self) -> float:
        return 1.0 if self.maximize else -1.0

    def _get_models(self) -> List[Any]:
        if self._dp_models is None:
            try:
                from deepmd.pt.infer.deep_eval import DeepProperty
            except ImportError as exc:
                raise ImportError(
                    "DPPropertyPredictor requires the deepmd-kit PyTorch backend "
                    "(`deepmd.pt.infer.deep_eval.DeepProperty`)."
                ) from exc
            self._dp_models = [
                DeepProperty(model_file=p, auto_batch_size=False, head=self.dp_head)
                for p in self.dp_models_paths
            ]
            type_map = self._dp_models[0].get_type_map()
            self._elem_to_type = {el: i for i, el in enumerate(type_map)}
        return self._dp_models
