"""DPStructurePredictor — generic DeepMD-ensemble property predictor.

Subsumes the previous HEA / perovskite predictors: they were byte-identical
except for the default ``site_symbol``. Now both are thin subclasses.

Behavior
--------
1. ``substitute_sites`` places target-composition elements on the placeholder
   ``site_symbol`` positions in the template POSCAR (creates
   ``n_random_configs`` random site assignments).
2. Each (structure, DP-model) pair is evaluated via the ASE DPCalculator
   path (``atoms.get_potential_energy()``).
3. Mean and std across all (structure, model) values are combined into a
   reward via ``objective_from_mean_std(-mean, std, objective, k)``.

The negation is because lower formation energy is more thermodynamically
stable but the framework always maximises reward. If your model returns
something that should be maximised directly (e.g. an elastic constant),
override :meth:`_value_sign` to return ``+1``.

Configuration keys
------------------
Required:
    - ``dp_models``: list of DeepMD ``.pt`` checkpoint paths.
    - ``base_poscar`` *(or legacy ``poscar`` / ``poscar_template``)*: template
      POSCAR with placeholder sites marked by ``site_symbol``.
Optional:
    - ``site_symbol`` (default depends on subclass; ``"X"`` for the base class).
    - ``objective`` (default ``"mean_minus_kstd"``).
    - ``k`` (default ``1.0``).
    - ``n_random_configs`` (default ``5``).
    - ``energy_per_atom`` (default ``True``).
    - ``dp_head`` (default ``None``): which head of a multi-task DP checkpoint
      to evaluate (e.g. ``"Omat24"``). Required when the checkpoint exposes
      multiple heads; single-task checkpoints can leave this unset.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class DPStructurePredictor:
    """Generic substitute-and-evaluate DP predictor.

    Constructed from a YAML-style ``cfg`` dict so the registry can drop a
    config straight in without per-system kwarg lists.
    """

    DEFAULT_SITE_SYMBOL: str = "X"

    def __init__(self, cfg: Dict[str, Any], *, seed: Optional[int] = None) -> None:
        self.cfg = cfg
        # Accept three historical names for the POSCAR template path.
        poscar = cfg.get("base_poscar") or cfg.get("poscar") or cfg.get("poscar_template")
        if not poscar:
            raise ValueError(
                "DPStructurePredictor needs a POSCAR template — set 'base_poscar' "
                "(or legacy 'poscar' / 'poscar_template') in the config."
            )
        if "dp_models" not in cfg:
            raise ValueError("DPStructurePredictor needs 'dp_models' in the config.")

        self.poscar_template: str = str(poscar)
        self.dp_models: List[str] = list(cfg["dp_models"])
        self.site_symbol: str = cfg.get("site_symbol", self.DEFAULT_SITE_SYMBOL)
        self.objective: str = cfg.get("objective", "mean_minus_kstd")
        self.k: float = float(cfg.get("k", 1.0))
        self.n_random_configs: int = int(cfg.get("n_random_configs", 5))
        self.energy_per_atom: bool = bool(cfg.get("energy_per_atom", True))
        head_val = cfg.get("dp_head")
        self.dp_head: Optional[str] = str(head_val) if head_val else None
        self._rng = np.random.default_rng(seed)
        self._dp_calculators: Optional[List[Any]] = None

    # ------------------------------------------------------------------
    # PropertyPredictor Protocol
    # ------------------------------------------------------------------

    def predict(self, composition: Dict[str, float]) -> Tuple[float, float]:
        """Return ``(reward, std)`` for *composition*."""
        from ..utils.structure import substitute_sites
        from ..utils.dp_eval import eval_energy_ase
        from ..training import objective_from_mean_std

        structures = substitute_sites(
            template_poscar=self.poscar_template,
            composition=composition,
            site_symbol=self.site_symbol,
            n_configs=self.n_random_configs,
            rng=self._rng,
        )
        values = eval_energy_ase(
            structures,
            self._get_calculators(),
            energy_per_atom=self.energy_per_atom,
        )
        mean_v = float(np.mean(values))
        std_v = float(np.std(values))
        reward = objective_from_mean_std(
            self._value_sign() * mean_v, std_v, self.objective, self.k,
        )
        return reward, std_v

    def batch_predict(
        self, compositions: List[Dict[str, float]]
    ) -> List[Tuple[float, float]]:
        return [self.predict(c) for c in compositions]

    # ------------------------------------------------------------------
    # Hooks subclasses may override
    # ------------------------------------------------------------------

    def _value_sign(self) -> float:
        """Return ``-1.0`` (default) so lower energy → higher reward.

        Override to ``+1.0`` for properties that should be maximised directly.
        """
        return -1.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_calculators(self) -> List[Any]:
        if self._dp_calculators is None:
            from ..utils.dp_eval import load_ase_calculators
            self._dp_calculators = load_ase_calculators(
                self.dp_models, head=self.dp_head,
            )
        return self._dp_calculators
