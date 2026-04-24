"""PerovskitePropertyPredictor — DPA-ensemble reward oracle for perovskite oxides.

Designed for high-entropy perovskite B-site doping (e.g. La(ABCDE)O₃).
Given a composition dict for the B-site elements and a LaBO₃ POSCAR template,
this predictor substitutes the B-site placeholder atoms and evaluates the
structure with a DeepMD ensemble to predict formation enthalpy.

Reward = ``objective_from_mean_std(-mean_Hf, std_Hf, objective, k)``
so that lower formation enthalpy (more stable) maps to higher reward.

Requirements
------------
    pip install ase deepmd-kit
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


class PerovskitePropertyPredictor:
    """DeepMD ensemble predictor for perovskite B-site formation enthalpy.

    Parameters
    ----------
    poscar_template:
        Path to a LaBO₃ POSCAR with B-site atoms marked as *site_symbol*.
        The A-site (La) and anion (O) atoms are left unchanged.
    dp_models:
        List of DeepMD model checkpoint paths (.pt) for the ensemble.
    objective:
        ``"mean"`` | ``"mean_minus_kstd"`` (exploit) | ``"mean_plus_kstd"`` (explore).
    k:
        Coefficient on the std term.
    n_random_configs:
        Number of random B-site configurations (for ``structure_mode="random"``).
    site_symbol:
        Placeholder element in the template marking B-sites to substitute.
        Typically ``"Fe"`` or another single symbol.
    structure_mode:
        ``"random"`` or ``"sqs"``.  Use ``"sqs"`` for DFT validation.
    energy_per_atom:
        Normalise energy by number of atoms (default True).
    rng_seed:
        Optional seed for the random structure generator.
    """

    def __init__(
        self,
        poscar_template: str,
        dp_models: List[str],
        *,
        objective: str = "mean_minus_kstd",
        k: float = 1.0,
        n_random_configs: int = 5,
        site_symbol: str = "Fe",
        structure_mode: str = "random",
        energy_per_atom: bool = True,
        rng_seed: Optional[int] = None,
    ) -> None:
        self.poscar_template = poscar_template
        self.dp_models = dp_models
        self.objective = objective
        self.k = k
        self.n_random_configs = n_random_configs
        self.site_symbol = site_symbol
        self.structure_mode = structure_mode
        self.energy_per_atom = energy_per_atom
        self._rng = np.random.default_rng(rng_seed)
        self._dp_calculators: Optional[List] = None

    # ------------------------------------------------------------------
    # PropertyPredictor Protocol
    # ------------------------------------------------------------------

    def predict(self, composition: Dict[str, float]) -> Tuple[float, float]:
        """Return (mean_reward, std_reward) for *composition*.

        Only the B-site composition is passed; La and O are fixed by the template.
        """
        from ..utils.structure import substitute_sites
        from ..training import objective_from_mean_std

        structures = substitute_sites(
            template_poscar=self.poscar_template,
            composition=composition,
            site_symbol=self.site_symbol,
            mode=self.structure_mode,
            n_configs=self.n_random_configs,
            rng=self._rng,
        )

        all_energies: List[float] = []
        calcs = self._get_calculators()
        for atoms in structures:
            for calc in calcs:
                atoms_copy = atoms.copy()
                atoms_copy.calc = calc
                e = atoms_copy.get_potential_energy()
                if self.energy_per_atom:
                    e /= len(atoms_copy)
                all_energies.append(float(e))

        mean_energy = float(np.mean(all_energies))
        std_energy = float(np.std(all_energies))

        reward = objective_from_mean_std(-mean_energy, std_energy, self.objective, self.k)
        return reward, std_energy

    def batch_predict(
        self, compositions: List[Dict[str, float]]
    ) -> List[Tuple[float, float]]:
        return [self.predict(c) for c in compositions]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_calculators(self) -> List:
        if self._dp_calculators is not None:
            return self._dp_calculators

        try:
            from deepmd.calculator import DP as DPCalculator
        except ImportError as exc:
            raise ImportError(
                "PerovskitePropertyPredictor requires deepmd-kit: pip install deepmd-kit"
            ) from exc

        self._dp_calculators = [DPCalculator(model=p) for p in self.dp_models]
        return self._dp_calculators
