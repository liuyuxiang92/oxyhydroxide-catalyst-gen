"""HEAPropertyPredictor — DPA-ensemble reward oracle for High-Entropy Alloys.

Given a composition dict and a FCC/BCC POSCAR template, this predictor:
1. Substitutes the placeholder sites with elements according to the composition
   fractions using :func:`~rl_matdesign.utils.structure.substitute_sites`.
2. Evaluates each structure with a DeepMD (DPA) ensemble.
3. Returns (reward, std) where reward = f(-mean_Ef_per_atom, std, objective).

Lower formation energy = more thermodynamically stable = higher reward.
The caller sets the sign convention by choosing an appropriate ``objective``.

Requirements
------------
    pip install ase deepmd-kit
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


class HEAPropertyPredictor:
    """DeepMD ensemble predictor for high-entropy alloy formation energy.

    Parameters
    ----------
    poscar_template:
        Path to a POSCAR/CONTCAR containing the FCC or BCC supercell.
        All sites with *site_symbol* will be substituted according to the
        target composition.
    dp_models:
        List of paths to DeepMD model checkpoint files (.pt).
        Ensemble variance is used as uncertainty estimate.
    objective:
        How to aggregate (mean, std) into a scalar reward:
        ``"mean"`` | ``"mean_minus_kstd"`` (exploit) | ``"mean_plus_kstd"`` (explore).
    k:
        Coefficient on the std term.
    n_random_configs:
        Number of random solid-solution configurations to generate and average.
        Ignored when ``structure_mode="sqs"``.
    site_symbol:
        Element symbol marking substitution sites in the template.
    structure_mode:
        ``"random"`` (fast, for RL training) or ``"sqs"`` (rigorous, for DFT
        validation).  See :func:`~rl_matdesign.utils.structure.substitute_sites`.
    energy_per_atom:
        If True (default), normalise energies by number of atoms.
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
        site_symbol: str = "X",
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

        reward = objective_from_mean_std(-mean_energy, energy_std, objective, k)

        The negation is because lower formation energy is better but the
        framework always maximises reward.
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

        # Negate energy: lower energy (more stable) → higher reward.
        reward = objective_from_mean_std(-mean_energy, std_energy, self.objective, self.k)
        return reward, std_energy

    def batch_predict(
        self, compositions: List[Dict[str, float]]
    ) -> List[Tuple[float, float]]:
        """Loop over predict(); override for batched efficiency if needed."""
        return [self.predict(c) for c in compositions]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_calculators(self) -> List:
        """Lazy-load DeepMD calculators (one per model file)."""
        if self._dp_calculators is not None:
            return self._dp_calculators

        try:
            from deepmd.calculator import DP as DPCalculator
        except ImportError as exc:
            raise ImportError(
                "HEAPropertyPredictor requires deepmd-kit: pip install deepmd-kit"
            ) from exc

        self._dp_calculators = [DPCalculator(model=p) for p in self.dp_models]
        return self._dp_calculators
