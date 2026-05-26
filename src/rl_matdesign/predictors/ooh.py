"""OOHCatalystPredictor — wraps DeepMDOverpotentialPredictor for general-framework."""
from __future__ import annotations

from typing import Dict, List, Tuple


class OOHCatalystPredictor:
    """PropertyPredictor for oxyhydroxide (ABCDE-OOH) overpotential.

    Parameters
    ----------
    base_poscar : str
        Path to the base slab POSCAR template.
    dp_models : List[str]
        Paths to DeepMD model checkpoint files (.pt) for the ensemble.
    objective : str
        "mean" | "mean_minus_kstd" (exploit) | "mean_plus_kstd" (explore).
    k : float
        Coefficient on the std term.
    n_random_configs : int
        Number of random alloy configs per composition.
    ads_height : float
        Adsorbate height (Å) above the surface.
    ads_dz : float
        Vertical shift (Å) between OOH adsorbate atoms.
    geo_opt : bool
        Whether to run geometry optimisation before property evaluation.
    geo_opt_model : str
        Path to DPA model for geometry optimisation (only used if geo_opt=True).
    rng_seed : int
        Seed forwarded to DPConfig.seed for structure generation.
    uncertainty : str
        "models" | "configs" | "total" — how ensemble std is computed.
    """

    def __init__(
        self,
        base_poscar: str,
        dp_models: List[str],
        *,
        objective: str = "mean_minus_kstd",
        k: float = 1.0,
        n_random_configs: int = 10,
        ads_height: float = 1.9,
        ads_dz: float = 1.0,
        geo_opt: bool = False,
        geo_opt_model: str = "./DPA-3.1-3M_1.pt",
        rng_seed: int = 123,
        uncertainty: str = "models",
    ) -> None:
        from abcde_ooh.dp_predictor import DPConfig, DeepMDOverpotentialPredictor
        from ..training import objective_from_mean_std

        self._objective_fn = objective_from_mean_std
        self.objective = objective
        self.k = k
        self.uncertainty = uncertainty

        cfg = DPConfig(
            base_poscar=base_poscar,
            model_files=tuple(dp_models),
            n_random_configs=n_random_configs,
            ads_height=ads_height,
            ads_dz=ads_dz,
            seed=rng_seed,
            geo_opt=geo_opt,
            geo_opt_model=geo_opt_model,
        )
        self._predictor = DeepMDOverpotentialPredictor(cfg)

    def predict(self, composition: Dict[str, float]) -> Tuple[float, float]:
        """Return (reward, std) for *composition*.

        Lower overpotential is better, so mean_overpotential is negated before
        the objective function, mirroring how HEAPropertyPredictor negates energy.
        """
        mean_overpotential, std = self._predictor.predict_overpotential(
            composition, uncertainty=self.uncertainty
        )
        reward = self._objective_fn(-mean_overpotential, std, self.objective, self.k)
        return reward, std

    def batch_predict(self, compositions: List[Dict[str, float]]) -> List[Tuple[float, float]]:
        return [self.predict(c) for c in compositions]
