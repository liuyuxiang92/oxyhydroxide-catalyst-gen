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
        output_index: int = 0,
    ) -> None:
        from abcde_ooh.dp_predictor import DPConfig, DeepMDOverpotentialPredictor
        from ..training import objective_from_mean_std

        self._objective_fn = objective_from_mean_std
        self.objective = objective
        self.k = k
        self.uncertainty = uncertainty
        self.output_index = output_index

        cfg = DPConfig(
            base_poscar=base_poscar,
            model_files=tuple(dp_models),
            n_random_configs=n_random_configs,
            ads_height=ads_height,
            ads_dz=ads_dz,
            seed=rng_seed,
            geo_opt=geo_opt,
            geo_opt_model=geo_opt_model,
            output_index=output_index,
        )
        self._predictor = DeepMDOverpotentialPredictor(cfg)
        # Composition cache: comp_key → (raw_mean_overpotential, std)
        # Shared across predict_raw() calls to avoid redundant DeepMD evaluations.
        self._cache: Dict[tuple, Tuple[float, float]] = {}

    @staticmethod
    def _comp_key(composition: Dict[str, float]) -> tuple:
        """Canonical cache key matching classical-dqn _comp_key (units of 1/20).

        Note: this key is composition-only and does NOT include ``output_index``.
        If you load a `dp_cache` saved with one ``output_index`` and resume
        training under a different ``output_index``, the cached values may be
        stale. Start a fresh run when changing ``output_index``.
        """
        return tuple(sorted(
            (k, int(round(v * 20)))
            for k, v in composition.items()
            if int(round(v * 20)) > 0
        ))

    def predict_raw(self, composition: Dict[str, float]) -> Tuple[float, float]:
        """Return (raw_mean_overpotential, std) — untransformed DeepMD values.

        Caches by composition key so repeated calls (e.g. once in env.reward_fn
        during training/generation and once in generate_candidates for CSV output)
        never hit DeepMD twice for the same composition.
        """
        key = self._comp_key(composition)
        if key in self._cache:
            return self._cache[key]
        raw_mean, std = self._predictor.predict_overpotential(
            composition, uncertainty=self.uncertainty
        )
        self._cache[key] = (raw_mean, std)
        return raw_mean, std

    def predict(self, composition: Dict[str, float]) -> Tuple[float, float]:
        """Return (reward, std) for *composition*.

        Lower overpotential is better, so mean_overpotential is negated before
        the objective function, mirroring how an energy-backend objective uses
        direction: min.
        Internally calls predict_raw() so results are cached.
        """
        raw_mean, std = self.predict_raw(composition)
        reward = self._objective_fn(-raw_mean, std, self.objective, self.k)
        return reward, std

    def check_phase(self, composition: Dict[str, float]) -> Tuple[bool, str]:
        """Return (primary_ok, primary_label) matching classical-dqn generate_pg output."""
        from abcde_ooh.constraints.primary_phase import check_primary_phase
        ok, label = check_primary_phase(composition)
        return bool(ok), label or ""

    def batch_predict(self, compositions: List[Dict[str, float]]) -> List[Tuple[float, float]]:
        return [self.predict(c) for c in compositions]
