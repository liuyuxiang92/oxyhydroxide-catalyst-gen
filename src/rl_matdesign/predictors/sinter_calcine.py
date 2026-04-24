"""SinterCalcineRFPredictor — RandomForest predictor for sintering and
calcination temperatures.

Loads the pretrained sklearn ``RandomForestRegressor`` checkpoints from
the reference repos (``optimal_sinter_RF.joblib`` /
``optimal_calcine_RF.joblib``) and predicts the temperature (K) from a
matminer/Magpie feature vector of the composition.

The reference featurization pipeline is identical to
:data:`rl_matdesign.featurization.feature_calculators` — we reuse that
object so feature order matches the model's training-time order exactly.

Reward convention
-----------------
Lower T is better, but our RL framework maximises reward, so we return
``(-T, std_T)``.  To recover the raw temperature from a generated CSV,
compute ``T = -dp_mean``.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


class SinterCalcineRFPredictor:
    """Predict sinter or calcine temperature from a composition dict.

    Parameters
    ----------
    rf_model_path:
        Path to the pretrained sklearn joblib checkpoint.
    mode:
        ``"sinter"`` or ``"calcine"`` — informational only; labels which
        temperature the model predicts.  Used for clearer logging.
    """

    def __init__(self, rf_model_path: str, mode: str = "sinter") -> None:
        if mode not in {"sinter", "calcine"}:
            raise ValueError(f"mode must be 'sinter' or 'calcine', got {mode!r}.")
        self.rf_model_path = rf_model_path
        self.mode = mode
        self._rf = None
        self._cache: Dict[Tuple[Tuple[str, float], ...], Tuple[float, float]] = {}

    # ------------------------------------------------------------------
    # Lazy model + featurizer loaders
    # ------------------------------------------------------------------

    def _load_rf(self):
        if self._rf is None:
            import joblib

            self._rf = joblib.load(self.rf_model_path)
        return self._rf

    @staticmethod
    def _get_featurizer():
        from ..featurization import feature_calculators

        if feature_calculators is None:
            raise ImportError(
                "matminer featurizer unavailable — SinterCalcineRFPredictor "
                "requires matminer + pymatgen. Install with "
                "`pip install matminer pymatgen`."
            )
        return feature_calculators

    # ------------------------------------------------------------------
    # PropertyPredictor Protocol
    # ------------------------------------------------------------------

    def predict(self, composition: Dict[str, float]) -> Tuple[float, float]:
        """Return ``(-T, std_T)`` for *composition*.

        *composition* is a dict of ``{element: fraction}`` (fractions need
        not sum to 1 — pymatgen ``Composition`` renormalises internally).
        """
        key = tuple(sorted((str(el), float(f)) for el, f in composition.items()))
        if key in self._cache:
            return self._cache[key]

        from pymatgen.core.composition import Composition

        rf = self._load_rf()
        feat_calc = self._get_featurizer()

        try:
            comp_obj = Composition(dict(composition))
            features = np.asarray(feat_calc.featurize(comp_obj), dtype=float).reshape(1, -1)
        except Exception:
            # Invalid composition — return a very high T penalty (mimics the
            # reference "invalid_reward = -1" behaviour in normalised units).
            result = (float(-2000.0), 0.0)
            self._cache[key] = result
            return result

        T = float(rf.predict(features)[0])

        # Ensemble std from the RF trees (free uncertainty estimate).
        try:
            per_tree = np.asarray([est.predict(features)[0] for est in rf.estimators_])
            std_T = float(per_tree.std())
        except Exception:
            std_T = 0.0

        result = (-T, std_T)
        self._cache[key] = result
        return result

    def batch_predict(
        self, compositions: List[Dict[str, float]]
    ) -> List[Tuple[float, float]]:
        return [self.predict(c) for c in compositions]
