from __future__ import annotations

from typing import Sequence

import warnings

import numpy as np

# Matminer may emit a verbose warning from `matminer.utils.data.PymatgenData`
# about `impute_nan=False`, sometimes repeatedly during featurization.
# Suppress it to keep stdout/stderr and tqdm output clean.
warnings.filterwarnings(
    "ignore",
    message=r"^PymatgenData\(impute_nan=False\):.*",
    category=UserWarning,
    module=r"matminer\.utils\.data",
    append=False,
)

try:
    from matminer.featurizers.base import MultipleFeaturizer
    import matminer.featurizers.composition as cf
    from pymatgen.core.composition import Composition

    feature_calculators = MultipleFeaturizer(
        [
            cf.element.Stoichiometry(),
            cf.composite.ElementProperty.from_preset("magpie"),
            cf.orbital.ValenceOrbital(props=["avg"]),
            cf.ion.IonProperty(fast=True),
        ]
    )
    _NUM_FEATURES = len(feature_calculators.feature_labels())
except Exception:  # pragma: no cover
    # Allow a minimal install to run the pipeline (useful for smoke tests / CI).
    feature_calculators = None
    _NUM_FEATURES = 8


def featurize_formula(formula: str) -> np.ndarray:
    """Featurize a (possibly partial) formula into a fixed-length float vector.

    Returns zeros if featurization fails (e.g., empty/invalid partial state).
    """

    if not formula:
        return np.zeros(_NUM_FEATURES, dtype=float)

    # Preferred: Magpie/matminer features.
    if feature_calculators is not None:
        try:
            chemical = Composition(formula)
            feats: Sequence[float] = feature_calculators.featurize(chemical)
            return np.asarray(feats, dtype=float)
        except Exception:
            return np.zeros(_NUM_FEATURES, dtype=float)

    # Fallback: deterministic lightweight features from the string.
    import zlib

    h = zlib.adler32(formula.encode("utf-8"))
    # Spread bits into a small fixed vector in [0, 1).
    out = np.zeros(_NUM_FEATURES, dtype=float)
    for i in range(_NUM_FEATURES):
        out[i] = ((h >> (i * 3)) & 0xFF) / 255.0
    return out
