"""DeepMD evaluation helpers shared across predictors.

Two evaluation paths are supported because the ecosystem uses both:

* **ASE calculator** (``deepmd.calculator.DP``) — wraps a DP model as an ASE
  Calculator. Use ``atoms.get_potential_energy()`` for one scalar per
  (structure, model) pair. This is the path HEA / perovskite / Ti-alloy-style
  bulk-energy predictors take.

* **Raw DeepProperty** (``deepmd.pt.infer.deep_eval.DeepProperty``) — direct
  inference call that returns a property vector (possibly multi-output). Pick
  one scalar via ``output_index``. This is the path the OOH overpotential
  pipeline takes (multiple binding energies → Sabatier formula).

Both paths are kept lazy so importing this module does not require deepmd-kit.
"""
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np


def load_ase_calculators(dp_models: List[str]) -> List[Any]:
    """Load a list of ASE-compatible DeepMD calculators (one per checkpoint)."""
    try:
        from deepmd.calculator import DP as DPCalculator
    except ImportError as exc:
        raise ImportError(
            "DPStructurePredictor requires deepmd-kit (ASE calculator): "
            "`pip install deepmd-kit`."
        ) from exc
    return [DPCalculator(model=p) for p in dp_models]


def eval_energy_ase(
    atoms_list: List[Any],
    calc_list: List[Any],
    *,
    energy_per_atom: bool = True,
) -> List[float]:
    """Evaluate every (structure, calculator) pair → list of scalar energies."""
    values: List[float] = []
    for atoms in atoms_list:
        for calc in calc_list:
            atoms_copy = atoms.copy()
            atoms_copy.calc = calc
            e = atoms_copy.get_potential_energy()
            if energy_per_atom:
                e /= len(atoms_copy)
            values.append(float(e))
    return values


def pick_scalar(
    vector: np.ndarray,
    *,
    output_index: int = 0,
    output_aggregator: str = "index",
) -> float:
    """Pick one scalar from a possibly-multi-output prediction vector.

    Parameters
    ----------
    vector:
        1-D array of model outputs.
    output_index:
        Used only when ``output_aggregator == "index"``.
    output_aggregator:
        ``"index"`` (default) selects ``vector[output_index]``; ``"mean"``
        returns the mean over all outputs; ``"max"`` returns the maximum.
    """
    v = np.asarray(vector).reshape(-1)
    if v.size == 0:
        raise ValueError("Empty prediction vector.")
    if output_aggregator == "index":
        if output_index < 0 or output_index >= v.size:
            raise IndexError(
                f"output_index={output_index} is out of range for a length-{v.size} output vector."
            )
        return float(v[output_index])
    if output_aggregator == "mean":
        return float(np.mean(v))
    if output_aggregator == "max":
        return float(np.max(v))
    raise ValueError(
        f"Unknown output_aggregator {output_aggregator!r}; expected one of: index, mean, max."
    )
