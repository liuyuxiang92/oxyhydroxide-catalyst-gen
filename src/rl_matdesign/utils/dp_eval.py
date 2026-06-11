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


def load_ase_calculators(
    dp_models: List[str], *, head: Optional[str] = None
) -> List[Any]:
    """Load a list of ASE-compatible DeepMD calculators (one per checkpoint).

    ``head`` selects which output head of a multi-task DP checkpoint to use
    (e.g. ``"Omat24"`` for the energy head of an Omat24+property model). Leave
    as ``None`` for single-task checkpoints — DPCalculator's own default.
    """
    try:
        from deepmd.calculator import DP as DPCalculator
    except ImportError as exc:
        raise ImportError(
            "The energy backend requires deepmd-kit (ASE calculator): "
            "`pip install deepmd-kit`."
        ) from exc
    kwargs = {"head": head} if head is not None else {}
    return [DPCalculator(model=p, **kwargs) for p in dp_models]


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


def _broadcast_frame_param(fparam: Any, nframes: int) -> np.ndarray:
    """Broadcast a scalar / length-d vector / (nframes, d) array to (nframes, d)."""
    arr = np.atleast_1d(np.asarray(fparam, dtype=np.float64))
    if arr.ndim == 1:
        return np.tile(arr.reshape(1, -1), (nframes, 1))
    if arr.ndim == 2:
        if arr.shape[0] == nframes:
            return arr
        if arr.shape[0] == 1:
            return np.tile(arr, (nframes, 1))
    raise ValueError(
        f"fparam must be a scalar, a length-d vector, or an (nframes, d) array; "
        f"got shape {arr.shape} for nframes={nframes}."
    )


def _broadcast_atomic_param(aparam: Any, nframes: int, natoms: int) -> np.ndarray:
    """Broadcast atomic params to (nframes, natoms, d)."""
    arr = np.asarray(aparam, dtype=np.float64)
    if arr.ndim == 1:                       # per-atom scalar or single d-vector
        if arr.shape[0] == natoms:
            arr = arr.reshape(natoms, 1)
        else:
            arr = np.tile(arr.reshape(1, -1), (natoms, 1))
        return np.tile(arr.reshape(1, natoms, -1), (nframes, 1, 1))
    if arr.ndim == 2 and arr.shape[0] == natoms:
        return np.tile(arr.reshape(1, natoms, -1), (nframes, 1, 1))
    if arr.ndim == 3 and arr.shape[:2] == (nframes, natoms):
        return arr
    raise ValueError(
        f"aparam must broadcast to (nframes={nframes}, natoms={natoms}, d); "
        f"got shape {arr.shape}."
    )


def eval_property_ensemble(
    structures: List[Any],
    models: List[Any],
    elem_to_type: dict,
    *,
    output_index: int = 0,
    output_aggregator: str = "index",
    fparam: Optional[Any] = None,
    aparam: Optional[Any] = None,
) -> List[float]:
    """Evaluate a ``DeepProperty`` ensemble on *structures* → flat list of scalars.

    Returns one scalar per (structure, model) pair (so an ensemble of M models on
    N structures yields N*M values), each collapsed from the model's output
    vector via :func:`pick_scalar`. Used by the ``structure_score`` predictor's
    ``property`` backend.

    ``fparam`` / ``aparam`` supply frame / atomic parameters for models trained
    with them (``numb_fparam > 0`` / ``numb_aparam > 0``). ``fparam`` is broadcast
    to one row per frame; pass a scalar, a length-``numb_fparam`` vector, or a
    full ``(nframes, numb_fparam)`` array.
    """
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
                f"({sorted(elem_to_type)}). Check the checkpoint covers every species."
            )
        atom_types[i] = np.fromiter(
            (elem_to_type[s] for s in syms), dtype=np.int32, count=natoms,
        )

    fparam_arr = _broadcast_frame_param(fparam, nframes) if fparam is not None else None
    aparam_arr = (
        _broadcast_atomic_param(aparam, nframes, natoms) if aparam is not None else None
    )

    eval_kwargs: dict = {"mixed_type": True}
    if fparam_arr is not None:
        eval_kwargs["fparam"] = fparam_arr
    if aparam_arr is not None:
        eval_kwargs["aparam"] = aparam_arr

    values: List[float] = []
    for dp in models:
        res = np.asarray(
            dp.eval(coords=coords, cells=cells, atom_types=atom_types, **eval_kwargs)
        )
        if res.ndim == 1:
            res = res.reshape(1, -1) if nframes == 1 else res.reshape(nframes, -1)
        for frame_vec in res:
            values.append(
                pick_scalar(
                    frame_vec, output_index=output_index, output_aggregator=output_aggregator
                )
            )
    return values
