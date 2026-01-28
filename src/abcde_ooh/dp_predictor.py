# dp_predictor.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from ase import Atoms  # type: ignore
    from deepmd.pt.infer.deep_eval import DeepProperty  # type: ignore


class DPDependencyError(ImportError):
    pass


def _lazy_import_ase_deepmd():
    try:
        from ase import Atoms  # type: ignore[import-not-found]  # noqa: F401
        from ase.io import read as ase_read  # type: ignore[import-not-found]  # noqa: F401
        from ase.data import chemical_symbols as _ASE_CHEMICAL_SYMBOLS  # type: ignore[import-not-found]  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise DPDependencyError(
            "Missing ASE dependency. Install `ase` in the environment running dp reward."
        ) from e

    try:
        from deepmd.pt.infer.deep_eval import DeepProperty  # type: ignore[import-not-found]  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise DPDependencyError(
            "Missing DeepMD dependency. Ensure `deepmd` (PyTorch infer) is installed."
        ) from e

    from ase import Atoms  # type: ignore[import-not-found]
    from ase.io import read as ase_read  # type: ignore[import-not-found]
    from ase.data import chemical_symbols as _ASE_CHEMICAL_SYMBOLS  # type: ignore[import-not-found]
    from deepmd.pt.infer.deep_eval import DeepProperty  # type: ignore[import-not-found]

    return Atoms, ase_read, _ASE_CHEMICAL_SYMBOLS, DeepProperty


def stable_species_order(symbols: Sequence[str]) -> List[str]:
    seen = set()
    order: List[str] = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            order.append(s)
    return order


def _choose_counts_from_fractions(n: int, fracs: Dict[str, float]) -> Dict[str, int]:
    raw = {k: fracs[k] * n for k in fracs}
    counts = {k: int(math.floor(raw[k])) for k in fracs}
    rem = n - sum(counts.values())
    if rem > 0:
        order = sorted(
            fracs.keys(),
            key=lambda k: (raw[k] - math.floor(raw[k])),
            reverse=True,
        )
        for k in order:
            if rem == 0:
                break
            counts[k] += 1
            rem -= 1
    return counts


@dataclass
class DPConfig:
    base_poscar: str = "POSCAR"
    model_files: Tuple[str, ...] = (
        "model_1.ckpt.pt",
        "model_2.ckpt.pt",
        "model_3.ckpt.pt",
        "model_4.ckpt.pt",
        "model_5.ckpt.pt",
    )
    n_random_configs: int = 10
    ads_height: float = 1.9
    ads_dz: float = 1.0
    seed: int = 123


class DeepMDOverpotentialPredictor:
    """Predict overpotential for alloyed (M)OOH slabs using an ensemble of DeepMD models.

    This is a refactor of your previous `dp_predictor.py` so it can be used as a library:
    - no global side effects at import time
    - models and POSCAR are loaded once per instance

    Public API:
      predict_overpotential(comp) -> (mean, std)

    Where:
      comp is a dict of metal-site fractions (sum==1), e.g. {"Fe":0.2,"Co":0.2,"Ni":0.2,"Cu":0.2,"Ti":0.2}
    """

    def __init__(self, cfg: DPConfig):
        Atoms, ase_read, _ASE_CHEMICAL_SYMBOLS, DeepProperty = _lazy_import_ase_deepmd()

        self._Atoms = Atoms
        self._ase_read = ase_read
        self._DeepProperty = DeepProperty

        # periodic type map in periodic-table order
        periodic_type_map: List[str] = [s for s in _ASE_CHEMICAL_SYMBOLS if s != "X"]
        self._periodic_index: Dict[str, int] = {el: i for i, el in enumerate(periodic_type_map)}

        self.cfg = cfg
        self.base_slab = self._ase_read(cfg.base_poscar, format="vasp")
        self.nat_slab = len(self.base_slab)
        self.slab_species_order = stable_species_order(self.base_slab.get_chemical_symbols())

        self.dp_models = [
            self._DeepProperty(model_file=f, auto_batch_size=False, head="property")
            for f in cfg.model_files
        ]

    def _infer_metal_elem_from_slab(self, exclude: set = {"H", "O"}) -> str:
        syms = self.base_slab.get_chemical_symbols()
        counts: Dict[str, int] = {}
        for s in syms:
            if s in exclude:
                continue
            counts[s] = counts.get(s, 0) + 1
        if not counts:
            raise RuntimeError("Cannot infer metal element: no non-(H/O) atoms found in slab.")
        return max(counts.items(), key=lambda kv: kv[1])[0]

    def _find_species_indices(self, symbols: List[str], target: str) -> List[int]:
        return [i for i, s in enumerate(symbols) if s == target]

    def _do_random_alloying_on_metal_sites(
        self,
        base,
        metal_elem: str,
        fracs: Dict[str, float],
        rng: random.Random,
    ):
        atoms = base.copy()
        symbols = atoms.get_chemical_symbols()

        metal_idx = [i for i, s in enumerate(symbols) if s == metal_elem]
        if not metal_idx:
            raise RuntimeError(f"No {metal_elem} atoms found in base POSCAR.")
        rng.shuffle(metal_idx)

        counts = _choose_counts_from_fractions(len(metal_idx), fracs)
        keys = sorted(counts.keys())

        cursor = 0
        for el in keys:
            k = counts[el]
            sel = metal_idx[cursor : cursor + k]
            cursor += k
            for idx in sel:
                symbols[idx] = el

        atoms.set_chemical_symbols(symbols)
        return atoms

    def _get_surface_normal(self, atoms) -> np.ndarray:
        cvec = atoms.get_cell().array[2]
        return cvec / np.linalg.norm(cvec)

    def _pick_anchor_atom(
        self,
        atoms,
        anchor_elems=None,
        topk: int = 20,
        rng: Optional[random.Random] = None,
    ) -> Tuple[int, np.ndarray]:
        if anchor_elems is None:
            anchor_elems = {"Co"}
        symbols = atoms.get_chemical_symbols()
        pos = atoms.get_positions()
        cand = [(i, pos[i, 2]) for i, s in enumerate(symbols) if s in anchor_elems]
        if not cand:
            cand = [(i, p[2]) for i, p in enumerate(pos)]
        cand.sort(key=lambda t: t[1], reverse=True)
        top = cand[: min(topk, len(cand))]
        if rng is None:
            rng = random
        idx = rng.choice([i for i, _ in top])
        return idx, pos[idx].copy()

    def _add_adsorbates_equalized(
        self,
        atoms,
        mode: str,
        height: float,
        dz_chain: float,
        anchor_elems=None,
        rng: Optional[random.Random] = None,
    ):
        if mode not in {"O*", "OH*", "OOH*"}:
            raise ValueError(f"Unknown mode: {mode}")
        a = atoms.copy()
        n_hat = self._get_surface_normal(a)
        nat = len(a)
        _, r0 = self._pick_anchor_atom(a, anchor_elems=anchor_elems, topk=20, rng=rng)

        O1 = r0 + height * n_hat
        O2 = O1 + dz_chain * n_hat
        H = O2 + dz_chain * n_hat

        a += self._Atoms("O", positions=[O1])
        a += self._Atoms("O", positions=[O2])
        a += self._Atoms("H", positions=[H])

        if mode == "O*":
            masked = [nat + 1, nat + 2]
        elif mode == "OH*":
            masked = [nat + 1]
        else:
            masked = []
        return a, masked

    def _reorder_slab_then_adsorbates(self, fr, ads_indices_in_frame: List[int]):
        symbols = fr.get_chemical_symbols()
        slab_idx_all = list(range(self.nat_slab))
        ads_idx_all = [i for i in range(len(fr)) if i >= self.nat_slab]

        grouped: List[int] = []
        for sp in self.slab_species_order:
            grouped.extend([i for i in slab_idx_all if symbols[i] == sp])
        others = [i for i in slab_idx_all if symbols[i] not in self.slab_species_order]
        grouped.extend(others)

        ads_sorted = sorted(ads_idx_all)
        new_order = grouped + ads_sorted

        reordered = fr[new_order]
        inv = {old_i: new_i for new_i, old_i in enumerate(new_order)}
        new_mask = [inv[i] for i in ads_indices_in_frame]
        return reordered, new_mask

    def _build_dp_inputs_for_one_doped_slab(self, doped, anchor_elems: List[str], rng: random.Random):
        anchor_set = set(anchor_elems)

        frames_raw = []
        masks_raw = []
        for mode in ("O*", "OH*", "OOH*"):
            fr, masked = self._add_adsorbates_equalized(
                doped,
                mode,
                height=self.cfg.ads_height,
                dz_chain=self.cfg.ads_dz,
                anchor_elems=anchor_set,
                rng=rng,
            )
            frames_raw.append(fr)
            masks_raw.append(masked)

        frames = []
        masks = []
        for fr, masked in zip(frames_raw, masks_raw):
            fr_re, new_mask = self._reorder_slab_then_adsorbates(fr, masked)
            frames.append(fr_re)
            masks.append(new_mask)

        nframes = 3
        natoms = len(frames[0])
        for fr in frames[1:]:
            if len(fr) != natoms:
                raise RuntimeError("Different natoms across frames; check adsorbate builder/reordering.")

        coords = np.zeros((nframes, natoms, 3), dtype=np.float64)
        cells = np.zeros((nframes, 3, 3), dtype=np.float64)
        atom_types = np.zeros((nframes, natoms), dtype=np.int32)

        symbols0 = frames[0].get_chemical_symbols()
        for fr in frames[1:]:
            if fr.get_chemical_symbols() != symbols0:
                raise RuntimeError("Atom symbols/order differ across frames; ensure identical reordering.")

        unknown = sorted({s for s in symbols0 if s not in self._periodic_index})
        if unknown:
            raise RuntimeError(f"Found elements not in periodic map: {unknown}")

        for i, (fr, masked) in enumerate(zip(frames, masks)):
            coords[i] = fr.get_positions()
            cells[i] = fr.get_cell().array

            types = np.fromiter(
                (self._periodic_index[s] for s in fr.get_chemical_symbols()),
                dtype=np.int32,
                count=natoms,
            )
            for idx in masked:
                types[idx] = -1
            atom_types[i] = types

        return coords, cells, atom_types

    def _eval_models_on_prepared_inputs(self, coords: np.ndarray, cells: np.ndarray, atom_types: np.ndarray) -> List[float]:
        values: List[float] = []
        for dp in self.dp_models:
            res = dp.eval(coords=coords, cells=cells, atom_types=atom_types, mixed_type=True)
            res = np.asarray(res).reshape(-1)
            values.append(float(res[0]))
        return values

    def predict_overpotential(
        self,
        comp: Dict[str, float],
        *,
        uncertainty: str = "models",
        return_per_model: bool = False,
    ):
        fracs = {el: float(fr) for el, fr in comp.items()}
        metal_elem = self._infer_metal_elem_from_slab(self.base_slab)

        base_seed = self.cfg.seed + (hash(tuple(sorted(fracs.items()))) % 100000)
        rng = random.Random(base_seed)

        n_models = len(self.dp_models)
        values_per_model: List[List[float]] = [[] for _ in range(n_models)]

        anchor_elems = list(fracs.keys())

        for _ in range(self.cfg.n_random_configs):
            doped = self._do_random_alloying_on_metal_sites(self.base_slab, metal_elem, fracs, rng)
            coords, cells, atom_types = self._build_dp_inputs_for_one_doped_slab(
                doped,
                anchor_elems=anchor_elems,
                rng=rng,
            )
            vals = self._eval_models_on_prepared_inputs(coords, cells, atom_types)
            for i, v in enumerate(vals):
                values_per_model[i].append(v)

        per_model_means: List[float] = [
            float(np.mean(vs)) if len(vs) > 0 else float("nan") for vs in values_per_model
        ]
        per_model_stds: List[float] = [
            float(np.std(vs, ddof=1)) if len(vs) > 1 else 0.0 for vs in values_per_model
        ]

        # Mean is defined as the mean of per-model means (typical ensemble mean).
        ensemble_mean = float(np.mean(per_model_means))

        # Uncertainty choices:
        # - models: std across the 5 model means (epistemic / model disagreement)
        # - configs: mean over models of std across random doped configs (structure randomness)
        # - total: std across all (model, config) predictions pooled together
        if uncertainty == "models":
            ensemble_std = float(np.std(per_model_means, ddof=1)) if n_models > 1 else 0.0
        elif uncertainty == "configs":
            ensemble_std = float(np.mean(per_model_stds))
        elif uncertainty == "total":
            pooled = [v for vs in values_per_model for v in vs]
            ensemble_std = float(np.std(pooled, ddof=1)) if len(pooled) > 1 else 0.0
        else:
            raise ValueError("uncertainty must be one of: 'models', 'configs', 'total'")

        details = {
            "per_model_means": per_model_means,
            "per_model_stds_over_configs": per_model_stds,
        }

        if return_per_model:
            return ensemble_mean, ensemble_std, details
        return ensemble_mean, ensemble_std, None


def objective_from_mean_std(mean: float, std: float, mode: str, k: float) -> float:
    """Smaller objective is better."""
    if mode == "mean_minus_kstd":
        return mean - k * std
    if mode == "mean_plus_kstd":
        return mean + k * std
    raise ValueError(f"Unknown objective mode: {mode}")
