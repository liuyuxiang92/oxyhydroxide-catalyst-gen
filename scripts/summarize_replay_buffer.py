from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# Avoid noisy matminer warnings triggered by importing abcde_ooh.env (which imports featurization).
warnings.filterwarnings(
    "ignore",
    message=r"^MagpieData\(impute_nan=False\):.*",
)
warnings.filterwarnings(
    "ignore",
    message=r"^PymatgenData\(impute_nan=False\):.*",
)
warnings.filterwarnings(
    "ignore",
    message=r"^ValenceOrbital\(impute_nan=False\):.*",
)
warnings.filterwarnings(
    "ignore",
    message=r"^IonProperty\(impute_nan=False\):.*",
)


def _load_run_config(run_dir: str) -> dict:
    cfg_path = os.path.join(run_dir, "run_config.json")
    if not os.path.exists(cfg_path):
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _decode_episode_formulas(
    *,
    a_elem: np.ndarray,
    a_comp: np.ndarray,
    max_steps: int,
    anion_formula: str,
) -> Tuple[List[str], List[Dict[str, float]]]:
    from abcde_ooh.env import DEFAULT_CATION_SET, DEFAULT_FRACTIONS
    from abcde_ooh.encoding import decode_one_hot

    n = int(a_elem.shape[0])
    if n % max_steps != 0:
        raise ValueError(f"Expected a multiple of {max_steps} steps, got {n} rows")
    neps = n // max_steps

    formulas: List[str] = []
    comps: List[Dict[str, float]] = []

    for e in range(neps):
        state = ""
        comp: Dict[str, float] = {}
        for t in range(max_steps):
            i = e * max_steps + t
            el = decode_one_hot(a_elem[i], DEFAULT_CATION_SET)
            frac_str = decode_one_hot(a_comp[i], DEFAULT_FRACTIONS)
            if t == 0:
                state = f"{el}{frac_str}"
            else:
                state = f"{state}{el}{frac_str}"
            comp[el] = float(frac_str)
        formulas.append(f"{state}{anion_formula}")
        comps.append(comp)

    return formulas, comps


def _terminal_reward_from_y(y: np.ndarray, max_steps: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        y = y.reshape(-1)
    if y.size % max_steps != 0:
        raise ValueError(f"Expected y to be multiple of {max_steps}, got {y.size}")
    neps = y.size // max_steps
    Y = y.reshape(neps, max_steps)
    return Y[:, max_steps - 1]


@dataclass(frozen=True)
class DPSpec:
    model_files: Tuple[str, ...]
    base_poscar: str
    n_random_configs: int
    ads_height: float
    ads_dz: float
    seed: int
    uncertainty: str
    objective_mode: str
    k: float


def _maybe_build_dp_predictor(dp: Optional[DPSpec]):
    if dp is None:
        return None, None

    from abcde_ooh.dp_predictor import DPConfig, DeepMDOverpotentialPredictor, objective_from_mean_std

    cfg = DPConfig(
        base_poscar=dp.base_poscar,
        model_files=dp.model_files,
        n_random_configs=dp.n_random_configs,
        ads_height=dp.ads_height,
        ads_dz=dp.ads_dz,
        seed=dp.seed,
    )
    return DeepMDOverpotentialPredictor(cfg), objective_from_mean_std


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize a saved replay buffer (random_dataset.npz) into formulas and DP objective values. "
            "Optionally recompute DeepMD mean/std if model checkpoints are available."
        )
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run directory containing random_dataset.npz (and optionally run_config.json)",
    )
    parser.add_argument(
        "--out-csv",
        default=None,
        help="Output CSV path (default: <run-dir>/replay_buffer_summary.csv)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on number of episodes written (for quick inspection)",
    )

    # Optional DP recompute path
    parser.add_argument(
        "--recompute-dp",
        action="store_true",
        help="If set, recompute dp_mean/dp_std for each unique composition using DeepMD models.",
    )
    parser.add_argument(
        "--dp-model",
        action="append",
        default=[],
        help="Path to a DeepMD .pt checkpoint. Repeat for ensemble (overrides run_config.json).",
    )
    parser.add_argument("--dp-poscar", type=str, default=None)
    parser.add_argument("--dp-n-random-configs", type=int, default=None)
    parser.add_argument("--dp-ads-height", type=float, default=None)
    parser.add_argument("--dp-ads-dz", type=float, default=None)
    parser.add_argument("--dp-uncertainty", type=str, default=None)
    parser.add_argument("--dp-objective", type=str, default=None)
    parser.add_argument("--dp-k", type=float, default=None)

    args = parser.parse_args()

    # Ensure local imports work when invoked as a script.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, os.path.join(repo_root, "src"))

    run_cfg = _load_run_config(args.run_dir)

    anion_formula = run_cfg.get("anion_formula", "O2H1")
    max_steps = 5

    ds_path = os.path.join(args.run_dir, "random_dataset.npz")
    if not os.path.exists(ds_path):
        raise SystemExit(f"Not found: {ds_path}")

    data = np.load(ds_path)
    for k in ("a_elem", "a_comp", "y"):
        if k not in data.files:
            raise SystemExit(f"random_dataset.npz missing key '{k}'. Keys: {sorted(data.files)}")

    a_elem = data["a_elem"]
    a_comp = data["a_comp"]
    y = data["y"]

    formulas, comps = _decode_episode_formulas(
        a_elem=a_elem,
        a_comp=a_comp,
        max_steps=max_steps,
        anion_formula=anion_formula,
    )

    terminal_reward = _terminal_reward_from_y(y, max_steps)
    buffer_objective = -terminal_reward  # objective minimized by DP; reward is -objective

    # Decide DP settings if recompute requested.
    dp_spec: Optional[DPSpec] = None
    if args.recompute_dp:
        model_files = tuple(args.dp_model) if args.dp_model else tuple(run_cfg.get("dp_model", []))
        if not model_files:
            raise SystemExit("--recompute-dp requires at least one --dp-model or dp_model in run_config.json")

        dp_spec = DPSpec(
            model_files=tuple(model_files),
            base_poscar=str(args.dp_poscar or run_cfg.get("dp_poscar", "POSCAR")),
            n_random_configs=int(args.dp_n_random_configs or run_cfg.get("dp_n_random_configs", 10)),
            ads_height=float(args.dp_ads_height or run_cfg.get("dp_ads_height", 1.9)),
            ads_dz=float(args.dp_ads_dz or run_cfg.get("dp_ads_dz", 1.0)),
            seed=int(run_cfg.get("seed", 0)),
            uncertainty=str(args.dp_uncertainty or run_cfg.get("dp_uncertainty", "models")),
            objective_mode=str(args.dp_objective or run_cfg.get("dp_objective", "mean_minus_kstd")),
            k=float(args.dp_k or run_cfg.get("dp_k", 1.0)),
        )

    dp_predictor, objective_from_mean_std = _maybe_build_dp_predictor(dp_spec)

    out_csv = args.out_csv or os.path.join(args.run_dir, "replay_buffer_summary.csv")

    # DP recompute caching by composition key.
    dp_cache: Dict[Tuple[Tuple[str, float], ...], Tuple[float, float, float]] = {}

    n = len(formulas)
    limit = min(n, int(args.max_rows)) if args.max_rows is not None else n

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "formula",
            "buffer_objective",
            "buffer_terminal_reward",
            "dp_mean",
            "dp_std",
            "dp_mean_minus_kstd",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(limit):
            dp_mean = ""
            dp_std = ""
            dp_obj = ""

            if dp_predictor is not None and dp_spec is not None and objective_from_mean_std is not None:
                key = tuple(sorted((k, float(v)) for k, v in comps[i].items()))
                if key in dp_cache:
                    mean, std, obj = dp_cache[key]
                else:
                    mean, std = dp_predictor.predict_overpotential(comps[i], uncertainty=dp_spec.uncertainty)
                    obj = objective_from_mean_std(
                        float(mean),
                        float(std),
                        mode=dp_spec.objective_mode,
                        k=dp_spec.k,
                    )
                    dp_cache[key] = (float(mean), float(std), float(obj))

                dp_mean = mean
                dp_std = std
                dp_obj = obj

            writer.writerow(
                {
                    "formula": formulas[i],
                    "buffer_objective": float(buffer_objective[i]),
                    "buffer_terminal_reward": float(terminal_reward[i]),
                    "dp_mean": dp_mean,
                    "dp_std": dp_std,
                    "dp_mean_minus_kstd": dp_obj,
                }
            )

    # Lightweight console summary
    obj = buffer_objective[:limit]
    print(f"Wrote {limit} episodes -> {out_csv}")
    print(
        "buffer_objective stats:",
        {
            "min": float(np.min(obj)),
            "p10": float(np.quantile(obj, 0.10)),
            "median": float(np.median(obj)),
            "mean": float(np.mean(obj)),
            "p90": float(np.quantile(obj, 0.90)),
            "max": float(np.max(obj)),
        },
    )


if __name__ == "__main__":
    main()
