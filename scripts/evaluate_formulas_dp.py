from __future__ import annotations

import argparse
import csv
import os
import re
from typing import Dict, Iterable, List, Tuple


def parse_cation_fractions(formula: str, *, anion_formula: str) -> Dict[str, float]:
    """Parse a terminal formula like 'Ni0.70Fe0.15...O2H1' into {el: frac}.

    Notes:
    - This intentionally does NOT use pymatgen to avoid extra dependencies and warnings.
    - It assumes the formula uses explicit decimal fractions for each cation.
    """

    if anion_formula and formula.endswith(anion_formula):
        core = formula[: -len(anion_formula)]
    else:
        core = formula

    parts = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]+)", core)
    if not parts:
        raise ValueError(f"Could not parse any (element,fraction) pairs from: {formula}")

    comp: Dict[str, float] = {}
    for el, frac_str in parts:
        comp[el] = comp.get(el, 0.0) + float(frac_str)

    return comp


def iter_formulas(args: argparse.Namespace) -> Iterable[str]:
    for f in args.formula:
        if f.strip():
            yield f.strip()

    if args.formulas_file:
        with open(args.formulas_file, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # allow either "formula" alone, or "formula\tvalue" style
                yield line.split()[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate DeepMD dp_mean/dp_std/objective for one or more terminal formulas. "
            "Useful to check whether the DP model can reproduce low experimental candidates."
        )
    )
    parser.add_argument(
        "--formula",
        action="append",
        default=[],
        help="Terminal formula like 'Ni0.70Fe0.15Ce0.05Er0.05Tm0.05O2H1'. Repeatable.",
    )
    parser.add_argument(
        "--formulas-file",
        type=str,
        default=None,
        help="Text file with one formula per line (optionally followed by whitespace + value).",
    )
    parser.add_argument("--anion-formula", type=str, default="O2H1")

    parser.add_argument(
        "--dp-model",
        action="append",
        default=None,
        help=(
            "Path to a DeepMD .pt checkpoint. Repeat for ensemble. "
            "Defaults to model_1.ckpt.pt … model_5.ckpt.pt if not specified."
        ),
    )
    parser.add_argument("--dp-poscar", type=str, default="POSCAR")
    parser.add_argument("--dp-n-random-configs", type=int, default=10)
    parser.add_argument("--dp-ads-height", type=float, default=1.9)
    parser.add_argument("--dp-ads-dz", type=float, default=1.0)
    parser.add_argument(
        "--dp-uncertainty",
        type=str,
        default="models",
        choices=["models", "configs", "total"],
    )
    parser.add_argument(
        "--dp-objective",
        type=str,
        default="mean_minus_kstd",
        choices=["mean", "mean_minus_kstd", "mean_plus_kstd", "exp_scaled"],
    )
    parser.add_argument("--dp-k", type=float, default=1.0)
    parser.add_argument("--dp-exp-ref", type=float, default=230.0,
        help="Reference overpotential (mV) for exp_scaled objective. Default: 230.")
    parser.add_argument("--dp-exp-scale", type=float, default=10.0,
        help="Scale factor in denominator for exp_scaled objective. Default: 10.")

    parser.add_argument(
        "--dp-debug-dir",
        type=str,
        default=None,
        help="If set, dump the exact POSCAR frames used for DP evaluation into this directory.",
    )

    parser.add_argument("--out-csv", type=str, default=None)

    args = parser.parse_args()

    if args.dp_model is None:
        args.dp_model = [
            "model_1.ckpt.pt",
            "model_2.ckpt.pt",
            "model_3.ckpt.pt",
            "model_4.ckpt.pt",
            "model_5.ckpt.pt",
        ]

    # Local imports
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    import sys

    sys.path.insert(0, os.path.join(repo_root, "src"))

    from abcde_ooh.dp_predictor import DPConfig, DeepMDOverpotentialPredictor, objective_from_mean_std

    cfg = DPConfig(
        base_poscar=args.dp_poscar,
        model_files=tuple(args.dp_model),
        n_random_configs=args.dp_n_random_configs,
        ads_height=args.dp_ads_height,
        ads_dz=args.dp_ads_dz,
        seed=0,
    )
    predictor = DeepMDOverpotentialPredictor(cfg)

    rows: List[dict] = []

    skipped: List[Tuple[str, str]] = []

    for formula in iter_formulas(args):
        try:
            comp = parse_cation_fractions(formula, anion_formula=args.anion_formula)
            mean, std = predictor.predict_overpotential(
                comp,
                uncertainty=args.dp_uncertainty,
                debug_dir=args.dp_debug_dir,
            )
            obj = objective_from_mean_std(
                float(mean), float(std),
                mode=args.dp_objective, k=args.dp_k,
                exp_ref=args.dp_exp_ref, exp_scale=args.dp_exp_scale,
            )
            reward = -float(obj)

            rows.append(
                {
                    "formula": formula,
                    "dp_mean": float(mean),
                    "dp_std": float(std),
                    "dp_objective": float(obj),
                    "reward": float(reward),
                }
            )
        except Exception as exc:
            skipped.append((formula, str(exc)))
            print(f"[WARN] Skipping {formula}: {exc}", flush=True)

    if args.out_csv:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["formula", "dp_mean", "dp_std", "dp_objective", "reward"])
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {len(rows)} rows -> {args.out_csv}")
    else:
        w = csv.DictWriter(
            sys.stdout,
            fieldnames=["formula", "dp_mean", "dp_std", "dp_objective", "reward"],
        )
        w.writeheader()
        w.writerows(rows)

    if skipped:
        print(f"\n[WARN] {len(skipped)} formula(s) skipped due to errors:")
        for formula, reason in skipped:
            print(f"  {formula}: {reason}")


if __name__ == "__main__":
    main()
