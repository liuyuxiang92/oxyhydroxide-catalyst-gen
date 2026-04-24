#!/usr/bin/env python3
"""Multi-seed experiment runner.

Launches run_experiment.py for multiple seeds (sequentially or in parallel)
and aggregates all generated.csv files into a single all_seeds.csv with
mean ± std summary statistics.

Usage
-----
    python scripts/run_seeds.py --config configs/hea.yaml --method a2c \\
        --out runs/hea_a2c --seeds 0 1 2 3 4

    # Parallel execution (uses multiprocessing):
    python scripts/run_seeds.py --config configs/hea.yaml --method a2c \\
        --out runs/hea_a2c --seeds 0 1 2 3 4 --parallel
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from typing import List


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_RUN_SCRIPT = os.path.join(_SCRIPT_DIR, "run_experiment.py")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--method", default="a2c", choices=["dqn", "reinforce", "a2c"])
    p.add_argument("--out", required=True, help="Base output directory; seed dirs created inside")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--parallel", action="store_true",
                   help="Run seeds in parallel using multiprocessing.Pool")
    return p.parse_args()


def run_one_seed(args_tuple) -> None:
    config, method, base_out, seed = args_tuple
    seed_dir = os.path.join(base_out, f"seed_{seed}")
    cmd = [
        sys.executable, _RUN_SCRIPT,
        "--config", config,
        "--method", method,
        "--out", seed_dir,
        "--seed", str(seed),
    ]
    print(f"[INFO] Starting seed {seed}: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[WARN] Seed {seed} exited with code {result.returncode}.")


def aggregate_results(base_out: str, seeds: List[int]) -> None:
    """Concatenate all generated.csv files and write all_seeds.csv + summary."""
    all_rows = []
    for seed in seeds:
        path = os.path.join(base_out, f"seed_{seed}", "generated.csv")
        if not os.path.exists(path):
            print(f"[WARN] Missing: {path}")
            continue
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["seed"] = seed
                all_rows.append(row)

    if not all_rows:
        print("[WARN] No generated.csv files found; skipping aggregation.")
        return

    out_path = os.path.join(base_out, "all_seeds.csv")
    fieldnames = list(all_rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[INFO] Aggregated {len(all_rows)} rows → {out_path}")

    # Summary statistics per seed.
    import numpy as np
    seed_means = {}
    for seed in seeds:
        rows = [r for r in all_rows if r.get("seed") == seed]
        if rows:
            vals = [float(r["dp_mean"]) for r in rows if "dp_mean" in r]
            if vals:
                seed_means[seed] = float(np.mean(vals))

    if seed_means:
        all_vals = list(seed_means.values())
        print(
            f"[SUMMARY] dp_mean across seeds: "
            f"mean={float(np.mean(all_vals)):.4f} ± {float(np.std(all_vals)):.4f}"
        )
        for seed, val in sorted(seed_means.items()):
            print(f"  seed={seed}: dp_mean={val:.4f}")


def main() -> None:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    tasks = [(args.config, args.method, args.out, seed) for seed in args.seeds]

    if args.parallel:
        from multiprocessing import Pool
        with Pool(processes=len(args.seeds)) as pool:
            pool.map(run_one_seed, tasks)
    else:
        for task in tasks:
            run_one_seed(task)

    aggregate_results(args.out, args.seeds)
    print("[INFO] All seeds complete.")


if __name__ == "__main__":
    main()
