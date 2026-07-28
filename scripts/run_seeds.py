#!/usr/bin/env python3
"""Multi-seed experiment runner.

Launches run_experiment.py for multiple seeds (sequentially or in parallel)
and aggregates all generated.csv files into a single all_seeds.csv with
mean ± std summary statistics, plus all timing.json files into
all_seeds_timing.csv for cost comparisons across methods.

Usage
-----
    python scripts/run_seeds.py --config configs/hea.yaml --method a2c \\
        --out runs/hea_a2c --seeds 0 1 2 3 4

    # Parallel execution (uses multiprocessing):
    python scripts/run_seeds.py --config configs/hea.yaml --method a2c \\
        --out runs/hea_a2c --seeds 0 1 2 3 4 --parallel

    # DQN Monte-Carlo-target ablation arm:
    python scripts/run_seeds.py --config configs/ooh_dqn.yaml --method dqn \\
        --dqn-target-mode mc --out runs/ooh_dqn_mc --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from typing import List, Optional, Sequence, Tuple


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
    p.add_argument("--dqn-target-mode", choices=["bootstrap", "mc"], default=None,
                   dest="dqn_target_mode",
                   help="DQN regression target (ignored for reinforce/a2c). "
                        "Use 'mc' for the Monte-Carlo-return ablation arm.")
    p.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[],
                   dest="extra_args",
                   help="Everything after this flag is forwarded verbatim to "
                        "run_experiment.py (must come last).")
    return p.parse_args()


def run_one_seed(args_tuple) -> Tuple[int, int]:
    """Run one seed. Returns ``(seed, returncode)``."""
    config, method, base_out, seed, target_mode, extra = args_tuple
    seed_dir = os.path.join(base_out, f"seed_{seed}")
    cmd = [
        sys.executable, _RUN_SCRIPT,
        "--config", config,
        "--method", method,
        "--out", seed_dir,
        # run_experiment.py has no --seed flag; it takes three decorrelated
        # seeds. Passing all three the same value reproduces the single-seed
        # intent, and passing --train-seed explicitly is what enables GPU
        # determinism (see set_global_seed in run_experiment.main).
        "--dp-seed", str(seed),
        "--train-seed", str(seed),
        "--gen-seed", str(seed),
    ]
    if target_mode is not None:
        cmd += ["--dqn-target-mode", target_mode]
    cmd += list(extra)
    print(f"[INFO] Starting seed {seed}: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[ERROR] Seed {seed} exited with code {result.returncode}.")
    return seed, result.returncode


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


_TIMING_FIELDS = [
    "seed", "method", "dqn_target_mode", "device", "total_s", "overhead_s",
    "t_predict_s", "n_calls", "n_unique", "n_cache_hits", "cache_hit_rate",
    "mean_s_per_call", "mean_s_per_unique", "best_reward",
    "setup_s", "warmup_s", "train_s", "generate_s",
]


def aggregate_timing(base_out: str, seeds: List[int]) -> None:
    """Flatten each seed's timing.json into one row of all_seeds_timing.csv."""
    rows = []
    for seed in seeds:
        path = os.path.join(base_out, f"seed_{seed}", "timing.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            t = json.load(f)
        pred = t.get("predictor", {})
        ph = t.get("phases_s", {})
        rows.append({
            "seed": seed,
            "method": t.get("method"),
            "dqn_target_mode": t.get("dqn_target_mode"),
            "device": t.get("device"),
            "total_s": t.get("total_s"),
            "overhead_s": t.get("overhead_s"),
            **{k: pred.get(k) for k in (
                "t_predict_s", "n_calls", "n_unique", "n_cache_hits",
                "cache_hit_rate", "mean_s_per_call", "mean_s_per_unique",
                "best_reward")},
            **{f"{k}_s": ph.get(k) for k in ("setup", "warmup", "train", "generate")},
        })

    if not rows:
        print("[WARN] No timing.json files found; skipping timing aggregation.")
        return

    out_path = os.path.join(base_out, "all_seeds_timing.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_TIMING_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Aggregated timing for {len(rows)} seeds → {out_path}")

    import numpy as np
    totals = [r["total_s"] for r in rows if r.get("total_s") is not None]
    preds = [r["t_predict_s"] for r in rows if r.get("t_predict_s") is not None]
    if totals:
        print(f"[SUMMARY] total_s across seeds: "
              f"mean={float(np.mean(totals)):.1f} ± {float(np.std(totals)):.1f}")
    if preds and totals:
        print(f"[SUMMARY] predictor share of wall-clock: "
              f"{100.0 * float(np.sum(preds)) / float(np.sum(totals)):.1f}%")


def main() -> None:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.dqn_target_mode is not None and args.method != "dqn":
        print(f"[WARN] --dqn-target-mode is a DQN-only switch; ignoring it for "
              f"method={args.method!r}.")
        target_mode: Optional[str] = None
    else:
        target_mode = args.dqn_target_mode

    extra: Sequence[str] = args.extra_args
    tasks = [(args.config, args.method, args.out, seed, target_mode, extra)
             for seed in args.seeds]

    if args.parallel:
        from multiprocessing import Pool
        with Pool(processes=len(args.seeds)) as pool:
            results = pool.map(run_one_seed, tasks)
    else:
        results = [run_one_seed(task) for task in tasks]

    aggregate_results(args.out, args.seeds)
    aggregate_timing(args.out, args.seeds)

    failed = [s for s, rc in results if rc != 0]
    if failed:
        # A silent warning here previously hid the fact that *every* seed was
        # failing on an unrecognized --seed flag. Fail loudly instead.
        raise SystemExit(
            f"[ERROR] {len(failed)}/{len(results)} seeds failed: {failed}"
        )
    print("[INFO] All seeds complete.")


if __name__ == "__main__":
    main()
