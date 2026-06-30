#!/usr/bin/env python3
"""Compare reward distributions across optimization methods.

Aggregates the ``reward`` column of several ``generated.csv`` files (RL runs you
already have + BO/GA baseline runs) and produces a summary table and a comparison
figure.  ``reward`` is the apples-to-apples metric: for every method it is the
scalar ``predictor.predict(comp)[0]`` (the value RL maximizes and the value BO/GA
optimize), so distributions are directly comparable.  ``dp_mean`` is NOT used as
the comparison axis — its meaning differs per scenario (raw overpotential for OOH,
the reward itself for the baselines).

Usage
-----
    python scripts/baselines/compare_methods.py \\
        --run DQN:runs/oxides_sinter_dqn/generated.csv \\
        --run A2C:runs/oxides_sinter_a2c/generated.csv \\
        --run BO:runs/bo_sinter/generated.csv \\
        --run GA:runs/ga_sinter/generated.csv \\
        --out runs/compare/sinter --top-k 10
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare reward distributions across methods")
    p.add_argument("--run", action="append", required=True, metavar="LABEL:PATH",
                   help="Repeatable. LABEL is the method name, PATH points to its "
                        "generated.csv (e.g. DQN:runs/sinter_dqn/generated.csv)")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--top-k", type=int, default=10,
                   help="K for the top-K mean reward summary (default 10)")
    p.add_argument("--title", default=None, help="Optional figure title")
    p.add_argument("--no-plot", action="store_true", help="Skip the PNG (CSV only)")
    return p.parse_args()


def _split_run(spec: str) -> Tuple[str, str]:
    if ":" not in spec:
        raise SystemExit(f"--run must be LABEL:PATH, got {spec!r}")
    label, path = spec.split(":", 1)
    return label.strip(), path.strip()


def _load_rewards(path: str) -> List[float]:
    if not os.path.exists(path):
        raise SystemExit(f"generated.csv not found: {path}")
    rewards: List[float] = []
    with open(path) as f:
        reader = csv.DictReader(f)
        if "reward" not in (reader.fieldnames or []):
            raise SystemExit(f"{path} has no 'reward' column (columns: {reader.fieldnames})")
        for row in reader:
            try:
                rewards.append(float(row["reward"]))
            except (TypeError, ValueError):
                continue
    if not rewards:
        raise SystemExit(f"{path} has no usable reward rows")
    return rewards


def _summarize(label: str, rewards: List[float], top_k: int) -> Dict[str, object]:
    s = sorted(rewards, reverse=True)
    topk = s[:top_k]
    return {
        "method": label,
        "n_candidates": len(s),
        "best_reward": s[0],
        f"top{top_k}_mean": statistics.fmean(topk),
        "mean_reward": statistics.fmean(s),
        "median_reward": statistics.median(s),
        "min_reward": s[-1],
    }


def main() -> None:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    data: Dict[str, List[float]] = {}
    for spec in args.run:
        label, path = _split_run(spec)
        data[label] = _load_rewards(path)

    summaries = [_summarize(lbl, rw, args.top_k) for lbl, rw in data.items()]
    summaries.sort(key=lambda d: -d["best_reward"])

    fields = ["method", "n_candidates", "best_reward",
              f"top{args.top_k}_mean", "mean_reward", "median_reward", "min_reward"]
    summary_csv = os.path.join(args.out, "comparison_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in summaries:
            w.writerow(s)

    print(f"[compare] {len(data)} methods | top-{args.top_k} mean / best reward:")
    for s in summaries:
        print(f"  {s['method']:<8} n={s['n_candidates']:<5} "
              f"best={s['best_reward']:.3f}  top{args.top_k}={s[f'top{args.top_k}_mean']:.3f}")
    print(f"[compare] summary -> {summary_csv}")

    if args.no_plot:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[compare] matplotlib not available; skipped figure (CSV written).")
        return

    labels = [s["method"] for s in summaries]
    dists = [data[lbl] for lbl in labels]
    bests = [s["best_reward"] for s in summaries]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    parts = ax1.violinplot(dists, showmeans=True, showextrema=True)
    ax1.set_xticks(range(1, len(labels) + 1))
    ax1.set_xticklabels(labels, rotation=0)
    ax1.set_ylabel("reward (= predictor objective, higher is better)")
    ax1.set_title("Reward distribution per method")
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.bar(labels, bests, color="tab:green", alpha=0.8)
    ax2.set_ylabel("best reward")
    ax2.set_title("Best candidate per method")
    ax2.grid(True, axis="y", alpha=0.3)
    for i, b in enumerate(bests):
        ax2.text(i, b, f"{b:.1f}", ha="center", va="bottom", fontsize=8)

    if args.title:
        fig.suptitle(args.title)
    fig.tight_layout()
    out_png = os.path.join(args.out, "comparison.png")
    fig.savefig(out_png, dpi=150)
    print(f"[compare] figure -> {out_png}")


if __name__ == "__main__":
    main()
