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

Minimized objectives (``--minimize``)
------------------------------------
Scenarios whose properties carry ``direction: min`` (sintering / calcination
temperature) store ``reward = -objective``, so "best" is the *least negative*
reward.  Plotting that raw is unreadable for a physical quantity — a temperature
is positive and we want the lowest one.  ``--minimize`` flips the display axis to
``objective = -reward`` (positive, **lower is better**) while every ranking stays
on reward internally, so the winner is unchanged.  The summary CSV switches its
column names from ``*_reward`` to ``*_objective`` so a minimized table is never
mistaken for a maximized one::

    python scripts/baselines/compare_methods.py \\
        --run "DQN(bootstrap):runs/sinter_dqn" --run "DQN(mc):runs/sinter_mc" \\
        --run "A2C:runs/sinter_a2c" --minimize \\
        --y-label "sintering temperature (lower is better)" \\
        --out runs/compare/sinter
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
    p.add_argument("--minimize", action="store_true",
                   help="The scenario minimizes a positive physical quantity "
                        "(reward = -objective). Plot objective = -reward, lower "
                        "is better. Ranking is unchanged.")
    p.add_argument("--y-label", default=None,
                   help="Override the y-axis label of the distribution panel")
    p.add_argument("--sort-by-best", action="store_true",
                   help="Rank methods by best reward instead of keeping the order "
                        "the --run flags were given in (default: command-line order)")
    p.add_argument("--no-plot", action="store_true", help="Skip the PNG (CSV only)")
    return p.parse_args()


def _split_run(spec: str) -> Tuple[str, str]:
    if ":" not in spec:
        raise SystemExit(f"--run must be LABEL:PATH, got {spec!r}")
    label, path = spec.split(":", 1)
    return label.strip(), path.strip()


def _load_rewards(path: str) -> List[Tuple[float, str]]:
    """Return ``[(reward, formula), ...]``.  ``path`` may be a run directory."""
    if os.path.isdir(path):
        path = os.path.join(path, "generated.csv")
    if not os.path.exists(path):
        raise SystemExit(f"generated.csv not found: {path}")
    rows: List[Tuple[float, str]] = []
    with open(path) as f:
        reader = csv.DictReader(f)
        if "reward" not in (reader.fieldnames or []):
            raise SystemExit(f"{path} has no 'reward' column (columns: {reader.fieldnames})")
        for row in reader:
            try:
                rows.append((float(row["reward"]), row.get("formula", "")))
            except (TypeError, ValueError):
                continue
    if not rows:
        raise SystemExit(f"{path} has no usable reward rows")
    return rows


def _summarize(label: str, rows: List[Tuple[float, str]], top_k: int,
               minimize: bool) -> Dict[str, object]:
    # Ranking always happens on reward (higher = better); ``minimize`` only
    # changes how the numbers are *reported*, never who wins.
    s = sorted(rows, key=lambda t: -t[0])
    rw = [r for r, _ in s]
    sign = -1.0 if minimize else 1.0
    key = "objective" if minimize else "reward"
    return {
        "method": label,
        "n_candidates": len(s),
        "best_formula": s[0][1],
        f"best_{key}": sign * rw[0],
        f"top{top_k}_mean_{key}": sign * statistics.fmean(rw[:top_k]),
        f"mean_{key}": sign * statistics.fmean(rw),
        f"median_{key}": sign * statistics.median(rw),
        f"worst_{key}": sign * rw[-1],
        "_best_reward": rw[0],  # stable sort key, stripped before writing
    }


def main() -> None:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    data: Dict[str, List[Tuple[float, str]]] = {}
    for spec in args.run:
        label, path = _split_run(spec)
        data[label] = _load_rewards(path)

    key = "objective" if args.minimize else "reward"
    # ``data`` is insertion-ordered, so summaries follow the --run order the user
    # typed; that order carries through the CSV, the printout and the figure.
    summaries = [_summarize(lbl, rw, args.top_k, args.minimize) for lbl, rw in data.items()]
    if args.sort_by_best:
        summaries.sort(key=lambda d: -d["_best_reward"])

    fields = ["method", "n_candidates", f"best_{key}", "best_formula",
              f"top{args.top_k}_mean_{key}", f"mean_{key}", f"median_{key}",
              f"worst_{key}"]
    summary_csv = os.path.join(args.out, "comparison_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for s in summaries:
            w.writerow(s)

    direction = "lowest" if args.minimize else "best"
    print(f"[compare] {len(data)} methods | {direction} / top-{args.top_k} mean {key}:")
    for s in summaries:
        print(f"  {s['method']:<16} n={s['n_candidates']:<5} "
              f"{direction}={s[f'best_{key}']:10.3f}  "
              f"top{args.top_k}={s[f'top{args.top_k}_mean_{key}']:10.3f}  "
              f"{s['best_formula']}")
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

    sign = -1.0 if args.minimize else 1.0
    labels = [s["method"] for s in summaries]
    dists = [[sign * r for r, _ in data[lbl]] for lbl in labels]
    bests = [s[f"best_{key}"] for s in summaries]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    positions = list(range(1, len(labels) + 1))
    # A violin needs a KDE, which requires >=2 values with nonzero spread; a lone
    # (or constant) candidate can't form one. Draw violins only where possible and
    # always overlay the raw points so single-candidate methods stay visible.
    violin_data, violin_pos = [], []
    for pos, d in zip(positions, dists):
        if len(d) >= 2 and min(d) != max(d):
            violin_data.append(d)
            violin_pos.append(pos)
    if violin_data:
        ax1.violinplot(violin_data, positions=violin_pos, showmeans=True, showextrema=True)
    for pos, d in zip(positions, dists):
        ax1.scatter([pos] * len(d), d, s=18, color="tab:blue", alpha=0.6, zorder=3)
    ax1.set_xticks(positions)
    # Candidate counts vary hugely at a fixed budget (generated.csv is
    # deduplicated, so a peaked policy emits fewer rows); carry n in the tick so
    # the distributions are never compared as if they were equal-sized samples.
    ax1.set_xticklabels([f"{lbl}\nn={len(d)}" for lbl, d in zip(labels, dists)],
                        rotation=0)
    default_ylab = ("objective (lower is better)" if args.minimize
                    else "reward (= predictor objective, higher is better)")
    ax1.set_ylabel(args.y_label or default_ylab)
    dist_word = "Objective" if args.minimize else "Reward"
    ax1.set_title(f"{dist_word} distribution per method")
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.bar(labels, bests, color="tab:green", alpha=0.8)
    best_word = "lowest" if args.minimize else "best"
    ax2.set_ylabel(f"{best_word} {args.y_label or key}")
    ax2.set_title(f"{best_word.capitalize()} candidate per method")
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
