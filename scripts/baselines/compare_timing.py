#!/usr/bin/env python3
"""Compare optimization methods by *cost*, where cost includes the reward model.

Companion to ``compare_methods.py``, which compares the quality of the final
candidates.  This script answers the other half: how much wall-clock time and how
many predictor evaluations did each method spend to get there.  For an expensive
reward model (DeepMD ensemble, geometry-optimised structure score) the predictor
dominates the run, so "which method is faster" is mostly "which method wastes
fewer model calls".

It reads only files written by ``run_experiment.py`` — ``timing.json`` and
``training_log.csv`` — so the expensive runs can happen on a GPU box and the
figures can be made later on a laptop from a few hundred KB of text.

Usage
-----
    python scripts/baselines/compare_timing.py \\
        --run "DQN(bootstrap):runs/ooh_dqn_boot" \\
        --run "DQN(mc):runs/ooh_dqn_mc" \\
        --run "A2C:runs/ooh_a2c" \\
        --out runs/compare/cost --title "OOH: cost to best candidate"

Each PATH is a run directory containing ``timing.json``.  If it instead contains
``seed_*/`` subdirectories (the layout ``run_seeds.py`` produces), every seed is
loaded: curves are drawn as a median line with a min–max band, and the bar
panels show the mean across seeds.

Note on ``--resume-training``
-----------------------------
A resumed run restarts its predictor counters at zero while the predictor cache
is already warm, so its cache-hit rate reads artificially high and its
per-unique-call cost is understated.  Resumed runs are flagged in the printout;
for clean cost numbers, benchmark without ``--resume-training``.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
from typing import Dict, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare methods on wall-clock time and predictor-call cost")
    p.add_argument("--run", action="append", required=True, metavar="LABEL:PATH",
                   help="Repeatable. LABEL is the method name, PATH is a run "
                        "directory (or a parent holding seed_*/ dirs).")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--title", default=None, help="Optional figure title")
    p.add_argument("--sort-by-best", action="store_true",
                   help="Rank methods by best reward instead of keeping the order "
                        "the --run flags were given in (default: command-line order)")
    p.add_argument("--x-log", action="store_true",
                   help="Log-scale the wall-clock and call-count axes")
    p.add_argument("--no-plot", action="store_true", help="Skip the PNG (CSV only)")
    return p.parse_args()


def _split_run(spec: str) -> Tuple[str, str]:
    """Split ``LABEL:PATH``.

    Split from the right so labels may contain ':' — but not on Windows drive
    letters, which we do not support anyway.
    """
    if ":" not in spec:
        raise SystemExit(f"--run must be LABEL:PATH, got {spec!r}")
    label, path = spec.rsplit(":", 1)
    return label.strip(), path.strip()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _seed_dirs(path: str) -> List[str]:
    """Run directories under *path*: either *path* itself or its seed_*/ dirs."""
    if os.path.exists(os.path.join(path, "timing.json")):
        return [path]
    seeds = sorted(glob.glob(os.path.join(path, "seed_*")))
    seeds = [d for d in seeds if os.path.exists(os.path.join(d, "timing.json"))]
    if not seeds:
        raise SystemExit(
            f"no timing.json in {path!r} nor in any {path}/seed_*/ — did the run "
            "predate the timing instrumentation?")
    return seeds


def _load_curve(run_dir: str) -> List[Tuple[float, float, float]]:
    """``(t_wall, n_predict_unique, best_reward_so_far)`` from training_log.csv.

    Returns [] when the log lacks the timing columns (an older run, or one where
    ``--skip-generation``/``--only-generate`` meant no training happened).  The
    bar panels still work in that case; only the curves drop out.
    """
    path = os.path.join(run_dir, "training_log.csv")
    if not os.path.exists(path):
        return []
    out: List[Tuple[float, float, float]] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("phase") not in ("dqn_train", "pg_train"):
                continue
            try:
                t = float(row["t_wall"])
                n = float(row["n_predict_unique"])
                b = float(row["best_reward_so_far"])
            except (KeyError, TypeError, ValueError):
                continue
            out.append((t, n, b))
    # best_reward_so_far is already a running max inside a run, but guard anyway
    # so a resumed/concatenated log can't produce a dipping "best so far" line.
    running = float("-inf")
    fixed = []
    for t, n, b in out:
        running = max(running, b)
        fixed.append((t, n, running))
    return fixed


def _load_timing(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "timing.json")) as f:
        return json.load(f)


def load_method(label: str, path: str) -> dict:
    dirs = _seed_dirs(path)
    timings = [_load_timing(d) for d in dirs]
    curves = [c for c in (_load_curve(d) for d in dirs) if c]
    return {"label": label, "path": path, "dirs": dirs,
            "timings": timings, "curves": curves}


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _mean(vals: Sequence[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in vals if v is not None]
    return statistics.fmean(clean) if clean else None


def _time_to_fraction_of_best(curve: Sequence[Tuple[float, float, float]],
                              frac: float = 0.9) -> Optional[float]:
    """Wall-clock seconds until best-so-far first reaches *frac* of its final value.

    Rewards can be negative (overpotentials are negated, energies are negated),
    so "90% of best" is taken along the span from the first to the final best
    rather than as a raw multiplication — ``0.9 * -3.0`` would otherwise mean
    *better* than the target.
    """
    if not curve:
        return None
    first, final = curve[0][2], curve[-1][2]
    if final <= first:
        return curve[0][0]
    target = first + frac * (final - first)
    for t, _n, b in curve:
        if b >= target:
            return t
    return curve[-1][0]


def summarize(m: dict) -> Dict[str, object]:
    ts = m["timings"]
    preds = [t.get("predictor", {}) for t in ts]
    curves = m["curves"]
    t90 = [x for x in (_time_to_fraction_of_best(c) for c in curves) if x is not None]
    return {
        "method": m["label"],
        "n_runs": len(ts),
        "resumed": any(bool(t.get("resumed")) for t in ts),
        "total_s": _mean([t.get("total_s") for t in ts]),
        "predictor_s": _mean([p.get("t_predict_s") for p in preds]),
        "overhead_s": _mean([t.get("overhead_s") for t in ts]),
        "n_calls": _mean([p.get("n_calls") for p in preds]),
        "n_unique": _mean([p.get("n_unique") for p in preds]),
        "cache_hit_rate": _mean([p.get("cache_hit_rate") for p in preds]),
        "mean_s_per_unique": _mean([p.get("mean_s_per_unique") for p in preds]),
        "best_reward": _mean([p.get("best_reward") for p in preds]),
        "s_to_90pct_of_best": statistics.fmean(t90) if t90 else None,
    }


_FIELDS = ["method", "n_runs", "resumed", "total_s", "predictor_s", "overhead_s",
           "n_calls", "n_unique", "cache_hit_rate", "mean_s_per_unique",
           "best_reward", "s_to_90pct_of_best"]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _band(curves: List[List[Tuple[float, float, float]]], xi: int):
    """Median curve + min/max band on a common x grid.

    ``xi`` selects the x column (0 = wall-clock, 1 = unique predictor calls).
    Each run has its own x samples, so everything is interpolated onto a shared
    grid spanning the overlap of all runs.
    """
    import numpy as np
    xs = [np.asarray([p[xi] for p in c], dtype=float) for c in curves]
    ys = [np.asarray([p[2] for p in c], dtype=float) for c in curves]
    lo = max(float(x[0]) for x in xs)
    hi = min(float(x[-1]) for x in xs)
    if not (hi > lo):
        # No overlap (wildly different budgets) — fall back to the longest run.
        k = max(range(len(curves)), key=lambda i: len(curves[i]))
        return xs[k], ys[k], ys[k], ys[k]
    grid = np.linspace(lo, hi, 300)
    stack = np.vstack([np.interp(grid, x, y) for x, y in zip(xs, ys)])
    return grid, np.median(stack, axis=0), stack.min(axis=0), stack.max(axis=0)


def _plot_curves(ax, methods, colors, xi, xlabel, x_log):
    drew = False
    for m, color in zip(methods, colors):
        curves = m["curves"]
        if not curves:
            continue
        if len(curves) == 1:
            xs = [p[xi] for p in curves[0]]
            ys = [p[2] for p in curves[0]]
            # A single-point curve draws nothing as a bare line; add a marker so
            # short runs stay visible.
            ax.plot(xs, ys, label=m["label"], color=color, lw=1.8,
                    marker="o" if len(xs) < 3 else None, ms=4)
        else:
            g, med, lo, hi = _band(curves, xi)
            ax.plot(g, med, label=m["label"], color=color, lw=1.8)
            ax.fill_between(g, lo, hi, color=color, alpha=0.18, lw=0)
        drew = True
    ax.set_xlabel(xlabel)
    ax.set_ylabel("best reward so far")
    if x_log:
        ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    if drew:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no timing columns in training_log.csv",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
    return drew


def make_figure(methods: List[dict], summaries: List[Dict[str, object]],
                out_png: str, title: Optional[str], x_log: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(methods))]
    labels = [s["method"] for s in summaries]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    (ax1, ax2), (ax3, ax4) = axes

    _plot_curves(ax1, methods, colors, 0, "wall-clock seconds", x_log)
    ax1.set_title("Progress vs wall-clock time (predictor included)")

    _plot_curves(ax2, methods, colors, 1, "cumulative unique predictor calls", x_log)
    ax2.set_title("Progress vs predictor evaluations (hardware-independent)")

    # Panel 3 — where the wall-clock actually went.
    pred = [s["predictor_s"] or 0.0 for s in summaries]
    over = [s["overhead_s"] or 0.0 for s in summaries]
    ax3.bar(labels, pred, color="tab:red", alpha=0.85, label="predictor")
    ax3.bar(labels, over, bottom=pred, color="tab:blue", alpha=0.85, label="RL overhead")
    for i, (p, o) in enumerate(zip(pred, over)):
        ax3.text(i, p + o, f"{p + o:.0f}s", ha="center", va="bottom", fontsize=8)
    ax3.set_ylabel("seconds")
    ax3.set_title("Wall-clock breakdown")
    ax3.grid(True, axis="y", alpha=0.3)
    ax3.legend(fontsize=8)

    # Panel 4 — cost per real evaluation, with the cache-hit rate that explains
    # why a method may look cheap (it re-visited compositions it already knew).
    per_unique = [s["mean_s_per_unique"] or 0.0 for s in summaries]
    hit = [(s["cache_hit_rate"] or 0.0) * 100.0 for s in summaries]
    ax4.bar(labels, per_unique, color="tab:green", alpha=0.85)
    ax4.set_ylabel("seconds per unique predictor call")
    ax4.set_title("Cost per real evaluation (line = cache-hit rate)")
    ax4.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(per_unique):
        ax4.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax4b = ax4.twinx()
    ax4b.plot(labels, hit, color="tab:purple", marker="o", lw=1.5)
    ax4b.set_ylabel("cache-hit rate (%)", color="tab:purple")
    ax4b.tick_params(axis="y", labelcolor="tab:purple")
    ax4b.set_ylim(0, max(100.0, max(hit) * 1.1 if hit else 100.0))

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)


def main() -> None:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    methods = [load_method(*_split_run(spec)) for spec in args.run]
    # Insertion order = the order the --run flags were typed; that order carries
    # through the CSV, the printout and every figure panel.
    summaries = [summarize(m) for m in methods]
    if args.sort_by_best:
        order = sorted(range(len(summaries)),
                       key=lambda i: -(summaries[i]["best_reward"] or float("-inf")))
        methods = [methods[i] for i in order]
        summaries = [summaries[i] for i in order]

    summary_csv = os.path.join(args.out, "timing_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS, extrasaction="ignore")
        w.writeheader()
        for s in summaries:
            w.writerow(s)

    def _f(v, spec=".1f"):
        return "n/a" if v is None else format(v, spec)

    print(f"[timing] {len(methods)} methods")
    for s in summaries:
        print(f"  {s['method']:<18} runs={s['n_runs']} "
              f"total={_f(s['total_s'])}s  predictor={_f(s['predictor_s'])}s "
              f"({_f(s['cache_hit_rate'], '.1%')} cached)  "
              f"overhead={_f(s['overhead_s'])}s  "
              f"best={_f(s['best_reward'], '.4f')}  "
              f"t@90%={_f(s['s_to_90pct_of_best'])}s"
              + ("   [RESUMED — cost numbers understated]" if s["resumed"] else ""))
    print(f"[timing] summary -> {summary_csv}")

    if args.no_plot:
        return
    try:
        make_figure(methods, summaries,
                    os.path.join(args.out, "timing_comparison.png"),
                    args.title, args.x_log)
    except ImportError:
        print("[timing] matplotlib not available; skipped figure (CSV written).")
        return
    print(f"[timing] figure -> {os.path.join(args.out, 'timing_comparison.png')}")


if __name__ == "__main__":
    main()
