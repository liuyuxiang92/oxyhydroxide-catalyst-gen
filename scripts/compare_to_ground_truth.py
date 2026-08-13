#!/usr/bin/env python3
"""compare_to_ground_truth.py — gap-to-global-minimum vs. budget.

For a scenario small enough to enumerate exhaustively (e.g. perovskite Level 1,
73*73 = 5,329 candidates — see the separate, standalone
``../perovskite_ground_truth/enumerate.py``, which is NOT part of this repo),
this answers: at each sampling budget, how close did each method (DQN
bootstrap/mc, A2C, BO, GA) get to the TRUE global optimum, and did it find it
exactly?

Each (arm, budget, seed) is an INDEPENDENT run (this repo's established
convention — see ``submit_perovskite_sweep.sh`` / ``submit_sweep.sh`` headers:
episode budgets are set by hand per launch, not resumed across budgets), so
this reads the single best candidate each run's ``generated.csv`` found — not
a within-run "best so far" trajectory (that column, ``best_reward_so_far`` in
``training_log.csv``, only exists for the RL arms; BO/GA never write
``training_log.csv``, only the RL-compatible ``generated.csv`` every arm
shares, per ``scripts/baselines/_common.py``'s ``write_outputs``).

Sign convention: ``generated.csv``'s ``reward`` is ALWAYS "higher is better"
(``structure_score.py`` folds ``direction`` into it internally), while the
external ground-truth table stores the raw, untransformed predictor score
("lower is better" for this ``direction: min`` scenario). This script converts
one to the other — see ``_raw_score_from_generated_row`` — rather than
assuming the two files share a sign convention.

Usage::

    python scripts/compare_to_ground_truth.py \\
        --ground-truth /path/to/perovskite_ground_truth/ground_truth.csv \\
        --run "DQN(bootstrap):runs/perovskite_l1/dqn_bootstrap_eps*_seed*" \\
        --run "DQN(mc):runs/perovskite_l1/dqn_mc_eps*_seed*" \\
        --run "A2C:runs/perovskite_l1/a2c_eps*_seed*" \\
        --run "BO:runs/perovskite_l1/bo_eps*_seed*" \\
        --run "GA:runs/perovskite_l1/ga_eps*_seed*" \\
        --out runs/compare/perovskite_l1_ground_truth
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import statistics
from typing import Dict, List, Optional, Sequence, Tuple

_RUN_DIR_RE = re.compile(r"eps(?P<budget>\d+)_seed(?P<seed>\d+)$")


def _parse_run_spec(spec: str) -> Tuple[str, str]:
    if ":" not in spec:
        raise SystemExit(f"--run must be LABEL:GLOB_PATTERN, got {spec!r}")
    label, pattern = spec.split(":", 1)
    return label, pattern


def _load_ground_truth(path: str) -> Tuple[Dict[Tuple[str, str], float], float, Tuple[str, str]]:
    """Return ``({(A,B): score}, true_min, (A,B) of the true min)``."""
    scores: Dict[Tuple[str, str], float] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            scores[(row["A"], row["B"])] = float(row["score"])
    if not scores:
        raise SystemExit(f"{path}: no rows.")
    best_pair = min(scores, key=lambda k: scores[k])
    return scores, scores[best_pair], best_pair


def _raw_score_from_generated_row(row: dict) -> float:
    """Undo structure_score.py's ``direction`` sign flip.

    ``reward`` (and, when std==0 as it is here with ``n_random_configs: 1``,
    the numerically-identical ``dp_mean``) is ``direction * raw_score`` folded
    with an (here: zero) uncertainty penalty. This scenario's config uses
    ``direction: min``, so ``raw_score = -reward``.
    """
    return -float(row["reward"])


def _best_row_in_generated(run_dir: str) -> Optional[dict]:
    path = os.path.join(run_dir, "generated.csv")
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    # generated.csv is sorted by reward descending (write_outputs' own
    # convention) but don't rely on file order — recompute the max directly.
    return max(rows, key=lambda r: float(r["reward"]))


def _run_ab(run_dir: str, best_row: dict) -> Optional[Tuple[str, str]]:
    """Best-effort (A, B) for this run's winning candidate, from its formula.

    Only used for the "found the exact global optimum" check; a parse miss
    just means that check is skipped for this run (the gap number itself
    doesn't need it).
    """
    formula = best_row.get("formula", "")
    elems = re.findall(r"[A-Z][a-z]?", formula)
    if len(elems) < 2:
        return None
    return elems[0], elems[1]


def _collect(label: str, pattern: str, gt_scores: Dict[Tuple[str, str], float],
             true_min: float) -> List[dict]:
    records: List[dict] = []
    for run_dir in sorted(glob.glob(pattern)):
        if not os.path.isdir(run_dir):
            continue
        m = _RUN_DIR_RE.search(os.path.basename(run_dir.rstrip("/")))
        if not m:
            print(f"[warn] {run_dir}: dir name doesn't match '...eps<N>_seed<N>', skipping")
            continue
        budget, seed = int(m["budget"]), int(m["seed"])

        best_row = _best_row_in_generated(run_dir)
        if best_row is None:
            print(f"[warn] {run_dir}: no generated.csv / no rows, skipping")
            continue

        raw_score = _raw_score_from_generated_row(best_row)
        gap = raw_score - true_min
        ab = _run_ab(run_dir, best_row)
        found_optimum = bool(ab and gt_scores.get(ab) is not None and abs(gt_scores[ab] - true_min) < 1e-9)

        records.append({
            "method": label, "budget": budget, "seed": seed, "run_dir": run_dir,
            "best_score": raw_score, "gap": gap, "found_optimum": found_optimum,
        })
    return records


def _summarize(records: Sequence[dict]) -> List[dict]:
    by_key: Dict[Tuple[str, int], List[dict]] = {}
    for r in records:
        by_key.setdefault((r["method"], r["budget"]), []).append(r)

    out: List[dict] = []
    for (method, budget), rs in sorted(by_key.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        gaps = [r["gap"] for r in rs]
        out.append({
            "method": method, "budget": budget, "n_seeds": len(rs),
            "mean_gap": statistics.fmean(gaps), "min_gap": min(gaps), "max_gap": max(gaps),
            "std_gap": statistics.pstdev(gaps) if len(gaps) > 1 else 0.0,
            "frac_found_optimum": sum(r["found_optimum"] for r in rs) / len(rs),
        })
    return out


def _write_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot(summary: List[dict], out_png: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = sorted({r["method"] for r in summary})
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, method in enumerate(methods):
        rows = sorted((r for r in summary if r["method"] == method), key=lambda r: r["budget"])
        budgets = [r["budget"] for r in rows]
        means = [r["mean_gap"] for r in rows]
        lo = [r["min_gap"] for r in rows]
        hi = [r["max_gap"] for r in rows]
        color = cmap(i % 10)
        ax.plot(budgets, means, marker="o", label=method, color=color)
        ax.fill_between(budgets, lo, hi, alpha=0.15, color=color)

    ax.set_xlabel("Budget (predictor calls)")
    ax.set_ylabel("Gap to global minimum (raw score units)")
    ax.set_title("Perovskite Level 1: distance from the true global minimum")
    ax.axhline(0.0, color="black", linewidth=0.5, linestyle="--")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f"[compare_to_ground_truth] wrote {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ground-truth", required=True, help="Path to ground_truth.csv from the standalone enumerate.py.")
    ap.add_argument("--run", action="append", required=True, metavar="LABEL:GLOB",
                     help="e.g. 'DQN(bootstrap):runs/perovskite_l1/dqn_bootstrap_eps*_seed*'")
    ap.add_argument("--out", required=True, help="Output directory for the summary CSV + plot.")
    args = ap.parse_args()

    gt_scores, true_min, true_min_ab = _load_ground_truth(args.ground_truth)
    print(f"[compare_to_ground_truth] true global minimum: A={true_min_ab[0]} B={true_min_ab[1]} "
          f"score={true_min:+.4f} (of {len(gt_scores)} candidates)")

    records: List[dict] = []
    for spec in args.run:
        label, pattern = _parse_run_spec(spec)
        recs = _collect(label, pattern, gt_scores, true_min)
        print(f"[compare_to_ground_truth] {label}: {len(recs)} runs matched '{pattern}'")
        records.extend(recs)

    if not records:
        raise SystemExit("No runs matched any --run pattern.")

    summary = _summarize(records)
    os.makedirs(args.out, exist_ok=True)
    _write_csv(os.path.join(args.out, "per_run.csv"), records)
    _write_csv(os.path.join(args.out, "summary.csv"), summary)
    _plot(summary, os.path.join(args.out, "gap_vs_budget.png"))

    print("\nmethod            budget  n_seeds  mean_gap    min_gap   frac_found_optimum")
    for r in summary:
        print(f"{r['method']:<16}  {r['budget']:>6}  {r['n_seeds']:>7}  "
              f"{r['mean_gap']:>9.4f}  {r['min_gap']:>8.4f}  {r['frac_found_optimum']:>18.2f}")


if __name__ == "__main__":
    main()
