#!/usr/bin/env python3
"""Plot the reward of every training episode, per method.

``compare_methods.py`` compares the *generated* candidates — what the trained
agent produces at the end. This script looks at the search itself: one point per
sampled episode over the whole training run, so you can see whether a method kept
probing new regions or converged onto a single composition and idled.

That distinction is the whole story on the oxide benchmark. A2C's mean return
improves monotonically while its *best* candidate freezes ~20% in — a plot of the
batch mean looks like success, a plot of the point cloud shows the tail dying.

Granularity caveat
------------------
DQN writes one ``dqn_train`` row per episode. PG writes one ``pg_train`` row per
*iteration*, holding only the batch mean; ``pg_episode`` rows (one per sampled
episode) were added later. Runs predating that only have batch means, which are
not distributionally comparable with raw episodes — a mean over 25 episodes has
√25 = 5× less spread by construction. This script detects which is present, uses
per-episode rows when it can, and labels every panel with what it actually drew.
Summary statistics that would be misleading across granularities (spread, worst)
are still written to CSV but flagged there.

Usage
-----
    python scripts/baselines/compare_training_rewards.py \
        --run "DQN(bootstrap):runs/sinter_dqn_eps_7500" \
        --run "DQN(mc):runs/sinter_mc_eps_7500" \
        --run "A2C:runs/sinter_a2c_eps_7500" \
        --out runs/compare/train_rewards \
        --minimize --y-label "Sintering temperature (K)" \
        --title "Sinter, 7.5k episodes"
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from typing import Dict, List, Tuple

import numpy as np

# Phases that carry one row per sampled training episode, in preference order.
_EPISODE_PHASES = ("dqn_train", "pg_episode")
# Fallback: one row per PG iteration, holding the batch mean.
_BATCH_PHASE = "pg_train"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-episode training reward distribution")
    p.add_argument("--run", action="append", required=True, metavar="LABEL:PATH",
                   help="Repeatable. PATH is a run directory or its training_log.csv")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--title", default=None, help="Optional figure title")
    p.add_argument("--minimize", action="store_true",
                   help="The scenario minimizes a positive physical quantity "
                        "(reward = -objective). Plot objective = -reward, lower "
                        "is better. Ranking is unchanged.")
    p.add_argument("--y-label", default=None, help="Override the y-axis label")
    p.add_argument("--max-points", type=int, default=60000,
                   help="Thin the scatter above this many points. Stats and the "
                        "best-so-far line always use every point. The default "
                        "clears the 45k-episode arm, so nothing is thinned in the "
                        "standard sweep. 0 disables thinning entirely.")
    p.add_argument("--no-plot", action="store_true", help="Skip the PNG (CSV only)")
    return p.parse_args()


def _split_run(spec: str) -> Tuple[str, str]:
    if ":" not in spec:
        raise SystemExit(f"--run must be LABEL:PATH, got {spec!r}")
    label, path = spec.split(":", 1)
    return label.strip(), path.strip()


def _discount_correction(run_dir: str) -> Tuple[float, str]:
    """``gamma ** (n_components - 1)`` for a legacy PG log, from run_config.json.

    PG logs the *discounted* return G_0 while DQN logs the undiscounted terminal
    reward, both under the column name ``return``. For a terminal-only reward (all
    envs in this repo) G_0 = gamma^(n-1) * r_T exactly, so dividing recovers the
    property's own units. Newer runs carry ``terminal_reward`` and skip this.
    """
    cfg_path = os.path.join(run_dir, "run_config.json")
    if not os.path.exists(cfg_path):
        return 1.0, "no run_config.json — left as discounted return"
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
        gamma = float(cfg.get("gamma", 1.0))
        n = int(cfg.get("n_components", 1))
    except (ValueError, TypeError, json.JSONDecodeError):
        return 1.0, "unreadable run_config.json — left as discounted return"
    if gamma >= 1.0 or n <= 1:
        return 1.0, ""
    factor = gamma ** (n - 1)
    return factor, (f"un-discounted by gamma^{n - 1} = {factor:.4f} "
                    f"(gamma={gamma}, n_components={n})")


def _load_training_rewards(path: str) -> Dict[str, object]:
    """Return ``{episodes, rewards, granularity, batch_eps, note}`` for one run."""
    run_dir = path if os.path.isdir(path) else os.path.dirname(path)
    if os.path.isdir(path):
        path = os.path.join(path, "training_log.csv")
    if not os.path.exists(path):
        raise SystemExit(f"training_log.csv not found: {path}")

    by_phase: Dict[str, List[dict]] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            by_phase.setdefault(row.get("phase", ""), []).append(row)

    # DQN's warmup rolls out random episodes that each pay a real predictor call,
    # so they are part of the training cost and belong on the same axis, ahead of
    # the trained episodes. (PG's warmup neutralises reward_fn — no predictor calls,
    # no rewards, nothing to prepend.)
    warmup = by_phase.get("dqn_warmup") or []

    phase = next((p for p in _EPISODE_PHASES if by_phase.get(p)), None)
    granularity, batch_eps = "episode", 1
    if phase is None:
        if not by_phase.get(_BATCH_PHASE):
            raise SystemExit(
                f"{path} has no training rows "
                f"(phases present: {sorted(by_phase) or 'none'})"
            )
        phase = _BATCH_PHASE
        granularity = "batch-mean"
        try:
            batch_eps = int(float(by_phase[phase][0].get("batch_eps") or 1))
        except (TypeError, ValueError):
            batch_eps = 1

    # DQN's `return` IS the undiscounted terminal reward; PG's is the discounted
    # return G_0. Prefer the explicit column, and correct legacy PG logs that
    # predate it — otherwise the two methods land on one axis in different units.
    rows = by_phase[phase]
    is_pg = phase in (_BATCH_PHASE, "pg_episode")
    n_warmup = 0
    if warmup and not is_pg:
        n_warmup = len(warmup)
        rows = warmup + rows
    has_terminal = any((r.get("terminal_reward") or "") != "" for r in rows)
    col, factor, note = "return", 1.0, ""
    if has_terminal:
        col = "terminal_reward"
    elif is_pg:
        factor, note = _discount_correction(run_dir)

    def _value(row: dict) -> float:
        """Read one row's reward, falling back per-row rather than per-file.

        A log can mix schemas: a run started before `terminal_reward` existed and
        resumed after it has rows of both kinds. Choosing one column for the whole
        file would make every row of the other kind unparseable — and they would be
        dropped silently by the caller's `except`, not raise.
        """
        raw = row.get(col)
        if raw not in (None, ""):
            return float(raw) / (1.0 if col == "terminal_reward" else factor)
        return float(row["return"]) / factor

    episodes: List[float] = []
    rewards: List[float] = []
    for i, row in enumerate(rows, start=1):
        try:
            rewards.append(_value(row))
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            continue
        try:
            ep = float(row["episode"])
        except (KeyError, TypeError, ValueError):
            ep = float(i)
        # Both phases number their episodes from 1, so shift the trained ones past
        # the warmup block; the x axis is then the true cumulative budget spent.
        if n_warmup and i > n_warmup:
            ep += n_warmup
        episodes.append(ep)

    if not rewards:
        raise SystemExit(f"{path}: phase {phase!r} has no usable {col!r} values")
    return {
        "episodes": np.asarray(episodes, dtype=float),
        "rewards": np.asarray(rewards, dtype=float),
        "granularity": granularity,
        "batch_eps": batch_eps,
        "reward_col": col,
        "note": note,
        "n_warmup": n_warmup,
    }


def _summarize(label: str, data: Dict[str, object], minimize: bool) -> Dict[str, object]:
    """Rank on reward (higher = better); ``minimize`` only changes reporting."""
    rw = np.asarray(data["rewards"], dtype=float)
    eps = np.asarray(data["episodes"], dtype=float)
    sign = -1.0 if minimize else 1.0
    key = "objective" if minimize else "reward"

    best_i = int(np.argmax(rw))
    # Where in the budget the best sample was first drawn. A small fraction with a
    # long tail after it is the freeze signature: the agent stopped finding better.
    frac_at_best = float(eps[best_i] / eps[-1]) if eps[-1] > 0 else float("nan")
    return {
        "method": label,
        "granularity": data["granularity"],
        "reward_col": data.get("reward_col", "return"),
        "discount_note": data.get("note", ""),
        "batch_eps": data["batch_eps"],
        "n_points": int(rw.size),
        "n_episodes": int(eps[-1]),
        f"best_{key}": sign * float(rw[best_i]),
        "best_at_episode": int(eps[best_i]),
        "best_at_budget_frac": round(frac_at_best, 4),
        f"mean_{key}": sign * float(statistics.fmean(rw.tolist())),
        f"median_{key}": sign * float(statistics.median(rw.tolist())),
        # Spread is granularity-dependent — a batch mean over B episodes has ~1/sqrt(B)
        # the spread of raw episodes. Only compare these within a granularity.
        f"p05_{key}": sign * float(np.percentile(rw, 95 if minimize else 5)),
        f"p95_{key}": sign * float(np.percentile(rw, 5 if minimize else 95)),
        f"std_{key}": float(np.std(rw)),
        "_best_reward": float(rw[best_i]),
    }


def _best_noun(data: Dict[str, object]) -> str:
    """What the extremum actually *is*, so it never reads as a best episode.

    Under batch-mean granularity the minimum is the best mean over ``batch_eps``
    episodes — a fundamentally different (and much less extreme) quantity than the
    single best episode a per-episode log reports.
    """
    return "best episode" if data["granularity"] == "episode" else "best batch mean"


def _subsample(eps: np.ndarray, rw: np.ndarray, max_points: int):
    """Even thinning for the scatter only. Deterministic, and keeps the endpoints."""
    if max_points <= 0 or rw.size <= max_points:
        return eps, rw
    idx = np.unique(np.linspace(0, rw.size - 1, max_points).astype(int))
    return eps[idx], rw[idx]


def main() -> None:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    sign = -1.0 if args.minimize else 1.0
    key = "objective" if args.minimize else "reward"

    runs = [_split_run(s) for s in args.run]
    loaded = [(label, _load_training_rewards(path)) for label, path in runs]
    summaries = [_summarize(label, data, args.minimize) for label, data in loaded]

    granularities = {s["granularity"] for s in summaries}
    if len(granularities) > 1:
        print("[WARN] Mixed log granularity across runs: "
              + ", ".join(f"{s['method']}={s['granularity']}" for s in summaries)
              + ".\n[WARN] Batch means are averages over batch_eps episodes, so their "
                "spread is compressed by ~1/sqrt(batch_eps). Best-so-far and "
                "best_at_budget_frac stay comparable; distribution width does not.")

    for (_, data), s in zip(loaded, summaries):
        if data.get("note"):
            print(f"[INFO] {s['method']}: {data['note']}")

    csv_path = os.path.join(args.out, "training_reward_summary.csv")
    fieldnames = [k for k in summaries[0] if not k.startswith("_")]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    print(f"[INFO] Summary written to {csv_path}")
    for (_, data), s in zip(loaded, summaries):
        print(f"  {s['method']:<18} n={s['n_points']:>6} ({s['granularity']})  "
              f"{_best_noun(data)} {key}={s[f'best_{key}']:.2f} at episode "
              f"{s['best_at_episode']} ({100 * s['best_at_budget_frac']:.0f}% of budget)")

    if args.no_plot:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(loaded)
    fig, axes = plt.subplots(1, n + 1, figsize=(4.2 * n + 4.5, 5.0), sharey=True)
    axes = np.atleast_1d(axes)

    ylabel = args.y_label or (f"Training episode {key}")

    for i, ((label, data), summary) in enumerate(zip(loaded, summaries)):
        ax = axes[i]
        eps = np.asarray(data["episodes"], dtype=float)
        rw = np.asarray(data["rewards"], dtype=float)

        # Running best is computed on every point, then plotted; only the cloud is
        # thinned, so the frontier line is never distorted by subsampling.
        running_best = sign * np.maximum.accumulate(rw)
        eps_s, rw_s = _subsample(eps, rw, args.max_points)

        ax.scatter(eps_s, sign * rw_s, s=6, color="tab:blue", alpha=0.25,
                   linewidths=0, rasterized=True, zorder=2)
        ax.plot(eps, running_best, color="tab:red", lw=1.8, zorder=4,
                label="best so far")
        ax.axhline(summary[f"best_{key}"], color="tab:green", ls="--", lw=1.2,
                   zorder=3,
                   label=f"{_best_noun(data)} = {summary[f'best_{key}']:.1f}")

        # Say how many points EXIST first; the thinning count is a display detail
        # and reads as missing data if it comes first.
        gran = ("per episode" if data["granularity"] == "episode"
                else f"batch mean of {data['batch_eps']} eps")
        drawn = f"{rw.size:,} pts ({gran})"
        if data.get("n_warmup"):
            drawn += f", incl. {data['n_warmup']:,} warmup"
        if rw_s.size < rw.size:
            drawn += f"\n{rw_s.size:,} drawn — cloud thinned, curve uses all"
        ax.set_title(f"{label}\n{drawn}", fontsize=10)
        if data.get("n_warmup"):
            ax.axvline(data["n_warmup"], color="0.4", ls=":", lw=1.2, zorder=1)
        ax.set_xlabel("training episode")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right" if args.minimize else "lower right")
        if i == 0:
            ax.set_ylabel(ylabel)

    # Marginal distribution, matching compare_methods.py's panel style.
    ax_d = axes[n]
    dists = [sign * np.asarray(d["rewards"], dtype=float) for _, d in loaded]
    labels = [lb for lb, _ in loaded]
    positions = list(range(1, n + 1))
    violin_data, violin_pos = [], []
    for d, pos in zip(dists, positions):
        if d.size >= 2 and float(np.std(d)) > 0:
            violin_data.append(d)
            violin_pos.append(pos)
    if violin_data:
        ax_d.violinplot(violin_data, positions=violin_pos, showmeans=True,
                        showextrema=True)
    for d, pos, s in zip(dists, positions, summaries):
        ax_d.scatter([pos], [s[f"best_{key}"]], marker="*", s=140,
                     color="tab:green", zorder=5)
    ax_d.set_xticks(positions)
    ax_d.set_xticklabels(
        [f"{lb}\nn={s['n_points']:,}" for lb, s in zip(labels, summaries)],
        fontsize=9,
    )
    ax_d.set_title("Distribution over training\n(★ = best point in that log)", fontsize=10)
    ax_d.grid(alpha=0.3, axis="y")
    if len(granularities) > 1:
        ax_d.text(0.5, 0.02,
                  "mixed granularity — widths and ★ not comparable",
                  transform=ax_d.transAxes, ha="center", fontsize=8,
                  color="tab:red")

    if args.title:
        fig.suptitle(args.title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95) if args.title else None)
    png = os.path.join(args.out, "training_rewards.png")
    fig.savefig(png, dpi=150)
    print(f"[INFO] Figure written to {png}")


if __name__ == "__main__":
    main()
