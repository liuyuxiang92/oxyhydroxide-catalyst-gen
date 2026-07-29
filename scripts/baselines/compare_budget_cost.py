#!/usr/bin/env python3
"""Compare methods across *training-budget* sweeps: what does more sampling buy?

``compare_methods.py`` answers "which method wins at a fixed budget".  This one
answers the orthogonal question: as the number of episodes sampled during policy /
value-net training grows (2500 -> 7500 -> 45000), how does **wall-clock cost** and
**best candidate quality** move, per method?

Each ``--run`` is a finished run directory.  The x-axis budget is read from
``run_config.json`` — the *configured* episode count, never the directory name —
so a mislabelled sweep directory shows up as two points stacked on one x value
instead of silently faking a trend.

Usage
-----
    python scripts/baselines/compare_budget_cost.py \\
        --run "DQN(bootstrap):runs/sinter_dqn_eps_2500" \\
        --run "DQN(bootstrap):runs/sinter_dqn_eps_7500" \\
        --run "DQN(MC):runs/sinter_mc_eps_2500" \\
        --run "DQN(MC):runs/sinter_mc_eps_7500" \\
        --run "A2C:runs/sinter_a2c_eps_2500" \\
        --run "A2C:runs/sinter_a2c_eps_7500" \\
        --minimize --y-label "sintering temperature" \\
        --out runs/compare/sinter_cost --title "Sinter: cost vs budget"

Panels
------
1. wall-clock vs training budget            (what more sampling costs)
2. best candidate vs training budget        (what more sampling buys)
3. best candidate vs wall-clock             (the cost/quality frontier)
4. wall-clock split into predictor vs RL overhead, per run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Optional, Tuple

_SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
_TEXT_PRIMARY = "#0b0b0b"
_TEXT_SECONDARY = "#52514e"
_SURFACE = "#fcfcfb"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Budget sweep: cost and quality per method")
    p.add_argument("--run", action="append", required=True, metavar="LABEL:PATH",
                   help="Repeatable. LABEL is the method; PATH is a run directory. "
                        "Repeat the same LABEL once per budget.")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--title", default=None, help="Optional figure title")
    p.add_argument("--minimize", action="store_true",
                   help="reward = -objective (temperature scenarios); plot the "
                        "positive objective, lower is better")
    p.add_argument("--y-label", default="objective",
                   help="Name of the physical quantity, e.g. 'sintering temperature'")
    p.add_argument("--series-color", action="append", default=[], metavar="LABEL:HEX",
                   help="Repeatable. Pin a method to a colour so it keeps the same "
                        "hue across a figure set.")
    p.add_argument("--hours", action="store_true",
                   help="Plot wall-clock in hours instead of seconds")
    p.add_argument("--no-plot", action="store_true", help="Skip the PNG (CSV only)")
    return p.parse_args()


def _split_run(spec: str) -> Tuple[str, str]:
    if ":" not in spec:
        raise SystemExit(f"--run must be LABEL:PATH, got {spec!r}")
    label, path = spec.split(":", 1)
    return label.strip(), path.strip()


def _budget(cfg: dict) -> int:
    """Episodes sampled during training (warmup + on-policy/online rollouts).

    This is the number the sweep varies, and the only budget that is comparable
    across DQN and PG — ``num_gen_eps`` is a separate, post-training phase.
    """
    if str(cfg.get("method")) == "dqn":
        return int(cfg.get("dqn_warmup_eps", 0)) + int(cfg.get("dqn_num_train_eps", 0))
    return (int(cfg.get("pg_warmup_eps", 0))
            + int(cfg.get("pg_num_iters", 0)) * int(cfg.get("pg_batch_eps", 0)))


def _load_run(path: str, minimize: bool) -> Optional[Dict[str, object]]:
    """Read one run directory. Returns ``None`` (with a warning) if incomplete."""
    cfg_p = os.path.join(path, "run_config.json")
    tim_p = os.path.join(path, "timing.json")
    gen_p = os.path.join(path, "generated.csv")
    missing = [os.path.basename(p) for p in (cfg_p, tim_p, gen_p) if not os.path.exists(p)]
    if missing:
        print(f"[budget] SKIP {path}: incomplete run, missing {', '.join(missing)}")
        return None

    with open(cfg_p) as f:
        cfg = json.load(f)
    with open(tim_p) as f:
        tim = json.load(f)
    with open(gen_p) as f:
        rows = [r for r in csv.DictReader(f)]

    rewards = []
    for r in rows:
        try:
            rewards.append((float(r["reward"]), r.get("formula", "")))
        except (TypeError, ValueError):
            continue
    if not rewards:
        print(f"[budget] SKIP {path}: generated.csv has no usable reward rows")
        return None
    rewards.sort(key=lambda t: -t[0])

    sign = -1.0 if minimize else 1.0
    pred = tim.get("predictor", {})
    total = float(tim.get("total_s", 0.0))
    t_pred = float(pred.get("t_predict_s", 0.0))
    return {
        "run": os.path.basename(os.path.normpath(path)),
        "budget_eps": _budget(cfg),
        "num_gen_eps": int(cfg.get("num_gen_eps", 0)),
        "total_s": total,
        "t_predict_s": t_pred,
        "overhead_s": total - t_pred,
        "n_predict_calls": int(pred.get("n_calls", 0)),
        "n_predict_unique": int(pred.get("n_unique", 0)),
        "n_candidates": len(rewards),
        "best_generated": sign * rewards[0][0],
        "best_formula": rewards[0][1],
        # timing.json tracks the best value seen by *any* predictor call, so it
        # can beat generated.csv when training stumbled on a candidate that
        # generation never re-emitted.
        "best_any_phase": sign * float(pred["best_reward"]) if pred.get("best_reward") is not None else None,
        "resumed": bool(tim.get("resumed", False)),
    }


def main() -> None:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    series: Dict[str, List[Dict[str, object]]] = {}
    for spec in args.run:
        label, path = _split_run(spec)
        rec = _load_run(path, args.minimize)
        if rec is None:
            continue
        rec["method"] = label
        series.setdefault(label, []).append(rec)
    if not series:
        raise SystemExit("[budget] no complete runs to plot")
    for recs in series.values():
        recs.sort(key=lambda r: r["budget_eps"])

    dup_labels: List[str] = []
    for label, recs in series.items():
        budgets = [r["budget_eps"] for r in recs]
        if len(set(budgets)) != len(budgets):
            dup_labels.append(label)
            print(f"[budget] WARNING {label}: duplicate configured budgets {budgets} — "
                  f"the sweep directories disagree with run_config.json")

    fields = ["method", "run", "budget_eps", "num_gen_eps", "total_s", "t_predict_s",
              "overhead_s", "n_predict_calls", "n_predict_unique", "n_candidates",
              "best_generated", "best_any_phase", "best_formula", "resumed"]
    out_csv = os.path.join(args.out, "budget_cost_summary.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for recs in series.values():
            for r in recs:
                w.writerow(r)

    best_word = "lowest" if args.minimize else "best"
    print(f"[budget] {len(series)} methods | budget -> wall-clock / {best_word} {args.y_label}:")
    for label, recs in series.items():
        for r in recs:
            print(f"  {label:<16} eps={r['budget_eps']:<6} "
                  f"{r['total_s']/3600:6.2f} h  "
                  f"{best_word}={r['best_generated']:9.2f}  "
                  f"n_cand={r['n_candidates']:<5} {r['best_formula']}")
    print(f"[budget] summary -> {out_csv}")

    if args.no_plot:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[budget] matplotlib not available; skipped figure (CSV written).")
        return

    tdiv, tunit = (3600.0, "hours") if args.hours else (1.0, "seconds")
    pinned = dict(_split_run(spec) for spec in args.series_color)
    colors = {lbl: pinned.get(lbl, _SERIES_COLORS[i % len(_SERIES_COLORS)])
              for i, lbl in enumerate(series)}

    def _kfmt(n: int) -> str:
        return f"{n/1000:g}k" if n >= 1000 else str(n)

    # A log budget axis over {2.5k, 7.5k, 45k} otherwise draws a single "10^4"
    # decade tick and no way to read which point is which. Methods don't land on
    # identical budgets either (A2C's warmup makes 2700 where DQN makes 2500), so
    # cluster budgets within 20% into one tick rather than overprinting labels.
    all_budgets = sorted({r["budget_eps"] for recs in series.values() for r in recs})
    clusters: List[List[int]] = [[all_budgets[0]]]
    for b in all_budgets[1:]:
        if b <= clusters[-1][-1] * 1.2:
            clusters[-1].append(b)
        else:
            clusters.append([b])
    tick_pos = [(min(c) * max(c)) ** 0.5 for c in clusters]
    tick_lab = [_kfmt(c[0]) if len(set(c)) == 1
                else f"{_kfmt(min(c))}–{_kfmt(max(c))}" for c in clusters]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    (ax_cost, ax_qual), (ax_front, ax_split) = axes

    # --- 1. cost vs budget -------------------------------------------------
    for lbl, recs in series.items():
        x = [r["budget_eps"] for r in recs]
        y = [r["total_s"] / tdiv for r in recs]
        ax_cost.plot(x, y, "-o", color=colors[lbl], linewidth=2.0,
                     markersize=8, label=lbl, zorder=3)
    ax_cost.set_xlabel("training episodes sampled", color=_TEXT_SECONDARY)
    ax_cost.set_ylabel(f"total wall-clock ({tunit})", color=_TEXT_SECONDARY)
    ax_cost.set_title("Cost: wall-clock vs sampling budget", color=_TEXT_PRIMARY)

    # --- 2. quality vs budget ----------------------------------------------
    for si, (lbl, recs) in enumerate(series.items()):
        x = [r["budget_eps"] for r in recs]
        y = [r["best_generated"] for r in recs]
        ax_qual.plot(x, y, "-o", color=colors[lbl], linewidth=2.0,
                     markersize=8, label=lbl, zorder=3)
        # Methods routinely converge on the *same* best candidate, so their n=
        # labels land on one point; give each series its own vertical lane and
        # tint it with the series colour so the stack stays attributable.
        for r in recs:
            ax_qual.annotate(f"n={r['n_candidates']}",
                             (r["budget_eps"], r["best_generated"]),
                             xytext=(0, 11 + 11 * si), textcoords="offset points",
                             ha="center", fontsize=7.5, color=colors[lbl])
    ax_qual.set_xlabel("training episodes sampled", color=_TEXT_SECONDARY)
    ax_qual.set_ylabel(f"{best_word} {args.y_label}", color=_TEXT_SECONDARY)
    ax_qual.set_title(f"Quality: {best_word} candidate vs sampling budget",
                      color=_TEXT_PRIMARY)

    # --- 3. the frontier ----------------------------------------------------
    for lbl, recs in series.items():
        x = [r["total_s"] / tdiv for r in recs]
        y = [r["best_generated"] for r in recs]
        ax_front.plot(x, y, "-o", color=colors[lbl], linewidth=2.0,
                      markersize=8, label=lbl, zorder=3)
        for r in recs:
            ax_front.annotate(_kfmt(r["budget_eps"]),
                              (r["total_s"] / tdiv, r["best_generated"]),
                              xytext=(0, 9), textcoords="offset points",
                              ha="center", fontsize=7.5, color=_TEXT_SECONDARY)
    ax_front.set_xlabel(f"total wall-clock ({tunit})", color=_TEXT_SECONDARY)
    ax_front.set_ylabel(f"{best_word} {args.y_label}", color=_TEXT_SECONDARY)
    arrow = "down-left is better" if args.minimize else "up-left is better"
    ax_front.set_title(f"Frontier: quality per unit time ({arrow})", color=_TEXT_PRIMARY)

    # --- 4. where the wall-clock goes --------------------------------------
    flat = [r for recs in series.values() for r in recs]
    flat.sort(key=lambda r: (r["budget_eps"], r["method"]))
    ypos = list(range(len(flat)))[::-1]
    for y, r in zip(ypos, flat):
        c = colors[r["method"]]
        ax_split.barh(y, r["t_predict_s"] / tdiv, height=0.62, color=c, zorder=3)
        # 2px surface gap between the two segments keeps them readable when the
        # predictor slice is thin.
        ax_split.barh(y, r["overhead_s"] / tdiv, height=0.62,
                      left=r["t_predict_s"] / tdiv, color=c, alpha=0.28,
                      edgecolor=_SURFACE, linewidth=2.0, zorder=3)
    ax_split.set_yticks(ypos)
    ax_split.set_yticklabels([f"{r['method']}  {_kfmt(r['budget_eps'])}" for r in flat],
                             fontsize=8.5)
    ax_split.set_xlabel(f"wall-clock ({tunit}) — solid: predictor, faded: RL overhead",
                        color=_TEXT_SECONDARY)
    ax_split.set_title("Where the time goes", color=_TEXT_PRIMARY)
    ax_split.grid(True, axis="x", alpha=0.25, linewidth=0.6)

    # Log spacing keeps 2.5k / 7.5k from collapsing onto each other next to 45k,
    # but the default decade locator would draw a lone "10^4" tick; pin the ticks
    # to the budgets actually run.
    for ax in (ax_cost, ax_qual):
        ax.set_xscale("log")
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lab)
        ax.minorticks_off()

    for ax in (ax_cost, ax_qual, ax_front):
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.legend(frameon=False, fontsize=9)
    for ax in axes.ravel():
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for side in ("left",):
        ax_split.spines[side].set_visible(False)

    ax_qual.margins(y=0.22)  # headroom for the stacked n= lanes

    if args.title:
        fig.suptitle(args.title, color=_TEXT_PRIMARY, fontsize=14)
    fig.tight_layout()
    if dup_labels:
        # A duplicate budget means two sweep directories hold the same configured
        # episode count — they plot as a vertical segment, not a trend. Say so on
        # the figure itself; a console warning does not survive into a slide deck.
        fig.subplots_adjust(bottom=0.11)
        fig.text(0.5, 0.012,
                 "⚠ duplicate configured budgets for " + ", ".join(dup_labels) +
                 " — those sweep directories hold the same run_config.json episode "
                 "count, so the paired points are repeats, not a budget trend.",
                 ha="center", fontsize=9, color="#b3521f", wrap=True)
    out_png = os.path.join(args.out, "budget_cost.png")
    fig.savefig(out_png, dpi=150, facecolor=_SURFACE)
    print(f"[budget] figure -> {out_png}")


if __name__ == "__main__":
    main()
