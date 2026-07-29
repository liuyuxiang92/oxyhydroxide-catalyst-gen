#!/usr/bin/env python3
"""Driver: build the full method x budget figure set for a sinter/calcine sweep.

Walks a directory of runs named ``<scenario>_<method>_eps_<N>_temp1`` and emits

* per (scenario, budget):  ``compare_methods.py --minimize``  — reward
  distribution and lowest-temperature candidate for DQN(bootstrap) / DQN(MC) / A2C
* per scenario:            ``compare_budget_cost.py --minimize`` — wall-clock and
  best temperature as the training budget grows

Both scenarios' objectives are minimised temperatures (``direction: min`` in the
scenario YAML, so ``reward = -T``); every figure is therefore drawn on the
positive temperature axis with *lower is better*.

    python scripts/baselines/make_compare_time_figures.py \\
        --root runs/compare_time_extract/compare_time --out runs/compare/time

Runs missing ``timing.json`` / ``generated.csv`` are reported and skipped — an
interrupted arm never becomes a silent gap in a line.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from typing import Dict, List, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))

# Method token in the directory name -> figure label. ``mc`` is the DQN
# Monte-Carlo-target *ablation*, not a fourth algorithm; the labels say so.
_METHODS = [("dqn", "DQN(bootstrap)"), ("mc", "DQN(MC)"), ("a2c", "A2C")]

# Slots 1-3 of the validated categorical palette, pinned per method for the
# budget-sweep figures. The 45k panels are missing DQN(bootstrap); without
# pinning, DQN(MC) would inherit slot 1 there and silently change colour between
# figures of the same set. (compare_methods.py doesn't colour by method — its
# distribution panel is single-hue and its bars are keyed by the x axis.)
_METHOD_COLORS = {"DQN(bootstrap)": "#2a78d6", "DQN(MC)": "#eb6834", "A2C": "#1baf7a"}
_COLOR_ARGS = sum((["--series-color", f"{k}:{v}"] for k, v in _METHOD_COLORS.items()), [])

# Scenario token -> (pretty name, y-axis quantity).
_SCENARIOS = {
    "sinter": ("Sinter", "sintering temperature"),
    "calcine": ("Calcine", "calcination temperature"),
    "sinter_calcine": ("Sinter+Calcine", "T(sinter) + T(calcine)"),
}

_DIR_RE = re.compile(r"^(?P<scenario>.+)_(?P<method>dqn|mc|a2c)_eps_(?P<eps>\d+)_temp\d+$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="Directory holding the run dirs")
    p.add_argument("--out", required=True, help="Output directory for the figures")
    p.add_argument("--top-k", type=int, default=10, help="K for the top-K summary")
    return p.parse_args()


def _discover(root: str) -> Dict[Tuple[str, str, int], str]:
    """``(scenario, method_token, eps) -> path``."""
    found: Dict[Tuple[str, str, int], str] = {}
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        m = _DIR_RE.match(name)
        if not m:
            print(f"[driver] skipping unrecognised directory name: {name}")
            continue
        scen = m.group("scenario")
        if scen not in _SCENARIOS:
            print(f"[driver] skipping unknown scenario {scen!r} in {name}")
            continue
        found[(scen, m.group("method"), int(m.group("eps")))] = path
    return found


def _complete(path: str) -> bool:
    return all(os.path.exists(os.path.join(path, f))
               for f in ("run_config.json", "timing.json", "generated.csv"))


def _run(cmd: List[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    found = _discover(args.root)
    if not found:
        raise SystemExit(f"[driver] no run directories found under {args.root}")

    incomplete = sorted(p for p in found.values() if not _complete(p))
    if incomplete:
        print("[driver] INCOMPLETE runs (excluded from every figure):")
        for p in incomplete:
            print(f"    {os.path.basename(p)}")

    scenarios = sorted({k[0] for k in found}, key=lambda s: list(_SCENARIOS).index(s))
    budgets = sorted({k[2] for k in found})

    for scen in scenarios:
        pretty, quantity = _SCENARIOS[scen]

        # --- per-budget method comparison ---------------------------------
        for eps in budgets:
            runs: List[str] = []
            for tok, label in _METHODS:
                path = found.get((scen, tok, eps))
                if path and _complete(path):
                    runs.append(f"{label}:{path}")
            if len(runs) < 2:
                print(f"[driver] {pretty} @ {eps} eps: only {len(runs)} complete "
                      f"method(s); skipping the comparison figure")
                continue
            _run([sys.executable, os.path.join(_HERE, "compare_methods.py"),
                  *sum(([f"--run", r] for r in runs), []),
                  "--minimize",
                  "--y-label", f"{quantity} (lower is better)",
                  "--top-k", str(args.top_k),
                  "--title", f"{pretty} — {eps} training episodes "
                             f"({len(runs)}/3 methods complete)",
                  "--out", os.path.join(args.out, f"{scen}_eps{eps}")])

            # --- per-episode training reward cloud (same output dir) -------
            # Needs training_log.csv, which _complete() doesn't require; a run can
            # have finished generation without it if the log write failed.
            with_log = [r for r in runs
                        if os.path.exists(os.path.join(r.split(":", 1)[1],
                                                       "training_log.csv"))]
            if len(with_log) >= 2:
                _run([sys.executable,
                      os.path.join(_HERE, "compare_training_rewards.py"),
                      *sum(([f"--run", r] for r in with_log), []),
                      "--minimize",
                      "--y-label", f"{quantity} (lower is better)",
                      "--title", f"{pretty} — training-episode rewards, "
                                 f"{eps} episode budget",
                      "--out", os.path.join(args.out, f"{scen}_eps{eps}")])
            else:
                print(f"[driver] {pretty} @ {eps} eps: only {len(with_log)} run(s) "
                      f"have training_log.csv; skipping the training-reward figure")

        # --- budget sweep --------------------------------------------------
        sweep: List[str] = []
        for tok, label in _METHODS:
            for eps in budgets:
                path = found.get((scen, tok, eps))
                if path and _complete(path):
                    sweep.append(f"{label}:{path}")
        if sweep:
            _run([sys.executable, os.path.join(_HERE, "compare_budget_cost.py"),
                  *sum(([f"--run", r] for r in sweep), []),
                  *_COLOR_ARGS, "--minimize", "--hours",
                  "--y-label", quantity,
                  "--title", f"{pretty} — cost and quality vs sampling budget",
                  "--out", os.path.join(args.out, f"{scen}_budget")])

    print(f"\n[driver] figures under {args.out}")


if __name__ == "__main__":
    main()
