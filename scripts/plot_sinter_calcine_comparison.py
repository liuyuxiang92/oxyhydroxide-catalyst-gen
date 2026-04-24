#!/usr/bin/env python3
"""Compare sinter/calcine temperature ranges across RL methods.

Reads the ``generated.csv`` file from each run directory produced by
``scripts/run_experiment.py`` and plots a multi-panel figure that
mirrors the reference figure style:
    - Violin / box plot of temperature distribution per run.
    - Bar chart of validity (SMACT charge-neutrality, if smact is installed).
    - Bar chart of diversity (unique_formulas / total).

Recovers raw temperature as ``T = -dp_mean`` because the predictor
returns ``-T`` so that "higher reward = lower T" in our framework.

Usage
-----
    python scripts/plot_sinter_calcine_comparison.py \\
        --run runs/oxides_sinter_dqn_s0:DQN-sinter \\
        --run runs/oxides_sinter_a2c_s0:A2C-sinter \\
        --run runs/oxides_calcine_dqn_s0:DQN-calcine \\
        --run runs/oxides_calcine_a2c_s0:A2C-calcine \\
        --out figures/sinter_calcine_comparison.png
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RunResult:
    label: str
    path: str
    temperatures: List[float] = field(default_factory=list)
    formulas: List[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.temperatures)

    @property
    def diversity(self) -> float:
        if not self.formulas:
            return 0.0
        return len(set(self.formulas)) / len(self.formulas)

    @property
    def validity(self) -> float:
        """Fraction charge-neutral (via SMACT); 1.0 if smact is unavailable."""
        try:
            import smact
            from smact.screening import pauling_test
        except ImportError:
            return float("nan")

        from pymatgen.core.composition import Composition

        ok = 0
        for f in self.formulas:
            try:
                comp = Composition(f)
                els = [str(e) for e in comp.elements]
                species = [smact.Element(e) for e in els]
                ox_states = [s.oxidation_states for s in species]
                counts = [comp[e] for e in els]
                valid = False
                for combo in _product(ox_states):
                    if abs(sum(c * o for c, o in zip(counts, combo))) < 1e-6:
                        valid = True
                        break
                if valid:
                    ok += 1
            except Exception:
                continue
        return ok / len(self.formulas) if self.formulas else 0.0


def _product(lists):
    import itertools
    return itertools.product(*lists)


def load_run(spec: str) -> RunResult:
    if ":" in spec:
        path, label = spec.split(":", 1)
    else:
        path = spec
        label = os.path.basename(path.rstrip("/"))

    csv_path = os.path.join(path, "generated.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No generated.csv in {path}")

    temps: List[float] = []
    formulas: List[str] = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("phase") and row["phase"] != "generate":
                continue
            try:
                dp_mean = float(row["dp_mean"])
            except (KeyError, ValueError):
                continue
            # Skip invalid penalty rows (we return -2000 for unparseable).
            if dp_mean <= -1999.0:
                continue
            temps.append(-dp_mean)
            formulas.append(row.get("formula", ""))

    return RunResult(label=label, path=path, temperatures=temps, formulas=formulas)


def plot(runs: List[RunResult], out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={"width_ratios": [2, 1, 1]})
    labels = [r.label for r in runs]
    data = [r.temperatures for r in runs]

    # --- Temperature distribution (violin + box) ---
    ax = axes[0]
    parts = ax.violinplot(data, showmeans=False, showmedians=True, widths=0.8)
    for pc in parts["bodies"]:
        pc.set_alpha(0.5)
    ax.boxplot(data, positions=range(1, len(data) + 1), widths=0.2,
               patch_artist=True, boxprops=dict(facecolor="white", alpha=0.7),
               medianprops=dict(color="red"))
    ax.set_xticks(range(1, len(data) + 1))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("Sinter / calcine temperature distribution (lower is better)")
    ax.grid(axis="y", alpha=0.3)

    # --- Validity ---
    ax = axes[1]
    vals = [r.validity for r in runs]
    bars = ax.bar(range(len(runs)), vals, color="tab:green", alpha=0.75)
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Validity (charge-neutral)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Validity (SMACT)")
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, vals):
        if v == v:  # not NaN
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                    f"{v:.2f}", ha="center", fontsize=9)

    # --- Diversity ---
    ax = axes[2]
    divs = [r.diversity for r in runs]
    bars = ax.bar(range(len(runs)), divs, color="tab:blue", alpha=0.75)
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Diversity (unique / total)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Diversity")
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, divs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                f"{v:.2f}", ha="center", fontsize=9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"[INFO] Saved figure to {out_path}")

    # Print a text summary.
    print("\n=== Summary ===")
    print(f"{'label':<20}  {'n':>5}  {'T_mean':>9}  {'T_std':>9}  {'T_min':>9}"
          f"  {'valid':>6}  {'diverse':>7}")
    for r in runs:
        if not r.temperatures:
            continue
        arr = np.asarray(r.temperatures)
        print(f"{r.label:<20}  {r.n:>5}  {arr.mean():>9.1f}  {arr.std():>9.1f}"
              f"  {arr.min():>9.1f}  {r.validity:>6.2f}  {r.diversity:>7.2f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Plot sinter/calcine comparison across runs.")
    p.add_argument("--run", action="append", required=True,
                   help="Run spec in the form 'path:label' (repeatable).")
    p.add_argument("--out", default="figures/sinter_calcine_comparison.png",
                   help="Output figure path.")
    args = p.parse_args()

    runs = [load_run(spec) for spec in args.run]
    if not runs:
        raise SystemExit("No runs to plot.")

    plot(runs, args.out)


if __name__ == "__main__":
    main()
