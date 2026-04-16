"""Plot training diagnostics from a PG training log CSV.

Handles the warmup phase (return==0 episodes before DeepMD predictions
are available) by filtering them out before plotting so the y-axis scale
is not collapsed.

Usage:
    python plot_training_diagnostics.py --csv runs_bo/.../training_log.csv
    python plot_training_diagnostics.py --csv runs_bo/.../training_log.csv --out my_plot.png --window 100
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to training_log.csv")
    ap.add_argument("--out", default="training_diagnostics.png")
    ap.add_argument("--window", type=int, default=50, help="rolling mean window")
    ap.add_argument(
        "--keep-warmup",
        action="store_true",
        help="Include warmup episodes (return==0) in plots",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["phase"] == "pg_train"].reset_index(drop=True)

    if not args.keep_warmup:
        # Drop warmup episodes where the DP predictor hasn't returned a reward
        # yet (return==0 and actor_loss==0). These would compress the y-axis
        # and hide the actual learning signal.
        warmup_mask = (df["return"] == 0.0) & (df["actor_loss"] == 0.0)
        n_warmup = warmup_mask.sum()
        if n_warmup:
            print(f"Filtered {n_warmup} warmup episodes (return=0, actor_loss=0)")
        df = df[~warmup_mask].reset_index(drop=True)

    if df.empty:
        print("No training rows found after filtering. Check --csv path and phase column.")
        return

    ep = df["episode"].to_numpy()
    r = df["return"].to_numpy()
    entropy = df["entropy"].to_numpy()
    actor_loss = df["actor_loss"].to_numpy()

    critic_loss_raw = pd.to_numeric(df["critic_loss"], errors="coerce").to_numpy()

    roll = lambda x: pd.Series(x).rolling(args.window, min_periods=1).mean().to_numpy()
    roll_r = roll(r)
    roll_h = roll(entropy)
    roll_al = roll(actor_loss)

    # Critic loss rolling mean — skip NaN entries (rows without a critic)
    critic_valid = ~np.isnan(critic_loss_raw)
    has_critic = critic_valid.any()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    # --- Reward vs episode ---
    ax = axes[0, 0]
    ax.scatter(ep, r, s=4, alpha=0.25, color="tab:blue", label="per-episode")
    ax.plot(ep, roll_r, color="tab:red", lw=2, label=f"rolling mean (w={args.window})")
    ax.set_ylabel("return (= -overpotential)")
    ax.set_title("Reward vs episode")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- Critic loss vs episode ---
    ax = axes[0, 1]
    if has_critic:
        ep_c = ep[critic_valid]
        cl = critic_loss_raw[critic_valid]
        roll_cl = pd.Series(critic_loss_raw).rolling(args.window, min_periods=1).mean().to_numpy()
        ax.scatter(ep_c, cl, s=4, alpha=0.25, color="tab:orange", label="per-episode")
        ax.plot(ep[critic_valid], roll_cl[critic_valid], color="tab:red", lw=2,
                label=f"rolling mean (w={args.window})")
    else:
        ax.text(0.5, 0.5, "No critic loss (REINFORCE)", ha="center", va="center",
                transform=ax.transAxes, color="gray")
    ax.set_ylabel("critic loss")
    ax.set_title("Critic loss vs episode")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- Policy entropy vs episode ---
    ax = axes[1, 0]
    ax.scatter(ep, entropy, s=4, alpha=0.25, color="tab:green", label="per-episode")
    ax.plot(ep, roll_h, color="tab:red", lw=2, label=f"rolling mean (w={args.window})")
    ax.axhline(np.log(28), color="gray", ls="--", lw=1,
               label="log(28) = 3.33 (uniform over cations)")
    ax.set_xlabel("episode")
    ax.set_ylabel("entropy (nats)")
    ax.set_title("Policy entropy vs episode")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- Actor loss vs episode ---
    ax = axes[1, 1]
    ax.scatter(ep, actor_loss, s=4, alpha=0.25, color="tab:purple", label="per-episode")
    ax.plot(ep, roll_al, color="tab:red", lw=2, label=f"rolling mean (w={args.window})")
    ax.set_xlabel("episode")
    ax.set_ylabel("actor loss")
    ax.set_title("Actor loss vs episode")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")

    # --- Block diagnostics ---
    n = len(r)
    chunks = min(8, n)
    size = n // chunks
    if size > 0:
        print(f"\nBlock means ({chunks} blocks of ~{size} episodes):")
        for i in range(chunks):
            s, e = i * size, (i + 1) * size
            print(
                f"  ep {ep[s]:4.0f}-{ep[e-1]:4.0f}: "
                f"mean return = {r[s:e].mean():8.3f}  "
                f"best return = {r[s:e].max():8.3f}  "
                f"entropy = {entropy[s:e].mean():5.3f}  "
                f"|actor_loss| = {np.abs(actor_loss[s:e]).mean():9.2f}"
            )


if __name__ == "__main__":
    main()
