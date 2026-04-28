"""Plot training diagnostics from a training log CSV.

Auto-detects the run type from the ``phase`` column:
  * ``pg_train`` → 4 panels (return, critic loss, entropy, actor loss).
    Warmup episodes (return==0 ∧ actor_loss==0) are filtered by default;
    pass ``--keep-warmup`` to include them.
  * ``dqn_train`` → 4 panels (loss vs epoch, ε vs iter, buffer rows vs iter,
    per-iter convergence).

Usage:
    python plot_training_diagnostics.py --csv runs/.../training_log.csv
    python plot_training_diagnostics.py --csv runs/.../training_log.csv --out my_plot.png --window 100
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _plot_pg(df: pd.DataFrame, args) -> None:
    if not args.keep_warmup:
        warmup_mask = (df["return"] == 0.0) & (df["actor_loss"] == 0.0)
        n_warmup = int(warmup_mask.sum())
        if n_warmup:
            print(f"Filtered {n_warmup} warmup episodes (return=0, actor_loss=0)")
        df = df[~warmup_mask].reset_index(drop=True)

    if df.empty:
        print("No PG training rows after filtering.")
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

    critic_valid = ~np.isnan(critic_loss_raw)
    has_critic = critic_valid.any()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    ax = axes[0, 0]
    ax.scatter(ep, r, s=4, alpha=0.25, color="tab:blue", label="per-episode")
    ax.plot(ep, roll_r, color="tab:red", lw=2, label=f"rolling mean (w={args.window})")
    ax.set_ylabel("return (= -overpotential)")
    ax.set_title("Reward vs episode")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

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
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.scatter(ep, entropy, s=4, alpha=0.25, color="tab:green", label="per-episode")
    ax.plot(ep, roll_h, color="tab:red", lw=2, label=f"rolling mean (w={args.window})")
    ax.axhline(np.log(28), color="gray", ls="--", lw=1,
               label="log(28) = 3.33 (uniform over cations)")
    ax.set_xlabel("episode"); ax.set_ylabel("entropy (nats)")
    ax.set_title("Policy entropy vs episode")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.scatter(ep, actor_loss, s=4, alpha=0.25, color="tab:purple", label="per-episode")
    ax.plot(ep, roll_al, color="tab:red", lw=2, label=f"rolling mean (w={args.window})")
    ax.set_xlabel("episode"); ax.set_ylabel("actor loss")
    ax.set_title("Actor loss vs episode")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")

    n = len(r)
    chunks = min(8, n)
    size = n // chunks if chunks else 0
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


def _plot_dqn(df: pd.DataFrame, args) -> None:
    if df.empty:
        print("No DQN rows.")
        return

    train_df = df[df["phase"] == "dqn_train"].reset_index(drop=True)
    collect_df = df[df["phase"] == "dqn_collect"].reset_index(drop=True)

    if train_df.empty:
        print("No 'dqn_train' rows; cannot plot loss panels.")
        return

    iteration = pd.to_numeric(train_df["iteration"], errors="coerce").fillna(0).astype(int).to_numpy()
    train_loss = pd.to_numeric(train_df["train_loss"], errors="coerce").to_numpy()
    val_loss = pd.to_numeric(train_df.get("val_loss", pd.Series([np.nan] * len(train_df))), errors="coerce").to_numpy()
    has_val = not np.all(np.isnan(val_loss))
    loss_name = (
        train_df["loss_name"].dropna().iloc[0]
        if "loss_name" in train_df.columns and train_df["loss_name"].notna().any()
        else "loss"
    )

    agg_kwargs = {
        "last_train_loss": ("_train", "last"),
        "last_val_loss": ("_val", "last"),
    }
    if "epsilon" in train_df.columns:
        agg_kwargs["eps"] = ("epsilon", "first")

    by_iter = (
        train_df.assign(_train=train_loss, _val=val_loss)
        .groupby("iteration", as_index=False)
        .agg(**agg_kwargs)
    )
    if "eps" not in by_iter.columns:
        by_iter["eps"] = np.nan
    iters = by_iter["iteration"].to_numpy()
    x = np.arange(len(train_df))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # --- Loss vs cumulative epoch ---
    ax = axes[0, 0]
    ax.plot(x, train_loss, label="train", color="tab:blue", lw=1, alpha=0.85)
    if has_val:
        ax.plot(x, val_loss, label="val", color="tab:red", lw=1, alpha=0.85)
    iter_starts = np.where(np.diff(np.concatenate([[-1], iteration])) != 0)[0]
    for s in iter_starts[1:]:
        ax.axvline(s - 0.5, color="gray", lw=0.3, alpha=0.4)
    ax.set_xlabel("cumulative epoch (across iterations)")
    ax.set_ylabel(loss_name)
    ax.set_title(f"Loss vs epoch ({len(iters)} iterations)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # --- Epsilon vs iteration ---
    ax = axes[0, 1]
    if "epsilon" in df.columns and by_iter["eps"].notna().any():
        ax.plot(iters, by_iter["eps"], color="tab:purple", lw=1.2, marker="o", ms=3)
        ax.set_ylabel("ε")
        ax.set_title("ε schedule")
    else:
        ax.text(0.5, 0.5, "No epsilon column", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        ax.set_title("ε schedule")
    ax.set_xlabel("iteration"); ax.grid(alpha=0.3)

    # --- Reward vs episode (replaces replay-buffer panel) ---
    ax = axes[1, 0]
    if not collect_df.empty:
        ep = pd.to_numeric(collect_df["episode"], errors="coerce").to_numpy()
        r = pd.to_numeric(collect_df["return"], errors="coerce").to_numpy()
        roll_r = pd.Series(r).rolling(args.window, min_periods=1).mean().to_numpy()
        ax.scatter(ep, r, s=4, alpha=0.25, color="tab:blue", label="per-episode")
        ax.plot(ep, roll_r, color="tab:red", lw=2, label=f"rolling mean (w={args.window})")
        ax.set_xlabel("episode"); ax.set_ylabel("terminal reward")
        ax.set_title(f"Reward vs episode ({len(ep)} episodes)")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No 'dqn_collect' rows\n(rerun with reward logging)",
                ha="center", va="center", transform=ax.transAxes, color="gray")
        ax.set_xlabel("episode"); ax.set_ylabel("terminal reward")
        ax.set_title("Reward vs episode")
    ax.grid(alpha=0.3)

    # --- Per-iteration final train/val loss ---
    ax = axes[1, 1]
    ax.plot(iters, by_iter["last_train_loss"], color="tab:blue", lw=1.2,
            marker="o", ms=3, label="train (last epoch)")
    if has_val and by_iter["last_val_loss"].notna().any():
        ax.plot(iters, by_iter["last_val_loss"], color="tab:red", lw=1.2,
                marker="o", ms=3, label="val (last epoch)")
    ax.set_xlabel("iteration"); ax.set_ylabel(loss_name)
    ax.set_title("Per-iteration convergence (last epoch)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")

    n = len(iters)
    chunks = min(8, n)
    size = n // chunks if chunks else 0
    if size > 0:
        print(f"\nBlock means ({chunks} blocks of ~{size} iterations):")
        for i in range(chunks):
            s, e = i * size, (i + 1) * size
            block = by_iter.iloc[s:e]
            eps_str = f"{block['eps'].mean():.3f}" if block["eps"].notna().any() else "  -  "
            print(
                f"  iter {iters[s]:4d}-{iters[e-1]:4d}: "
                f"mean train = {block['last_train_loss'].mean():8.4f}  "
                f"mean val = {block['last_val_loss'].mean():8.4f}  "
                f"ε = {eps_str}"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to training_log.csv")
    ap.add_argument("--out", default="training_diagnostics.png")
    ap.add_argument("--window", type=int, default=50, help="(PG) rolling mean window")
    ap.add_argument(
        "--keep-warmup",
        action="store_true",
        help="(PG only) Include warmup episodes (return==0) in plots",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "phase" not in df.columns:
        print("CSV is missing the 'phase' column.")
        return

    phases = sorted(df["phase"].dropna().unique().tolist())

    if "dqn_train" in phases or "dqn_collect" in phases:
        _plot_dqn(df[df["phase"].isin(["dqn_train", "dqn_collect"])].reset_index(drop=True), args)
    elif "pg_train" in phases:
        _plot_pg(df[df["phase"] == "pg_train"].reset_index(drop=True), args)
    else:
        print(f"No 'dqn_train' or 'pg_train' rows found. Phases present: {phases}")


if __name__ == "__main__":
    main()
