#!/usr/bin/env python3
"""Train a regressor to generate constrained ABCDEOOH-like compositions.

Offline workflow:
1) generate random episodes under the constraint mask
2) compute Monte-Carlo returns as Q targets
3) fit Q(s,a) regressor by supervised regression
4) generate new candidates by selecting actions maximizing predicted Q
"""

from __future__ import annotations

import os

# Mitigation for OpenMP runtime conflicts (common on macOS when numpy/sklearn
# and torch pull different OpenMP implementations). This is a best-effort guard
# to avoid segfaults; a clean conda env with consistent BLAS/OpenMP is better.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import csv
import json
import random
import sys
import warnings
from typing import List, Sequence, Tuple

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Matminer can emit a verbose warning like:
#   "PymatgenData(impute_nan=False): ..."
# which tends to spam logs and corrupt tqdm rendering on HPC.
warnings.filterwarnings(
    "ignore",
    message=r"^PymatgenData\(impute_nan=False\):.*",
    category=UserWarning,
    module=r"matminer\.utils\.data",
    append=False,
)

# Allow running without installing the package.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from abcde_ooh.env import ABCDEOOHEnv, DEFAULT_CATION_SET, DEFAULT_FRACTIONS  # noqa: E402
from abcde_ooh.constraints.primary_phase import check_primary_phase  # noqa: E402
from abcde_ooh.featurization import feature_calculators  # noqa: E402
from abcde_ooh.model import QRegressor  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def extract_mc_q_targets(episode, gamma: float):
    inputs = []
    q_targets: List[float] = []

    G = 0.0
    for step in reversed(episode):
        G = float(step.reward) + gamma * G
        q_targets.append(G)
        inputs.append(
            (
                np.asarray(step.state_material_features, dtype=float),
                np.asarray(step.state_step_onehot, dtype=float),
                np.asarray(step.action_elem_onehot, dtype=float),
                np.asarray(step.action_comp_onehot, dtype=float),
            )
        )

    inputs.reverse()
    q_targets.reverse()
    return inputs, q_targets


def train_q(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> None:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for _ in tqdm(range(epochs), desc="Q epochs"):
        for s_mat, s_step, a_elem, a_comp, y in loader:
            s_mat = s_mat.to(device)
            s_step = s_step.to(device)
            a_elem = a_elem.to(device)
            a_comp = a_comp.to(device)
            y = y.to(device)

            opt.zero_grad(set_to_none=True)
            pred = model(s_mat, s_step, a_elem, a_comp)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()


def choose_action(
    *,
    model: torch.nn.Module,
    device: torch.device,
    s_material: np.ndarray,
    s_step: np.ndarray,
    allowed_actions: Sequence[Tuple[Tuple[float, ...], Tuple[float, ...]]],
    stochastic_top_frac: float,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    if not allowed_actions:
        raise RuntimeError("No allowed actions.")

    a_elem = np.asarray([a[0] for a in allowed_actions], dtype=float)
    a_comp = np.asarray([a[1] for a in allowed_actions], dtype=float)

    s_mat_batch = np.repeat(s_material.reshape(1, -1), repeats=len(allowed_actions), axis=0)
    s_step_batch = np.repeat(s_step.reshape(1, -1), repeats=len(allowed_actions), axis=0)

    with torch.no_grad():
        q = model(
            torch.tensor(s_mat_batch, dtype=torch.float32, device=device),
            torch.tensor(s_step_batch, dtype=torch.float32, device=device),
            torch.tensor(a_elem, dtype=torch.float32, device=device),
            torch.tensor(a_comp, dtype=torch.float32, device=device),
        ).reshape(-1)

    order_t = torch.argsort(q, descending=True)
    order = order_t.detach().cpu().tolist()

    if stochastic_top_frac <= 0.0:
        return allowed_actions[int(order[0])]

    k = max(1, int(round(stochastic_top_frac * len(allowed_actions))))
    topk = order[:k]
    idx = int(np.random.choice(topk))
    return allowed_actions[idx]


def _rollout_random_episode(env: ABCDEOOHEnv) -> None:
    env.initialize()
    for _t in range(env.max_steps):
        a = env.sample_random_action()
        env.step(a)


def _rollout_policy_episode(
    *,
    env: ABCDEOOHEnv,
    qnet: torch.nn.Module,
    scaler: StandardScaler,
    device: torch.device,
    stochastic_top_frac: float,
) -> None:
    env.initialize()
    for _t in range(env.max_steps):
        allowed = env.allowed_actions()
        s_mat = env.state_featurizer(env.state)
        s_mat = scaler.transform(s_mat.reshape(1, -1))[0]

        step_onehot = np.zeros(env.max_steps, dtype=float)
        if env.counter < env.max_steps:
            step_onehot[env.counter] = 1.0

        a = choose_action(
            model=qnet,
            device=device,
            s_material=s_mat,
            s_step=step_onehot,
            allowed_actions=allowed,
            stochastic_top_frac=stochastic_top_frac,
        )
        env.step(a)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--num-random-eps", type=int, default=5000)
    parser.add_argument("--max-random-attempts", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=0.9)

    parser.add_argument("--dqn-epochs", dest="dqn_epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--num-gen-eps", type=int, default=500)
    parser.add_argument("--max-gen-attempts", type=int, default=None)
    parser.add_argument("--stochastic-top-frac", type=float, default=0.0)

    # Replay buffer construction
    parser.add_argument(
        "--buffer-mode",
        type=str,
        default="offline",
        choices=["offline", "iterative"],
        help="How to build the replay buffer: offline random only, or iterative (Option B) which adds on-the-fly episodes generated by the current Q network.",
    )
    parser.add_argument(
        "--num-online-eps",
        type=int,
        default=0,
        help="When --buffer-mode iterative, number of additional on-the-fly episodes to add to the buffer using the learned policy.",
    )

    parser.add_argument("--anion-formula", type=str, default="O2H1")

    parser.add_argument(
        "--reward-mode",
        type=str,
        default="none",
        choices=["none", "dp"],
    )

    parser.add_argument(
        "--primary-phase-filter",
        type=str,
        default="none",
        choices=["none", "buffer", "generated", "both"],
    )

    parser.add_argument(
        "--use-saved-random-dataset",
        action="store_true",
        help="Load random_dataset.npz from --out instead of regenerating.",
    )

    # DeepMD options
    parser.add_argument("--dp-poscar", type=str, default="POSCAR")
    parser.add_argument(
        "--dp-model",
        action="append",
        default=[],
        help="Path to a DeepMD .pt checkpoint. Repeat for ensemble.",
    )
    parser.add_argument("--dp-n-random-configs", type=int, default=10)
    parser.add_argument("--dp-ads-height", type=float, default=1.9)
    parser.add_argument("--dp-ads-dz", type=float, default=1.0)
    parser.add_argument(
        "--dp-objective",
        dest="dp_objective",
        default="mean_minus_kstd",
        choices=["mean_minus_kstd", "mean_plus_kstd"],
    )
    parser.add_argument("--dp-k", type=float, default=1.0)
    parser.add_argument(
        "--dp-uncertainty",
        dest="dp_uncertainty",
        default="models",
        choices=["models", "configs", "total"],
    )

    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dp_predictor = None
    dp_cache = {}
    if args.reward_mode == "dp":
        if not args.dp_model:
            raise SystemExit("--reward-mode dp requires at least one --dp-model")

        from abcde_ooh.dp_predictor import DPConfig, DeepMDOverpotentialPredictor, objective_from_mean_std

        cfg = DPConfig(
            base_poscar=args.dp_poscar,
            model_files=tuple(args.dp_model) if args.dp_model else (),
            n_random_configs=args.dp_n_random_configs,
            ads_height=args.dp_ads_height,
            ads_dz=args.dp_ads_dz,
            seed=args.seed,
        )
        dp_predictor = DeepMDOverpotentialPredictor(cfg)

    def reward_fn(formula: str) -> float:
        if args.reward_mode == "none":
            return 0.0
        if args.reward_mode == "dp":
            raise RuntimeError("DP reward is bound via env-aware closure")
        raise RuntimeError("Unknown reward mode")

    env = ABCDEOOHEnv(
        cation_set=DEFAULT_CATION_SET,
        fraction_set=DEFAULT_FRACTIONS,
        anion_formula=args.anion_formula,
        reward_fn=reward_fn,
    )

    current_phase = "random"  # random | generate

    if args.reward_mode == "dp":
        assert dp_predictor is not None
        from abcde_ooh.dp_predictor import objective_from_mean_std

        def dp_reward_fn(_terminal_formula: str) -> float:
            comp = env.terminal_cation_fractions()

            # Skip expensive DeepMD if this comp would be discarded.
            skip_for_constraints = False
            if args.primary_phase_filter in {"buffer", "both"}:
                skip_for_constraints = True
            elif current_phase == "generate" and args.primary_phase_filter in {"generated", "both"}:
                skip_for_constraints = True

            if skip_for_constraints:
                ok, label = check_primary_phase(comp)
                if not ok:
                    return 0.0

            key = tuple(sorted((k, float(v)) for k, v in comp.items()))
            if key in dp_cache:
                entry = dp_cache[key]
                mean = entry["mean"]
                std = entry["std"]
            else:
                pred = dp_predictor.predict_overpotential(comp, uncertainty=args.dp_uncertainty)
                mean, std = pred[0], pred[1]
                dp_cache[key] = {"mean": float(mean), "std": float(std)}

            obj = objective_from_mean_std(float(mean), float(std), mode=args.dp_objective, k=args.dp_k)
            return -float(obj)

        env.reward_fn = dp_reward_fn

    # 1) Build or load offline random dataset
    if args.use_saved_random_dataset:
        ds_path = os.path.join(args.out, "random_dataset.npz")
        if not os.path.exists(ds_path):
            raise SystemExit(f"{ds_path} not found")
        data = np.load(ds_path)
        s_mat = data["s_mat"]
        s_step = data["s_step"]
        a_elem = data["a_elem"]
        a_comp = data["a_comp"]
        y = data["y"]
    else:
        need_buffer_filter = args.primary_phase_filter in {"buffer", "both"}
        target_accepted = int(args.num_random_eps)
        max_attempts = int(args.max_random_attempts) if args.max_random_attempts is not None else None

        all_inputs = []
        all_targets = []

        accepted = 0
        attempted = 0
        rejected = 0

        current_phase = "random"

        pbar = tqdm(
            total=target_accepted,
            desc="Random episodes (accepted)",
        )

        while accepted < target_accepted and (max_attempts is None or attempted < max_attempts):
            attempted += 1
            _rollout_random_episode(env)

            comp = env.terminal_cation_fractions()
            if need_buffer_filter:
                ok, _label = check_primary_phase(comp)
                if not ok:
                    rejected += 1
                    if rejected <= 20 or rejected % 2000 == 0:
                        tqdm.write(
                            f"[REJECT] buffer filter: attempt={attempted} rejected={rejected} comp={comp}"
                        )
                    continue

            inputs, q_targets = extract_mc_q_targets(env.path, gamma=args.gamma)
            all_inputs.extend(inputs)
            all_targets.extend(q_targets)
            accepted += 1

            pbar.update(1)
            if attempted % 2000 == 0:
                rate = (accepted / attempted) if attempted else 0.0
                pbar.set_postfix(attempts=attempted, rejected=rejected, rate=f"{rate:.3f}")

        pbar.close()

        rate = (accepted / attempted) if attempted else 0.0
        print(
            f"[INFO] Random episodes: accepted {accepted}/{target_accepted} after {attempted} attempts (rate={rate:.4f}).",
            flush=True,
        )

        if max_attempts is not None and accepted < target_accepted:
            print(
                f"[WARN] Only accepted {accepted}/{target_accepted} random episodes after {attempted} attempts.",
                flush=True,
            )

        s_mat = np.asarray([x[0] for x in all_inputs], dtype=float)
        s_step = np.asarray([x[1] for x in all_inputs], dtype=float)
        a_elem = np.asarray([x[2] for x in all_inputs], dtype=float)
        a_comp = np.asarray([x[3] for x in all_inputs], dtype=float)
        y = np.asarray(all_targets, dtype=float).reshape(-1, 1)

        np.savez_compressed(
            os.path.join(args.out, "random_dataset.npz"),
            s_mat=s_mat,
            s_step=s_step,
            a_elem=a_elem,
            a_comp=a_comp,
            y=y,
        )

    # 2) Scale state features
    scaler = StandardScaler()
    s_mat_scaled = scaler.fit_transform(s_mat)
    joblib.dump(scaler, os.path.join(args.out, "std_scaler.bin"), compress=True)

    # 3) Train Q regressor
    ds = TensorDataset(
        torch.tensor(s_mat_scaled, dtype=torch.float32),
        torch.tensor(s_step, dtype=torch.float32),
        torch.tensor(a_elem, dtype=torch.float32),
        torch.tensor(a_comp, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    qnet = QRegressor(
        state_dim=s_mat_scaled.shape[1],
        step_dim=s_step.shape[1],
        elem_dim=a_elem.shape[1],
        frac_dim=a_comp.shape[1],
    ).to(device)

    train_q(model=qnet, loader=loader, device=device, epochs=args.dqn_epochs, lr=args.lr)

    # Option B (iterative): grow the buffer with on-the-fly episodes from the learned policy, then retrain.
    if args.buffer_mode == "iterative" and int(args.num_online_eps) > 0:
        need_buffer_filter = args.primary_phase_filter in {"buffer", "both"}
        target_online = int(args.num_online_eps)
        max_attempts = int(args.max_random_attempts) if args.max_random_attempts is not None else None

        accepted = 0
        attempted = 0
        rejected = 0

        current_phase = "random"  # treat as buffer phase for DP skip logic

        pbar = tqdm(
            total=target_online,
            desc="Online episodes (accepted)",
        )

        qnet.eval()
        all_inputs = []
        all_targets = []

        while accepted < target_online and (max_attempts is None or attempted < max_attempts):
            attempted += 1
            _rollout_policy_episode(
                env=env,
                qnet=qnet,
                scaler=scaler,
                device=device,
                stochastic_top_frac=args.stochastic_top_frac,
            )

            comp = env.terminal_cation_fractions()
            if need_buffer_filter:
                ok, _label = check_primary_phase(comp)
                if not ok:
                    rejected += 1
                    if rejected <= 20 or rejected % 2000 == 0:
                        tqdm.write(
                            f"[REJECT] buffer filter (online): attempt={attempted} rejected={rejected} comp={comp}"
                        )
                    continue

            inputs, q_targets = extract_mc_q_targets(env.path, gamma=args.gamma)
            all_inputs.extend(inputs)
            all_targets.extend(q_targets)
            accepted += 1

            pbar.update(1)
            if attempted % 2000 == 0:
                rate = (accepted / attempted) if attempted else 0.0
                pbar.set_postfix(attempts=attempted, rejected=rejected, rate=f"{rate:.3f}")

        pbar.close()

        if max_attempts is not None and accepted < target_online:
            print(
                f"[WARN] Only accepted {accepted}/{target_online} online episodes after {attempted} attempts.",
                flush=True,
            )

        if all_inputs:
            s_mat_new = np.asarray([x[0] for x in all_inputs], dtype=float)
            s_step_new = np.asarray([x[1] for x in all_inputs], dtype=float)
            a_elem_new = np.asarray([x[2] for x in all_inputs], dtype=float)
            a_comp_new = np.asarray([x[3] for x in all_inputs], dtype=float)
            y_new = np.asarray(all_targets, dtype=float).reshape(-1, 1)

            s_mat = np.concatenate([s_mat, s_mat_new], axis=0)
            s_step = np.concatenate([s_step, s_step_new], axis=0)
            a_elem = np.concatenate([a_elem, a_elem_new], axis=0)
            a_comp = np.concatenate([a_comp, a_comp_new], axis=0)
            y = np.concatenate([y, y_new], axis=0)

            # Keep the original scaler (fit on initial random buffer) for consistency.
            s_mat_scaled = scaler.transform(s_mat)

            ds = TensorDataset(
                torch.tensor(s_mat_scaled, dtype=torch.float32),
                torch.tensor(s_step, dtype=torch.float32),
                torch.tensor(a_elem, dtype=torch.float32),
                torch.tensor(a_comp, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32),
            )
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
            qnet.train()
            train_q(model=qnet, loader=loader, device=device, epochs=args.dqn_epochs, lr=args.lr)

            np.savez_compressed(
                os.path.join(args.out, "random_dataset.npz"),
                s_mat=s_mat,
                s_step=s_step,
                a_elem=a_elem,
                a_comp=a_comp,
                y=y,
            )

    torch.save(qnet.state_dict(), os.path.join(args.out, "qnet.pt"))

    # 4) Generate candidates
    qnet.eval()

    rows = []
    need_generated_filter = args.primary_phase_filter in {"generated", "both"}
    target_gen = int(args.num_gen_eps)
    max_gen_attempts = int(args.max_gen_attempts) if args.max_gen_attempts is not None else None

    current_phase = "generate"

    accepted = 0
    attempted = 0
    rejected = 0

    pbar = tqdm(
        total=target_gen,
        desc="Generate (accepted)",
    )

    while accepted < target_gen and (max_gen_attempts is None or attempted < max_gen_attempts):
        attempted += 1
        env.initialize()

        for _t in range(env.max_steps):
            allowed = env.allowed_actions()
            s_mat = env.state_featurizer(env.state)
            s_mat = scaler.transform(s_mat.reshape(1, -1))[0]

            step_onehot = np.zeros(env.max_steps, dtype=float)
            if env.counter < env.max_steps:
                step_onehot[env.counter] = 1.0

            a = choose_action(
                model=qnet,
                device=device,
                s_material=s_mat,
                s_step=step_onehot,
                allowed_actions=allowed,
                stochastic_top_frac=args.stochastic_top_frac,
            )
            env.step(a)

        comp = env.terminal_cation_fractions()
        ok, label = check_primary_phase(comp)
        if need_generated_filter and not ok:
            rejected += 1
            if rejected <= 20 or rejected % 2000 == 0:
                tqdm.write(
                    f"[REJECT] generated filter: attempt={attempted} rejected={rejected} comp={comp}"
                )
            continue

        reward = float(env.path[-1].reward) if env.path else 0.0

        row = {
            "formula": env.terminal_formula,
            "primary_phase": label or "none",
            "reward": reward,
        }

        if args.reward_mode == "dp":
            from abcde_ooh.dp_predictor import objective_from_mean_std

            key = tuple(sorted((k, float(v)) for k, v in comp.items()))
            if key in dp_cache:
                entry = dp_cache[key]
                mean = float(entry["mean"])
                std = float(entry["std"])
            else:
                assert dp_predictor is not None
                pred = dp_predictor.predict_overpotential(comp, uncertainty=args.dp_uncertainty)
                mean, std = float(pred[0]), float(pred[1])
                dp_cache[key] = {"mean": mean, "std": std}

            # Keep "mean/std/mean-std" style columns for readability.
            row.update(
                {
                    "dp_mean": mean,
                    "dp_std": std,
                    "dp_mean_minus_std": mean - std,
                    "dp_mean_minus_kstd": mean - float(args.dp_k) * std,
                    "dp_mean_plus_kstd": mean + float(args.dp_k) * std,
                    "dp_objective": float(
                        objective_from_mean_std(mean, std, mode=args.dp_objective, k=float(args.dp_k))
                    ),
                }
            )

        row.update({k: float(v) for k, v in sorted(comp.items())})

        rows.append(row)
        accepted += 1

        pbar.update(1)
        if attempted % 2000 == 0:
            rate = (accepted / attempted) if attempted else 0.0
            pbar.set_postfix(attempts=attempted, rejected=rejected, rate=f"{rate:.3f}")

    pbar.close()

    rate = (accepted / attempted) if attempted else 0.0
    print(
        f"[INFO] Generated candidates: accepted {accepted}/{target_gen} after {attempted} attempts (rate={rate:.4f}).",
        flush=True,
    )

    if max_gen_attempts is not None and accepted < target_gen:
        print(f"[WARN] Only accepted {accepted}/{target_gen} generated candidates after {attempted} attempts.")

    out_csv = os.path.join(args.out, "generated.csv")
    preferred = ["formula", "primary_phase", "reward"]
    if args.reward_mode == "dp":
        preferred += [
            "dp_mean",
            "dp_std",
            "dp_mean_minus_std",
            "dp_mean_minus_kstd",
            "dp_mean_plus_kstd",
            "dp_objective",
        ]

    # Put known columns first, then everything else (e.g., element fractions).
    all_keys = []
    for r in rows:
        for k in r.keys():
            if k not in all_keys:
                all_keys.append(k)
    keys = [k for k in preferred if k in all_keys] + [k for k in all_keys if k not in preferred]

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with open(os.path.join(args.out, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
