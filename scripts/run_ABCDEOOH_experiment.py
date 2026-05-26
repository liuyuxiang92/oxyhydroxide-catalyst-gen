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
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import collections
import copy
import csv
import json
import math
import random
import sys
import warnings
from collections import Counter
from typing import Callable, List, Optional, Sequence, Tuple

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
from abcde_ooh.model import PolicyNet, QRegressor, ValueNet  # noqa: E402


def _comp_key(comp: dict) -> tuple:
    """Canonical hashable key for a terminal cation composition dict.

    Quantizes each fraction to integer units of 1/20 (5%), drops zero-units,
    and sorts. Used both for `dp_cache` lookups and for visit-count tracking
    in the repeat-penalty shaping (`train_pg`).
    """
    items = []
    for el, frac in comp.items():
        units = int(round(float(frac) * 20))
        if units > 0:
            items.append((str(el), units))
    return tuple(sorted(items))


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)


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


def _save_checkpoint(path: str, data: dict, numbered_path: Optional[str] = None) -> None:
    """Atomically write a checkpoint dict.

    If numbered_path is given, the data is written there and path is updated as
    a relative symlink pointing to the new file (so checkpoint.pt always resolves
    to the most-recently-saved numbered checkpoint).
    """
    target = numbered_path if numbered_path else path
    tmp = target + ".tmp"
    torch.save(data, tmp)
    os.replace(tmp, target)
    if numbered_path:
        link_tmp = path + ".lnk"
        if os.path.lexists(link_tmp):
            os.remove(link_tmp)
        os.symlink(os.path.basename(numbered_path), link_tmp)
        os.replace(link_tmp, path)


def _make_loss_fn(name: str) -> torch.nn.Module:
    name = (name or "smoothl1").lower()
    if name == "mse":
        return torch.nn.MSELoss()
    if name in ("smoothl1", "huber"):
        return torch.nn.SmoothL1Loss()
    raise ValueError(f"Unknown DQN loss: {name!r} (expected 'mse' or 'smoothl1')")


def _anneal_epsilon(eps: float, eps_min: float, decay: float) -> float:
    return max(float(eps_min), float(eps) * float(decay))


def _fifo_cap_arrays(arrays: dict, max_rows: int) -> dict:
    """Keep only the last `max_rows` rows of every array in `arrays`."""
    if max_rows is None or max_rows <= 0:
        return arrays
    n = len(next(iter(arrays.values())))
    if n <= max_rows:
        return arrays
    start = n - max_rows
    return {k: v[start:] for k, v in arrays.items()}


def _split_train_val(n_rows: int, val_frac: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Random index split. Returns (train_idx, val_idx)."""
    val_frac = float(val_frac)
    if val_frac <= 0.0 or n_rows < 2:
        idx = np.arange(n_rows)
        return idx, np.array([], dtype=int)
    n_val = max(1, int(round(n_rows * val_frac)))
    n_val = min(n_val, n_rows - 1)
    perm = rng.permutation(n_rows)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx, val_idx


def _build_train_val_loaders(
    s_mat_scaled: np.ndarray,
    s_step: np.ndarray,
    a_elem: np.ndarray,
    a_comp: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    val_frac: float,
    rng: np.random.Generator,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Random 1-(val_frac) / val_frac split into train/val DataLoaders."""
    n_rows = int(s_mat_scaled.shape[0])
    train_idx, val_idx = _split_train_val(n_rows, val_frac, rng)

    def _make_loader(idx: np.ndarray, shuffle: bool) -> Optional[DataLoader]:
        if idx.size == 0:
            return None
        ds = TensorDataset(
            torch.tensor(s_mat_scaled[idx], dtype=torch.float32),
            torch.tensor(s_step[idx], dtype=torch.float32),
            torch.tensor(a_elem[idx], dtype=torch.float32),
            torch.tensor(a_comp[idx], dtype=torch.float32),
            torch.tensor(y[idx], dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    return _make_loader(train_idx, shuffle=True), _make_loader(val_idx, shuffle=False)


def _eval_q_loss(model: torch.nn.Module, loader: DataLoader, loss_fn: torch.nn.Module, device: torch.device) -> float:
    if loader is None:
        return float("nan")
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for s_mat, s_step, a_elem, a_comp, y in loader:
            s_mat = s_mat.to(device)
            s_step = s_step.to(device)
            a_elem = a_elem.to(device)
            a_comp = a_comp.to(device)
            y = y.to(device)
            pred = model(s_mat, s_step, a_elem, a_comp)
            losses.append(float(loss_fn(pred, y).item()))
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def train_q(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    iteration: int = 0,
    checkpoint_cfg: Optional[dict] = None,
    loss_name: str = "smoothl1",
    val_loader: Optional[DataLoader] = None,
) -> List[dict]:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = _make_loss_fn(loss_name)
    metrics: List[dict] = []

    ckpt_path: Optional[str] = None
    ckpt_freq: int = 0
    start_epoch: int = 0
    if checkpoint_cfg:
        ckpt_path = checkpoint_cfg.get("path")
        ckpt_freq = int(checkpoint_cfg.get("freq", 0))
        start_epoch = int(checkpoint_cfg.get("start_epoch", 0))
        if checkpoint_cfg.get("opt_state"):
            opt.load_state_dict(checkpoint_cfg["opt_state"])

    pbar = tqdm(range(start_epoch, epochs), desc="Q epochs")
    for epoch_idx in pbar:
        batch_losses: List[float] = []
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
            batch_losses.append(float(loss.item()))

        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        val_loss = _eval_q_loss(model, val_loader, loss_fn, device) if val_loader is not None else float("nan")

        metrics.append({
            "phase": "dqn_train",
            "iteration": iteration,
            "epoch": epoch_idx + 1,
            "loss_name": loss_name,
            "train_loss": train_loss,
            "val_loss": val_loss,
            # Back-compat alias so existing plotting scripts that read 'mse_loss' don't break.
            "mse_loss": train_loss,
        })
        if val_loader is not None:
            pbar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}")
            tqdm.write(
                f"[DQN train] iter={iteration} epoch={epoch_idx + 1}/{epochs} | "
                f"{loss_name}_train={train_loss:.4f} | {loss_name}_val={val_loss:.4f}"
            )
        else:
            pbar.set_postfix(train=f"{train_loss:.4f}")
            tqdm.write(f"[DQN train] iter={iteration} epoch={epoch_idx + 1}/{epochs} | {loss_name}_train={train_loss:.4f}")

        if ckpt_freq > 0 and ckpt_path and (epoch_idx + 1) % ckpt_freq == 0:
            _numbered = ckpt_path.replace(".pt", f"-{epoch_idx + 1}.pt")
            _save_checkpoint(ckpt_path, {
                "type": "dqn",
                "rl_method": "dqn",
                "epochs_completed": epoch_idx + 1,
                "dqn_iteration": iteration,
                "qnet_state": model.state_dict(),
                "opt_state": opt.state_dict(),
            }, numbered_path=_numbered)
            tqdm.write(f"[INFO] Checkpoint saved at epoch {epoch_idx + 1} → {_numbered}")

    return metrics


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


# ── Classical DQN helpers ──────────────────────────────────────────────────


def _precompute_elem_features(
    cation_set: List[str],
    featurizer: Callable,
) -> Tuple[np.ndarray, "StandardScaler"]:
    """Precompute and scale element Magpie feature vectors for all cations.

    Uses a separate scaler from s_mat because single-element Magpie
    statistics have zero range features that differ from composites.
    """
    raw = np.asarray([featurizer(el + "1.00") for el in cation_set], dtype=float)
    elem_scaler = StandardScaler()
    elem_scaler.fit(raw)
    return elem_scaler.transform(raw), elem_scaler


def add_episode_to_buffer(
    path: list,
    buffer: collections.deque,
    elem_feats_scaled: np.ndarray,
    fraction_set: List[str],
) -> None:
    """Convert an episode path into 5 buffer rows and append them to buffer."""
    for k, step in enumerate(path):
        done = k == len(path) - 1
        next_step = path[k + 1] if not done else None

        next_allowed_idx: List[Tuple[int, int]] = []
        if not done:
            for (elem_oh, comp_oh) in next_step.allowed_actions:
                next_allowed_idx.append(
                    (int(np.argmax(elem_oh)), int(np.argmax(comp_oh)))
                )

        buffer.append({
            "s_mat_raw":        np.asarray(step.state_material_features, dtype=float),
            "s_step":           np.asarray(step.state_step_onehot, dtype=float),
            "a_elem_idx":       int(np.argmax(step.action_elem_onehot)),
            "a_comp_val":       float(fraction_set[int(np.argmax(step.action_comp_onehot))]),
            "reward":           float(step.reward),
            "s_mat_next_raw":   np.asarray(next_step.state_material_features, dtype=float)
                                if not done else np.zeros(len(step.state_material_features), dtype=float),
            "s_step_next":      np.asarray(next_step.state_step_onehot, dtype=float)
                                if not done else np.zeros(len(step.state_step_onehot), dtype=float),
            "next_allowed_idx": next_allowed_idx,
            "done":             done,
        })


def _choose_action_dqn(
    qnet: torch.nn.Module,
    s_mat_scaled: np.ndarray,
    s_step: np.ndarray,
    allowed: list,
    elem_feats_scaled: np.ndarray,
    fraction_set: List[str],
    device: torch.device,
    *,
    gen_epsilon: float = 0.0,
    gen_top_frac: float = 0.0,
    gen_temperature: float = 0.0,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Return action via Q-network. Priority: epsilon-greedy > Boltzmann > top-k > greedy."""
    n = len(allowed)

    if gen_epsilon > 0.0 and np.random.random() < gen_epsilon:
        return allowed[int(np.random.randint(n))]

    a_elem_b = np.asarray(
        [elem_feats_scaled[int(np.argmax(a[0]))] for a in allowed], dtype=float
    )
    a_comp_b = np.asarray(
        [[float(fraction_set[int(np.argmax(a[1]))])] for a in allowed], dtype=float
    )
    s_mat_b = np.repeat(s_mat_scaled.reshape(1, -1), n, axis=0)
    s_step_b = np.repeat(s_step.reshape(1, -1), n, axis=0)

    with torch.no_grad():
        q = qnet(
            torch.tensor(s_mat_b, dtype=torch.float32, device=device),
            torch.tensor(s_step_b, dtype=torch.float32, device=device),
            torch.tensor(a_elem_b, dtype=torch.float32, device=device),
            torch.tensor(a_comp_b, dtype=torch.float32, device=device),
        ).reshape(-1)

    if gen_temperature > 0.0:
        q_np = q.cpu().numpy().astype(float) / gen_temperature
        q_np -= q_np.max()
        probs = np.exp(q_np)
        probs /= probs.sum()
        return allowed[int(np.random.choice(n, p=probs))]

    if gen_top_frac > 0.0:
        k = max(1, int(round(gen_top_frac * n)))
        topk = torch.argsort(q, descending=True).cpu().tolist()[:k]
        return allowed[int(np.random.choice(topk))]

    raise ValueError("No generation strategy active: set --gen-temperature, --gen-top-frac, or --gen-epsilon.")


def _compute_td_target(
    row: dict,
    target_net: torch.nn.Module,
    s_mat_scaler: StandardScaler,
    elem_feats_scaled: np.ndarray,
    fraction_set: List[str],
    device: torch.device,
    gamma: float,
) -> float:
    """TD target for one buffer row: r if done, else r + γ·max_a' Q_target(s',a')."""
    if row["done"] or not row["next_allowed_idx"]:
        return float(row["reward"])

    next_idxs = row["next_allowed_idx"]
    n = len(next_idxs)
    ns_mat = s_mat_scaler.transform(row["s_mat_next_raw"].reshape(1, -1))[0]
    a_e = np.asarray([elem_feats_scaled[ei] for ei, _ in next_idxs], dtype=float)
    a_c = np.asarray([[float(fraction_set[fi])] for _, fi in next_idxs], dtype=float)
    ns_mat_b = np.repeat(ns_mat.reshape(1, -1), n, axis=0)
    ns_step_b = np.repeat(row["s_step_next"].reshape(1, -1), n, axis=0)

    with torch.no_grad():
        q_next = target_net(
            torch.tensor(ns_mat_b, dtype=torch.float32, device=device),
            torch.tensor(ns_step_b, dtype=torch.float32, device=device),
            torch.tensor(a_e, dtype=torch.float32, device=device),
            torch.tensor(a_c, dtype=torch.float32, device=device),
        ).reshape(-1)

    return float(row["reward"]) + gamma * float(q_next.max().item())


def _dqn_gradient_step(
    qnet: torch.nn.Module,
    batch: list,
    target_net: torch.nn.Module,
    s_mat_scaler: StandardScaler,
    elem_feats_scaled: np.ndarray,
    fraction_set: List[str],
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    gamma: float,
) -> float:
    """One DQN gradient step: compute TD targets, forward, SmoothL1, backward."""
    y = [
        _compute_td_target(row, target_net, s_mat_scaler, elem_feats_scaled, fraction_set, device, gamma)
        for row in batch
    ]

    s_mat_b = torch.tensor(
        s_mat_scaler.transform(np.asarray([r["s_mat_raw"] for r in batch], dtype=float)),
        dtype=torch.float32, device=device,
    )
    s_step_b = torch.tensor(
        np.asarray([r["s_step"] for r in batch], dtype=float),
        dtype=torch.float32, device=device,
    )
    a_elem_b = torch.tensor(
        np.asarray([elem_feats_scaled[r["a_elem_idx"]] for r in batch], dtype=float),
        dtype=torch.float32, device=device,
    )
    a_comp_b = torch.tensor(
        np.asarray([[r["a_comp_val"]] for r in batch], dtype=float),
        dtype=torch.float32, device=device,
    )
    y_t = torch.tensor(y, dtype=torch.float32, device=device).reshape(-1, 1)

    qnet.train()
    pred = qnet(s_mat_b, s_step_b, a_elem_b, a_comp_b)
    loss = loss_fn(pred, y_t)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def _rollout_policy_episode(
    *,
    env: ABCDEOOHEnv,
    qnet: torch.nn.Module,
    scaler: StandardScaler,
    device: torch.device,
    stochastic_top_frac: float,
    online_epsilon: float,
) -> None:
    env.initialize()
    for _t in range(env.max_steps):
        allowed = env.allowed_actions()
        s_mat = env.state_featurizer(env.state)
        s_mat = scaler.transform(s_mat.reshape(1, -1))[0]

        step_onehot = np.zeros(env.max_steps, dtype=float)
        if env.counter < env.max_steps:
            step_onehot[env.counter] = 1.0

        # Online collection policy:
        # - If online_epsilon > 0: epsilon-greedy (random with prob epsilon, otherwise greedy-best Q)
        # - Else: greedy-best Q (classic DQN online collection)
        if online_epsilon > 0.0 and float(np.random.rand()) < float(online_epsilon):
            a = random.choice(allowed)
        else:
            a = choose_action(
                model=qnet,
                device=device,
                s_material=s_mat,
                s_step=step_onehot,
                allowed_actions=allowed,
                stochastic_top_frac=(0.0 if online_epsilon > 0.0 else stochastic_top_frac),
            )
        env.step(a)


def _fit_scaler_from_warmup(env: ABCDEOOHEnv, n_warmup_eps: int) -> StandardScaler:
    """Roll out random episodes and fit a StandardScaler on material features.

    Warmup only needs partial-state composition features, so we temporarily
    swap env.reward_fn for a no-op to avoid paying DeepMD (and, under
    --geo-opt, LBFGS) cost on rewards that are immediately discarded.
    """
    all_s_mat = []
    accepted = 0
    pbar = tqdm(total=n_warmup_eps, desc="PG warmup (scaler fit)")
    original_reward_fn = env.reward_fn
    env.reward_fn = lambda _f: 0.0
    try:
        while accepted < n_warmup_eps:
            _rollout_random_episode(env)
            for step in env.path:
                all_s_mat.append(np.asarray(step.state_material_features, dtype=float))
            accepted += 1
            pbar.update(1)
    finally:
        env.reward_fn = original_reward_fn
        pbar.close()
    print(f"[INFO] PG warmup: accepted {accepted}/{n_warmup_eps}.")
    scaler = StandardScaler()
    scaler.fit(np.asarray(all_s_mat, dtype=float))
    return scaler


def _rollout_pg_episode(
    *,
    env: ABCDEOOHEnv,
    policy: torch.nn.Module,
    scaler: StandardScaler,
    elem_feats_scaled: np.ndarray,
    fraction_set,
    device: torch.device,
) -> None:
    """Roll out one episode using the policy network (softmax sampling)."""
    env.initialize()
    for _t in range(env.max_steps):
        allowed = env.allowed_actions()
        s_mat = env.state_featurizer(env.state)
        s_mat_scaled = scaler.transform(s_mat.reshape(1, -1))[0]

        step_onehot = np.zeros(env.max_steps, dtype=float)
        if env.counter < env.max_steps:
            step_onehot[env.counter] = 1.0

        n = len(allowed)
        a_elem_batch = np.asarray(
            [elem_feats_scaled[int(np.argmax(a[0]))] for a in allowed], dtype=float
        )
        a_comp_batch = np.asarray(
            [[float(fraction_set[int(np.argmax(a[1]))])] for a in allowed], dtype=float
        )
        s_mat_batch = np.repeat(s_mat_scaled.reshape(1, -1), n, axis=0)
        s_step_batch = np.repeat(step_onehot.reshape(1, -1), n, axis=0)
        with torch.no_grad():
            logits = policy(
                torch.tensor(s_mat_batch, dtype=torch.float32, device=device),
                torch.tensor(s_step_batch, dtype=torch.float32, device=device),
                torch.tensor(a_elem_batch, dtype=torch.float32, device=device),
                torch.tensor(a_comp_batch, dtype=torch.float32, device=device),
            ).reshape(-1)
        probs = torch.softmax(logits, dim=0).cpu().numpy()
        idx = int(np.random.choice(n, p=probs))
        a = allowed[idx]

        env.step(a)


def _episode_pg_terms(
    *,
    path,
    returns: List[float],
    returns_shaped: List[float],
    policy: torch.nn.Module,
    value_net,
    scaler: StandardScaler,
    elem_feats_scaled: np.ndarray,
    fraction_set,
    device: torch.device,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """Build per-step actor/entropy/critic loss tensors for one episode.

    Tensors carry autograd state so they can be stacked with terms from other
    episodes in the same batch and backproped together.
    """
    actor_losses: List[torch.Tensor] = []
    entropy_terms: List[torch.Tensor] = []
    critic_losses: List[torch.Tensor] = []

    for step, G_t, G_t_shaped in zip(path, returns, returns_shaped):
        allowed = step.allowed_actions
        if not allowed:
            continue

        s_mat_raw = np.asarray(step.state_material_features, dtype=float)
        s_mat = scaler.transform(s_mat_raw.reshape(1, -1))[0]
        s_step = np.asarray(step.state_step_onehot, dtype=float)

        n = len(allowed)
        a_elem_batch = np.asarray(
            [elem_feats_scaled[int(np.argmax(a[0]))] for a in allowed], dtype=float
        )
        a_comp_batch = np.asarray(
            [[float(fraction_set[int(np.argmax(a[1]))])] for a in allowed], dtype=float
        )
        s_mat_batch = np.repeat(s_mat.reshape(1, -1), n, axis=0)
        s_step_batch = np.repeat(s_step.reshape(1, -1), n, axis=0)

        s_mat_t = torch.tensor(s_mat_batch, dtype=torch.float32, device=device)
        s_step_t = torch.tensor(s_step_batch, dtype=torch.float32, device=device)
        a_elem_t = torch.tensor(a_elem_batch, dtype=torch.float32, device=device)
        a_comp_t = torch.tensor(a_comp_batch, dtype=torch.float32, device=device)

        logits = policy(s_mat_t, s_step_t, a_elem_t, a_comp_t).reshape(-1)
        log_probs = torch.log_softmax(logits, dim=0)
        probs = torch.softmax(logits, dim=0)

        taken_elem = np.asarray(step.action_elem_onehot)
        taken_comp = np.asarray(step.action_comp_onehot)
        taken_idx = None
        for i, a in enumerate(allowed):
            if np.array_equal(np.asarray(a[0]), taken_elem) and np.array_equal(np.asarray(a[1]), taken_comp):
                taken_idx = i
                break
        if taken_idx is None:
            continue

        G_raw_t = torch.tensor(G_t, dtype=torch.float32, device=device)
        G_shaped_t = torch.tensor(G_t_shaped, dtype=torch.float32, device=device)
        if value_net is not None:
            s_mat_single = torch.tensor(s_mat.reshape(1, -1), dtype=torch.float32, device=device)
            s_step_single = torch.tensor(s_step.reshape(1, -1), dtype=torch.float32, device=device)
            value = value_net(s_mat_single, s_step_single).reshape(-1)[0]
            advantage = G_shaped_t - value.detach()
            critic_losses.append((value - G_raw_t) ** 2)
        else:
            advantage = G_shaped_t

        actor_losses.append(-log_probs[taken_idx] * advantage)
        entropy_terms.append(-(probs * log_probs).sum())

    return actor_losses, entropy_terms, critic_losses


def train_pg(
    *,
    policy: torch.nn.Module,
    value_net,
    env: ABCDEOOHEnv,
    scaler: StandardScaler,
    elem_feats_scaled: np.ndarray,
    fraction_set,
    device: torch.device,
    num_iters: int,
    batch_eps: int,
    gamma: float,
    lr_actor: float,
    lr_critic: float,
    entropy_coef: float,
    rl_method: str,
    repeat_penalty_coef: float = 0.0,
    repeat_penalty_shape: str = "log",
    max_train_attempts=None,
    checkpoint_cfg: Optional[dict] = None,
    checkpoint_iter_freq: int = 0,
) -> List[dict]:
    """Batched PG training loop.

    Structure:
        for it in range(num_iters):
            collect `batch_eps` episodes
            accumulate per-step actor/entropy/critic terms across all of them
            one optimizer step over the entire batch
            (optionally checkpoint)

    Repeat-penalty shaping (`repeat_penalty_coef > 0`) and entropy bonus carry
    over from the previous implementation; only the unit of update changed
    from "1 episode" to "batch of `batch_eps` episodes".
    """
    policy.train()
    opt_actor = torch.optim.Adam(policy.parameters(), lr=lr_actor)
    if value_net is not None:
        value_net.train()
        opt_critic = torch.optim.Adam(value_net.parameters(), lr=lr_critic)
    else:
        opt_critic = None

    ckpt_path: Optional[str] = None
    ckpt_freq_eps: int = 0  # legacy per-episode checkpoint frequency (still honoured)
    start_episode: int = 0
    if checkpoint_cfg:
        ckpt_path = checkpoint_cfg.get("path")
        ckpt_freq_eps = int(checkpoint_cfg.get("freq", 0))
        start_episode = int(checkpoint_cfg.get("start_episode", 0))
        if checkpoint_cfg.get("opt_actor_state"):
            opt_actor.load_state_dict(checkpoint_cfg["opt_actor_state"])
        if opt_critic is not None and checkpoint_cfg.get("opt_critic_state"):
            opt_critic.load_state_dict(checkpoint_cfg["opt_critic_state"])
        if checkpoint_cfg.get("visit_counts"):
            visit_counts: Counter = Counter(checkpoint_cfg["visit_counts"])
        else:
            visit_counts = Counter()
    else:
        visit_counts = Counter()

    metrics: List[dict] = []
    accepted = start_episode  # episodes consumed in total
    attempted = 0
    update_idx = 0  # outer*inner counter for diagnostics

    total_updates = max(1, int(num_iters))
    print(
        f"[INFO] PG: {num_iters} iters × "
        f"{batch_eps} eps/update = {total_updates} updates, {total_updates * batch_eps} total episodes."
    )

    outer_pbar = tqdm(range(int(num_iters)), desc=f"{rl_method.upper()} iters")
    for it in outer_pbar:
        # Collect a batch of `batch_eps` episodes, all rolled out under the *current* policy.
        batch_paths: List[list] = []
        batch_returns: List[List[float]] = []
        batch_returns_shaped: List[List[float]] = []
        batch_terminal_keys: List[tuple] = []
        batch_repeat_penalties: List[float] = []
        batch_visits_before: List[int] = []

        collected = 0
        while collected < int(batch_eps):
            attempted += 1
            if max_train_attempts is not None and attempted > max_train_attempts:
                break
            _rollout_pg_episode(
                env=env,
                policy=policy,
                scaler=scaler,
                elem_feats_scaled=elem_feats_scaled,
                fraction_set=fraction_set,
                device=device,
            )
            path = env.path
            if not path:
                continue

            # MC returns backwards
            G = 0.0
            returns: List[float] = []
            for step in reversed(path):
                G = float(step.reward) + gamma * G
                returns.append(G)
            returns.reverse()

            terminal_key = _comp_key(env.terminal_cation_fractions())
            n_visits_before = visit_counts[terminal_key]
            if repeat_penalty_coef > 0.0:
                if repeat_penalty_shape == "log":
                    repeat_penalty = repeat_penalty_coef * math.log1p(n_visits_before)
                elif repeat_penalty_shape == "sqrt":
                    repeat_penalty = repeat_penalty_coef * math.sqrt(n_visits_before)
                else:
                    repeat_penalty = repeat_penalty_coef * float(n_visits_before)
            else:
                repeat_penalty = 0.0
            visit_counts[terminal_key] += 1
            returns_shaped = [G_t - repeat_penalty for G_t in returns]

            batch_paths.append(path)
            batch_returns.append(returns)
            batch_returns_shaped.append(returns_shaped)
            batch_terminal_keys.append(terminal_key)
            batch_repeat_penalties.append(repeat_penalty)
            batch_visits_before.append(n_visits_before)
            collected += 1
            accepted += 1

        if not batch_paths:
            continue

        # Build losses for the entire batch in one autograd graph.
        all_actor: List[torch.Tensor] = []
        all_entropy: List[torch.Tensor] = []
        all_critic: List[torch.Tensor] = []
        for path, returns, returns_shaped in zip(batch_paths, batch_returns, batch_returns_shaped):
            a_terms, e_terms, c_terms = _episode_pg_terms(
                path=path,
                returns=returns,
                returns_shaped=returns_shaped,
                policy=policy,
                value_net=value_net,
                scaler=scaler,
                elem_feats_scaled=elem_feats_scaled,
                fraction_set=fraction_set,
                device=device,
            )
            all_actor.extend(a_terms)
            all_entropy.extend(e_terms)
            all_critic.extend(c_terms)

        if not all_actor:
            continue

        actor_loss = torch.stack(all_actor).mean()
        entropy_bonus = torch.stack(all_entropy).mean()
        total_actor_loss = actor_loss - entropy_coef * entropy_bonus

        opt_actor.zero_grad(set_to_none=True)
        total_actor_loss.backward()
        opt_actor.step()

        critic_loss_val: float | str = ""
        if opt_critic is not None and all_critic:
            critic_loss = torch.stack(all_critic).mean()
            opt_critic.zero_grad(set_to_none=True)
            critic_loss.backward()
            opt_critic.step()
            critic_loss_val = float(critic_loss.item())

        update_idx += 1
        ep_actor_loss = float(actor_loss.item())
        ep_entropy = float(entropy_bonus.item())
        mean_return_raw = float(np.mean([rs[0] for rs in batch_returns])) if batch_returns else 0.0
        mean_return_shaped = (
            float(np.mean([rs[0] for rs in batch_returns_shaped])) if batch_returns_shaped else 0.0
        )

        metrics.append({
            "phase": "pg_train",
            "iteration": it + 1,
            "update": update_idx,
            "episode": accepted,
            "batch_eps": int(batch_eps),
            "return": mean_return_raw,
            "return_raw": mean_return_raw,
            "return_shaped": mean_return_shaped,
            "repeat_penalty": float(np.mean(batch_repeat_penalties)) if batch_repeat_penalties else 0.0,
            "visit_count_before": int(np.max(batch_visits_before)) if batch_visits_before else 0,
            "unique_comps_seen": len(visit_counts),
            "max_visit_count": max(visit_counts.values()) if visit_counts else 0,
            "terminal_comp_key": ";".join(f"{el}:{u}" for el, u in batch_terminal_keys[-1]) if batch_terminal_keys else "",
            "actor_loss": ep_actor_loss,
            "entropy": ep_entropy,
            "critic_loss": critic_loss_val,
        })

        outer_pbar.set_postfix(
            ret=f"{mean_return_raw:.3f}",
            actor=f"{ep_actor_loss:.3f}",
            ent=f"{ep_entropy:.3f}",
            upd=update_idx,
        )

        # Legacy per-episode checkpoint (--save-checkpoint-freq) honoured here too.
        if ckpt_freq_eps > 0 and ckpt_path and accepted > 0 and accepted % ckpt_freq_eps == 0:
            _numbered = ckpt_path.replace(".pt", f"-{accepted}.pt")
            _save_checkpoint(ckpt_path, {
                "type": "pg",
                "rl_method": rl_method,
                "episodes_completed": accepted,
                "iteration": it + 1,
                "policy_state": policy.state_dict(),
                "value_net_state": value_net.state_dict() if value_net is not None else None,
                "opt_actor_state": opt_actor.state_dict(),
                "opt_critic_state": opt_critic.state_dict() if opt_critic is not None else None,
                "visit_counts": dict(visit_counts),
            }, numbered_path=_numbered)
            tqdm.write(f"[INFO] PG checkpoint saved at episode {accepted} → {_numbered}")

        # Outer-iter checkpoint (--pg-checkpoint-iter-freq).
        if checkpoint_iter_freq > 0 and ckpt_path and (it + 1) % checkpoint_iter_freq == 0:
            _numbered = ckpt_path.replace(".pt", f"-iter{it + 1}.pt")
            _save_checkpoint(ckpt_path, {
                "type": "pg",
                "rl_method": rl_method,
                "episodes_completed": accepted,
                "iteration": it + 1,
                "policy_state": policy.state_dict(),
                "value_net_state": value_net.state_dict() if value_net is not None else None,
                "opt_actor_state": opt_actor.state_dict(),
                "opt_critic_state": opt_critic.state_dict() if opt_critic is not None else None,
                "visit_counts": dict(visit_counts),
            }, numbered_path=_numbered)
            tqdm.write(f"[INFO] PG iter-checkpoint saved at iter {it + 1} → {_numbered}")

        if max_train_attempts is not None and attempted >= max_train_attempts:
            tqdm.write(f"[WARN] PG: max_train_attempts={max_train_attempts} reached at iter {it + 1}.")
            break

    outer_pbar.close()
    print(f"[INFO] PG training: {update_idx} updates over {accepted} episodes, {attempted} attempts.")
    return metrics


def generate_pg(
    *,
    policy: torch.nn.Module,
    env: ABCDEOOHEnv,
    scaler: StandardScaler,
    elem_feats_scaled: np.ndarray,
    fraction_set,
    device: torch.device,
    n_eps: int,
    gen_epsilon: float = 0.0,
    gen_top_frac: float = 0.0,
    gen_temperature: float = 0.0,
    dp_predictor,
    dp_cache: dict,
    max_gen_attempts,
) -> List[dict]:
    """Generate candidate compositions using a trained PolicyNet."""
    policy.eval()
    rows: List[dict] = []
    target_gen = n_eps
    accepted = 0
    attempted = 0
    dup_rejected = 0
    seen_comp_keys: set = set()

    pbar = tqdm(total=target_gen, desc="PG generate (accepted)")
    while accepted < target_gen and (max_gen_attempts is None or attempted < max_gen_attempts):
        attempted += 1
        env.initialize()
        for _t in range(env.max_steps):
            allowed = env.allowed_actions()
            s_mat = env.state_featurizer(env.state)
            s_mat_scaled = scaler.transform(s_mat.reshape(1, -1))[0]

            step_onehot = np.zeros(env.max_steps, dtype=float)
            if env.counter < env.max_steps:
                step_onehot[env.counter] = 1.0

            n = len(allowed)
            a_elem_batch = np.asarray(
                [elem_feats_scaled[int(np.argmax(a[0]))] for a in allowed], dtype=float
            )
            a_comp_batch = np.asarray(
                [[float(fraction_set[int(np.argmax(a[1]))])] for a in allowed], dtype=float
            )
            s_mat_batch = np.repeat(s_mat_scaled.reshape(1, -1), n, axis=0)
            s_step_batch = np.repeat(step_onehot.reshape(1, -1), n, axis=0)
            with torch.no_grad():
                logits = policy(
                    torch.tensor(s_mat_batch, dtype=torch.float32, device=device),
                    torch.tensor(s_step_batch, dtype=torch.float32, device=device),
                    torch.tensor(a_elem_batch, dtype=torch.float32, device=device),
                    torch.tensor(a_comp_batch, dtype=torch.float32, device=device),
                ).reshape(-1)

            if gen_epsilon > 0.0 and np.random.random() < gen_epsilon:
                idx = int(np.random.randint(n))
            elif gen_temperature > 0.0:
                lg = logits.cpu().numpy().astype(float) / gen_temperature
                lg -= lg.max()
                probs = np.exp(lg)
                probs /= probs.sum()
                idx = int(np.random.choice(n, p=probs))
            elif gen_top_frac > 0.0:
                k = max(1, int(round(gen_top_frac * n)))
                topk = torch.argsort(logits, descending=True).cpu().tolist()[:k]
                idx = int(np.random.choice(topk))
            else:
                raise ValueError("No generation strategy active: set --gen-temperature, --gen-top-frac, or --gen-epsilon.")

            env.step(allowed[idx])

        comp = env.terminal_cation_fractions()
        comp_key = tuple(
            sorted(
                (str(k), int(round(float(v) * 20)))
                for k, v in comp.items()
                if int(round(float(v) * 20)) > 0
            )
        )
        ok, label = check_primary_phase(comp)
        if comp_key in seen_comp_keys:
            dup_rejected += 1
            if dup_rejected <= 20 or dup_rejected % 500 == 0:
                tqdm.write(f"[REJECT] PG duplicate generated: attempt={attempted} dup_rejected={dup_rejected}")
            continue
        seen_comp_keys.add(comp_key)

        reward = float(env.path[-1].reward) if env.path else 0.0
        key = comp_key
        if key in dp_cache:
            entry = dp_cache[key]
            mean = float(entry["mean"])
            std = float(entry["std"])
        else:
            pred = dp_predictor.predict_overpotential(comp)
            mean, std = float(pred[0]), float(pred[1])
            dp_cache[key] = {"mean": mean, "std": std}
        dp_mean = mean
        dp_std = std
        dp_mean_minus_std = mean - std

        row = {
            "formula": env.terminal_formula,
            "reward": reward,
            "dp_mean": dp_mean,
            "dp_std": dp_std,
            "dp_mean_minus_std": dp_mean_minus_std,
            "primary_ok": bool(ok),
            "primary_label": label or "",
        }
        rows.append(row)
        accepted += 1
        pbar.update(1)
        rate = (accepted / attempted) if attempted else 0.0
        pbar.set_postfix(attempts=attempted, rate=f"{rate:.3f}", dups=dup_rejected)

    pbar.close()

    rate = (accepted / attempted) if attempted else 0.0
    print(
        f"[INFO] PG generated: accepted {accepted}/{target_gen} after {attempted} attempts "
        f"(rate={rate:.4f}, dup_rejected={dup_rejected}).",
        flush=True,
    )
    return rows


_DEFAULT_DP_MODELS = [
    "model_1.ckpt.pt",
    "model_2.ckpt.pt",
    "model_3.ckpt.pt",
    "model_4.ckpt.pt",
    "model_5.ckpt.pt",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--dp-seed",
        type=int,
        default=0,
        help=(
            "Seed for the DeepMD predictor (controls random alloy configurations and "
            "adsorbate placement). Also acts as fallback for --train-seed and "
            "--gen-seed if those are not set. Must be the same across runs for "
            "fully reproducible training."
        ),
    )
    parser.add_argument(
        "--train-seed",
        type=int,
        default=None,
        help=(
            "Seed for the training phase (DQN buffer collection / iterative loop, or "
            "PG warmup + training RNG). When set, also enables GPU deterministic mode "
            "(cudnn.deterministic, use_deterministic_algorithms). Falls back to "
            "--dp-seed if not provided. Fix this together with --dp-seed for "
            "reproducible training."
        ),
    )
    parser.add_argument(
        "--gen-seed",
        type=int,
        default=None,
        help=(
            "Seed applied just before the generation phase. Makes stochastic generation "
            "(--gen-temperature, --gen-top-frac, --gen-epsilon) reproducible "
            "independently of training. Falls back to --dp-seed if not provided."
        ),
    )

    parser.add_argument(
        "--only-generate",
        action="store_true",
        help=(
            "Skip buffer building + Q training; load a saved scaler and qnet and only generate candidates. "
            "Defaults to loading <out>/std_scaler.bin and <out>/qnet.pt."
        ),
    )
    parser.add_argument(
        "--load-qnet",
        type=str,
        default=None,
        help="Path to saved qnet.pt state_dict (default: <out>/qnet.pt).",
    )
    parser.add_argument(
        "--load-scaler",
        type=str,
        default=None,
        help="Path to saved std_scaler.bin (default: <out>/std_scaler.bin).",
    )
    parser.add_argument(
        "--load-policy",
        type=str,
        default=None,
        help="Path to saved policy.pt state_dict (default: <out>/policy.pt). Used with --only-generate or --resume-training for PG methods.",
    )
    parser.add_argument(
        "--load-value-net",
        type=str,
        default=None,
        help="Path to saved value_net.pt state_dict (default: <out>/value_net.pt). Used with --only-generate or --resume-training for A2C.",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help=(
            "Run training but skip candidate generation. "
            "Combine with --resume-training to extend training without generating."
        ),
    )
    parser.add_argument(
        "--resume-training",
        action="store_true",
        help=(
            "Load an existing model checkpoint and continue training. "
            "For PG: skips warmup, loads std_scaler.bin + policy.pt (+ value_net.pt for A2C) "
            "and runs --pg-train-eps more episodes. "
            "For DQN: loads std_scaler.bin + random_dataset.npz + qnet.pt and runs --dqn-epochs more epochs. "
            "All paths default to <out>/<filename> and can be overridden by "
            "--load-scaler / --load-policy / --load-qnet / --load-value-net. "
            "Cannot be combined with --only-generate."
        ),
    )

    parser.add_argument(
        "--save-checkpoint-freq",
        type=int,
        default=0,
        metavar="N",
        dest="save_checkpoint_freq",
        help=(
            "Save a mid-training checkpoint every N episodes (PG) or N epochs (DQN). "
            "0 disables. Saved atomically to <out>/checkpoint.pt. "
            "When --resume-training is used and checkpoint.pt exists, it is loaded "
            "instead of policy.pt/qnet.pt so training resumes from the exact interruption point."
        ),
    )

    parser.add_argument("--num-random-eps", type=int, default=5000)
    parser.add_argument("--max-random-attempts", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=0.9)

    parser.add_argument("--dqn-epochs", dest="dqn_epochs", type=int, default=50)
    parser.add_argument("--dqn-batch-size", type=int, default=256)
    parser.add_argument("--dqn-lr", type=float, default=1e-3)

    parser.add_argument("--num-gen-eps", type=int, default=500)
    parser.add_argument("--max-gen-attempts", type=int, default=None)
    parser.add_argument(
        "--gen-epsilon",
        type=float,
        default=0.0,
        help=(
            "Candidate generation: epsilon-greedy policy for all RL methods. "
            "With prob ε choose a random allowed action, otherwise use the best action. "
            "Takes priority over --gen-temperature and --gen-top-frac."
        ),
    )
    parser.add_argument(
        "--gen-temperature",
        type=float,
        default=1.0,
        help=(
            "Candidate generation: Boltzmann sampling temperature for all RL methods. "
            "Converts Q-values/logits to probabilities via softmax(x / τ). "
            "Ignored if --gen-epsilon > 0. Typical range: 0.5–3.0. Default: 1.0."
        ),
    )
    parser.add_argument(
        "--gen-top-frac",
        type=float,
        default=0.0,
        help=(
            "Candidate generation: uniformly sample from top-k%% of actions by Q/logit. "
            "0 = disabled. Ignored if --gen-epsilon or --gen-temperature is set. "
            "Example: 0.3 = top 30%%."
        ),
    )

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

    # Iterative buffer schedule (optional).
    parser.add_argument(
        "--iter-num-iters",
        type=int,
        default=1,
        help=(
            "When --buffer-mode iterative: how many rounds of (collect episodes -> retrain) to run. "
            "Default 1 matches the current behavior."
        ),
    )
    parser.add_argument(
        "--iter-online-eps-per-iter",
        type=int,
        default=0,
        help=(
            "When --buffer-mode iterative: how many accepted online episodes to collect per iteration. "
            "If 0, it is derived from --num-online-eps / --iter-num-iters."
        ),
    )
    parser.add_argument(
        "--iter-train-epochs",
        type=int,
        default=None,
        help=(
            "Training epochs per Phase-1 iteration after adding new episodes. "
            "If unset, defaults to 100 (the paper-aligned value)."
        ),
    )

    # Paper-aligned DQN Phase-1 loop (replaces / extends the older iterative-buffer flags).
    parser.add_argument(
        "--dqn-num-iters",
        type=int,
        default=500,
        help="DQN Phase-1 iterations (collect → buffer-cap → sample → train → ε-decay). 0 disables Phase-1.",
    )
    parser.add_argument(
        "--dqn-eps-per-iter",
        type=int,
        default=100,
        help="Episodes collected with ε-greedy policy per Phase-1 iteration.",
    )
    parser.add_argument(
        "--replay-buffer-size",
        type=int,
        default=50000,
        help="Maximum number of step-level rows kept in the replay buffer (FIFO eviction).",
    )
    parser.add_argument(
        "--train-sample-size",
        type=int,
        default=100,
        help="Number of step-level rows uniformly sampled from the replay buffer per Phase-1 iteration.",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.2,
        help="Train/val split fraction applied to the per-iteration sample (and to Phase-0 pre-training).",
    )
    parser.add_argument(
        "--eps-start",
        type=float,
        default=1.0,
        help="Initial ε for Phase-1 ε-greedy collection.",
    )
    parser.add_argument(
        "--eps-min",
        type=float,
        default=0.05,
        help="Floor for the annealed Phase-1 ε.",
    )
    parser.add_argument(
        "--eps-decay",
        type=float,
        default=0.99,
        help="Multiplicative per-iteration decay: ε ← max(eps_min, ε × decay).",
    )
    parser.add_argument(
        "--checkpoint-iter-freq",
        type=int,
        default=10,
        help="Save a numbered DQN checkpoint every N Phase-1 iterations (0 disables).",
    )
    parser.add_argument(
        "--dqn-loss",
        type=str,
        choices=["mse", "smoothl1"],
        default="smoothl1",
        help="Loss for DQN training. Defaults to SmoothL1 (Huber).",
    )

    # Classical DQN flags (replace the old Phase-0/Phase-1 flags when --rl-method dqn).
    # Old flags are kept below for backward compat but are ignored in classical DQN mode.
    parser.add_argument(
        "--dqn-warmup-eps",
        type=int,
        default=500,
        help="Random episodes to collect before training starts; used to fit the s_mat scaler.",
    )
    parser.add_argument(
        "--num-train-eps",
        type=int,
        default=20000,
        help="Total training episodes for classical DQN.",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=50000,
        help="Replay buffer capacity (FIFO). Replaces --replay-buffer-size for classical DQN.",
    )
    parser.add_argument(
        "--target-update-freq",
        type=int,
        default=100,
        help="Hard-copy qnet → target_net every N training episodes.",
    )
    parser.add_argument(
        "--eps-anneal-eps",
        type=int,
        default=10000,
        help="Linear epsilon annealing: ε reaches eps-min after this many training episodes.",
    )
    parser.add_argument(
        "--grad-steps-per-ep",
        type=int,
        default=5,
        help="Gradient steps per training episode (1:1 ratio with env steps in a 5-step episode).",
    )

    parser.add_argument("--anion-formula", type=str, default="O2H1")

    parser.add_argument(
        "--target-phase",
        type=str,
        nargs="+",
        default=["none"],
        choices=["none", "Ni", "Co", "NiFe", "CoFe", "NiFeCo", "any"],
        metavar="PHASE",
        help=(
            "Constrain action space at every step so all generated compositions satisfy "
            "the target phase(s). Eliminates post-generation filtering; 100%% acceptance "
            "rate. Multiple phases may be passed (e.g. --target-phase Ni Co); a "
            "composition is accepted if it satisfies *any* of the listed phases. Use "
            "'none' (the default) to disable. 'any' allows any of the five valid phase "
            "types."
        ),
    )

    parser.add_argument(
        "--exclude-elements",
        type=str,
        nargs="+",
        default=[],
        metavar="EL",
        help=(
            "Elements to remove from the candidate cation set for both training and "
            "generation (e.g. --exclude-elements Fe). Useful with --target-phase Ni Co "
            "to enforce Ni- or Co-majority compositions that contain no Fe at all. "
            "Errors out if an excluded element is required as a primary by any target "
            "phase (e.g. excluding Fe with NiFe/CoFe/NiFeCo, or excluding Ni/Co with "
            "the corresponding single-primary phase)."
        ),
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
        default=None,
        help=(
            "Path to a DeepMD .pt checkpoint. Repeat for ensemble. "
            "Defaults to model_1.ckpt.pt … model_5.ckpt.pt if not specified."
        ),
    )
    parser.add_argument("--dp-n-random-configs", type=int, default=10)
    parser.add_argument("--dp-ads-height", type=float, default=1.9)
    parser.add_argument("--dp-ads-dz", type=float, default=1.0)

    # Geometry optimization
    parser.add_argument(
        "--geo-opt",
        action="store_true",
        help="Geometry-optimize each structure (LBFGS, fmax=0.001, max 1000 steps) before DP property evaluation.",
    )
    parser.add_argument(
        "--geo-opt-model",
        type=str,
        default="./DPA-3.1-3M_1.pt",
        help="Path to DP model .pt for geometry optimization (default: ./DPA-3.1-3M_1.pt).",
    )

    # Policy gradient options
    parser.add_argument(
        "--rl-method",
        type=str,
        default="dqn",
        choices=["dqn", "reinforce", "a2c"],
        help="RL algorithm: 'dqn' uses the existing offline Q-learning path; 'reinforce'/'a2c' use online policy gradient.",
    )
    parser.add_argument(
        "--pg-warmup-eps",
        type=int,
        default=200,
        help="Random warmup episodes for scaler fitting (PG methods only).",
    )
    parser.add_argument(
        "--pg-train-eps",
        type=int,
        default=1000,
        help="Online PG training episodes.",
    )
    parser.add_argument(
        "--pg-lr-actor",
        type=float,
        default=1e-3,
        help="Adam learning rate for PolicyNet (PG methods).",
    )
    parser.add_argument(
        "--pg-lr-critic",
        type=float,
        default=1e-3,
        help="Adam learning rate for ValueNet (A2C only).",
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.01,
        help="Entropy regularization coefficient for actor loss (PG methods).",
    )
    parser.add_argument(
        "--repeat-penalty-coef",
        type=float,
        default=0.0,
        help=(
            "Count-based repeat penalty for PG training. If >0, subtract "
            "α·f(visit_count) from the return used in the actor gradient, "
            "where counts are tracked per terminal composition and "
            "accumulate across the training run. 0 disables (backward "
            "compatible). Suggested starting value: 20.0 for the current "
            "reward scale (~100 units of spread)."
        ),
    )
    parser.add_argument(
        "--repeat-penalty-shape",
        type=str,
        choices=["log", "sqrt", "linear"],
        default="log",
        help=(
            "Functional form f(n) for the visit-count penalty. 'log' uses "
            "log(1+n) — bounded growth, recommended default. 'sqrt' is "
            "more aggressive. 'linear' grows without bound."
        ),
    )
    parser.add_argument(
        "--pg-num-iters",
        type=int,
        default=500,
        help="PG training iterations. Each iteration collects --pg-batch-eps episodes and does one gradient step.",
    )
    parser.add_argument(
        "--pg-batch-eps",
        type=int,
        default=15,
        help="Episodes sampled per batched PG update (paper default: 15).",
    )
    parser.add_argument(
        "--pg-checkpoint-iter-freq",
        type=int,
        default=20,
        help="Save a numbered PG checkpoint every N outer iterations (0 disables).",
    )

    args = parser.parse_args()

    if args.only_generate and args.resume_training:
        raise SystemExit("--only-generate and --resume-training are mutually exclusive.")

    if args.dp_model is None:
        args.dp_model = _DEFAULT_DP_MODELS

    _train_seed = args.train_seed if args.train_seed is not None else args.dp_seed
    _gen_seed   = args.gen_seed   if args.gen_seed   is not None else args.dp_seed
    set_seed(_train_seed, deterministic=(args.train_seed is not None))
    ensure_dir(args.out)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dp_cache = {}
    if not args.dp_model:
        raise SystemExit("--dp-model is required")

    from abcde_ooh.dp_predictor import DPConfig, DeepMDOverpotentialPredictor

    cfg = DPConfig(
        base_poscar=args.dp_poscar,
        model_files=tuple(args.dp_model) if args.dp_model else (),
        n_random_configs=args.dp_n_random_configs,
        ads_height=args.dp_ads_height,
        ads_dz=args.dp_ads_dz,
        seed=args.dp_seed,
        geo_opt=args.geo_opt,
        geo_opt_model=args.geo_opt_model,
    )
    dp_predictor = DeepMDOverpotentialPredictor(cfg)

    # Resolve target-phase list (drop "none"; empty means no filter).
    target_phases: List[str] = [p for p in args.target_phase if p != "none"]

    # Apply --exclude-elements to the candidate cation set.
    excluded = {e.strip() for e in args.exclude_elements if e.strip()}
    unknown = excluded - set(DEFAULT_CATION_SET)
    if unknown:
        raise SystemExit(
            f"--exclude-elements: unknown element(s) {sorted(unknown)}. "
            f"Valid elements: {DEFAULT_CATION_SET}"
        )
    cation_set = [el for el in DEFAULT_CATION_SET if el not in excluded]
    if len(cation_set) < 5:
        raise SystemExit(
            f"--exclude-elements left only {len(cation_set)} cations; need at least 5 "
            "for the 5-step environment."
        )

    # Validate that excluded elements aren't required primaries for any target phase.
    _phase_required_primaries = {
        "Ni": {"Ni"},
        "Co": {"Co"},
        "NiFe": {"Ni", "Fe"},
        "CoFe": {"Co", "Fe"},
        "NiFeCo": {"Ni", "Fe", "Co"},
        "any": set(),  # 'any' is permissive; at least one phase must remain feasible
    }
    for phase in target_phases:
        missing = _phase_required_primaries[phase] & excluded
        if missing:
            raise SystemExit(
                f"--target-phase {phase} requires {sorted(_phase_required_primaries[phase])} "
                f"as primaries, but {sorted(missing)} were removed via --exclude-elements."
            )

    phase_filter = None
    if target_phases:
        from abcde_ooh.constraints.phase_sampler import PhaseActionFilter
        _tmp_env = ABCDEOOHEnv(cation_set=cation_set, fraction_set=DEFAULT_FRACTIONS)
        phase_filter = PhaseActionFilter(
            target_phase=target_phases,
            allowed_units=_tmp_env._allowed_units,
            possible_sums_by_k=_tmp_env._possible_sums_by_k,
        )
        print(f"[INFO] Target phase filter active: {target_phases}")
    if excluded:
        print(f"[INFO] Excluded cations: {sorted(excluded)} (cation set size: {len(cation_set)})")

    env = ABCDEOOHEnv(
        cation_set=cation_set,
        fraction_set=DEFAULT_FRACTIONS,
        anion_formula=args.anion_formula,
        phase_filter=phase_filter,
    )

    def dp_reward_fn(_terminal_formula: str) -> float:
        comp = env.terminal_cation_fractions()
        key = _comp_key(comp)
        if key in dp_cache:
            entry = dp_cache[key]
            mean = entry["mean"]
            std = entry["std"]
        else:
            pred = dp_predictor.predict_overpotential(comp)
            mean, std = pred[0], pred[1]
            dp_cache[key] = {"mean": float(mean), "std": float(std)}
        return -(float(mean) - float(std))

    env.reward_fn = dp_reward_fn

    # === Policy Gradient early exit (REINFORCE / A2C) ===
    if args.rl_method in {"reinforce", "a2c"}:
        if args.only_generate:
            scaler_path = args.load_scaler or os.path.join(args.out, "std_scaler.bin")
            if not os.path.exists(scaler_path):
                raise SystemExit(f"--only-generate requires {scaler_path} (use --load-scaler to override)")
            scaler = joblib.load(scaler_path)

            elem_feats_scaled, _elem_scaler = _precompute_elem_features(env.cation_set, env.state_featurizer)
            elem_dim = int(elem_feats_scaled.shape[1])
            fraction_set = env.fraction_set

            state_dim = int(scaler.mean_.shape[0])
            policy = PolicyNet(
                state_dim=state_dim,
                step_dim=env.max_steps,
                elem_dim=elem_dim,
                frac_dim=1,
            ).to(device)
            policy_path = args.load_policy or os.path.join(args.out, "policy.pt")
            if not os.path.exists(policy_path):
                raise SystemExit(f"--only-generate requires {policy_path} (use --load-policy to override)")
            _policy_raw = torch.load(policy_path, map_location=device)
            if isinstance(_policy_raw, dict) and "policy_state" in _policy_raw:
                policy.load_state_dict(_policy_raw["policy_state"])
            else:
                policy.load_state_dict(_policy_raw)
            print(f"[INFO] Loaded policy from {policy_path}", flush=True)

            # Re-seed generation phase for reproducible stochastic sampling.
            np.random.seed(_gen_seed)
            random.seed(_gen_seed)

            rows = generate_pg(
                policy=policy,
                env=env,
                scaler=scaler,
                elem_feats_scaled=elem_feats_scaled,
                fraction_set=fraction_set,
                device=device,
                n_eps=args.num_gen_eps,
                gen_epsilon=args.gen_epsilon,
                gen_top_frac=args.gen_top_frac,
                gen_temperature=args.gen_temperature,
                dp_predictor=dp_predictor,
                dp_cache=dp_cache,
                max_gen_attempts=args.max_gen_attempts if args.max_gen_attempts is not None else 10 * args.num_gen_eps,
            )

            def _sort_key_pg_only(r: dict) -> float:
                v = r.get("dp_mean_minus_std", "")
                try:
                    return float(v)
                except Exception:
                    return float("inf")
            rows.sort(key=_sort_key_pg_only)

            _pg_csv_keys = ["formula", "reward", "dp_mean", "dp_std", "dp_mean_minus_std", "primary_ok", "primary_label"]
            with open(os.path.join(args.out, "generated.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=_pg_csv_keys)
                w.writeheader()
                for r in rows:
                    w.writerow(r)

            with open(os.path.join(args.out, "run_config.json"), "w") as f:
                json.dump(vars(args), f, indent=2, sort_keys=True)

            return

        _pg_ckpt_path = os.path.join(args.out, "checkpoint.pt")
        _pg_mid_checkpoint = None  # set below if resuming from a mid-training checkpoint

        if args.resume_training:
            # --- Resume PG: load scaler; check for mid-training checkpoint first ---
            scaler_path = args.load_scaler or os.path.join(args.out, "std_scaler.bin")
            if not os.path.exists(scaler_path):
                raise SystemExit(
                    f"--resume-training requires {scaler_path} (use --load-scaler to override)"
                )
            scaler = joblib.load(scaler_path)
            print(f"[INFO] Loaded scaler from {scaler_path}", flush=True)

            elem_feats_scaled, _elem_scaler = _precompute_elem_features(env.cation_set, env.state_featurizer)
            elem_dim = int(elem_feats_scaled.shape[1])
            fraction_set = env.fraction_set

            state_dim = int(scaler.mean_.shape[0])
            policy = PolicyNet(
                state_dim=state_dim,
                step_dim=env.max_steps,
                elem_dim=elem_dim,
                frac_dim=1,
            ).to(device)
            value_net = None
            if args.rl_method == "a2c":
                value_net = ValueNet(state_dim=state_dim, step_dim=env.max_steps).to(device)

            # Determine which checkpoint file to load.
            # --load-policy takes priority when explicitly given; otherwise default to checkpoint.pt.
            _explicit_policy = args.load_policy
            _resume_path = _explicit_policy if _explicit_policy else _pg_ckpt_path

            if os.path.exists(_resume_path):
                _raw = torch.load(_resume_path, map_location=device)
                if _raw.get("type") == "pg":
                    _pg_mid_checkpoint = _raw
                    policy.load_state_dict(_raw["policy_state"])
                    if value_net is not None and _raw.get("value_net_state") is not None:
                        value_net.load_state_dict(_raw["value_net_state"])
                    print(
                        f"[INFO] Resuming from mid-training checkpoint: "
                        f"{_raw['episodes_completed']} episodes completed → {_resume_path}",
                        flush=True,
                    )
                else:
                    print(
                        f"[WARN] {_resume_path} exists but type={_raw.get('type')!r} "
                        "(expected 'pg'); falling back to policy.pt.",
                        flush=True,
                    )
            elif _explicit_policy:
                raise SystemExit(f"--load-policy path not found: {_explicit_policy}")

            if _pg_mid_checkpoint is None:
                # No valid mid-training checkpoint; fall back to policy.pt
                policy_path = os.path.join(args.out, "policy.pt")
                if not os.path.exists(policy_path):
                    raise SystemExit(
                        f"--resume-training requires {policy_path} (use --load-policy to override)"
                    )
                policy.load_state_dict(torch.load(policy_path, map_location=device))
                print(f"[INFO] Loaded policy from {policy_path}", flush=True)

                if value_net is not None:
                    vnet_path = args.load_value_net or os.path.join(args.out, "value_net.pt")
                    if not os.path.exists(vnet_path):
                        raise SystemExit(
                            f"--resume-training (a2c) requires {vnet_path} (use --load-value-net to override)"
                        )
                    value_net.load_state_dict(torch.load(vnet_path, map_location=device))
                    print(f"[INFO] Loaded value_net from {vnet_path}", flush=True)
        else:
            scaler = _fit_scaler_from_warmup(env, args.pg_warmup_eps)
            joblib.dump(scaler, os.path.join(args.out, "std_scaler.bin"), compress=True)

            elem_feats_scaled, _elem_scaler = _precompute_elem_features(env.cation_set, env.state_featurizer)
            elem_dim = int(elem_feats_scaled.shape[1])
            fraction_set = env.fraction_set

            state_dim = int(scaler.mean_.shape[0])
            policy = PolicyNet(
                state_dim=state_dim,
                step_dim=env.max_steps,
                elem_dim=elem_dim,
                frac_dim=1,
            ).to(device)
            value_net = (
                ValueNet(state_dim=state_dim, step_dim=env.max_steps).to(device)
                if args.rl_method == "a2c"
                else None
            )

        # Build checkpoint_cfg for train_pg (used when saving mid-training checkpoints
        # and/or when resuming from a mid-training checkpoint).
        _pg_checkpoint_cfg: Optional[dict] = None
        if args.save_checkpoint_freq > 0 or _pg_mid_checkpoint is not None:
            _pg_checkpoint_cfg = {
                "path": _pg_ckpt_path,
                "freq": args.save_checkpoint_freq,
                "start_episode": _pg_mid_checkpoint["episodes_completed"] if _pg_mid_checkpoint else 0,
                "opt_actor_state": _pg_mid_checkpoint.get("opt_actor_state") if _pg_mid_checkpoint else None,
                "opt_critic_state": _pg_mid_checkpoint.get("opt_critic_state") if _pg_mid_checkpoint else None,
                "visit_counts": _pg_mid_checkpoint.get("visit_counts") if _pg_mid_checkpoint else None,
                "rl_method": args.rl_method,
            }

        all_metrics: List[dict] = []
        all_metrics.extend(train_pg(
            policy=policy,
            value_net=value_net,
            env=env,
            scaler=scaler,
            elem_feats_scaled=elem_feats_scaled,
            fraction_set=fraction_set,
            device=device,
            num_iters=args.pg_num_iters,
            batch_eps=args.pg_batch_eps,
            gamma=args.gamma,
            lr_actor=args.pg_lr_actor,
            lr_critic=args.pg_lr_critic,
            entropy_coef=args.entropy_coef,
            rl_method=args.rl_method,
            repeat_penalty_coef=args.repeat_penalty_coef,
            repeat_penalty_shape=args.repeat_penalty_shape,
            max_train_attempts=None,
            checkpoint_cfg=_pg_checkpoint_cfg,
            checkpoint_iter_freq=int(args.pg_checkpoint_iter_freq),
        ))

        # Write training log CSV (append if resuming from a mid-training checkpoint
        # so the log remains continuous; otherwise overwrite).
        _log_path = os.path.join(args.out, "training_log.csv")
        _log_cols = [
            "phase", "iteration", "repeat", "update", "episode", "batch_eps", "epoch",
            "return", "return_raw", "return_shaped", "repeat_penalty",
            "visit_count_before", "unique_comps_seen", "max_visit_count",
            "terminal_comp_key",
            "actor_loss", "entropy", "critic_loss",
            "loss_name", "train_loss", "val_loss", "mse_loss",
            "epsilon", "buffer_rows", "sample_size",
        ]
        _log_mode = "a" if _pg_mid_checkpoint is not None else "w"
        with open(_log_path, _log_mode, newline="") as _f:
            _w = csv.DictWriter(_f, fieldnames=_log_cols, extrasaction="ignore")
            if _log_mode == "w":
                _w.writeheader()
            for _row in all_metrics:
                _w.writerow({c: _row.get(c, "") for c in _log_cols})
        print(f"[INFO] Training log {'appended to' if _log_mode == 'a' else 'written to'} {_log_path}", flush=True)

        torch.save(policy.state_dict(), os.path.join(args.out, "policy.pt"))
        if value_net is not None:
            torch.save(value_net.state_dict(), os.path.join(args.out, "value_net.pt"))

        if not args.skip_generation:
            # Re-seed generation phase so it is independent of training RNG state.
            np.random.seed(_gen_seed)
            random.seed(_gen_seed)

            rows = generate_pg(
                policy=policy,
                env=env,
                scaler=scaler,
                elem_feats_scaled=elem_feats_scaled,
                fraction_set=fraction_set,
                device=device,
                n_eps=args.num_gen_eps,
                gen_epsilon=args.gen_epsilon,
                gen_top_frac=args.gen_top_frac,
                gen_temperature=args.gen_temperature,
                dp_predictor=dp_predictor,
                dp_cache=dp_cache,
                max_gen_attempts=args.max_gen_attempts if args.max_gen_attempts is not None else 10 * args.num_gen_eps,
            )

            def _sort_key_pg(r: dict) -> float:
                v = r.get("dp_mean_minus_std", "")
                try:
                    return float(v)
                except Exception:
                    return float("inf")
            rows.sort(key=_sort_key_pg)

            _pg_csv_keys = ["formula", "reward", "dp_mean", "dp_std", "dp_mean_minus_std", "primary_ok", "primary_label"]
            with open(os.path.join(args.out, "generated.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=_pg_csv_keys)
                w.writeheader()
                for r in rows:
                    w.writerow(r)

        with open(os.path.join(args.out, "run_config.json"), "w") as f:
            json.dump(vars(args), f, indent=2, sort_keys=True)

        return

    # === Classical DQN path ===
    scaler_path = args.load_scaler or os.path.join(args.out, "std_scaler.bin")
    qnet_path = args.load_qnet or os.path.join(args.out, "qnet.pt")

    all_metrics: List[dict] = []

    # Precompute element Magpie feature lookup table — used for both training and generation.
    print("[INFO] Precomputing element Magpie features...", flush=True)
    elem_feats_scaled, _elem_scaler = _precompute_elem_features(env.cation_set, env.state_featurizer)
    elem_dim = int(elem_feats_scaled.shape[1])
    fraction_set = env.fraction_set

    if args.only_generate:
        if not os.path.exists(scaler_path):
            raise SystemExit(f"--only-generate requires {scaler_path} (use --load-scaler to override)")
        scaler = joblib.load(scaler_path)
        state_dim = int(getattr(scaler, "n_features_in_", scaler.mean_.shape[0]))
        qnet = QRegressor(
            state_dim=state_dim, step_dim=env.max_steps, elem_dim=elem_dim, frac_dim=1,
        ).to(device)
        if not os.path.exists(qnet_path):
            raise SystemExit(f"--only-generate requires {qnet_path} (use --load-qnet to override)")
        _raw = torch.load(qnet_path, map_location=device)
        if isinstance(_raw, dict) and "qnet_state" in _raw:
            qnet.load_state_dict(_raw["qnet_state"])
        else:
            qnet.load_state_dict(_raw)
        print(f"[INFO] Loaded qnet from {qnet_path}", flush=True)

    else:
        if args.resume_training:
            tqdm.write(
                "[WARN] --resume-training is not supported in classical DQN mode; starting fresh training."
            )

        # WARMUP: collect random episodes to fill buffer and fit the s_mat scaler.
        warmup_eps = int(args.dqn_warmup_eps)
        buffer_size = int(args.buffer_size)
        buffer: collections.deque = collections.deque(maxlen=buffer_size)

        # Warmup uses real DeepMD rewards so the buffer is correctly pre-filled.
        # This matches standard DQN practice and costs the same total DeepMD calls
        # as clearing the buffer would (early training with ε=1 would re-pay the cost).
        print(f"[INFO] DQN warmup: {warmup_eps} random episodes with real rewards...", flush=True)
        pbar = tqdm(total=warmup_eps, desc="DQN warmup")
        for _ in range(warmup_eps):
            _rollout_random_episode(env)
            add_episode_to_buffer(env.path, buffer, elem_feats_scaled, fraction_set)
            pbar.update(1)
        pbar.close()

        _warmup_s_mat = np.asarray([row["s_mat_raw"] for row in buffer], dtype=float)
        scaler = StandardScaler()
        scaler.fit(_warmup_s_mat)
        state_dim = int(_warmup_s_mat.shape[1])
        joblib.dump(scaler, os.path.join(args.out, "std_scaler.bin"), compress=True)
        print(f"[INFO] s_mat scaler fitted on {len(_warmup_s_mat)} warmup rows.", flush=True)

        # INIT: qnet, frozen target net, optimizer.
        qnet = QRegressor(
            state_dim=state_dim, step_dim=env.max_steps, elem_dim=elem_dim, frac_dim=1,
        ).to(device)
        target_net = copy.deepcopy(qnet)
        target_net.eval()
        optimizer = torch.optim.Adam(qnet.parameters(), lr=args.dqn_lr)
        loss_fn = _make_loss_fn(args.dqn_loss)

        num_train_eps = int(args.num_train_eps)
        batch_size = int(args.dqn_batch_size)
        grad_steps = int(args.grad_steps_per_ep)
        target_update_freq = int(args.target_update_freq)
        eps_anneal_eps = int(args.eps_anneal_eps)
        eps_min = float(args.eps_min)
        gamma = float(args.gamma)
        ckpt_every = int(args.checkpoint_iter_freq)
        dqn_ep_idx = 0
        eps = 1.0

        print(
            f"[INFO] Classical DQN: {num_train_eps} episodes, "
            f"buffer_size={buffer_size}, batch_size={batch_size}, "
            f"grad_steps_per_ep={grad_steps}, target_update_freq={target_update_freq}",
            flush=True,
        )

        # TRAINING LOOP
        outer_pbar = tqdm(range(num_train_eps), desc="DQN train")
        for ep in outer_pbar:
            dqn_ep_idx += 1

            # 1. Collect 1 episode with epsilon-greedy policy
            qnet.eval()
            env.initialize()
            for _t in range(env.max_steps):
                _allowed = env.allowed_actions()
                _s_mat_raw = env.state_featurizer(env.state)
                _s_mat_sc = scaler.transform(_s_mat_raw.reshape(1, -1))[0]
                _s_step_oh = np.zeros(env.max_steps, dtype=float)
                _s_step_oh[env.counter] = 1.0

                if float(np.random.rand()) < eps:
                    _a = random.choice(_allowed)
                else:
                    _a = _choose_action_dqn(
                        qnet, _s_mat_sc, _s_step_oh, _allowed, elem_feats_scaled, fraction_set, device
                    )
                env.step(_a)

            episode_reward = float(env.path[-1].reward)

            # 2. Add 5 rows to buffer
            add_episode_to_buffer(env.path, buffer, elem_feats_scaled, fraction_set)

            # 3. Gradient steps — each samples batch_size rows from the full buffer
            mean_loss = float("nan")
            if len(buffer) >= batch_size:
                _buf_list = list(buffer)
                _losses = []
                for _ in range(grad_steps):
                    _batch = random.sample(_buf_list, batch_size)
                    _losses.append(_dqn_gradient_step(
                        qnet, _batch, target_net, scaler, elem_feats_scaled,
                        fraction_set, optimizer, loss_fn, device, gamma,
                    ))
                mean_loss = float(np.mean(_losses))

            all_metrics.append({
                "phase": "dqn_train",
                "iteration": ep + 1,
                "episode": dqn_ep_idx,
                "return": episode_reward,
                "train_loss": mean_loss,
                "mse_loss": mean_loss,
                "epsilon": eps,
                "buffer_rows": len(buffer),
            })

            outer_pbar.set_postfix(
                eps=f"{eps:.3f}",
                buf=len(buffer),
                loss=f"{mean_loss:.4f}" if not math.isnan(mean_loss) else "warmup",
                ret=f"{episode_reward:.3f}",
            )

            # 4. Hard-copy target net every target_update_freq episodes
            if (ep + 1) % target_update_freq == 0:
                target_net.load_state_dict(qnet.state_dict())
                target_net.eval()

            # 5. Linear epsilon annealing: reaches eps_min at eps_anneal_eps episodes
            eps = max(eps_min, 1.0 - ep / eps_anneal_eps)

            # Periodic numbered checkpoint
            if ckpt_every > 0 and (ep + 1) % ckpt_every == 0:
                _ckpt_path = os.path.join(args.out, "checkpoint.pt")
                _numbered = _ckpt_path.replace(".pt", f"-ep{ep + 1}.pt")
                _save_checkpoint(_ckpt_path, {
                    "type": "dqn",
                    "rl_method": "dqn",
                    "episodes_completed": ep + 1,
                    "qnet_state": qnet.state_dict(),
                    "epsilon": eps,
                }, numbered_path=_numbered)

        outer_pbar.close()
        torch.save(qnet.state_dict(), os.path.join(args.out, "qnet.pt"))
        print(f"[INFO] qnet saved to {os.path.join(args.out, 'qnet.pt')}", flush=True)

    # Write training log
    _log_path = os.path.join(args.out, "training_log.csv")
    _log_cols = [
        "phase", "iteration", "repeat", "update", "episode", "batch_eps", "epoch",
        "return", "return_raw", "return_shaped", "repeat_penalty",
        "visit_count_before", "unique_comps_seen", "max_visit_count",
        "terminal_comp_key",
        "actor_loss", "entropy", "critic_loss",
        "loss_name", "train_loss", "val_loss", "mse_loss",
        "epsilon", "buffer_rows", "sample_size",
    ]
    with open(_log_path, "w", newline="") as _f:
        _w = csv.DictWriter(_f, fieldnames=_log_cols, extrasaction="ignore")
        _w.writeheader()
        for _row in all_metrics:
            _w.writerow({c: _row.get(c, "") for c in _log_cols})
    print(f"[INFO] Training log written to {_log_path}", flush=True)

    if not args.skip_generation:
        # Generate candidates — pure greedy (ε=0)
        qnet.eval()
        rows: List[dict] = []
        target_gen = int(args.num_gen_eps)
        max_gen_attempts = int(args.max_gen_attempts) if args.max_gen_attempts is not None else None
        accepted = 0
        attempted = 0
        dup_rejected = 0
        seen_comp_keys: set = set()

        np.random.seed(_gen_seed)
        random.seed(_gen_seed)

        pbar = tqdm(total=target_gen, desc="Generate (accepted)")
        while accepted < target_gen and (max_gen_attempts is None or attempted < max_gen_attempts):
            attempted += 1
            env.initialize()

            for _t in range(env.max_steps):
                _allowed_g = env.allowed_actions()
                _s_mat_raw_g = env.state_featurizer(env.state)
                _s_mat_sc_g = scaler.transform(_s_mat_raw_g.reshape(1, -1))[0]
                _s_step_oh_g = np.zeros(env.max_steps, dtype=float)
                _s_step_oh_g[env.counter] = 1.0
                _a_g = _choose_action_dqn(
                    qnet, _s_mat_sc_g, _s_step_oh_g, _allowed_g, elem_feats_scaled, fraction_set, device,
                    gen_epsilon=args.gen_epsilon,
                    gen_top_frac=args.gen_top_frac,
                    gen_temperature=args.gen_temperature,
                )
                env.step(_a_g)

            comp = env.terminal_cation_fractions()
            comp_key = tuple(sorted(
                (str(k), int(round(float(v) * 20)))
                for k, v in comp.items()
                if int(round(float(v) * 20)) > 0
            ))
            ok, label = check_primary_phase(comp)

            if comp_key in seen_comp_keys:
                dup_rejected += 1
                if dup_rejected <= 20 or dup_rejected % 500 == 0:
                    tqdm.write(
                        f"[REJECT] duplicate generated: attempt={attempted} dup_rejected={dup_rejected}"
                    )
                continue
            seen_comp_keys.add(comp_key)

            reward = float(env.path[-1].reward) if env.path else 0.0
            key = comp_key
            if key in dp_cache:
                entry = dp_cache[key]
                mean = float(entry["mean"])
                std = float(entry["std"])
            else:
                pred = dp_predictor.predict_overpotential(comp)
                mean, std = float(pred[0]), float(pred[1])
                dp_cache[key] = {"mean": mean, "std": std}

            rows.append({
                "formula": env.terminal_formula,
                "reward": reward,
                "dp_mean": mean,
                "dp_std": std,
                "dp_mean_minus_std": mean - std,
                "primary_ok": bool(ok),
                "primary_label": label or "",
            })
            accepted += 1
            pbar.update(1)
            rate = (accepted / attempted) if attempted else 0.0
            pbar.set_postfix(attempts=attempted, rate=f"{rate:.3f}", dups=dup_rejected)

        pbar.close()
        rate = (accepted / attempted) if attempted else 0.0
        print(
            f"[INFO] Generated: accepted {accepted}/{target_gen} after {attempted} attempts "
            f"(rate={rate:.4f}, dup_rejected={dup_rejected}).",
            flush=True,
        )
        if max_gen_attempts is not None and accepted < target_gen:
            print(f"[WARN] Only {accepted}/{target_gen} candidates generated after {attempted} attempts.")

        def _sort_key(r: dict) -> float:
            try:
                return float(r.get("dp_mean_minus_std", "inf"))
            except Exception:
                return float("inf")
        rows.sort(key=_sort_key)

        out_csv = os.path.join(args.out, "generated.csv")
        _csv_keys = ["formula", "reward", "dp_mean", "dp_std", "dp_mean_minus_std", "primary_ok", "primary_label"]
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_csv_keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    with open(os.path.join(args.out, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
