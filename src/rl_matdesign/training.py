"""RL training functions for rl_matdesign experiments.

All functions are domain-agnostic: they operate on a :class:`CompositionEnv`
and a :class:`~rl_matdesign.predictors.base.PropertyPredictor` and contain
no material-system-specific logic.

Key improvements
----------------
- Classical online DQN (``train_dqn_online``): FIFO replay buffer, TD Bellman
  targets, hard target-network copy, SmoothL1 loss.  Replaces offline MC
  Q-learning.
- Magpie element features: ``_precompute_elem_features`` computes per-element
  Magpie feature vectors (separately scaled), replacing one-hot element
  encoding everywhere.  Fraction encoding is a scalar float, replacing
  one-hot fraction encoding.
- Unified generation flags (``gen_temperature / gen_top_frac / gen_epsilon``):
  same three-mode exploration strategy for DQN and PG generation.  Default
  Boltzmann T=1.0 prevents pure-greedy composition collapse.
- ``pg_epsilon`` removed from PG rollout: epsilon-greedy is off-policy
  contamination in an on-policy method; removed entirely.
- Return normalisation in ``train_pg`` (default True): per-episode
  ``(G - mean) / (std + ε)`` reduces gradient variance.
- Dual-phase generation in ``generate_candidates``: exploitation + exploration
  candidates from one call, tagged with ``purpose`` column.
"""
from __future__ import annotations

import collections
import copy
import math
import random
import warnings
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from .env import CompositionEnv
from .predictors.base import PropertyPredictor


# ---------------------------------------------------------------------------
# Loss function factory
# ---------------------------------------------------------------------------

def _make_loss_fn(name: str) -> torch.nn.Module:
    name = (name or "smoothl1").lower()
    if name == "mse":
        return torch.nn.MSELoss()
    if name in ("smoothl1", "huber"):
        return torch.nn.SmoothL1Loss()
    raise ValueError(f"Unknown DQN loss: {name!r} (expected 'mse' or 'smoothl1')")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(path: str, data: dict, numbered_path: str | None = None) -> None:
    """Atomically write a checkpoint dict.

    If *numbered_path* is given, data is written there and *path* is updated
    as a relative symlink so ``checkpoint.pt`` always resolves to the latest.
    """
    import os
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


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _comp_key(comp: Dict[str, float], total_units: int = 20) -> tuple:
    """Canonical hashable key for a terminal cation composition dict."""
    items = []
    for el, frac in comp.items():
        units = int(round(float(frac) * total_units))
        if units > 0:
            items.append((str(el), units))
    return tuple(sorted(items))


def objective_from_mean_std(
    mean: float, std: float, objective: str = "mean", k: float = 1.0
) -> float:
    """Compute scalar reward from predictor (mean, std) output."""
    if objective == "mean":
        return mean
    elif objective == "mean_minus_kstd":
        return mean - k * std
    elif objective == "mean_plus_kstd":
        return mean + k * std
    else:
        raise ValueError(f"Unknown objective '{objective}'.")


# ---------------------------------------------------------------------------
# Element feature precomputation (Magpie)
# ---------------------------------------------------------------------------

def _precompute_elem_features(
    cation_set: List[str],
    featurizer: Callable,
) -> Tuple[np.ndarray, StandardScaler]:
    """Precompute and scale Magpie feature vectors for each cation.

    Uses a separate scaler from s_mat because single-element statistics differ
    from composite material statistics (some features have zero variance for
    single elements).

    Returns
    -------
    (elem_feats_scaled, elem_scaler):
        ``elem_feats_scaled`` has shape ``(n_elements, feature_dim)``.
        Index ``i`` corresponds to ``cation_set[i]``.
    """
    raw = np.asarray([featurizer(el + "1.00") for el in cation_set], dtype=float)
    elem_scaler = StandardScaler()
    elem_scaler.fit(raw)
    return elem_scaler.transform(raw), elem_scaler


# ---------------------------------------------------------------------------
# Replay buffer helpers (classical DQN)
# ---------------------------------------------------------------------------

def add_episode_to_buffer(
    path: list,
    buffer: collections.deque,
    elem_feats_scaled: np.ndarray,
    fraction_set: List[str],
) -> None:
    """Convert a completed episode path into buffer rows and append.

    Each row stores:
    - Raw (unscaled) s_mat and s_mat_next for lazy scaling at batch time.
    - Element index and comp scalar value instead of one-hot vectors.
    - ``next_allowed_idx``: list of (elem_idx, comp_idx) pairs for the next
      step, used to compute the max-Q bootstrap target.
    """
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
            "s_mat_raw":       np.asarray(step.state_material_features, dtype=float),
            "s_step":          np.asarray(step.state_step_onehot, dtype=float),
            "a_elem_idx":      int(np.argmax(step.action_elem_onehot)),
            "a_comp_val":      float(fraction_set[int(np.argmax(step.action_comp_onehot))]),
            "reward":          float(step.reward),
            "s_mat_next_raw":  np.asarray(next_step.state_material_features, dtype=float)
                               if not done else np.zeros(len(step.state_material_features), dtype=float),
            "s_step_next":     np.asarray(next_step.state_step_onehot, dtype=float)
                               if not done else np.zeros(len(step.state_step_onehot), dtype=float),
            "next_allowed_idx": next_allowed_idx,
            "done":            done,
        })


def _compute_td_target(
    row: dict,
    target_net: torch.nn.Module,
    s_mat_scaler: StandardScaler,
    elem_feats_scaled: np.ndarray,
    fraction_set: List[str],
    device: torch.device,
    gamma: float,
) -> float:
    """TD target for one buffer row: r if done, else r + γ·max_a' Q_target(s', a')."""
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
            torch.tensor(ns_mat_b,  dtype=torch.float32, device=device),
            torch.tensor(ns_step_b, dtype=torch.float32, device=device),
            torch.tensor(a_e,       dtype=torch.float32, device=device),
            torch.tensor(a_c,       dtype=torch.float32, device=device),
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
    """One DQN gradient step: SmoothL1(Q(s,a), TD-target)."""
    y = [
        _compute_td_target(r, target_net, s_mat_scaler, elem_feats_scaled,
                           fraction_set, device, gamma)
        for r in batch
    ]
    s_mat_b  = torch.tensor(
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
    loss = loss_fn(qnet(s_mat_b, s_step_b, a_elem_b, a_comp_b), y_t)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.item())


# ---------------------------------------------------------------------------
# Classical online DQN training
# ---------------------------------------------------------------------------

def train_dqn_online(
    *,
    env: CompositionEnv,
    elem_feats_scaled: np.ndarray,
    fraction_set: List[str],
    device: torch.device,
    hidden_dim: int = 128,
    n_warmup_eps: int = 500,
    n_train_eps: int = 20000,
    buffer_size: int = 50000,
    batch_size: int = 256,
    grad_steps_per_ep: int = 5,
    target_update_freq: int = 100,
    eps_anneal_eps: int = 10000,
    eps_min: float = 0.05,
    gamma: float = 0.9,
    lr: float = 1e-3,
    loss_name: str = "smoothl1",
    checkpoint_cfg: Optional[dict] = None,
) -> Tuple[torch.nn.Module, StandardScaler, List[dict]]:
    """Classical online DQN with FIFO replay buffer and TD targets.

    Phase 0 — Warmup:
        Roll out ``n_warmup_eps`` random episodes with real rewards to pre-fill
        the buffer and fit the s_mat StandardScaler.

    Phase 1 — Training loop (``n_train_eps`` episodes):
        Each episode:
        1. ε-greedy rollout → 5 new buffer rows.
        2. ``grad_steps_per_ep`` gradient steps, each sampling ``batch_size``
           rows from the full buffer (SmoothL1 loss on TD targets).
        3. Hard-copy qnet → target_net every ``target_update_freq`` episodes.
        4. Linear epsilon anneal: ε = max(eps_min, 1 − ep / eps_anneal_eps).

    Returns
    -------
    (qnet, scaler, metrics):
        Trained Q-network, fitted s_mat scaler, and list of per-episode metric
        dicts (phase="dqn_train").
    """
    from .model import QRegressor

    buffer: collections.deque = collections.deque(maxlen=buffer_size)

    print(f"[INFO] DQN warmup: {n_warmup_eps} random episodes with real rewards...")
    pbar = tqdm(total=n_warmup_eps, desc="DQN warmup")
    for _ in range(n_warmup_eps):
        _rollout_random_episode(env)
        add_episode_to_buffer(env.path, buffer, elem_feats_scaled, fraction_set)
        pbar.update(1)
    pbar.close()

    s_mat_all = np.asarray([r["s_mat_raw"] for r in buffer], dtype=float)
    scaler = StandardScaler().fit(s_mat_all)
    state_dim = int(s_mat_all.shape[1])
    elem_dim  = int(elem_feats_scaled.shape[1])
    step_dim  = env.n_components
    print(f"[INFO] s_mat scaler fitted on {len(s_mat_all)} warmup rows.")

    qnet = QRegressor(
        state_dim=state_dim, step_dim=step_dim,
        elem_dim=elem_dim, frac_dim=1, hidden_dim=hidden_dim,
    ).to(device)
    target_net = copy.deepcopy(qnet)
    target_net.eval()
    optimizer = torch.optim.Adam(qnet.parameters(), lr=lr)
    loss_fn = _make_loss_fn(loss_name)

    ckpt_path: Optional[str] = None
    ckpt_freq: int = 0
    if checkpoint_cfg:
        ckpt_path = checkpoint_cfg.get("path")
        ckpt_freq = int(checkpoint_cfg.get("freq", 0))

    eps = 1.0
    metrics: List[dict] = []
    pbar = tqdm(range(n_train_eps), desc="DQN train")

    for ep in pbar:
        # 1. Collect one episode with epsilon-greedy policy.
        qnet.eval()
        env.initialize()
        for _t in range(env.n_components):
            _allowed = env.allowed_actions()
            _s_mat_sc = scaler.transform(
                env.state_featurizer(env.state).reshape(1, -1))[0]
            _s_step = np.zeros(env.n_components, dtype=float)
            _s_step[env.counter] = 1.0
            if float(np.random.rand()) < eps:
                _a = random.choice(_allowed)
            else:
                _a = choose_action(
                    model=qnet, device=device,
                    s_material=_s_mat_sc, s_step=_s_step,
                    allowed_actions=_allowed,
                    elem_feats_scaled=elem_feats_scaled,
                    fraction_set=fraction_set,
                    gen_temperature=1.0,
                )
            env.step(_a)

        episode_reward = float(env.path[-1].reward)
        add_episode_to_buffer(env.path, buffer, elem_feats_scaled, fraction_set)

        # 2. Gradient steps.
        mean_loss = float("nan")
        if len(buffer) >= batch_size:
            buf_list = list(buffer)
            losses = []
            for _ in range(grad_steps_per_ep):
                _batch = random.sample(buf_list, batch_size)
                losses.append(_dqn_gradient_step(
                    qnet, _batch, target_net, scaler,
                    elem_feats_scaled, fraction_set, optimizer, loss_fn, device, gamma,
                ))
            mean_loss = float(np.mean(losses))

        # 3. Hard target-net copy.
        if (ep + 1) % target_update_freq == 0:
            target_net.load_state_dict(qnet.state_dict())
            target_net.eval()

        # 4. Linear epsilon anneal.
        eps = max(eps_min, 1.0 - ep / eps_anneal_eps)

        metrics.append({
            "phase": "dqn_train",
            "episode": ep + 1,
            "return": episode_reward,
            "train_loss": mean_loss,
            "epsilon": eps,
            "buffer_rows": len(buffer),
        })
        pbar.set_postfix(
            eps=f"{eps:.3f}",
            loss=f"{mean_loss:.4f}" if not math.isnan(mean_loss) else "warmup",
            ret=f"{episode_reward:.3f}",
        )

        # 5. Periodic checkpoint.
        if ckpt_freq > 0 and ckpt_path and (ep + 1) % ckpt_freq == 0:
            _numbered = ckpt_path.replace(".pt", f"-ep{ep + 1}.pt")
            _save_checkpoint(ckpt_path, {
                "type": "dqn",
                "episodes_completed": ep + 1,
                "qnet_state": qnet.state_dict(),
                "target_net_state": target_net.state_dict(),
                "opt_state": optimizer.state_dict(),
                "eps": eps,
            }, numbered_path=_numbered)
            tqdm.write(f"[INFO] DQN checkpoint saved at episode {ep + 1} → {_numbered}")

    pbar.close()
    return qnet, scaler, metrics


# ---------------------------------------------------------------------------
# Legacy offline DQN helpers (kept for backward compatibility)
# ---------------------------------------------------------------------------

def extract_mc_q_targets(
    episode: List, gamma: float = 0.99
) -> Tuple[List[Tuple], List[float]]:
    """Compute Monte-Carlo Q-targets from an episode path (legacy offline DQN)."""
    inputs = []
    q_targets: List[float] = []
    G = 0.0
    for step in reversed(episode):
        G = float(step.reward) + gamma * G
        q_targets.append(G)
        inputs.append((
            np.asarray(step.state_material_features, dtype=float),
            np.asarray(step.state_step_onehot, dtype=float),
            np.asarray(step.action_elem_onehot, dtype=float),
            np.asarray(step.action_comp_onehot, dtype=float),
        ))
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
    iteration: int = 0,
) -> List[dict]:
    """Supervised regression on MC Q-targets (MSELoss + Adam) — legacy offline DQN."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    metrics: List[dict] = []
    pbar = tqdm(range(epochs), desc="Q epochs")
    for epoch_idx in pbar:
        batch_losses: List[float] = []
        for s_mat, s_step, a_elem, a_comp, y in loader:
            opt.zero_grad(set_to_none=True)
            pred = model(s_mat.to(device), s_step.to(device), a_elem.to(device), a_comp.to(device))
            loss = loss_fn(pred, y.to(device))
            loss.backward()
            opt.step()
            batch_losses.append(float(loss.item()))
        epoch_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        metrics.append({"phase": "dqn_train", "iteration": iteration,
                        "epoch": epoch_idx + 1, "mse_loss": epoch_loss})
        pbar.set_postfix(mse_loss=f"{epoch_loss:.4f}")
    return metrics


# ---------------------------------------------------------------------------
# Action selection (DQN)
# ---------------------------------------------------------------------------

def choose_action(
    *,
    model: torch.nn.Module,
    device: torch.device,
    s_material: np.ndarray,
    s_step: np.ndarray,
    allowed_actions: Sequence[Tuple[Tuple[float, ...], Tuple[float, ...]]],
    elem_feats_scaled: np.ndarray,
    fraction_set: List[str],
    gen_epsilon: float = 0.0,
    gen_top_frac: float = 0.0,
    gen_temperature: float = 0.0,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Select action from Q-network using Magpie encoding.

    Priority: ε-greedy > Boltzmann > top-k > ValueError (no pure greedy).

    Parameters
    ----------
    gen_epsilon:
        ε-greedy probability. Short-circuits before Q computation.
    gen_temperature:
        Boltzmann temperature τ > 0. Ignored if gen_epsilon fires.
    gen_top_frac:
        Uniform sample from top-k% by Q-value. Ignored if higher-priority
        strategy fires.
    """
    if not allowed_actions:
        raise RuntimeError("No allowed actions.")
    n = len(allowed_actions)

    if gen_epsilon > 0.0 and float(np.random.rand()) < gen_epsilon:
        return allowed_actions[int(np.random.randint(n))]

    a_elem = np.asarray(
        [elem_feats_scaled[int(np.argmax(a[0]))] for a in allowed_actions], dtype=float
    )
    a_comp = np.asarray(
        [[float(fraction_set[int(np.argmax(a[1]))])] for a in allowed_actions], dtype=float
    )
    s_mat_b  = np.repeat(s_material.reshape(1, -1), n, axis=0)
    s_step_b = np.repeat(s_step.reshape(1, -1), n, axis=0)

    with torch.no_grad():
        q = model(
            torch.tensor(s_mat_b,  dtype=torch.float32, device=device),
            torch.tensor(s_step_b, dtype=torch.float32, device=device),
            torch.tensor(a_elem,   dtype=torch.float32, device=device),
            torch.tensor(a_comp,   dtype=torch.float32, device=device),
        ).reshape(-1)

    if gen_temperature > 0.0:
        q_np = q.cpu().numpy().astype(float) / gen_temperature
        q_np -= q_np.max()
        probs = np.exp(q_np); probs /= probs.sum()
        return allowed_actions[int(np.random.choice(n, p=probs))]

    if gen_top_frac > 0.0:
        k = max(1, int(round(gen_top_frac * n)))
        topk = torch.argsort(q, descending=True).cpu().tolist()[:k]
        return allowed_actions[int(np.random.choice(topk))]

    raise ValueError(
        "No generation strategy active: set gen_temperature, gen_top_frac, or gen_epsilon > 0."
    )


# ---------------------------------------------------------------------------
# Rollout functions
# ---------------------------------------------------------------------------

def _rollout_random_episode(env: CompositionEnv) -> None:
    """Roll out one episode with uniformly random action selection."""
    env.initialize()
    for _ in range(env.n_components):
        a = env.sample_random_action()
        env.step(a)


def _rollout_policy_episode(
    *,
    env: CompositionEnv,
    qnet: torch.nn.Module,
    scaler: StandardScaler,
    device: torch.device,
    elem_feats_scaled: np.ndarray,
    fraction_set: List[str],
    gen_epsilon: float = 0.0,
    gen_top_frac: float = 0.0,
    gen_temperature: float = 1.0,
) -> None:
    """Roll out one episode guided by a Q-network (used during generation)."""
    env.initialize()
    for _ in range(env.n_components):
        allowed = env.allowed_actions()
        s_mat = scaler.transform(env.state_featurizer(env.state).reshape(1, -1))[0]
        s_step = np.zeros(env.n_components, dtype=float)
        if env.counter < env.n_components:
            s_step[env.counter] = 1.0
        a = choose_action(
            model=qnet, device=device,
            s_material=s_mat, s_step=s_step,
            allowed_actions=allowed,
            elem_feats_scaled=elem_feats_scaled,
            fraction_set=fraction_set,
            gen_epsilon=gen_epsilon,
            gen_top_frac=gen_top_frac,
            gen_temperature=gen_temperature,
        )
        env.step(a)


def _fit_scaler_from_warmup(env: CompositionEnv, n_warmup_eps: int) -> StandardScaler:
    """Fit a StandardScaler on material features from random warmup episodes.

    The reward function is temporarily replaced with a no-op so warmup does not
    invoke the (potentially expensive) PropertyPredictor.  Used by PG methods
    only (DQN warmup is handled inside ``train_dqn_online``).
    """
    all_s_mat = []
    original_reward_fn = env.reward_fn
    env.reward_fn = lambda _f: 0.0
    pbar = tqdm(total=n_warmup_eps, desc="Warmup (scaler fit)")
    try:
        for _ in range(n_warmup_eps):
            _rollout_random_episode(env)
            for step in env.path:
                all_s_mat.append(np.asarray(step.state_material_features, dtype=float))
            pbar.update(1)
    finally:
        env.reward_fn = original_reward_fn
        pbar.close()
    scaler = StandardScaler()
    scaler.fit(np.asarray(all_s_mat, dtype=float))
    print(f"[INFO] Scaler fitted on {len(all_s_mat)} warmup states.")
    return scaler


def _rollout_pg_episode(
    *,
    env: CompositionEnv,
    policy: torch.nn.Module,
    scaler: StandardScaler,
    elem_feats_scaled: np.ndarray,
    fraction_set: List[str],
    device: torch.device,
) -> None:
    """Roll out one episode with pure softmax sampling (on-policy, no pg_epsilon)."""
    env.initialize()
    for _ in range(env.n_components):
        allowed = env.allowed_actions()
        s_mat = scaler.transform(env.state_featurizer(env.state).reshape(1, -1))[0]
        s_step = np.zeros(env.n_components, dtype=float)
        if env.counter < env.n_components:
            s_step[env.counter] = 1.0
        n = len(allowed)
        a_elem_batch = np.asarray(
            [elem_feats_scaled[int(np.argmax(a[0]))] for a in allowed], dtype=float
        )
        a_comp_batch = np.asarray(
            [[float(fraction_set[int(np.argmax(a[1]))])] for a in allowed], dtype=float
        )
        s_mat_batch  = np.repeat(s_mat.reshape(1, -1), n, axis=0)
        s_step_batch = np.repeat(s_step.reshape(1, -1), n, axis=0)
        with torch.no_grad():
            logits = policy(
                torch.tensor(s_mat_batch,  dtype=torch.float32, device=device),
                torch.tensor(s_step_batch, dtype=torch.float32, device=device),
                torch.tensor(a_elem_batch, dtype=torch.float32, device=device),
                torch.tensor(a_comp_batch, dtype=torch.float32, device=device),
            ).reshape(-1)
        probs = torch.softmax(logits, dim=0).cpu().numpy()
        env.step(allowed[int(np.random.choice(n, p=probs))])


# ---------------------------------------------------------------------------
# Policy gradient training (REINFORCE / A2C)
# ---------------------------------------------------------------------------

def _episode_pg_terms(
    *,
    path,
    returns: List[float],
    returns_shaped: List[float],
    policy: torch.nn.Module,
    value_net: Optional[torch.nn.Module],
    scaler: "StandardScaler",
    elem_feats_scaled: np.ndarray,
    fraction_set: List[str],
    device: "torch.device",
) -> tuple:
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
        s_mat_batch  = np.repeat(s_mat.reshape(1, -1), n, axis=0)
        s_step_batch = np.repeat(s_step.reshape(1, -1), n, axis=0)

        s_mat_t  = torch.tensor(s_mat_batch,  dtype=torch.float32, device=device)
        s_step_t = torch.tensor(s_step_batch, dtype=torch.float32, device=device)
        a_elem_t = torch.tensor(a_elem_batch, dtype=torch.float32, device=device)
        a_comp_t = torch.tensor(a_comp_batch, dtype=torch.float32, device=device)

        logits    = policy(s_mat_t, s_step_t, a_elem_t, a_comp_t).reshape(-1)
        log_probs = torch.log_softmax(logits, dim=0)
        probs     = torch.softmax(logits, dim=0)

        taken_elem = np.asarray(step.action_elem_onehot)
        taken_comp = np.asarray(step.action_comp_onehot)
        taken_idx = None
        for i, a in enumerate(allowed):
            if (np.array_equal(np.asarray(a[0]), taken_elem)
                    and np.array_equal(np.asarray(a[1]), taken_comp)):
                taken_idx = i
                break
        if taken_idx is None:
            continue

        G_raw_t    = torch.tensor(G_t,        dtype=torch.float32, device=device)
        G_shaped_t = torch.tensor(G_t_shaped, dtype=torch.float32, device=device)

        if value_net is not None:
            s_mat_single  = torch.tensor(s_mat.reshape(1, -1),  dtype=torch.float32, device=device)
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
    value_net: Optional[torch.nn.Module],
    env: CompositionEnv,
    scaler: "StandardScaler",
    elem_feats_scaled: np.ndarray,
    fraction_set: List[str],
    device: "torch.device",
    num_iters: int,
    batch_eps: int,
    gamma: float = 0.99,
    lr_actor: float = 1e-3,
    lr_critic: float = 1e-3,
    entropy_coef: float = 0.01,
    rl_method: str = "a2c",
    repeat_penalty_coef: float = 0.0,
    repeat_penalty_shape: str = "log",
    max_train_attempts: Optional[int] = None,
    checkpoint_cfg: Optional[dict] = None,
) -> List[dict]:
    """Batched REINFORCE / A2C training loop (matches feat/classical-dqn).

    Structure:
        for it in range(num_iters):
            collect batch_eps episodes under current policy
            accumulate per-step actor/entropy/critic terms across all of them
            one optimizer step over the entire batch
    """
    policy.train()
    opt_actor = torch.optim.Adam(policy.parameters(), lr=lr_actor)
    if value_net is not None:
        value_net.train()
        opt_critic = torch.optim.Adam(value_net.parameters(), lr=lr_critic)
    else:
        opt_critic = None

    ckpt_path: Optional[str] = None
    ckpt_freq: int = 0
    if checkpoint_cfg:
        ckpt_path = checkpoint_cfg.get("path")
        ckpt_freq = int(checkpoint_cfg.get("freq", 0))
        if checkpoint_cfg.get("opt_actor_state") is not None:
            opt_actor.load_state_dict(checkpoint_cfg["opt_actor_state"])
        if opt_critic is not None and checkpoint_cfg.get("opt_critic_state") is not None:
            opt_critic.load_state_dict(checkpoint_cfg["opt_critic_state"])

    visit_counts: Counter = Counter()
    if checkpoint_cfg and checkpoint_cfg.get("visit_counts"):
        visit_counts = Counter(checkpoint_cfg["visit_counts"])

    metrics: List[dict] = []
    accepted = int(checkpoint_cfg.get("start_episode", 0)) if checkpoint_cfg else 0
    attempted = 0
    update_idx = 0

    total_updates = max(1, int(num_iters))
    print(
        f"[INFO] PG: {num_iters} iters × "
        f"{batch_eps} eps/update = {total_updates} updates, {total_updates * batch_eps} total episodes."
    )

    outer_pbar = tqdm(range(int(num_iters)), desc=f"{rl_method.upper()} iters")
    for it in outer_pbar:
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
                env=env, policy=policy, scaler=scaler,
                elem_feats_scaled=elem_feats_scaled, fraction_set=fraction_set,
                device=device,
            )
            path = env.path
            if not path:
                continue

            G = 0.0
            returns: List[float] = []
            for step in reversed(path):
                G = float(step.reward) + gamma * G
                returns.append(G)
            returns.reverse()

            terminal_key = env.terminal_comp_key()
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

        actor_loss       = torch.stack(all_actor).mean()
        entropy_bonus    = torch.stack(all_entropy).mean()
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
        ep_entropy    = float(entropy_bonus.item())
        mean_return_raw    = float(np.mean([rs[0] for rs in batch_returns])) if batch_returns else 0.0
        mean_return_shaped = float(np.mean([rs[0] for rs in batch_returns_shaped])) if batch_returns_shaped else 0.0

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
            "terminal_comp_key": str(batch_terminal_keys[-1]) if batch_terminal_keys else "",
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

        if ckpt_freq > 0 and ckpt_path and (it + 1) % ckpt_freq == 0:
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
            tqdm.write(f"[INFO] PG checkpoint saved at iter {it + 1} → {_numbered}")

    outer_pbar.close()
    print(f"[INFO] PG training: {update_idx} updates over {accepted} episodes, {attempted} attempts.")
    return metrics


# ---------------------------------------------------------------------------
# Candidate generation (dual-phase, unified generation flags)
# ---------------------------------------------------------------------------

def _pg_single_episode_generate(
    *,
    env: CompositionEnv,
    policy: torch.nn.Module,
    scaler: StandardScaler,
    elem_feats_scaled: np.ndarray,
    fraction_set: List[str],
    device: torch.device,
    gen_epsilon: float = 0.0,
    gen_top_frac: float = 0.0,
    gen_temperature: float = 1.0,
) -> None:
    """Single generation episode for PG methods.

    Priority: ε-greedy > Boltzmann > top-k > ValueError.
    Default gen_temperature=1.0 prevents pure-greedy composition collapse.
    """
    env.initialize()
    for _ in range(env.n_components):
        allowed = env.allowed_actions()
        s_mat = scaler.transform(env.state_featurizer(env.state).reshape(1, -1))[0]
        s_step = np.zeros(env.n_components, dtype=float)
        if env.counter < env.n_components:
            s_step[env.counter] = 1.0

        n = len(allowed)
        a_elem_batch = np.asarray(
            [elem_feats_scaled[int(np.argmax(a[0]))] for a in allowed], dtype=float
        )
        a_comp_batch = np.asarray(
            [[float(fraction_set[int(np.argmax(a[1]))])] for a in allowed], dtype=float
        )
        s_mat_batch  = np.repeat(s_mat.reshape(1, -1), n, axis=0)
        s_step_batch = np.repeat(s_step.reshape(1, -1), n, axis=0)

        with torch.no_grad():
            logits = policy(
                torch.tensor(s_mat_batch,  dtype=torch.float32, device=device),
                torch.tensor(s_step_batch, dtype=torch.float32, device=device),
                torch.tensor(a_elem_batch, dtype=torch.float32, device=device),
                torch.tensor(a_comp_batch, dtype=torch.float32, device=device),
            ).reshape(-1)

        if gen_epsilon > 0.0 and float(np.random.rand()) < gen_epsilon:
            idx = int(np.random.randint(n))
        elif gen_temperature > 0.0:
            lg = logits.cpu().numpy().astype(float) / gen_temperature
            lg -= lg.max()
            probs = np.exp(lg); probs /= probs.sum()
            idx = int(np.random.choice(n, p=probs))
        elif gen_top_frac > 0.0:
            k = max(1, int(round(gen_top_frac * n)))
            topk = torch.argsort(logits, descending=True).cpu().tolist()[:k]
            idx = int(np.random.choice(topk))
        else:
            raise ValueError(
                "No generation strategy active: set gen_temperature, gen_top_frac, or gen_epsilon > 0."
            )
        env.step(allowed[idx])


def generate_candidates(
    *,
    env: CompositionEnv,
    predictor: PropertyPredictor,
    scaler: StandardScaler,
    device: torch.device,
    elem_feats_scaled: np.ndarray,
    fraction_set: List[str],
    policy: Optional[torch.nn.Module] = None,
    qnet: Optional[torch.nn.Module] = None,
    n_exploit: int = 200,
    n_explore: int = 0,
    gen_epsilon: float = 0.0,
    gen_top_frac: float = 0.0,
    gen_temperature: float = 1.0,
    exploit_objective: str = "mean_minus_kstd",
    explore_objective: str = "mean_plus_kstd",
    k: float = 1.0,
    max_attempts: Optional[int] = None,
) -> List[dict]:
    """Generate candidate compositions in dual-phase mode.

    Produces exploitation and/or exploration candidates from the same trained
    policy.  Unified generation flags (``gen_temperature``, ``gen_top_frac``,
    ``gen_epsilon``) apply identically to DQN and PG methods.

    Parameters
    ----------
    gen_temperature:
        Boltzmann temperature τ for action sampling.  Default 1.0 prevents
        pure-greedy composition collapse.  Set to 0 and use gen_top_frac or
        gen_epsilon for alternative strategies.
    gen_top_frac:
        Uniform sample from top-k% of actions by Q/logit.
    gen_epsilon:
        ε-greedy probability (highest priority, short-circuits Q computation).

    Notes
    -----
    ``exploit_objective`` / ``explore_objective`` are kept in the signature for
    backward compatibility but are no longer used to re-compute the reward here.
    The reward stored in each row comes directly from ``env.path[-1].reward``
    (set by ``env.reward_fn`` during the episode), matching classical-dqn
    ``generate_pg`` behaviour and avoiding double-application of the objective.

    Raw predictor values are fetched via ``predictor.predict_raw()`` when
    available (OOHCatalystPredictor exposes this), falling back to
    ``predictor.predict()`` otherwise.  This ensures ``dp_mean`` stores the
    raw property value (e.g. overpotential in mV for OOH), not a transformed
    reward scalar.
    """
    if policy is None and qnet is None:
        raise ValueError("Provide either policy (PG) or qnet (DQN).")
    use_pg = policy is not None

    if use_pg:
        policy.eval()
    else:
        qnet.eval()

    use_predict_raw = hasattr(predictor, "predict_raw")
    use_check_phase = hasattr(predictor, "check_phase")

    # Cache raw (mean, std) values: avoids a second DeepMD call for the same
    # composition when the first was already made inside env.reward_fn.
    predictor_raw_cache: Dict[tuple, Tuple[float, float]] = {}
    seen_comp_keys: set = set()
    rows: List[dict] = []

    phases = []
    if n_exploit > 0:
        phases.append(("exploit", n_exploit))
    if n_explore > 0:
        phases.append(("explore", n_explore))

    for purpose, n_target in phases:
        accepted = 0
        attempted = 0
        dup_rejected = 0
        max_att = max_attempts or (n_target * 20)

        pbar = tqdm(total=n_target, desc=f"Generate [{purpose}]")
        while accepted < n_target and attempted < max_att:
            attempted += 1

            if use_pg:
                _pg_single_episode_generate(
                    env=env, policy=policy, scaler=scaler,
                    elem_feats_scaled=elem_feats_scaled, fraction_set=fraction_set,
                    device=device,
                    gen_epsilon=gen_epsilon,
                    gen_top_frac=gen_top_frac,
                    gen_temperature=gen_temperature,
                )
            else:
                _rollout_policy_episode(
                    env=env, qnet=qnet, scaler=scaler, device=device,
                    elem_feats_scaled=elem_feats_scaled, fraction_set=fraction_set,
                    gen_epsilon=gen_epsilon,
                    gen_top_frac=gen_top_frac,
                    gen_temperature=gen_temperature,
                )

            comp     = env.terminal_cation_fractions()
            comp_key = env.terminal_comp_key()

            if comp_key in seen_comp_keys:
                dup_rejected += 1
                continue
            seen_comp_keys.add(comp_key)

            # Reward from the episode (env.reward_fn already called in env.step).
            reward = float(env.path[-1].reward) if env.path else 0.0

            # Raw predictor values for CSV columns. OOHCatalystPredictor caches
            # internally so this is a cache hit when env.reward_fn already ran.
            if comp_key in predictor_raw_cache:
                raw_mean, std = predictor_raw_cache[comp_key]
            elif use_predict_raw:
                raw_mean, std = predictor.predict_raw(comp)
                predictor_raw_cache[comp_key] = (raw_mean, std)
            else:
                raw_mean, std = predictor.predict(comp)
                predictor_raw_cache[comp_key] = (raw_mean, std)

            row: dict = {
                "formula": env.terminal_formula,
                "reward": reward,
                "dp_mean": raw_mean,
                "dp_std": std,
                "dp_mean_minus_std": raw_mean - k * std,
            }
            if use_check_phase:
                phase_ok, phase_label = predictor.check_phase(comp)
                row["primary_ok"] = bool(phase_ok)
                row["primary_label"] = phase_label or ""

            rows.append(row)
            accepted += 1
            pbar.update(1)
            pbar.set_postfix(attempts=attempted, dups=dup_rejected)

        pbar.close()
        rate = accepted / max(attempted, 1)
        print(
            f"[INFO] Generated {accepted}/{n_target} {purpose} candidates "
            f"({attempted} attempts, {dup_rejected} dups, rate={rate:.3f})"
        )

    return rows
