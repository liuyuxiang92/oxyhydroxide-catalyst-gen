"""RL training functions extracted from the monolithic experiment script.

All functions are domain-agnostic: they operate on a :class:`CompositionEnv`
and a :class:`~rl_matdesign.predictors.base.PropertyPredictor` and contain
no material-system-specific logic.

Key improvements over the original implementation
--------------------------------------------------
- ``predictor: PropertyPredictor`` parameter replaces the hardcoded
  ``dp_predictor.predict_overpotential()`` call.
- Return normalisation in ``train_pg``: per-batch ``(G - mean) / (std + ε)``
  before computing the policy gradient reduces gradient variance, especially
  when the reward scale is unknown at the start of training.
- Dual-phase generation in ``generate_candidates``: a single call can produce
  both exploitation candidates (``mean_minus_kstd`` objective) and exploration
  candidates (``mean_plus_kstd`` objective, high model uncertainty → DFT
  validation targets).  Both pools are tagged with a ``purpose`` column.
- ``_comp_key`` is a module-level utility so it can be shared by both training
  and generation without re-definition.
"""
from __future__ import annotations

import math
import random
import warnings
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from .env import CompositionEnv
from .predictors.base import PropertyPredictor


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _comp_key(comp: Dict[str, float], total_units: int = 20) -> tuple:
    """Canonical hashable key for a terminal cation composition dict.

    Quantises each fraction to integer units, drops zero-unit elements, and
    sorts alphabetically.  Used for deduplication and visit-count tracking.
    """
    items = []
    for el, frac in comp.items():
        units = int(round(float(frac) * total_units))
        if units > 0:
            items.append((str(el), units))
    return tuple(sorted(items))


def objective_from_mean_std(
    mean: float, std: float, objective: str = "mean", k: float = 1.0
) -> float:
    """Compute scalar reward from predictor (mean, std) output.

    Parameters
    ----------
    mean, std:
        Output of ``PropertyPredictor.predict()``.
    objective:
        One of ``"mean"``, ``"mean_minus_kstd"`` (exploitation),
        ``"mean_plus_kstd"`` (exploration / uncertainty-driven).
    k:
        Coefficient for the std term.
    """
    if objective == "mean":
        return mean
    elif objective == "mean_minus_kstd":
        return mean - k * std
    elif objective == "mean_plus_kstd":
        return mean + k * std
    else:
        raise ValueError(f"Unknown objective '{objective}'.")


# ---------------------------------------------------------------------------
# Replay buffer helpers (DQN)
# ---------------------------------------------------------------------------

def extract_mc_q_targets(
    episode: List, gamma: float = 0.99
) -> Tuple[List[Tuple], List[float]]:
    """Compute Monte-Carlo Q-targets from an episode path.

    Parameters
    ----------
    episode:
        List of :class:`~rl_matdesign.env.EpisodeStep` objects.
    gamma:
        Discount factor.

    Returns
    -------
    (inputs, q_targets):
        ``inputs`` is a list of ``(s_mat, s_step, a_elem, a_comp)`` numpy arrays.
        ``q_targets`` is the corresponding list of scalar Q targets.
    """
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


# ---------------------------------------------------------------------------
# DQN training
# ---------------------------------------------------------------------------

def train_q(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    iteration: int = 0,
    loss_type: str = "mse",
) -> List[dict]:
    """Supervised regression on MC Q-targets (Adam).

    ``loss_type`` selects the regression loss: ``"mse"`` (default) or
    ``"smoothl1"`` (Huber, matches the npj DQN reference).
    """
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if loss_type == "smoothl1":
        loss_fn = torch.nn.SmoothL1Loss()
    else:
        loss_fn = torch.nn.MSELoss()
    metrics: List[dict] = []

    pbar = tqdm(range(epochs), desc="Q epochs")
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

        epoch_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        metrics.append({
            "phase": "dqn_train",
            "iteration": iteration,
            "epoch": epoch_idx + 1,
            "train_loss": epoch_loss,
        })
        pbar.set_postfix(train_loss=f"{epoch_loss:.4f}")
        tqdm.write(
            f"[DQN train] iter={iteration} epoch={epoch_idx+1}/{epochs} "
            f"| train_loss={epoch_loss:.4f}"
        )

    return metrics


def choose_action(
    *,
    model: torch.nn.Module,
    device: torch.device,
    s_material: np.ndarray,
    s_step: np.ndarray,
    allowed_actions: Sequence[Tuple[Tuple[float, ...], Tuple[float, ...]]],
    stochastic_top_frac: float = 0.0,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Select an action from *allowed_actions* using a Q-network.

    Parameters
    ----------
    stochastic_top_frac:
        If > 0, sample uniformly from the top-k fraction of actions by Q-value.
        If 0 (default), return the greedy argmax action.
    """
    if not allowed_actions:
        raise RuntimeError("No allowed actions.")

    a_elem = np.asarray([a[0] for a in allowed_actions], dtype=float)
    a_comp = np.asarray([a[1] for a in allowed_actions], dtype=float)
    n = len(allowed_actions)
    s_mat_batch = np.repeat(s_material.reshape(1, -1), n, axis=0)
    s_step_batch = np.repeat(s_step.reshape(1, -1), n, axis=0)

    with torch.no_grad():
        q = model(
            torch.tensor(s_mat_batch, dtype=torch.float32, device=device),
            torch.tensor(s_step_batch, dtype=torch.float32, device=device),
            torch.tensor(a_elem, dtype=torch.float32, device=device),
            torch.tensor(a_comp, dtype=torch.float32, device=device),
        ).reshape(-1)

    order = torch.argsort(q, descending=True).cpu().tolist()

    if stochastic_top_frac <= 0.0:
        return allowed_actions[int(order[0])]

    k = max(1, int(round(stochastic_top_frac * n)))
    idx = int(np.random.choice(order[:k]))
    return allowed_actions[idx]


# ---------------------------------------------------------------------------
# Iterative DQN refinement (on-policy data collection)
# ---------------------------------------------------------------------------

def iterate_dqn(
    *,
    env: CompositionEnv,
    qnet: torch.nn.Module,
    scaler: StandardScaler,
    device: torch.device,
    buffer_inputs: List[Tuple],
    buffer_targets: List[float],
    n_iters: int,
    eps_per_iter: int,
    buffer_cap: int,
    sample_per_iter: int,
    train_batch_size: int,
    epochs_per_iter: int,
    lr: float,
    gamma: float,
    initial_epsilon: float,
    epsilon_decay: float,
    top_frac: float,
    loss_type: str = "smoothl1",
    checkpoint_every: int = 0,
    checkpoint_path: Optional[str] = None,
) -> List[dict]:
    """Iterative on-policy DQN refinement, matching the npj reference loop.

    Per iteration:
      A) Collect ``eps_per_iter`` episodes with ε-greedy + top-``top_frac``
         sampling on the non-random branch; append step-level MC targets to
         the shared ``buffer_inputs`` / ``buffer_targets`` lists.
      B) FIFO-evict down to ``buffer_cap`` step-level datapoints.
      C) Uniformly sample ``sample_per_iter`` datapoints; keep the first
         ``train_batch_size`` as the training batch (remainder = held-out).
      D) Train the Q-network for ``epochs_per_iter`` full-batch passes.
      E) Decay ε by ``epsilon_decay``.

    The buffer lists are mutated in place; a per-iter metrics list is returned.
    """
    loss_fn = torch.nn.SmoothL1Loss() if loss_type == "smoothl1" else torch.nn.MSELoss()
    opt = torch.optim.Adam(qnet.parameters(), lr=lr)
    metrics: List[dict] = []

    epsilon = float(initial_epsilon)
    pbar = tqdm(range(n_iters), desc="DQN iterate")
    for iter_idx in pbar:
        # --- A. Collect new episodes with ε-greedy + top-k ---
        qnet.eval()
        ep_returns: List[float] = []
        for _ in range(eps_per_iter):
            env.initialize()
            for _ in range(env.n_components):
                allowed = env.allowed_actions()
                s_mat = scaler.transform(
                    env.state_featurizer(env.state).reshape(1, -1)
                )[0]
                s_step = np.zeros(env.n_components, dtype=float)
                if env.counter < env.n_components:
                    s_step[env.counter] = 1.0
                if float(np.random.rand()) < epsilon:
                    a = random.choice(allowed)
                else:
                    a = choose_action(
                        model=qnet, device=device,
                        s_material=s_mat, s_step=s_step,
                        allowed_actions=allowed,
                        stochastic_top_frac=top_frac,
                    )
                env.step(a)
            inp, tgt = extract_mc_q_targets(env.path, gamma=gamma)
            buffer_inputs.extend(inp)
            buffer_targets.extend(tgt)
            if tgt:
                ep_returns.append(float(tgt[0]))

        # --- B. FIFO eviction to buffer_cap ---
        if len(buffer_inputs) > buffer_cap:
            del buffer_inputs[: len(buffer_inputs) - buffer_cap]
            del buffer_targets[: len(buffer_targets) - buffer_cap]

        # --- C. Sample ``sample_per_iter`` datapoints ---
        N = len(buffer_inputs)
        k_sample = min(sample_per_iter, N)
        idx = np.random.choice(N, size=k_sample, replace=False)

        s_mat_arr = np.asarray([buffer_inputs[i][0] for i in idx], dtype=float)
        s_mat_arr = scaler.transform(s_mat_arr)
        s_step_arr = np.asarray([buffer_inputs[i][1] for i in idx], dtype=float)
        a_elem_arr = np.asarray([buffer_inputs[i][2] for i in idx], dtype=float)
        a_comp_arr = np.asarray([buffer_inputs[i][3] for i in idx], dtype=float)
        y_arr = np.asarray([buffer_targets[i] for i in idx], dtype=float)

        n_train = min(train_batch_size, k_sample)
        s_mat_t = torch.tensor(s_mat_arr[:n_train], dtype=torch.float32, device=device)
        s_step_t = torch.tensor(s_step_arr[:n_train], dtype=torch.float32, device=device)
        a_elem_t = torch.tensor(a_elem_arr[:n_train], dtype=torch.float32, device=device)
        a_comp_t = torch.tensor(a_comp_arr[:n_train], dtype=torch.float32, device=device)
        y_t = torch.tensor(y_arr[:n_train], dtype=torch.float32, device=device).unsqueeze(1)

        # --- D. Train qnet ``epochs_per_iter`` times on the train slice ---
        qnet.train()
        epoch_losses: List[float] = []
        for _ in range(epochs_per_iter):
            opt.zero_grad(set_to_none=True)
            pred = qnet(s_mat_t, s_step_t, a_elem_t, a_comp_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.item()))

        avg_loss = float(np.mean(epoch_losses))
        mean_return = float(np.mean(ep_returns)) if ep_returns else float("nan")

        metrics.append({
            "phase": "dqn_iterate",
            "iteration": iter_idx + 1,
            "epsilon": epsilon,
            "buffer_size": N,
            "train_loss": avg_loss,
            "mean_return": mean_return,
        })
        pbar.set_postfix(
            eps=f"{epsilon:.3f}", buf=N,
            loss=f"{avg_loss:.4f}", ret=f"{mean_return:.2f}",
        )

        # --- E. Decay ε ---
        epsilon *= epsilon_decay

        if (checkpoint_every > 0 and checkpoint_path
                and (iter_idx + 1) % checkpoint_every == 0):
            torch.save(qnet.state_dict(), checkpoint_path)

    pbar.close()
    return metrics


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
    stochastic_top_frac: float = 0.0,
    online_epsilon: float = 0.0,
) -> None:
    """Roll out one episode guided by a Q-network (with optional ε-greedy)."""
    env.initialize()
    for _ in range(env.n_components):
        allowed = env.allowed_actions()
        s_mat = scaler.transform(env.state_featurizer(env.state).reshape(1, -1))[0]
        s_step = np.zeros(env.n_components, dtype=float)
        if env.counter < env.n_components:
            s_step[env.counter] = 1.0

        if online_epsilon > 0.0 and float(np.random.rand()) < online_epsilon:
            a = random.choice(allowed)
        else:
            a = choose_action(
                model=qnet,
                device=device,
                s_material=s_mat,
                s_step=s_step,
                allowed_actions=allowed,
                stochastic_top_frac=(0.0 if online_epsilon > 0.0 else stochastic_top_frac),
            )
        env.step(a)


def _fit_scaler_from_warmup(env: CompositionEnv, n_warmup_eps: int) -> StandardScaler:
    """Fit a StandardScaler on material features from random warmup episodes.

    The reward function is temporarily replaced with a no-op so warmup does not
    invoke the (potentially expensive) PropertyPredictor.
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
    device: torch.device,
    pg_epsilon: float = 0.0,
) -> None:
    """Roll out one episode using the PolicyNet (softmax sampling)."""
    env.initialize()
    for _ in range(env.n_components):
        allowed = env.allowed_actions()
        s_mat = scaler.transform(env.state_featurizer(env.state).reshape(1, -1))[0]
        s_step = np.zeros(env.n_components, dtype=float)
        if env.counter < env.n_components:
            s_step[env.counter] = 1.0

        if pg_epsilon > 0.0 and float(np.random.rand()) < pg_epsilon:
            a = random.choice(allowed)
        else:
            n = len(allowed)
            a_elem_batch = np.asarray([a[0] for a in allowed], dtype=float)
            a_comp_batch = np.asarray([a[1] for a in allowed], dtype=float)
            s_mat_batch = np.repeat(s_mat.reshape(1, -1), n, axis=0)
            s_step_batch = np.repeat(s_step.reshape(1, -1), n, axis=0)
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


# ---------------------------------------------------------------------------
# Policy gradient training (REINFORCE / A2C)
# ---------------------------------------------------------------------------

def train_pg(
    *,
    policy: torch.nn.Module,
    value_net: Optional[torch.nn.Module],
    env: CompositionEnv,
    scaler: StandardScaler,
    device: torch.device,
    n_episodes: int,
    gamma: float = 0.99,
    lr_actor: float = 1e-3,
    lr_critic: float = 1e-3,
    entropy_coef: float = 0.01,
    pg_epsilon: float = 0.0,
    rl_method: str = "a2c",
    repeat_penalty_coef: float = 0.0,
    repeat_penalty_shape: str = "log",
    normalise_returns: bool = True,
    max_train_attempts: Optional[int] = None,
) -> List[dict]:
    """Online REINFORCE or A2C training loop.

    Improvements over the original implementation
    ----------------------------------------------
    ``normalise_returns`` (new, default True):
        Normalise returns per-episode as ``(G - mean(G)) / (std(G) + 1e-8)``
        before computing the policy gradient.  This reduces gradient variance
        substantially when the reward scale is unknown at training start and is
        a standard PG improvement missing from the original script.

    The *actor* gradient uses shaped returns (with repeat penalty).
    The *critic* (A2C only) learns on raw returns so V(s) stays stationary.
    """
    policy.train()
    opt_actor = torch.optim.Adam(policy.parameters(), lr=lr_actor)
    if value_net is not None:
        value_net.train()
        opt_critic = torch.optim.Adam(value_net.parameters(), lr=lr_critic)
    else:
        opt_critic = None

    visit_counts: Counter = Counter()
    metrics: List[dict] = []
    _roll_returns: List[float] = []
    _roll_actor: List[float] = []
    _roll_entropy: List[float] = []
    _roll_critic: List[float] = []
    _PRINT_INTERVAL = 50

    accepted = 0
    attempted = 0

    pbar = tqdm(total=n_episodes, desc=f"{rl_method.upper()} training")
    while accepted < n_episodes and (max_train_attempts is None or attempted < max_train_attempts):
        attempted += 1
        _rollout_pg_episode(env=env, policy=policy, scaler=scaler, device=device, pg_epsilon=pg_epsilon)

        path = env.path
        if not path:
            continue

        # Monte-Carlo returns (backwards pass).
        G = 0.0
        returns: List[float] = []
        for step in reversed(path):
            G = float(step.reward) + gamma * G
            returns.append(G)
        returns.reverse()
        episode_return = returns[0] if returns else 0.0

        # Return normalisation (new): reduces gradient variance.
        if normalise_returns and len(returns) > 1:
            ret_arr = np.asarray(returns, dtype=float)
            returns_norm = ((ret_arr - ret_arr.mean()) / (ret_arr.std() + 1e-8)).tolist()
        else:
            returns_norm = returns

        # Repeat penalty on the terminal composition key.
        terminal_comp_key = env.terminal_comp_key()
        n_visits_before = visit_counts[terminal_comp_key]
        if repeat_penalty_coef > 0.0:
            if repeat_penalty_shape == "log":
                repeat_penalty = repeat_penalty_coef * math.log1p(n_visits_before)
            elif repeat_penalty_shape == "sqrt":
                repeat_penalty = repeat_penalty_coef * math.sqrt(n_visits_before)
            else:
                repeat_penalty = repeat_penalty_coef * float(n_visits_before)
        else:
            repeat_penalty = 0.0
        visit_counts[terminal_comp_key] += 1

        returns_shaped = [g - repeat_penalty for g in returns_norm]

        actor_losses: List[torch.Tensor] = []
        entropy_terms: List[torch.Tensor] = []
        critic_losses: List[torch.Tensor] = []

        for step, G_t_raw, G_t_shaped in zip(path, returns, returns_shaped):
            allowed = step.allowed_actions
            if not allowed:
                continue

            s_mat = scaler.transform(
                np.asarray(step.state_material_features, dtype=float).reshape(1, -1)
            )[0]
            s_step = np.asarray(step.state_step_onehot, dtype=float)

            n = len(allowed)
            a_elem_batch = np.asarray([a[0] for a in allowed], dtype=float)
            a_comp_batch = np.asarray([a[1] for a in allowed], dtype=float)
            s_mat_batch = np.repeat(s_mat.reshape(1, -1), n, axis=0)
            s_step_batch = np.repeat(s_step.reshape(1, -1), n, axis=0)

            s_mat_t = torch.tensor(s_mat_batch, dtype=torch.float32, device=device)
            s_step_t = torch.tensor(s_step_batch, dtype=torch.float32, device=device)
            a_elem_t = torch.tensor(a_elem_batch, dtype=torch.float32, device=device)
            a_comp_t = torch.tensor(a_comp_batch, dtype=torch.float32, device=device)

            logits = policy(s_mat_t, s_step_t, a_elem_t, a_comp_t).reshape(-1)
            log_probs = torch.log_softmax(logits, dim=0)
            probs = torch.softmax(logits, dim=0)

            # Find index of taken action.
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

            G_shaped_t = torch.tensor(G_t_shaped, dtype=torch.float32, device=device)
            G_raw_t = torch.tensor(G_t_raw, dtype=torch.float32, device=device)

            if value_net is not None:
                s_mat_single = torch.tensor(s_mat.reshape(1, -1), dtype=torch.float32, device=device)
                s_step_single = torch.tensor(s_step.reshape(1, -1), dtype=torch.float32, device=device)
                value = value_net(s_mat_single, s_step_single).reshape(-1)[0]
                advantage = G_shaped_t - value.detach()
                # Critic learns on raw returns so V(s) stays stationary.
                critic_losses.append((value - G_raw_t) ** 2)
            else:
                advantage = G_shaped_t

            actor_losses.append(-log_probs[taken_idx] * advantage)
            entropy_terms.append(-(probs * log_probs).sum())

        if not actor_losses:
            continue

        actor_loss = torch.stack(actor_losses).mean()
        entropy_bonus = torch.stack(entropy_terms).mean()
        total_actor_loss = actor_loss - entropy_coef * entropy_bonus

        opt_actor.zero_grad(set_to_none=True)
        total_actor_loss.backward()
        opt_actor.step()

        ep_critic_loss: float | str = ""
        if opt_critic is not None and critic_losses:
            critic_loss = torch.stack(critic_losses).mean()
            opt_critic.zero_grad(set_to_none=True)
            critic_loss.backward()
            opt_critic.step()
            ep_critic_loss = float(critic_loss.item())

        ep_actor_loss = float(actor_loss.item())
        ep_entropy = float(entropy_bonus.item())
        accepted += 1
        pbar.update(1)

        metrics.append({
            "phase": "pg_train",
            "episode": accepted,
            "formula": env.terminal_formula,
            "return": episode_return,
            "return_shaped": returns_shaped[0] if returns_shaped else 0.0,
            "repeat_penalty": repeat_penalty,
            "visit_count_before": n_visits_before,
            "unique_comps_seen": len(visit_counts),
            "actor_loss": ep_actor_loss,
            "entropy": ep_entropy,
            "critic_loss": ep_critic_loss,
        })

        _roll_returns.append(episode_return)
        _roll_actor.append(ep_actor_loss)
        _roll_entropy.append(ep_entropy)
        if ep_critic_loss != "":
            _roll_critic.append(float(ep_critic_loss))

        for lst in (_roll_returns, _roll_actor, _roll_entropy):
            if len(lst) > _PRINT_INTERVAL:
                lst.pop(0)
        if _roll_critic and len(_roll_critic) > _PRINT_INTERVAL:
            _roll_critic.pop(0)

        if accepted % _PRINT_INTERVAL == 0:
            mean_ret = float(np.mean(_roll_returns))
            mean_al = float(np.mean(_roll_actor))
            mean_ent = float(np.mean(_roll_entropy))
            pbar.set_postfix(ret=f"{mean_ret:.3f}", actor=f"{mean_al:.3f}", ent=f"{mean_ent:.3f}")
            suffix = (
                f" | critic_loss={float(np.mean(_roll_critic)):.4f}" if _roll_critic else ""
            )
            tqdm.write(
                f"[{rl_method.upper()}] ep={accepted}/{n_episodes} | "
                f"return={mean_ret:.3f} | actor_loss={mean_al:.3f} | entropy={mean_ent:.3f}{suffix}"
            )

    pbar.close()
    print(f"[INFO] PG training: accepted {accepted}/{n_episodes} in {attempted} attempts.")
    return metrics


# ---------------------------------------------------------------------------
# Candidate generation (general, dual-phase)
# ---------------------------------------------------------------------------

def _pg_single_episode_generate(
    *,
    env: CompositionEnv,
    policy: torch.nn.Module,
    scaler: StandardScaler,
    device: torch.device,
    stochastic: bool,
    temperature: float,
) -> None:
    """Single generation episode for PG methods."""
    env.initialize()
    for _ in range(env.n_components):
        allowed = env.allowed_actions()
        s_mat = scaler.transform(env.state_featurizer(env.state).reshape(1, -1))[0]
        s_step = np.zeros(env.n_components, dtype=float)
        if env.counter < env.n_components:
            s_step[env.counter] = 1.0

        n = len(allowed)
        a_elem_batch = np.asarray([a[0] for a in allowed], dtype=float)
        a_comp_batch = np.asarray([a[1] for a in allowed], dtype=float)
        s_mat_batch = np.repeat(s_mat.reshape(1, -1), n, axis=0)
        s_step_batch = np.repeat(s_step.reshape(1, -1), n, axis=0)
        with torch.no_grad():
            logits = policy(
                torch.tensor(s_mat_batch, dtype=torch.float32, device=device),
                torch.tensor(s_step_batch, dtype=torch.float32, device=device),
                torch.tensor(a_elem_batch, dtype=torch.float32, device=device),
                torch.tensor(a_comp_batch, dtype=torch.float32, device=device),
            ).reshape(-1)

        if stochastic:
            probs = torch.softmax(logits / temperature, dim=0).cpu().numpy()
            idx = int(np.random.choice(n, p=probs))
        else:
            idx = int(torch.argmax(logits).item())
        env.step(allowed[idx])


def generate_candidates(
    *,
    env: CompositionEnv,
    predictor: PropertyPredictor,
    scaler: StandardScaler,
    device: torch.device,
    # Policy network (PG methods) or Q-network (DQN).
    policy: Optional[torch.nn.Module] = None,
    qnet: Optional[torch.nn.Module] = None,
    # Generation settings.
    n_exploit: int = 200,
    n_explore: int = 0,
    stochastic: bool = True,
    temperature: float = 1.0,
    stochastic_top_frac: float = 0.0,
    exploit_objective: str = "mean_minus_kstd",
    explore_objective: str = "mean_plus_kstd",
    k: float = 1.0,
    max_attempts: Optional[int] = None,
) -> List[dict]:
    """Generate candidate compositions in dual-phase mode.

    Produces two pools from the same trained policy:
    - **Exploitation** (``purpose="exploit"``): high predicted reward, good
      candidates for synthesis.
    - **Exploration** (``purpose="explore"``): high model uncertainty, good
      candidates for DFT validation to improve the DPA model.

    Both pools are returned in a single list with a ``purpose`` column.
    Duplicates are rejected globally across both pools.

    Parameters
    ----------
    n_exploit:
        Number of exploitation candidates to generate.
    n_explore:
        Number of exploration candidates (0 = exploitation only).
    exploit_objective / explore_objective:
        Reward objective used to compute the ``reward`` column for each pool.
        Exploration uses ``"mean_plus_kstd"`` by default (UCB-style).
    """
    if policy is None and qnet is None:
        raise ValueError("Provide either policy (PG) or qnet (DQN).")
    use_pg = policy is not None

    if use_pg:
        policy.eval()
    else:
        qnet.eval()

    predictor_cache: Dict[tuple, Tuple[float, float]] = {}
    seen_comp_keys: set = set()
    rows: List[dict] = []

    phases = []
    if n_exploit > 0:
        phases.append(("exploit", n_exploit, exploit_objective))
    if n_explore > 0:
        phases.append(("explore", n_explore, explore_objective))

    for purpose, n_target, objective in phases:
        accepted = 0
        attempted = 0
        dup_rejected = 0
        max_att = max_attempts or (n_target * 20)

        pbar = tqdm(total=n_target, desc=f"Generate [{purpose}]")
        while accepted < n_target and attempted < max_att:
            attempted += 1

            if use_pg:
                _pg_single_episode_generate(
                    env=env, policy=policy, scaler=scaler, device=device,
                    stochastic=stochastic, temperature=temperature,
                )
            else:
                _rollout_policy_episode(
                    env=env, qnet=qnet, scaler=scaler, device=device,
                    stochastic_top_frac=stochastic_top_frac,
                )

            comp = env.terminal_cation_fractions()
            comp_key = env.terminal_comp_key()

            if comp_key in seen_comp_keys:
                dup_rejected += 1
                continue
            seen_comp_keys.add(comp_key)

            if comp_key in predictor_cache:
                mean, std = predictor_cache[comp_key]
            else:
                mean, std = predictor.predict(comp)
                predictor_cache[comp_key] = (mean, std)

            reward = objective_from_mean_std(mean, std, objective, k)

            rows.append({
                "formula": env.terminal_formula,
                "purpose": purpose,
                "reward": reward,
                "dp_mean": mean,
                "dp_std": std,
                "dp_mean_minus_std": mean - k * std,
                "dp_mean_plus_std": mean + k * std,
            })
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
