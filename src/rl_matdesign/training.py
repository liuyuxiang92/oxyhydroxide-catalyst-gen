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
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .env import CompositionEnv
from .predictors.base import PropertyPredictor


# ---------------------------------------------------------------------------
# Candidate-evaluation failures (build/predictor errors unrelated to whether
# the composition itself is valid — e.g. every random MD/relax realization
# failing for a structure builder). Raised by ``reward_fn``/``mg_reward_fn``
# (scripts/run_experiment.py); rollout call sites below catch it and retry
# with a fresh candidate instead of either crashing the run or burning a
# real episode-budget slot on a candidate that was never actually scored.
# ---------------------------------------------------------------------------

class CandidateEvaluationFailed(Exception):
    """A well-formed candidate's predictor/builder call failed; retry, don't
    count it against the episode budget."""


# How many consecutive candidates may fail evaluation before a rollout call
# site gives up and lets the failure surface as a hard error. A handful of
# retries absorbs occasional transient/geometric failures (e.g. a random
# interstitial placement, or a random solid-solution decoration, that happens
# to destabilize an MD run); exhausting this many in a row means something
# systemic is wrong (a broken binary path, a missing model file, ...) that no
# amount of retrying will fix — better to fail loudly than spend the rest of
# the training budget on candidates that never get scored.
_MAX_CANDIDATE_RETRIES = 10


def _rollout_with_retry(rollout_fn: Callable[[], None]) -> None:
    """Call ``rollout_fn()`` (which must drive one full episode via
    ``env.initialize()`` + ``env.step()`` to a terminal reward); on
    :class:`CandidateEvaluationFailed`, discard the attempt and retry with a
    fresh candidate — up to :data:`_MAX_CANDIDATE_RETRIES` times — so the
    caller's episode-budget counter never advances for a candidate that
    failed evaluation.
    """
    for attempt in range(_MAX_CANDIDATE_RETRIES + 1):
        try:
            rollout_fn()
            return
        except CandidateEvaluationFailed as exc:
            if attempt == _MAX_CANDIDATE_RETRIES:
                raise RuntimeError(
                    f"{_MAX_CANDIDATE_RETRIES + 1} consecutive candidates all "
                    "failed evaluation -- giving up rather than spending the "
                    f"rest of the run on unscored candidates. Last failure: {exc}"
                ) from exc
            print(f"[WARN] candidate evaluation failed (retry "
                  f"{attempt + 1}/{_MAX_CANDIDATE_RETRIES}), discarding and "
                  f"trying a new candidate: {exc}")


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
    species_set: List[str],
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
        Index ``i`` corresponds to ``species_set[i]``.
    """
    raw = np.asarray([featurizer(el + "1.00") for el in species_set], dtype=float)
    elem_scaler = StandardScaler()
    elem_scaler.fit(raw)
    return elem_scaler.transform(raw), elem_scaler


# ---------------------------------------------------------------------------
# Replay buffer helpers (classical DQN)
# ---------------------------------------------------------------------------

# --- BEGIN mc-target experiment (removable) ---
def _attach_mc_returns(rows: list, path: list, gamma: float) -> None:
    """Attach the discounted Monte-Carlo return G to each buffer row in place.

    Non-bootstrap regression target, used only when ``dqn_target_mode == 'mc'``
    (mirrors the reference npj DQN). ``rows`` are in episode order, aligned 1:1
    with ``path``. With a terminal-only reward this reduces to
    ``G_t = gamma**(T-1-t) * R_terminal``; the backward recursion below stays
    correct even if a config ever emits intermediate rewards.
    """
    G = 0.0
    for k in reversed(range(len(path))):
        G = float(path[k].reward) + gamma * G   # terminal step: G = reward
        rows[k]["mc_return"] = G
# --- END mc-target experiment ---


def add_episode_to_buffer(
    path: list,
    buffer: collections.deque,
    elem_feats_scaled: np.ndarray,
    fraction_set: List[str],
    gamma: float = 0.9,
) -> None:
    """Convert a completed episode path into buffer rows and append.

    Each row stores:
    - Raw (unscaled) s_mat and s_mat_next for lazy scaling at batch time.
    - Element index and comp scalar value instead of one-hot vectors.
    - ``next_allowed_idx``: list of (elem_idx, comp_idx) pairs for the next
      step, used to compute the max-Q bootstrap target.
    """
    rows: List[dict] = []
    for k, step in enumerate(path):
        done = k == len(path) - 1
        next_step = path[k + 1] if not done else None
        next_allowed_idx: List[Tuple[int, int]] = []
        if not done:
            for (elem_oh, comp_oh) in next_step.allowed_actions:
                next_allowed_idx.append(
                    (int(np.argmax(elem_oh)), int(np.argmax(comp_oh)))
                )
        rows.append({
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
    # --- BEGIN mc-target experiment (removable) ---
    _attach_mc_returns(rows, path, gamma)
    # --- END mc-target experiment ---
    buffer.extend(rows)


def _row_fingerprint(row: dict) -> tuple:
    """Hashable fingerprint of a buffer row, used for within-episode dedup."""
    return (
        tuple(np.round(row["s_mat_raw"], 6).tolist()),
        tuple(row["s_step"].tolist()),
        int(row["a_elem_idx"]),
        float(row["a_comp_val"]),
        bool(row["done"]),
    )


def _detect_last_position_pin(env) -> Optional[set]:
    """If env.phase_filter pins certain elements at the terminal action,
    return the set of pinned element symbols. Otherwise return None.

    Walks ``ChainConstraintFilter.children`` so a filter nested inside a
    chain is detected correctly.
    """
    from .constraints.last_step_element import LastStepElementFilter

    if env.phase_filter is None:
        return None

    def _walk(f):
        if isinstance(f, LastStepElementFilter):
            if f.reserve_for_last:
                return set(f.required_elements)
            return None
        if hasattr(f, "children"):
            for child in f.children:
                pinned = _walk(child)
                if pinned is not None:
                    return pinned
        return None

    return _walk(env.phase_filter)


def _path_to_rows(path: list, fraction_set: List[str]) -> List[dict]:
    """Convert an env.path into row dicts (matching add_episode_to_buffer)."""
    rows: List[dict] = []
    n = len(path)
    for k, step in enumerate(path):
        done = k == n - 1
        next_step = path[k + 1] if not done else None
        next_allowed_idx: List[Tuple[int, int]] = []
        if not done:
            for (elem_oh, comp_oh) in next_step.allowed_actions:
                next_allowed_idx.append(
                    (int(np.argmax(elem_oh)), int(np.argmax(comp_oh)))
                )
        rows.append({
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
    return rows


# Module-level guards so the "no alternatives possible" / "non-DQN method"
# warnings fire only once per training run, not once per episode.
_AUG_WARNED_NO_ALTS = False
_AUG_WARNED_FIXED_ORDER = False


def _augment_episode_in_buffer(
    env,
    original_path: list,
    buffer: collections.deque,
    elem_feats_scaled: np.ndarray,
    fraction_set: List[str],
    *,
    K: int,
) -> None:
    """Insert *original_path* plus up to *K* permutation-augmented copies.

    Always inserts the original first (preserving exact ``K=0`` behavior).
    For each augmentation, re-drives the env with a permuted action sequence
    using ``env.step``, which keeps featurization, allowed_actions, and the
    constraint-filter outputs consistent with production code. The expensive
    predictor call is avoided by temporarily swapping ``env.reward_fn`` for
    a constant returning the original terminal reward.

    Dedup is scoped to this episode's augmentation pass: rows whose
    fingerprint already exists in the original or earlier-augmented rows
    of *this* episode are skipped. Cross-episode coincidences are kept,
    since they are legitimate independent observations.
    """
    global _AUG_WARNED_NO_ALTS, _AUG_WARNED_FIXED_ORDER

    # Always insert the original path first (zero risk when K == 0).
    add_episode_to_buffer(original_path, buffer, elem_feats_scaled, fraction_set)
    if K <= 0:
        return

    # episode_style: 'fixed_order_amount' forces a single ordering — no permutation valid.
    episode_style = getattr(env, "episode_style", None)
    if episode_style == "fixed_order_amount":
        if not _AUG_WARNED_FIXED_ORDER:
            print(
                "[WARN] dqn_augment_permutations is a no-op when "
                "episode_style='fixed_order_amount' (element order is forced).",
                flush=True,
            )
            _AUG_WARNED_FIXED_ORDER = True
        return

    N = len(original_path)
    if N < 2:
        return

    # Detect whether a LastStepElementFilter pins the terminal action.
    pinned_last_elements = _detect_last_position_pin(env)
    if pinned_last_elements is not None:
        permutable = list(range(N - 1))   # keep position N-1 fixed
    else:
        permutable = list(range(N))

    import math
    import itertools

    if len(permutable) < 2:
        if not _AUG_WARNED_NO_ALTS:
            print(
                f"[WARN] dqn_augment_permutations: only {math.factorial(len(permutable))} "
                "permutation(s) of the permutable positions; augmentation is a no-op "
                "for this episode shape.",
                flush=True,
            )
            _AUG_WARNED_NO_ALTS = True
        return

    perm_count = math.factorial(len(permutable))
    alternatives_count = perm_count - 1
    if alternatives_count == 0:
        return
    num_to_sample = min(K, alternatives_count)
    if K > alternatives_count and not _AUG_WARNED_NO_ALTS:
        print(
            f"[WARN] dqn_augment_permutations: requested K={K} but only "
            f"{alternatives_count} alternative permutation(s) exist; using "
            f"K={alternatives_count}.",
            flush=True,
        )
        _AUG_WARNED_NO_ALTS = True

    all_perms = list(itertools.permutations(permutable))
    identity = tuple(permutable)
    all_perms.remove(identity)
    random.shuffle(all_perms)
    chosen_perms = all_perms[:num_to_sample]

    # Build the action sequence from original_path.
    original_actions = [
        (step.action_elem_onehot, step.action_comp_onehot)
        for step in original_path
    ]

    # Build fingerprint set seeded with the original rows we just inserted.
    seen_fingerprints: set = set()
    for row in list(buffer)[-N:]:
        seen_fingerprints.add(_row_fingerprint(row))

    # Stash the original terminal reward and predictor.
    original_R = float(original_path[-1].reward)
    saved_reward_fn = env.reward_fn

    try:
        env.reward_fn = lambda _formula: original_R
        for perm in chosen_perms:
            if pinned_last_elements is not None:
                permuted_actions = [original_actions[i] for i in perm] + [original_actions[-1]]
            else:
                permuted_actions = [original_actions[i] for i in perm]

            env.initialize()
            try:
                for action in permuted_actions:
                    env.step(action)
            except (ValueError, RuntimeError) as _e:
                # An ordering that the env rejects (e.g., a constraint we
                # didn't anticipate): silently skip this augmentation.
                continue

            for row in _path_to_rows(env.path, fraction_set):
                fp = _row_fingerprint(row)
                if fp in seen_fingerprints:
                    continue
                seen_fingerprints.add(fp)
                buffer.append(row)
    finally:
        env.reward_fn = saved_reward_fn


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
    target_mode: str = "bootstrap",
) -> float:
    """One DQN gradient step: SmoothL1(Q(s,a), target).

    ``target_mode='bootstrap'`` (default) uses the one-step TD target
    ``r + gamma*max_a' Q_target(s',a')``. ``target_mode='mc'`` regresses to the
    fixed discounted Monte-Carlo return stored on each row (no bootstrap).
    """
    # --- BEGIN mc-target experiment (removable) ---
    if target_mode == "mc":
        y = [float(r["mc_return"]) for r in batch]
    else:
    # --- END mc-target experiment ---
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
    dqn_num_train_eps: int = 20000,
    dqn_buffer_size: int = 50000,
    batch_size: int = 256,
    dqn_grad_steps_per_ep: int = 5,
    dqn_target_update_freq: int = 100,
    dqn_eps_anneal_eps: int = 10000,
    dqn_eps_min: float = 0.05,
    gamma: float = 0.9,
    lr: float = 1e-3,
    loss_name: str = "smoothl1",
    checkpoint_cfg: Optional[dict] = None,
    resume_state: Optional[dict] = None,
    augment_permutations: int = 0,
    timer: Optional[object] = None,
    dqn_target_mode: str = "bootstrap",
) -> Tuple[torch.nn.Module, StandardScaler, List[dict]]:
    """Classical online DQN with FIFO replay buffer and TD targets.

    Phase 0 — Warmup (skipped when ``resume_state`` is provided):
        Roll out ``n_warmup_eps`` random episodes with real rewards to pre-fill
        the buffer and fit the s_mat StandardScaler.

    Phase 1 — Training loop (``dqn_num_train_eps`` episodes):
        Each episode:
        1. ε-greedy rollout → 5 new buffer rows.
        2. ``dqn_grad_steps_per_ep`` gradient steps, each sampling ``batch_size``
           rows from the full buffer (SmoothL1 loss on TD targets).
        3. Hard-copy qnet → target_net every ``dqn_target_update_freq`` episodes.
        4. Linear epsilon anneal: ε = max(dqn_eps_min, 1 − ep / dqn_eps_anneal_eps).

    Resume:
        Pass ``resume_state`` with pre-built ``{scaler, qnet, target_net,
        optimizer, buffer, start_ep, eps}`` to skip warmup and continue from
        a prior run. The loop iterates over ``range(start_ep, dqn_num_train_eps)``
        so the ε schedule (function of ``ep``) resumes correctly.

    Returns
    -------
    (qnet, scaler, metrics):
        Trained Q-network, fitted s_mat scaler, and list of per-episode metric
        dicts (phase="dqn_train").

    ``timer``:
        Optional :class:`~rl_matdesign.utils.timing.PredictorTimer`. When given,
        every per-episode row also carries cumulative wall-clock and
        predictor-call counters, which is what makes a best-reward-vs-time curve
        plottable after the run.
    """
    from .model import QRegressor

    # --- BEGIN mc-target experiment (removable) ---
    if dqn_target_mode == "mc":
        print(
            "[INFO] DQN target_mode='mc': regressing to fixed discounted "
            "Monte-Carlo returns (no bootstrap); target network and "
            "dqn_target_update_freq are inert.",
            flush=True,
        )
        if augment_permutations > 0:
            raise ValueError(
                "dqn_target_mode='mc' does not support permutation augmentation "
                "(augment_permutations>0); the mc-return field is only attached on "
                "the non-augmented path. Set dqn_augment_permutations=0 for the "
                "mc comparison run."
            )
    # --- END mc-target experiment ---

    # Warmup pays one real predictor call per episode (see below), so those
    # episodes are part of the training cost and belong in the log. Collected
    # here because `metrics` isn't built until after the warmup branch.
    warmup_rows: List[dict] = []

    if resume_state is not None:
        scaler     = resume_state["scaler"]
        qnet       = resume_state["qnet"]
        target_net = resume_state["target_net"]
        optimizer  = resume_state["optimizer"]
        buffer     = resume_state["buffer"]
        start_ep   = int(resume_state.get("start_ep", 0))
        eps        = float(resume_state.get(
            "eps", max(dqn_eps_min, 1.0 - start_ep / dqn_eps_anneal_eps),
        ))
        loss_fn    = _make_loss_fn(loss_name)
        print(
            f"[INFO] DQN resume: start_ep={start_ep}, eps={eps:.4f}, "
            f"buffer_rows={len(buffer)}",
            flush=True,
        )
    else:
        buffer: collections.deque = collections.deque(maxlen=dqn_buffer_size)

        if augment_permutations > 0:
            print(
                f"[INFO] DQN warmup: {n_warmup_eps} random episodes with real rewards "
                f"(K={augment_permutations} permutation augmentations per episode)...",
                flush=True,
            )
        else:
            print(f"[INFO] DQN warmup: {n_warmup_eps} random episodes with real rewards...")
        pbar = tqdm(total=n_warmup_eps, desc="DQN warmup")
        for _w in range(n_warmup_eps):
            _rollout_with_retry(lambda: _rollout_random_episode(env))
            if env.path:
                warmup_rows.append({
                    "phase": "dqn_warmup",
                    "episode": _w + 1,
                    "return": float(env.path[-1].reward),
                    "terminal_reward": float(env.path[-1].reward),
                    "epsilon": 1.0,          # warmup is uniformly random by definition
                    "terminal_comp_key": str(env.terminal_comp_key()),
                })
            if augment_permutations > 0:
                _augment_episode_in_buffer(
                    env, env.path, buffer, elem_feats_scaled, fraction_set,
                    K=augment_permutations,
                )
            else:
                add_episode_to_buffer(env.path, buffer, elem_feats_scaled, fraction_set, gamma=gamma)
            pbar.update(1)
        pbar.close()
        # Unlike PG warmup (which neutralises reward_fn), DQN warmup pays a real
        # predictor call per episode. The mark makes that cost visible.
        if timer is not None:
            timer.mark("warmup_end")

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
        start_ep = 0
        eps = 1.0

    ckpt_path: Optional[str] = None
    ckpt_freq: int = 0
    dp_cache_ref: Optional[dict] = None
    if checkpoint_cfg:
        ckpt_path    = checkpoint_cfg.get("path")
        ckpt_freq    = int(checkpoint_cfg.get("freq", 0))
        dp_cache_ref = checkpoint_cfg.get("dp_cache")

    metrics: List[dict] = []
    metrics.extend(warmup_rows)

    if start_ep >= dqn_num_train_eps:
        print(
            f"[WARN] DQN: start_ep ({start_ep}) >= dqn_num_train_eps ({dqn_num_train_eps}); "
            "skipping training loop.",
            flush=True,
        )
        return qnet, scaler, metrics

    pbar = tqdm(range(start_ep, dqn_num_train_eps), desc="DQN train")

    def _rollout_eps_greedy_episode() -> None:
        # Collect one episode with epsilon-greedy policy.
        env.initialize()
        for _t in range(env.n_components):
            _allowed = env.allowed_actions()
            _s_mat_sc = scaler.transform(
                env.current_state_features().reshape(1, -1))[0]
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
                )
            env.step(_a)

    for ep in pbar:
        qnet.eval()
        _rollout_with_retry(_rollout_eps_greedy_episode)

        episode_reward = float(env.path[-1].reward)
        if augment_permutations > 0:
            _augment_episode_in_buffer(
                env, env.path, buffer, elem_feats_scaled, fraction_set,
                K=augment_permutations,
            )
        else:
            add_episode_to_buffer(env.path, buffer, elem_feats_scaled, fraction_set, gamma=gamma)

        # 2. Gradient steps.
        mean_loss = float("nan")
        if len(buffer) >= batch_size:
            buf_list = list(buffer)
            losses = []
            for _ in range(dqn_grad_steps_per_ep):
                _batch = random.sample(buf_list, batch_size)
                losses.append(_dqn_gradient_step(
                    qnet, _batch, target_net, scaler,
                    elem_feats_scaled, fraction_set, optimizer, loss_fn, device, gamma,
                    target_mode=dqn_target_mode,
                ))
            mean_loss = float(np.mean(losses))

        # 3. Hard target-net copy.
        if (ep + 1) % dqn_target_update_freq == 0:
            target_net.load_state_dict(qnet.state_dict())
            target_net.eval()

        # 4. Linear epsilon anneal.
        eps = max(dqn_eps_min, 1.0 - ep / dqn_eps_anneal_eps)

        _row = {
            "phase": "dqn_train",
            "iteration": ep + 1,
            "episode": ep + 1,
            "return": episode_reward,
            # Identical to `return` for DQN (it already logs the undiscounted
            # terminal reward), but present so every phase carries the column that
            # is comparable across methods — see the PG rows, where they differ.
            "terminal_reward": episode_reward,
            "train_loss": mean_loss,
            "mse_loss": mean_loss,
            "epsilon": eps,
            "buffer_rows": len(buffer),
        }
        if timer is not None:
            _row.update(timer.snapshot())
        metrics.append(_row)
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
                "buffer": list(buffer),
                "dqn_buffer_size": dqn_buffer_size,
                "eps": eps,
                "dp_cache": dict(dp_cache_ref) if dp_cache_ref is not None else {},
            }, numbered_path=_numbered)
            tqdm.write(f"[INFO] DQN checkpoint saved at episode {ep + 1} → {_numbered}")

    pbar.close()
    return qnet, scaler, metrics


# ---------------------------------------------------------------------------
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

    Priority: ε-greedy > Boltzmann > top-k > greedy argmax (default).

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

    # Pure greedy argmax (default for DQN training rollouts).
    return allowed_actions[int(torch.argmax(q).item())]


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
        s_mat = scaler.transform(env.current_state_features().reshape(1, -1))[0]
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
        s_mat = scaler.transform(env.current_state_features().reshape(1, -1))[0]
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

# Entropy-floor controller. The failure it exists to prevent: with a fixed
# coefficient, how far the policy collapses is set by the *number of gradient
# updates*, so the same YAML that behaves at 100 updates goes deterministic at
# 3000 (observed: entropy 0.00 with one composition sampled 31,843 times out of
# 45,200 episodes). A proportional controller on normalised entropy removes that
# dependence on run length, so one setting serves every budget.
_ENTROPY_CTRL_GAIN = 2.0    # how hard coef_eff reacts to a floor violation
_ENTROPY_CTRL_MAX  = 100.0  # coef_eff ceiling, as a multiple of pg_entropy_coef


def entropy_coef_update(coef_eff: float, entropy_norm: float,
                        base_coef: float, floor: float) -> float:
    """One step of the entropy-floor controller.

    ``entropy_norm`` is ``H / ln|A|`` in [0, 1]; ``floor`` is ``pg_entropy_min``.
    Below the floor the weight grows, above it decays back toward ``base_coef``,
    which acts as the lower clamp so the controller can only ever *add* pressure
    to explore. ``floor <= 0`` disables it and pins the weight at ``base_coef``.
    """
    if floor <= 0.0:
        return base_coef
    coef_eff *= math.exp(_ENTROPY_CTRL_GAIN * (floor - entropy_norm))
    return min(max(coef_eff, base_coef), base_coef * _ENTROPY_CTRL_MAX)


def _episode_pg_terms(
    *,
    path,
    returns: List[float],
    policy: torch.nn.Module,
    value_net: Optional[torch.nn.Module],
    scaler: "StandardScaler",
    elem_feats_scaled: np.ndarray,
    fraction_set: List[str],
    device: "torch.device",
) -> tuple:
    """Build per-step policy-gradient *components* for one episode.

    Returns ``(logp_taken, advantage_raw, entropy, max_entropy, critic_loss)``
    rather than a finished actor loss. The advantage is deliberately left
    un-scaled and separate from ``logp`` so :func:`train_pg` can standardise it
    across the whole batch before forming ``-logp * adv`` — folding the two
    together here would make batch-level normalisation impossible.

    ``logp_taken`` and ``entropy`` carry autograd state; ``advantage_raw`` and
    ``max_entropy`` are plain floats (the advantage is a constant w.r.t. the
    actor by construction — ``G`` is data and ``V`` is detached).
    """
    logp_taken: List[torch.Tensor] = []
    advantages_raw: List[float] = []
    entropy_terms: List[torch.Tensor] = []
    max_entropies: List[float] = []
    critic_losses: List[torch.Tensor] = []

    for step, G_t in zip(path, returns):
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

        G_raw_t = torch.tensor(G_t, dtype=torch.float32, device=device)

        if value_net is not None:
            s_mat_single  = torch.tensor(s_mat.reshape(1, -1),  dtype=torch.float32, device=device)
            s_step_single = torch.tensor(s_step.reshape(1, -1), dtype=torch.float32, device=device)
            value = value_net(s_mat_single, s_step_single).reshape(-1)[0]
            # The critic regresses the *raw* return: it is a value function for the
            # real objective, not for the exploration-shaped one.
            critic_losses.append((value - G_raw_t) ** 2)
            advantage = G_t - float(value.detach().item())
        else:
            # REINFORCE: no baseline. Batch standardisation in train_pg supplies
            # the variance reduction a critic would otherwise give.
            advantage = G_t

        logp_taken.append(log_probs[taken_idx])
        advantages_raw.append(float(advantage))
        entropy_terms.append(-(probs * log_probs).sum())
        # Max entropy of this step's action set. |A| varies per step (the env
        # prunes infeasible actions), so the floor has to be measured against the
        # live action count, not a global constant.
        max_entropies.append(math.log(n) if n > 1 else 0.0)

    return logp_taken, advantages_raw, entropy_terms, max_entropies, critic_losses


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
    pg_entropy_coef: float = 0.01,
    pg_entropy_min: float = 0.3,
    rl_method: str = "a2c",
    pg_repeat_penalty_coef: float = 0.0,
    pg_repeat_penalty_shape: str = "log",
    max_train_attempts: Optional[int] = None,
    checkpoint_cfg: Optional[dict] = None,
    timer: Optional[object] = None,
) -> List[dict]:
    """Batched REINFORCE / A2C training loop (matches feat/classical-dqn).

    Structure:
        for it in range(num_iters):
            collect batch_eps episodes under current policy
            standardise advantages across the batch
            one optimizer step over the entire batch

    Scale invariance
    ----------------
    Advantages are **standardised across the batch** before forming the actor
    loss, unconditionally. Without it the actor term scales with the raw reward
    (sintering temperatures are 400-700, so |advantage| is O(hundreds)) while the
    entropy bonus is ``pg_entropy_coef * H`` with ``H <= ln|A| ~ 5.6``. The
    entropy term is then ~2 orders of magnitude too small to influence the update
    and the policy collapses to a single composition. After standardisation both
    ``pg_entropy_coef`` and ``pg_repeat_penalty_coef`` are expressed in standard
    deviations of batch return, so they mean the same thing in every scenario
    regardless of the property's units.

    ``pg_entropy_min`` is a floor on *normalised* entropy ``H / ln|A|`` in [0, 1],
    held by a proportional controller (see ``_ENTROPY_CTRL_GAIN``). Set it to 0 to
    disable the floor and use the bare ``pg_entropy_coef``.

    .. note::
       In the metrics rows ``repeat_penalty`` is now in σ, while ``return_shaped``
       converts it back to the property's units so it stays comparable against
       ``return_raw``. Logs written before this change are not comparable to
       either.
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

    # Effective entropy weight, carried across updates by the floor controller.
    # Restored on resume so a resumed run doesn't snap back to the base weight and
    # re-collapse the policy it had just re-opened.
    coef_eff = float(pg_entropy_coef)
    if checkpoint_cfg and checkpoint_cfg.get("entropy_coef_eff") is not None:
        coef_eff = float(checkpoint_cfg["entropy_coef_eff"])

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
        batch_terminal_keys: List[tuple] = []
        batch_repeat_penalties: List[float] = []
        batch_visits_before: List[int] = []
        # One record per *sampled episode*, so the PG training-reward distribution is
        # comparable with DQN's (which logs every episode). The pg_train row below
        # only carries the batch mean, which hides the low-temperature tail entirely.
        batch_ep_rows: List[dict] = []

        collected = 0
        while collected < int(batch_eps):
            attempted += 1
            if max_train_attempts is not None and attempted > max_train_attempts:
                break
            _rollout_with_retry(lambda: _rollout_pg_episode(
                env=env, policy=policy, scaler=scaler,
                elem_feats_scaled=elem_feats_scaled, fraction_set=fraction_set,
                device=device,
            ))
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
            if pg_repeat_penalty_coef > 0.0:
                if pg_repeat_penalty_shape == "log":
                    repeat_penalty = pg_repeat_penalty_coef * math.log1p(n_visits_before)
                elif pg_repeat_penalty_shape == "sqrt":
                    repeat_penalty = pg_repeat_penalty_coef * math.sqrt(n_visits_before)
                else:
                    repeat_penalty = pg_repeat_penalty_coef * float(n_visits_before)
            else:
                repeat_penalty = 0.0
            visit_counts[terminal_key] += 1
            # The penalty is applied *after* advantage standardisation (below), so
            # the coefficient is in σ of batch return rather than in the property's
            # own units. Subtracting it from the raw return here would make it a
            # no-op: 0.1*ln(1+4472) = 0.84 against returns of ~436 is 0.2%.

            batch_paths.append(path)
            batch_returns.append(returns)
            batch_terminal_keys.append(terminal_key)
            batch_repeat_penalties.append(repeat_penalty)
            batch_visits_before.append(n_visits_before)
            collected += 1
            accepted += 1
            batch_ep_rows.append({
                "phase": "pg_episode",
                "iteration": it + 1,
                "episode": accepted,
                "return": returns[0],
                "return_raw": returns[0],
                # `return` is the *discounted* return G_0 = gamma^(n-1) * r_T for a
                # terminal-only reward — the training signal, not a temperature.
                # DQN's `return` column is the undiscounted terminal reward, so the
                # two are NOT comparable without this column. Log it explicitly
                # rather than making every reader re-derive gamma from run_config.
                "terminal_reward": float(path[-1].reward),
                "repeat_penalty": repeat_penalty,
                "visit_count_before": n_visits_before,
                "terminal_comp_key": str(terminal_key),
            })

        if not batch_paths:
            continue

        all_logp: List[torch.Tensor] = []
        all_adv_raw: List[float] = []
        all_penalty: List[float] = []
        all_entropy: List[torch.Tensor] = []
        all_max_entropy: List[float] = []
        all_critic: List[torch.Tensor] = []
        for path, returns, penalty in zip(batch_paths, batch_returns, batch_repeat_penalties):
            lp, adv, e_terms, h_max, c_terms = _episode_pg_terms(
                path=path,
                returns=returns,
                policy=policy,
                value_net=value_net,
                scaler=scaler,
                elem_feats_scaled=elem_feats_scaled,
                fraction_set=fraction_set,
                device=device,
            )
            all_logp.extend(lp)
            all_adv_raw.extend(adv)
            all_penalty.extend([penalty] * len(adv))
            all_entropy.extend(e_terms)
            all_max_entropy.extend(h_max)
            all_critic.extend(c_terms)

        if not all_logp:
            continue

        # --- advantage standardisation (the whole point; see docstring) -------
        adv_raw = torch.tensor(all_adv_raw, dtype=torch.float32, device=device)
        adv_mean = float(adv_raw.mean().item())
        adv_std  = float(adv_raw.std(unbiased=False).item())
        if adv_std > 1e-8:
            adv = (adv_raw - adv_mean) / adv_std
        else:
            # Every episode in the batch scored identically — there is no signal to
            # extract. Dividing by an epsilon here would amplify float noise into a
            # full-magnitude gradient, so emit a zero update instead.
            adv = torch.zeros_like(adv_raw)
        # Repeat penalty now lives in σ units, applied to the standardised value.
        adv = adv - torch.tensor(all_penalty, dtype=torch.float32, device=device)

        logp_t        = torch.stack(all_logp)
        entropy_t     = torch.stack(all_entropy)
        actor_loss    = -(logp_t * adv).mean()
        entropy_bonus = entropy_t.mean()

        # --- entropy floor controller ----------------------------------------
        mean_max_entropy = float(np.mean(all_max_entropy)) if all_max_entropy else 0.0
        entropy_norm = (float(entropy_bonus.item()) / mean_max_entropy
                        if mean_max_entropy > 1e-8 else 0.0)
        coef_eff = entropy_coef_update(coef_eff, entropy_norm,
                                       pg_entropy_coef, pg_entropy_min)
        total_actor_loss = actor_loss - coef_eff * entropy_bonus

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
        mean_return_raw = float(np.mean([rs[0] for rs in batch_returns])) if batch_returns else 0.0
        # Undiscounted terminal reward, in the property's own units — the only
        # column comparable with DQN's `return` (see the pg_episode row above).
        mean_terminal_reward = (float(np.mean([float(p[-1].reward) for p in batch_paths]))
                                if batch_paths else 0.0)
        # In σ units now, like the penalty itself — see the class docstring.
        mean_return_shaped = mean_return_raw - float(np.mean(all_penalty)) * adv_std

        _row = {
            "phase": "pg_train",
            "iteration": it + 1,
            "update": update_idx,
            "episode": accepted,
            "batch_eps": int(batch_eps),
            "return": mean_return_raw,
            "return_raw": mean_return_raw,
            "return_shaped": mean_return_shaped,
            "terminal_reward": mean_terminal_reward,
            "repeat_penalty": float(np.mean(batch_repeat_penalties)) if batch_repeat_penalties else 0.0,
            "visit_count_before": int(np.max(batch_visits_before)) if batch_visits_before else 0,
            "unique_comps_seen": len(visit_counts),
            "max_visit_count": max(visit_counts.values()) if visit_counts else 0,
            "terminal_comp_key": str(batch_terminal_keys[-1]) if batch_terminal_keys else "",
            "actor_loss": ep_actor_loss,
            "entropy": ep_entropy,
            # entropy_norm is directly comparable to pg_entropy_min — this is the
            # column to watch for collapse, since raw entropy depends on |A|.
            "entropy_norm": entropy_norm,
            "entropy_coef_eff": coef_eff,
            "adv_mean_raw": adv_mean,
            "adv_std_raw": adv_std,
            "critic_loss": critic_loss_val,
        }
        if timer is not None:
            _row.update(timer.snapshot())
        metrics.append(_row)
        metrics.extend(batch_ep_rows)

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
                "entropy_coef_eff": coef_eff,
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

    Priority: ε-greedy > Boltzmann > top-k > greedy argmax (default).
    Default gen_temperature=1.0 prevents pure-greedy composition collapse.
    """
    env.initialize()
    for _ in range(env.n_components):
        allowed = env.allowed_actions()
        s_mat = scaler.transform(env.current_state_features().reshape(1, -1))[0]
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
            # Pure greedy argmax — default when no gen strategy is set.
            idx = int(torch.argmax(logits).item())
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
    k: float = 1.0,
    max_attempts: Optional[int] = None,
    charge_filter: bool = False,
    charge_use_pauling: bool = False,
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
    use_per_obj_stats = hasattr(predictor, "per_objective_stats")

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
        charge_rejected = 0
        seen_nonneutral: set = set()  # unique non-neutral keys, for diagnosis
        max_att = max_attempts or (n_target * 20)

        pbar = tqdm(total=n_target, desc=f"Generate [{purpose}]")
        while accepted < n_target and attempted < max_att:
            attempted += 1

            try:
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
            except CandidateEvaluationFailed as exc:
                # Unlike training, generation already has a generous, separate
                # attempts budget (max_att) distinct from `n_target` -- treat
                # this exactly like a rejected duplicate/infeasible candidate
                # (already counted via `attempted += 1` above) rather than
                # retrying without counting.
                print(f"[WARN] generation candidate failed evaluation, "
                      f"skipping: {exc}")
                continue

            comp     = env.terminal_cation_fractions()
            comp_key = env.terminal_comp_key()

            if comp_key in seen_comp_keys:
                dup_rejected += 1
                continue

            # Charge-neutrality gate on the WHOLE built formula (covers
            # builder-derived anions, e.g. SSE). Gated entirely by config: when
            # charge_filter is False this block is skipped (no smact import).
            # Checked BEFORE adding to seen_comp_keys and BEFORE the (cached)
            # predictor call so that (a) repeated non-neutral picks are counted
            # as charge rejections — not duplicates — giving a clean diagnosis of
            # why yield is low, and (b) we don't pay a predictor call on a
            # candidate we're about to drop.
            if charge_filter:
                from .constraints.charge import charge_neutral

                # Prefer the builder's full composition label (host/derived
                # anions the env never sees); fall back to the env pick-only
                # formula. Same resolution as the accepted-row path below.
                _formula = env.terminal_formula
                _cf = getattr(predictor, "composition_formula", None)
                if callable(_cf):
                    try:
                        _full = _cf(comp)
                        if _full:
                            _formula = _full
                    except Exception:
                        pass
                try:
                    neutral = charge_neutral(_formula, use_pauling=charge_use_pauling)
                except Exception:
                    neutral = True  # lenient: never crash generation on a bad parse
                if not neutral:
                    seen_nonneutral.add(comp_key)
                    charge_rejected += 1
                    continue

            # Reward from the episode (env.reward_fn already called in env.step).
            reward = float(env.path[-1].reward) if env.path else 0.0

            # Raw predictor values for CSV columns. OOHCatalystPredictor caches
            # internally so this is usually a cache hit when env.reward_fn
            # already ran -- but for a non-caching predictor, or a builder
            # whose cache doesn't cover this call shape, it is a genuinely
            # separate call and can independently fail (e.g. a re-run MD
            # realization). Skip the candidate rather than crash generation;
            # don't mark comp_key as seen, so a later attempt on the same
            # composition can still succeed and be reported.
            try:
                if comp_key in predictor_raw_cache:
                    raw_mean, std = predictor_raw_cache[comp_key]
                elif use_predict_raw:
                    raw_mean, std = predictor.predict_raw(comp)
                    predictor_raw_cache[comp_key] = (raw_mean, std)
                else:
                    raw_mean, std = predictor.predict(comp)
                    predictor_raw_cache[comp_key] = (raw_mean, std)
            except Exception as exc:  # noqa: BLE001 - any predictor/builder failure
                print(f"[WARN] generation candidate failed fetching raw stats, "
                      f"skipping: {type(exc).__name__}: {exc}")
                continue

            seen_comp_keys.add(comp_key)

            # Prefer the builder's full composition label (includes host/derived
            # elements the env never sees — e.g. SSE's surviving S, derived Br,
            # charge-balanced Li); fall back to the env's pick-only formula.
            formula = env.terminal_formula
            comp_formula = getattr(predictor, "composition_formula", None)
            if callable(comp_formula):
                try:
                    full = comp_formula(comp)
                    if full:
                        formula = full
                except Exception:
                    pass

            row: dict = {
                "formula": formula,
                "reward": reward,
                "dp_mean": raw_mean,
            }
            if use_per_obj_stats:
                # Composite predictor: each property keeps its own std in
                # native physical units. A joint dp_std across mixed units
                # would be meaningless, so dp_std / dp_mean_minus_std stay
                # absent (pandas writes NaN). Per-property columns below.
                for obj_name, (m, s) in predictor.per_objective_stats(comp).items():
                    row[f"obj_{obj_name}_mean"] = float(m)
                    row[f"obj_{obj_name}_std"] = float(s)
            else:
                row["dp_std"] = std
                row["dp_mean_minus_std"] = raw_mean - k * std
            if use_check_phase:
                phase_ok, phase_label = predictor.check_phase(comp)
                row["primary_ok"] = bool(phase_ok)
                row["primary_label"] = phase_label or ""

            rows.append(row)
            accepted += 1
            pbar.update(1)
            pbar.set_postfix(
                attempts=attempted, dups=dup_rejected, charge=charge_rejected
            )

        pbar.close()
        rate = accepted / max(attempted, 1)
        print(
            f"[INFO] Generated {accepted}/{n_target} {purpose} candidates "
            f"({attempted} attempts, {dup_rejected} dups, "
            f"{charge_rejected} non-neutral [{len(seen_nonneutral)} unique], "
            f"rate={rate:.3f})"
        )
        if charge_filter and accepted < n_target:
            print(
                f"[INFO] Shortfall diagnosis [{purpose}]: of {attempted} attempts, "
                f"{dup_rejected} were duplicates and {charge_rejected} were "
                f"non-charge-neutral. "
                + (
                    "Duplication dominates: the policy keeps proposing the same "
                    "few compositions (raise gen_temperature / gen_epsilon, or the "
                    "neutral subspace may simply be small)."
                    if dup_rejected >= charge_rejected
                    else "Charge-neutrality dominates: most proposed compositions "
                    "aren't neutral on the 0.05 grid (the neutral subspace is the "
                    "bottleneck, not exploration)."
                )
            )

    return rows
