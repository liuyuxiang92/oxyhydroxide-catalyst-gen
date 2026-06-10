#!/usr/bin/env python3
"""Config-driven general entry point for rl_matdesign experiments.

Replaces the OOH-specific run_ABCDEOOH_experiment.py with a single script
that works for any material system configured via a YAML file.

Usage
-----
    python scripts/run_experiment.py --config configs/hea.yaml --method a2c \\
        --out runs/hea_a2c_seed0 --seed 0

    python scripts/run_experiment.py --config configs/perovskite.yaml --method dqn \\
        --out runs/perovskite_dqn_seed1 --seed 1
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import os
import random
import warnings

# Required for torch.use_deterministic_algorithms(True) with CUDA >= 10.2.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

warnings.filterwarnings(
    "ignore",
    message=r"^PymatgenData\(impute_nan=False\):.*",
    category=UserWarning,
)

import joblib
import numpy as np
import torch
import yaml

from rl_matdesign.env import CompositionEnv
from rl_matdesign.env_integer import IntegerRatioEnv
from rl_matdesign.model import PolicyNet, QRegressor, ValueNet
from rl_matdesign.training import (
    _fit_scaler_from_warmup,
    _precompute_elem_features,
    generate_candidates,
    train_dqn_online,
    train_pg,
)
from rl_matdesign.utils.metrics import RunMetrics
from rl_matdesign.utils.seeding import set_global_seed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="General RL material design experiment")
    p.add_argument("--config", required=True, help="Path to YAML config file")
    p.add_argument("--method", choices=["dqn", "reinforce", "a2c"], default=None,
                   help="RL method (overrides config 'method' field)")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--dp-seed", type=int, default=0,
                   help="Base RNG seed for the predictor and fallback for --train-seed/--gen-seed")
    p.add_argument("--train-seed", type=int, default=None,
                   help="RNG seed for the training phase. Enables GPU determinism when set. Falls back to --dp-seed.")
    p.add_argument("--gen-seed", type=int, default=None,
                   help="RNG seed for the generation phase. Falls back to --dp-seed.")
    p.add_argument("--device", default=None, help="torch device (default: auto)")
    p.add_argument("--save-checkpoint-freq", type=int, default=0, metavar="N",
                   dest="save_checkpoint_freq",
                   help="Save mid-training checkpoint every N episodes (DQN) or N iterations (PG). 0 disables. "
                        "Saved atomically to <out>/checkpoint.pt.")
    p.add_argument("--only-generate", action="store_true",
                   help="Skip training; load saved model + scaler and generate candidates only. "
                        "Defaults to loading <out>/policy.pt (PG) or <out>/qnet.pt (DQN) and <out>/std_scaler.bin.")
    p.add_argument("--resume-training", action="store_true",
                   help="Load an existing model/checkpoint and continue training. "
                        "For PG: loads std_scaler.bin + checkpoint.pt (or policy.pt) and runs more iterations. "
                        "For DQN: loads std_scaler.bin + checkpoint.pt (qnet, target_net, optimizer, replay "
                        "buffer, dp_cache, epsilon, episode counter) for exact-state resume; falls back to "
                        "qnet.pt (weights only — buffer/epsilon/episode reset) when no mid-checkpoint exists. "
                        "Cannot be combined with --only-generate.")
    p.add_argument("--skip-generation", action="store_true",
                   help="Run training but skip candidate generation.")
    p.add_argument("--load-qnet", type=str, default=None,
                   help="Path to saved qnet.pt state dict (default: <out>/qnet.pt).")
    p.add_argument("--load-policy", type=str, default=None,
                   help="Path to saved policy.pt state dict (default: <out>/policy.pt).")
    p.add_argument("--load-scaler", type=str, default=None,
                   help="Path to saved std_scaler.bin (default: <out>/std_scaler.bin).")
    p.add_argument("--load-value-net", type=str, default=None,
                   help="Path to saved value_net.pt state dict (default: <out>/value_net.pt).")
    p.add_argument("--dqn-loss", choices=["mse", "smoothl1"], default=None,
                   dest="dqn_loss",
                   help="Loss function for DQN training (default: from config or smoothl1).")
    p.add_argument("--dqn-augment-permutations", type=int, default=None, metavar="K",
                   dest="dqn_augment_permutations",
                   help="DQN-only: insert K permutation-augmented copies of each completed "
                        "episode into the replay buffer (terminal reward is reused — no extra "
                        "predictor calls). Within-episode duplicate rows are skipped. Ignored "
                        "with a warning for REINFORCE/A2C (on-policy methods can't use "
                        "permuted trajectories). Default 0 (off). Overrides config "
                        "'dqn_augment_permutations'.")
    p.add_argument("--max-gen-attempts", type=int, default=None,
                   dest="max_gen_attempts",
                   help="Max generation attempts before stopping. Default: 10 × num_gen_eps.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

# Old YAML key → new YAML key. Old keys still work but emit a one-time
# deprecation warning. Conventions:
#   dqn_*  → used only by train_dqn_online
#   pg_*   → used only by train_pg (REINFORCE / A2C)
#   (no prefix) → shared across both methods
_FLAG_ALIASES = {
    "buffer_size":          "dqn_buffer_size",
    "num_train_eps":        "dqn_num_train_eps",
    "grad_steps_per_ep":    "dqn_grad_steps_per_ep",
    "target_update_freq":   "dqn_target_update_freq",
    "eps_anneal_eps":       "dqn_eps_anneal_eps",
    "eps_min":              "dqn_eps_min",
    "entropy_coef":         "pg_entropy_coef",
    "repeat_penalty_coef":  "pg_repeat_penalty_coef",
    "repeat_penalty_shape": "pg_repeat_penalty_shape",
}


def _apply_flag_aliases(cfg: dict) -> dict:
    """Map deprecated flag names to their new prefixed forms.

    If both old and new appear, the new value wins and a warning is emitted.
    """
    for old, new in _FLAG_ALIASES.items():
        if old in cfg:
            if new in cfg:
                warnings.warn(
                    f"Config key '{old}' is deprecated (renamed to '{new}'); "
                    f"both keys are present — using '{new}' and ignoring '{old}'.",
                    DeprecationWarning, stacklevel=2,
                )
            else:
                warnings.warn(
                    f"Config key '{old}' is deprecated; rename to '{new}'.",
                    DeprecationWarning, stacklevel=2,
                )
                cfg[new] = cfg[old]
            del cfg[old]
    return cfg


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return _apply_flag_aliases(cfg)


# ---------------------------------------------------------------------------
# Predictor / constraint factories — thin wrappers around the registry.
# YAML keys 'predictor' and 'constraint_filter' accept either a built-in
# short name or a fully-qualified 'pkg.module:ClassName' for user plug-ins.
# See src/rl_matdesign/registry.py for the dispatch logic.
# ---------------------------------------------------------------------------

def build_predictor(cfg: dict, seed: int = None):
    from rl_matdesign.registry import resolve_predictor
    return resolve_predictor(cfg.get("predictor", "dummy"), cfg, seed=seed)


def build_constraint_filter(cfg: dict, env=None):
    from rl_matdesign.registry import resolve_constraint
    return resolve_constraint(cfg.get("constraint_filter"), cfg, env=env)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.only_generate and args.resume_training:
        raise SystemExit("--only-generate and --resume-training are mutually exclusive.")

    cfg = load_config(args.config)

    # User-friendly scenario configs expand into a full multi_group config here,
    # before anything reads env/predictor keys.
    if cfg.get("env_type") == "lips":
        from rl_matdesign.scenarios.lips import expand as _expand_lips
        cfg = _expand_lips(cfg)

    method = args.method or cfg.get("method", "a2c")
    train_seed = args.train_seed if args.train_seed is not None else args.dp_seed
    gen_seed   = args.gen_seed   if args.gen_seed   is not None else args.dp_seed

    set_global_seed(train_seed, deterministic=(args.train_seed is not None))
    os.makedirs(args.out, exist_ok=True)

    run_config = {
        "config_file": args.config, "method": method,
        "dp_seed": args.dp_seed, "train_seed": train_seed, "gen_seed": gen_seed,
        **cfg,
    }
    with open(os.path.join(args.out, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"[INFO] device={device}, method={method}, dp_seed={args.dp_seed}, "
          f"train_seed={train_seed}, gen_seed={gen_seed}")

    predictor = build_predictor(cfg, seed=args.dp_seed)
    env_type  = cfg.get("env_type", "fraction")

    def reward_fn(formula: str) -> float:
        if env_type == "integer_ratio":
            from pymatgen.core.composition import Composition
            try:
                comp = dict(Composition(formula).fractional_composition.as_dict())
                comp = {str(k): float(v) for k, v in comp.items()}
            except Exception:
                return -2000.0
        else:
            import re
            parts = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]+)", formula)
            comp = {el: float(frac) for el, frac in parts}
            anion_formula = cfg.get("anion_formula", "")
            if anion_formula:
                for el, _ in re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]+)", anion_formula):
                    comp.pop(el, None)
        mean, _ = predictor.predict(comp)
        return mean

    def mg_reward_fn(groups: dict) -> float:
        # MultiGroupEnv hands the predictor the structured {group: {el: frac}}
        # mapping; the (recipe) predictor assembles the full structure from it.
        mean, _ = predictor.predict(groups)
        return mean

    # Build env first (without filter), then build the filter using env's
    # feasibility tables (_allowed_units, _possible_sums_by_k), then attach.
    # This is required for the ooh_phase filter and avoids the previous probe
    # episode that burned RNG state before warmup (Bug 3).
    if env_type == "integer_ratio":
        env = IntegerRatioEnv(
            cation_set=cfg["cation_set"],
            ratio_set=cfg.get("ratio_set", None) or _default_digits(),
            n_components=int(cfg.get("n_components", 5)),
            reward_fn=reward_fn,
            phase_filter=None,
        )
    elif env_type == "multi_group":
        from rl_matdesign.env_multigroup import MultiGroupEnv

        # Each group carries its own CompositionEnv-style spec plus an optional
        # constraint_filter; build the per-group filter instance here (env=None —
        # multi-group filters mask by element/level/prior-group, not env tables).
        built_groups = []
        for g in cfg["groups"]:
            gspec = dict(g)
            gspec["constraint_filter"] = build_constraint_filter(g, env=None)
            built_groups.append(gspec)
        env = MultiGroupEnv(groups=built_groups, reward_fn=mg_reward_fn)
    else:
        env = CompositionEnv(
            cation_set=cfg["cation_set"],
            fraction_set=cfg.get("fraction_set", None) or _default_fractions(),
            anion_formula=cfg.get("anion_formula", ""),
            n_components=int(cfg.get("n_components", 5)),
            reward_fn=reward_fn,
            phase_filter=None,
            total_units=int(cfg.get("total_units", 20)),
            element_bounds=cfg.get("element_bounds"),
            episode_style=cfg.get("episode_style", "element_then_amount"),
        )

    # Single-group envs attach a top-level filter post-construction (some filters
    # need the env's feasibility tables). multi_group filters live per-group.
    if env_type != "multi_group":
        env.phase_filter = build_constraint_filter(cfg, env=env)

    step_dim     = env.n_components
    fraction_set = list(env.fraction_set)

    # Precompute Magpie element features (replaces one-hot element encoding).
    elem_feats_scaled, _elem_scaler = _precompute_elem_features(
        env.cation_set, env.state_featurizer
    )
    elem_dim = int(elem_feats_scaled.shape[1])
    frac_dim = 1

    print(f"[INFO] step_dim={step_dim}, elem_dim(Magpie)={elem_dim}, frac_dim={frac_dim}")

    metrics = RunMetrics()

    ckpt_path = os.path.join(args.out, "checkpoint.pt")
    checkpoint_cfg = {"path": ckpt_path, "freq": args.save_checkpoint_freq} if args.save_checkpoint_freq > 0 else None

    # Generation strategy (unified across DQN and PG).
    gen_temperature = float(cfg.get("gen_temperature", 1.0))
    gen_top_frac    = float(cfg.get("gen_top_frac", 0.0))
    gen_epsilon_gen = float(cfg.get("gen_epsilon", 0.0))

    # Warn-and-ignore: augmentation only makes sense for off-policy DQN.
    _aug_K_request = args.dqn_augment_permutations
    if _aug_K_request is None:
        _aug_K_request = int(cfg.get("dqn_augment_permutations", 0))
    if method != "dqn" and int(_aug_K_request) > 0:
        print(
            f"[WARN] dqn_augment_permutations={_aug_K_request} is ignored for "
            f"method={method!r}: PG/A2C are on-policy and permuted trajectories "
            "break the log-prob attribution.",
            flush=True,
        )

    # ------------------------------------------------------------------
    # DQN path — classical online DQN
    # ------------------------------------------------------------------
    if method == "dqn":
        scaler_path = args.load_scaler or os.path.join(args.out, "std_scaler.bin")
        qnet_path   = args.load_qnet   or os.path.join(args.out, "qnet.pt")

        if args.only_generate:
            if not os.path.exists(scaler_path):
                raise SystemExit(f"--only-generate requires {scaler_path} (use --load-scaler to override)")
            if not os.path.exists(qnet_path):
                raise SystemExit(f"--only-generate requires {qnet_path} (use --load-qnet to override)")
            scaler = joblib.load(scaler_path)
            state_dim = int(getattr(scaler, "n_features_in_", scaler.mean_.shape[0]))
            from rl_matdesign.model import QRegressor
            qnet = QRegressor(
                state_dim=state_dim, step_dim=step_dim,
                elem_dim=elem_dim, frac_dim=1,
            ).to(device)
            _raw = torch.load(qnet_path, map_location=device)
            qnet.load_state_dict(_raw["qnet_state"] if isinstance(_raw, dict) and "qnet_state" in _raw else _raw)
            print(f"[INFO] Loaded qnet from {qnet_path}", flush=True)
        else:
            # Resolve hyperparameters early so resume + fresh paths share them.
            _hidden       = int(cfg.get("dqn_hidden_dim", 256))
            _lr_dqn       = float(cfg.get("dqn_lr", 1e-3))
            _buf_size     = int(cfg.get("dqn_buffer_size", 50000))
            _eps_anneal   = int(cfg.get("dqn_eps_anneal_eps", 10000))
            _eps_min      = float(cfg.get("dqn_eps_min", 0.05))
            _n_train_eps  = int(cfg.get("dqn_num_train_eps", 20000))

            resume_state = None
            if args.resume_training:
                if not os.path.exists(scaler_path):
                    raise SystemExit(
                        f"--resume-training (dqn) requires {scaler_path} (use --load-scaler to override)"
                    )
                scaler = joblib.load(scaler_path)
                state_dim_loaded = int(getattr(scaler, "n_features_in_", scaler.mean_.shape[0]))
                print(f"[INFO] Loaded scaler from {scaler_path}", flush=True)

                qnet = QRegressor(
                    state_dim=state_dim_loaded, step_dim=step_dim,
                    elem_dim=elem_dim, frac_dim=1, hidden_dim=_hidden,
                ).to(device)
                target_net = copy.deepcopy(qnet)
                target_net.eval()
                optimizer = torch.optim.Adam(qnet.parameters(), lr=_lr_dqn)
                buffer: collections.deque = collections.deque(maxlen=_buf_size)

                _resume_src = args.load_qnet if args.load_qnet else ckpt_path
                _mid = None
                if os.path.exists(_resume_src):
                    _raw_ckpt = torch.load(_resume_src, map_location=device, weights_only=False)
                    if (
                        isinstance(_raw_ckpt, dict)
                        and _raw_ckpt.get("type") == "dqn"
                        and "buffer" in _raw_ckpt
                    ):
                        _mid = _raw_ckpt
                    else:
                        _type_str = _raw_ckpt.get("type") if isinstance(_raw_ckpt, dict) else type(_raw_ckpt).__name__
                        print(
                            f"[WARN] {_resume_src} is not a valid DQN mid-checkpoint "
                            f"(type={_type_str!r}); falling back to qnet.pt",
                            flush=True,
                        )
                elif args.load_qnet:
                    raise SystemExit(f"--load-qnet path not found: {args.load_qnet}")

                _predictor_cache = getattr(predictor, "_cache", None)

                if _mid is not None:
                    qnet.load_state_dict(_mid["qnet_state"])
                    target_net.load_state_dict(_mid["target_net_state"])
                    optimizer.load_state_dict(_mid["opt_state"])
                    for _row in _mid["buffer"]:
                        buffer.append(_row)
                    # Old checkpoints used "buffer_size"; new use "dqn_buffer_size". Read both.
                    _saved_size = int(_mid.get("dqn_buffer_size", _mid.get("buffer_size", _buf_size)))
                    if _saved_size != _buf_size:
                        print(
                            f"[WARN] dqn_buffer_size changed: saved={_saved_size}, "
                            f"current={_buf_size}; kept loaded rows under new maxlen.",
                            flush=True,
                        )
                    if _predictor_cache is not None:
                        _predictor_cache.update(_mid.get("dp_cache", {}))
                    start_ep = int(_mid["episodes_completed"])
                    eps_val  = float(_mid.get("eps", max(_eps_min, 1.0 - start_ep / _eps_anneal)))
                    print(
                        f"[INFO] DQN resume from {_resume_src}: ep={start_ep}, "
                        f"eps={eps_val:.4f}, buffer_rows={len(buffer)}, "
                        f"dp_cache_rows={len(_predictor_cache) if _predictor_cache is not None else 0}",
                        flush=True,
                    )
                else:
                    if not os.path.exists(qnet_path):
                        raise SystemExit(
                            f"--resume-training (dqn) requires checkpoint.pt or {qnet_path} "
                            "(use --load-qnet to override)"
                        )
                    _raw_q = torch.load(qnet_path, map_location=device)
                    qnet.load_state_dict(
                        _raw_q["qnet_state"] if isinstance(_raw_q, dict) and "qnet_state" in _raw_q else _raw_q
                    )
                    target_net.load_state_dict(qnet.state_dict())
                    start_ep = 0
                    eps_val  = 1.0
                    print(
                        f"[WARN] DQN resume: no mid-checkpoint; loaded {qnet_path} "
                        "(weights only) — buffer/eps/episode counter reset.",
                        flush=True,
                    )

                resume_state = {
                    "scaler": scaler,
                    "qnet": qnet,
                    "target_net": target_net,
                    "optimizer": optimizer,
                    "buffer": buffer,
                    "start_ep": start_ep,
                    "eps": eps_val,
                }

            # Thread the predictor cache into checkpoint_cfg so periodic saves include it.
            if checkpoint_cfg is not None:
                _pcache = getattr(predictor, "_cache", None)
                if _pcache is not None:
                    checkpoint_cfg["dp_cache"] = _pcache

            _dqn_loss = args.dqn_loss or cfg.get("dqn_loss", "smoothl1")
            _aug_K = args.dqn_augment_permutations
            if _aug_K is None:
                _aug_K = int(cfg.get("dqn_augment_permutations", 0))
            qnet, scaler, train_rows = train_dqn_online(
                env=env,
                elem_feats_scaled=elem_feats_scaled,
                fraction_set=fraction_set,
                device=device,
                n_warmup_eps=int(cfg.get("dqn_warmup_eps", cfg.get("pg_warmup_eps", 500))),
                dqn_num_train_eps=_n_train_eps,
                dqn_buffer_size=_buf_size,
                batch_size=int(cfg.get("dqn_batch_size", 256)),
                dqn_grad_steps_per_ep=int(cfg.get("dqn_grad_steps_per_ep", 5)),
                dqn_target_update_freq=int(cfg.get("dqn_target_update_freq", 100)),
                dqn_eps_anneal_eps=_eps_anneal,
                dqn_eps_min=_eps_min,
                gamma=float(cfg.get("dqn_gamma", cfg.get("gamma", 0.9))),
                hidden_dim=_hidden,
                lr=_lr_dqn,
                loss_name=_dqn_loss,
                checkpoint_cfg=checkpoint_cfg,
                resume_state=resume_state,
                augment_permutations=_aug_K,
            )
            for r in train_rows:
                metrics.log(**r)
            torch.save(qnet.state_dict(), os.path.join(args.out, "qnet.pt"))
            joblib.dump(scaler, os.path.join(args.out, "std_scaler.bin"))

        if not args.skip_generation:
            np.random.seed(gen_seed)
            random.seed(gen_seed)
            _n_exploit = int(cfg.get("num_gen_eps", 200))
            _max_attempts = args.max_gen_attempts if args.max_gen_attempts is not None else 10 * _n_exploit
            gen_rows = generate_candidates(
                env=env, predictor=predictor, scaler=scaler, device=device,
                qnet=qnet,
                elem_feats_scaled=elem_feats_scaled,
                fraction_set=fraction_set,
                n_exploit=_n_exploit,
                n_explore=int(cfg.get("exploration_gen_eps", 0)),
                gen_temperature=gen_temperature,
                gen_top_frac=gen_top_frac,
                gen_epsilon=gen_epsilon_gen,
                k=float(cfg.get("k", 1.0)),
                max_attempts=_max_attempts,
            )
            for r in gen_rows:
                metrics.log(phase="generate", **r)

    # ------------------------------------------------------------------
    # PG paths (REINFORCE / A2C)
    # ------------------------------------------------------------------
    else:
        pg_warmup            = int(cfg.get("pg_warmup_eps", 200))
        pg_num_iters         = int(cfg.get("pg_num_iters", 1000))
        pg_batch_eps         = int(cfg.get("pg_batch_eps", 15))
        lr_actor             = float(cfg.get("pg_lr_actor", 1e-3))
        lr_critic            = float(cfg.get("pg_lr_critic", 1e-3))
        pg_entropy_coef         = float(cfg.get("pg_entropy_coef", 0.01))
        gamma                   = float(cfg.get("pg_gamma", cfg.get("gamma", 0.9)))
        pg_repeat_penalty_coef  = float(cfg.get("pg_repeat_penalty_coef", 0.0))
        pg_repeat_penalty_shape = cfg.get("pg_repeat_penalty_shape", "log")

        scaler_path  = args.load_scaler    or os.path.join(args.out, "std_scaler.bin")
        policy_path  = args.load_policy    or os.path.join(args.out, "policy.pt")
        vnet_path    = args.load_value_net or os.path.join(args.out, "value_net.pt")
        ckpt_path_pg = os.path.join(args.out, "checkpoint.pt")

        if args.only_generate:
            if not os.path.exists(scaler_path):
                raise SystemExit(f"--only-generate requires {scaler_path} (use --load-scaler to override)")
            if not os.path.exists(policy_path):
                raise SystemExit(f"--only-generate requires {policy_path} (use --load-policy to override)")
            scaler = joblib.load(scaler_path)
            state_dim_loaded = int(getattr(scaler, "n_features_in_", scaler.mean_.shape[0]))
            policy = PolicyNet(
                state_dim=state_dim_loaded, step_dim=step_dim,
                elem_dim=elem_dim, frac_dim=frac_dim,
            ).to(device)
            _raw = torch.load(policy_path, map_location=device)
            policy.load_state_dict(_raw["policy_state"] if isinstance(_raw, dict) and "policy_state" in _raw else _raw)
            print(f"[INFO] Loaded policy from {policy_path}", flush=True)
            value_net = None

        elif args.resume_training:
            if not os.path.exists(scaler_path):
                raise SystemExit(f"--resume-training requires {scaler_path} (use --load-scaler to override)")
            scaler = joblib.load(scaler_path)
            print(f"[INFO] Loaded scaler from {scaler_path}", flush=True)
            state_dim_loaded = int(getattr(scaler, "n_features_in_", scaler.mean_.shape[0]))
            policy = PolicyNet(
                state_dim=state_dim_loaded, step_dim=step_dim,
                elem_dim=elem_dim, frac_dim=frac_dim,
            ).to(device)
            value_net = ValueNet(state_dim=state_dim_loaded, step_dim=step_dim).to(device) if method == "a2c" else None

            # Prefer mid-training checkpoint.pt; fall back to policy.pt.
            _mid_ckpt = None
            _resume_src = args.load_policy if args.load_policy else ckpt_path_pg
            if os.path.exists(_resume_src):
                _raw = torch.load(_resume_src, map_location=device)
                if isinstance(_raw, dict) and _raw.get("type") == "pg":
                    _mid_ckpt = _raw
                    policy.load_state_dict(_raw["policy_state"])
                    if value_net is not None and _raw.get("value_net_state") is not None:
                        value_net.load_state_dict(_raw["value_net_state"])
                    print(f"[INFO] Resuming from mid-training checkpoint ({_raw['episodes_completed']} eps) → {_resume_src}", flush=True)
                else:
                    _type_str = _raw.get("type") if isinstance(_raw, dict) else "raw"
                    print(f"[WARN] {_resume_src} type={_type_str!r}, expected 'pg'; falling back to policy.pt", flush=True)
            elif args.load_policy:
                raise SystemExit(f"--load-policy path not found: {args.load_policy}")

            if _mid_ckpt is None:
                if not os.path.exists(policy_path):
                    raise SystemExit(f"--resume-training requires {policy_path} (use --load-policy to override)")
                policy.load_state_dict(torch.load(policy_path, map_location=device))
                print(f"[INFO] Loaded policy from {policy_path}", flush=True)
                if value_net is not None:
                    if not os.path.exists(vnet_path):
                        raise SystemExit(f"--resume-training (a2c) requires {vnet_path} (use --load-value-net to override)")
                    value_net.load_state_dict(torch.load(vnet_path, map_location=device))
                    print(f"[INFO] Loaded value_net from {vnet_path}", flush=True)

            # Build checkpoint_cfg with resume data so train_pg restores optimizer + visit_counts.
            checkpoint_cfg = {
                "path": ckpt_path_pg,
                "freq": args.save_checkpoint_freq,
                "start_episode": _mid_ckpt["episodes_completed"] if _mid_ckpt else 0,
                "opt_actor_state": _mid_ckpt.get("opt_actor_state") if _mid_ckpt else None,
                "opt_critic_state": _mid_ckpt.get("opt_critic_state") if _mid_ckpt else None,
                "visit_counts": _mid_ckpt.get("visit_counts") if _mid_ckpt else None,
            }

        else:
            scaler = _fit_scaler_from_warmup(env, pg_warmup)
            joblib.dump(scaler, os.path.join(args.out, "std_scaler.bin"))
            state_dim = int(scaler.n_features_in_)
            print(f"[INFO] state_dim={state_dim} (from warmup scaler)")
            policy = PolicyNet(
                state_dim=state_dim, step_dim=step_dim,
                elem_dim=elem_dim, frac_dim=frac_dim,
            ).to(device)
            value_net = ValueNet(state_dim=state_dim, step_dim=step_dim).to(device) if method == "a2c" else None

        if not args.only_generate:
            train_rows = train_pg(
                policy=policy,
                value_net=value_net,
                env=env,
                scaler=scaler,
                elem_feats_scaled=elem_feats_scaled,
                fraction_set=fraction_set,
                device=device,
                num_iters=pg_num_iters,
                batch_eps=pg_batch_eps,
                gamma=gamma,
                lr_actor=lr_actor,
                lr_critic=lr_critic,
                pg_entropy_coef=pg_entropy_coef,
                rl_method=method,
                pg_repeat_penalty_coef=pg_repeat_penalty_coef,
                pg_repeat_penalty_shape=pg_repeat_penalty_shape,
                checkpoint_cfg=checkpoint_cfg,
            )
            for r in train_rows:
                metrics.log(**r)
            torch.save(policy.state_dict(), os.path.join(args.out, "policy.pt"))
            if value_net is not None:
                torch.save(value_net.state_dict(), os.path.join(args.out, "value_net.pt"))

        if not args.skip_generation:
            np.random.seed(gen_seed)
            random.seed(gen_seed)
            _n_exploit = int(cfg.get("num_gen_eps", 200))
            _max_attempts = args.max_gen_attempts if args.max_gen_attempts is not None else 10 * _n_exploit
            gen_rows = generate_candidates(
                env=env, predictor=predictor, scaler=scaler, device=device,
                policy=policy,
                elem_feats_scaled=elem_feats_scaled,
                fraction_set=fraction_set,
                n_exploit=_n_exploit,
                n_explore=int(cfg.get("exploration_gen_eps", 0)),
                gen_temperature=gen_temperature,
                gen_top_frac=gen_top_frac,
                gen_epsilon=gen_epsilon_gen,
                k=float(cfg.get("k", 1.0)),
                max_attempts=_max_attempts,
            )
            for r in gen_rows:
                metrics.log(phase="generate", **r)

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    _log_path = os.path.join(args.out, "training_log.csv")
    _log_mode = "a" if args.resume_training else "w"
    metrics.to_csv(_log_path, mode=_log_mode)
    print(f"[INFO] Training log {'appended to' if _log_mode == 'a' else 'written to'} {_log_path}", flush=True)

    gen_rows_only = sorted(
        [r for r in metrics.rows if r.get("phase") == "generate"],
        key=lambda r: float(r["reward"]),
        reverse=True,
    )
    if gen_rows_only:
        import csv
        fieldnames = [k for k in gen_rows_only[0].keys() if k != "phase"]
        with open(os.path.join(args.out, "generated.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(gen_rows_only)

    top10     = metrics.top_k("dp_mean", k=10, phase="generate")
    diversity = metrics.diversity(phase="generate")
    pareto    = metrics.pareto_front(phase="generate")
    if top10:
        print(f"\n[SUMMARY] diversity={diversity} | top-10 mean dp_mean="
              f"{float(np.mean([float(r['dp_mean']) for r in top10])):.4f}")
    print(f"[SUMMARY] Pareto front size: {len(pareto)} candidates")
    print(f"[INFO] Results saved to {args.out}")


def _default_fractions():
    return [
        "0.05", "0.10", "0.15", "0.20", "0.25", "0.30", "0.35",
        "0.40", "0.45", "0.50", "0.55", "0.60", "0.65", "0.70", "0.75", "0.80",
    ]


def _default_digits():
    return ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


if __name__ == "__main__":
    main()
