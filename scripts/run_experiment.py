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
import json
import os
import random
import sys
import warnings

# Mitigate OpenMP runtime conflicts (common on macOS).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

warnings.filterwarnings(
    "ignore",
    message=r"^PymatgenData\(impute_nan=False\):.*",
    category=UserWarning,
)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

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
    _rollout_random_episode,
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
    return p.parse_args()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


# ---------------------------------------------------------------------------
# Predictor factory
# ---------------------------------------------------------------------------

def build_predictor(cfg: dict, seed: int = None):
    """Instantiate the correct PropertyPredictor from the config."""
    kind = cfg.get("predictor", "dummy").lower()

    if kind == "hea":
        from rl_matdesign.predictors.hea import HEAPropertyPredictor
        return HEAPropertyPredictor(
            poscar_template=cfg["poscar"],
            dp_models=cfg["dp_models"],
            objective=cfg.get("objective", "mean_minus_kstd"),
            k=float(cfg.get("k", 1.0)),
            n_random_configs=int(cfg.get("n_random_configs", 5)),
            site_symbol=cfg.get("site_symbol", "X"),
            rng_seed=seed,
        )

    elif kind == "perovskite":
        from rl_matdesign.predictors.perovskite import PerovskitePropertyPredictor
        return PerovskitePropertyPredictor(
            poscar_template=cfg["poscar"],
            dp_models=cfg["dp_models"],
            objective=cfg.get("objective", "mean_minus_kstd"),
            k=float(cfg.get("k", 1.0)),
            n_random_configs=int(cfg.get("n_random_configs", 5)),
            site_symbol=cfg.get("site_symbol", "Fe"),
            rng_seed=seed,
        )

    elif kind == "sinter_calcine":
        from rl_matdesign.predictors.sinter_calcine import SinterCalcineRFPredictor
        return SinterCalcineRFPredictor(
            rf_model_path=cfg["rf_model"],
            mode=cfg.get("mode", "sinter"),
        )

    elif kind == "dummy":
        import random as _random

        class _DummyPredictor:
            def predict(self, composition):
                return float(_random.gauss(0.5, 0.1)), float(abs(_random.gauss(0.05, 0.01)))

            def batch_predict(self, compositions):
                return [self.predict(c) for c in compositions]

        return _DummyPredictor()

    else:
        raise ValueError(f"Unknown predictor type '{kind}'. Options: hea, perovskite, sinter_calcine, dummy.")


# ---------------------------------------------------------------------------
# Constraint filter factory
# ---------------------------------------------------------------------------

def build_constraint_filter(cfg: dict):
    kind = cfg.get("constraint_filter", None)
    if kind is None:
        return None
    if kind == "smact_charge":
        from rl_matdesign.constraints.smact_filter import SMACTChargeFilter
        if "smact_anions" in cfg:
            anions = cfg["smact_anions"]
        else:
            anions = [{
                "symbol": cfg.get("smact_anion", "O"),
                "charge": int(cfg.get("smact_anion_charge", -2)),
                "stoich": float(cfg.get("smact_anion_stoich", 1.5)),
            }]
        return SMACTChargeFilter(anions=anions)
    if kind == "last_step_element":
        from rl_matdesign.constraints.last_step_element import LastStepElementFilter
        return LastStepElementFilter(
            required_elements=cfg["required_elements"],
            nonzero_ratio=bool(cfg.get("nonzero_ratio_at_last", True)),
            reserve_for_last=bool(cfg.get("reserve_for_last", True)),
        )
    raise ValueError(f"Unknown constraint_filter '{kind}'.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

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

    predictor    = build_predictor(cfg, seed=args.dp_seed)
    phase_filter = build_constraint_filter(cfg)
    env_type     = cfg.get("env_type", "fraction")

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

    if env_type == "integer_ratio":
        env = IntegerRatioEnv(
            cation_set=cfg["cation_set"],
            ratio_set=cfg.get("ratio_set", None) or _default_digits(),
            n_components=int(cfg.get("n_components", 5)),
            reward_fn=reward_fn,
            phase_filter=phase_filter,
        )
    else:
        env = CompositionEnv(
            cation_set=cfg["cation_set"],
            fraction_set=cfg.get("fraction_set", None) or _default_fractions(),
            anion_formula=cfg.get("anion_formula", ""),
            n_components=int(cfg.get("n_components", 5)),
            reward_fn=reward_fn,
            phase_filter=phase_filter,
        )

    # Probe dims with one random episode.
    _rollout_random_episode(env)
    state_dim = len(env.path[0].state_material_features)
    step_dim  = env.n_components
    fraction_set = list(env.fraction_set)

    # Precompute Magpie element features (replaces one-hot element encoding).
    elem_feats_scaled, _elem_scaler = _precompute_elem_features(
        env.cation_set, env.state_featurizer
    )
    elem_dim = int(elem_feats_scaled.shape[1])
    frac_dim = 1

    print(f"[INFO] state_dim={state_dim}, step_dim={step_dim}, "
          f"elem_dim(Magpie)={elem_dim}, frac_dim={frac_dim}")

    hidden_dim = int(cfg.get("hidden_dim", 128))
    metrics = RunMetrics()

    # Generation strategy (unified across DQN and PG).
    gen_temperature = float(cfg.get("gen_temperature", 1.0))
    gen_top_frac    = float(cfg.get("gen_top_frac", 0.0))
    gen_epsilon_gen = float(cfg.get("gen_epsilon", 0.0))

    # ------------------------------------------------------------------
    # DQN path — classical online DQN
    # ------------------------------------------------------------------
    if method == "dqn":
        qnet, scaler, train_rows = train_dqn_online(
            env=env,
            elem_feats_scaled=elem_feats_scaled,
            fraction_set=fraction_set,
            device=device,
            hidden_dim=hidden_dim,
            n_warmup_eps=int(cfg.get("dqn_warmup_eps", cfg.get("pg_warmup_eps", 500))),
            n_train_eps=int(cfg.get("num_train_eps", 20000)),
            buffer_size=int(cfg.get("buffer_size", 50000)),
            batch_size=int(cfg.get("dqn_batch_size", 256)),
            grad_steps_per_ep=int(cfg.get("grad_steps_per_ep", 5)),
            target_update_freq=int(cfg.get("target_update_freq", 100)),
            eps_anneal_eps=int(cfg.get("eps_anneal_eps", 10000)),
            eps_min=float(cfg.get("eps_min", 0.05)),
            gamma=float(cfg.get("dqn_gamma", cfg.get("gamma", 0.9))),
            lr=float(cfg.get("dqn_lr", 1e-3)),
        )
        for r in train_rows:
            metrics.log(**r)

        torch.save(qnet.state_dict(), os.path.join(args.out, "qnet.pt"))
        joblib.dump(scaler, os.path.join(args.out, "std_scaler.bin"))

        np.random.seed(gen_seed)
        random.seed(gen_seed)
        gen_rows = generate_candidates(
            env=env, predictor=predictor, scaler=scaler, device=device,
            qnet=qnet,
            elem_feats_scaled=elem_feats_scaled,
            fraction_set=fraction_set,
            n_exploit=int(cfg.get("num_gen_eps", 200)),
            n_explore=int(cfg.get("exploration_gen_eps", 0)),
            gen_temperature=gen_temperature,
            gen_top_frac=gen_top_frac,
            gen_epsilon=gen_epsilon_gen,
            k=float(cfg.get("k", 1.0)),
        )
        for r in gen_rows:
            metrics.log(phase="generate", **r)

    # ------------------------------------------------------------------
    # PG paths (REINFORCE / A2C)
    # ------------------------------------------------------------------
    else:
        pg_warmup           = int(cfg.get("pg_warmup_eps", 200))
        pg_train            = int(cfg.get("pg_train_eps", 1000))
        lr_actor            = float(cfg.get("pg_lr_actor", 1e-3))
        lr_critic           = float(cfg.get("pg_lr_critic", 1e-3))
        entropy_coef        = float(cfg.get("entropy_coef", 0.01))
        gamma               = float(cfg.get("pg_gamma", cfg.get("gamma", 0.99)))
        repeat_penalty_coef = float(cfg.get("repeat_penalty_coef", 0.0))
        repeat_penalty_shape = cfg.get("repeat_penalty_shape", "log")

        scaler = _fit_scaler_from_warmup(env, pg_warmup)
        joblib.dump(scaler, os.path.join(args.out, "std_scaler.bin"))

        policy = PolicyNet(
            state_dim=state_dim, step_dim=step_dim,
            elem_dim=elem_dim, frac_dim=frac_dim, hidden_dim=hidden_dim,
        ).to(device)
        value_net = None
        if method == "a2c":
            value_net = ValueNet(
                state_dim=state_dim, step_dim=step_dim, hidden_dim=hidden_dim,
            ).to(device)

        train_rows = train_pg(
            policy=policy,
            value_net=value_net,
            env=env,
            scaler=scaler,
            elem_feats_scaled=elem_feats_scaled,
            fraction_set=fraction_set,
            device=device,
            n_episodes=pg_train,
            gamma=gamma,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            entropy_coef=entropy_coef,
            rl_method=method,
            repeat_penalty_coef=repeat_penalty_coef,
            repeat_penalty_shape=repeat_penalty_shape,
            normalise_returns=bool(cfg.get("normalise_returns", True)),
        )
        for r in train_rows:
            metrics.log(**r)

        torch.save(policy.state_dict(), os.path.join(args.out, "policy.pt"))
        if value_net is not None:
            torch.save(value_net.state_dict(), os.path.join(args.out, "value_net.pt"))

        np.random.seed(gen_seed)
        random.seed(gen_seed)
        gen_rows = generate_candidates(
            env=env, predictor=predictor, scaler=scaler, device=device,
            policy=policy,
            elem_feats_scaled=elem_feats_scaled,
            fraction_set=fraction_set,
            n_exploit=int(cfg.get("num_gen_eps", 200)),
            n_explore=int(cfg.get("exploration_gen_eps", 0)),
            gen_temperature=gen_temperature,
            gen_top_frac=gen_top_frac,
            gen_epsilon=gen_epsilon_gen,
            k=float(cfg.get("k", 1.0)),
            exploit_objective=cfg.get("objective", "mean_minus_kstd"),
        )
        for r in gen_rows:
            metrics.log(phase="generate", **r)

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    metrics.to_csv(os.path.join(args.out, "training_log.csv"))

    gen_rows_only = [r for r in metrics.rows if r.get("phase") == "generate"]
    if gen_rows_only:
        import csv
        fieldnames = list(gen_rows_only[0].keys())
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
