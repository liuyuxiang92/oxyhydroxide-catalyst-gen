#!/usr/bin/env python3
"""Bayesian-Optimization baseline for material composition design.

Uses Optuna's TPE sampler (a sequential model-based / Bayesian optimizer) to
search the **same constrained space** the RL agent searches, against the **same**
``PropertyPredictor`` (``build_predictor`` -> ``registry.build_reward``).  Validity
is guaranteed by replaying each suggested trajectory through the real env's
``allowed_actions()`` (see ``scripts/baselines/_common.py``), so BO never scores a
composition the RL agent could not have produced.

Search space
------------
Per env step ``k`` the trial suggests an integer ``step_k`` in ``[0, UB)``; the
decoder folds it into the live allowed-action set via modulo.  This fixed-width
integer parameterisation works uniformly for the integer-ratio (oxides) and
fraction (OOH) envs and keeps every trial feasible.

Usage
-----
    python scripts/baselines/run_bo.py --config configs/oxides_sinter.yaml \\
        --out runs/bo_sinter_seed0 --seed 0 --budget 1000

Requirements
------------
    pip install "optuna>=3.0" pyyaml
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import _common as C  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BO (Optuna TPE) baseline for composition design")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--budget", type=int, default=None,
                   help="Total trials = predictor-evaluation budget "
                        "(default: derived from config; set to match your RL run)")
    p.add_argument("--n-startup-trials", type=int, default=20,
                   help="Random warmup trials before TPE modelling kicks in")
    p.add_argument("--no-study-db", action="store_true",
                   help="Do not persist the Optuna study to <out>/study.db")
    return p.parse_args()


def main() -> None:
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover
        raise ImportError("BO baseline requires Optuna: pip install 'optuna>=3.0'") from exc

    args = parse_args()

    from rl_matdesign.utils.seeding import set_global_seed
    set_global_seed(args.seed)

    cfg = C.load_scenario(args.config)
    os.makedirs(args.out, exist_ok=True)

    predictor, env, env_type = C.make_predictor_and_env(cfg, seed=args.seed)
    n = C.n_steps(env)
    ub = C.action_upper_bound(env)
    budget = args.budget or C.default_budget(cfg)
    print(f"[BO] env_type={env_type}, n_components={n}, action_ub={ub}, "
          f"budget(valid trials)={budget}, startup={args.n_startup_trials}")

    cache: dict = {}
    eval_count = [0]
    # Unique candidates keyed on canonical composition -> best row.
    results: dict = {}
    valid_count = [0]
    invalid_count = [0]

    def objective(trial: "optuna.Trial") -> float:
        choices = [trial.suggest_int(f"step_{k}", 0, ub - 1) for k in range(n)]
        decoded = C.decode_choices(env, choices)
        if decoded is None:
            # Dead-end trajectory (a custom filter emptied the action set). The RL
            # agent's env guarantees every episode completes, so a dead-end here is
            # not something an RL episode would ever spend on — prune it and don't
            # let it consume the valid-trial budget below; a duplicate of an
            # already-seen (valid) composition still does, since RL pays for that
            # too (one episode, cached predictor call).
            invalid_count[0] += 1
            raise optuna.TrialPruned()
        valid_count[0] += 1
        mean, std = C.score_composition(predictor, decoded["comp"], cache, eval_count)
        key = decoded["key"]
        if key not in results or mean > results[key]["reward"]:
            results[key] = {
                "formula": decoded["formula"],
                "purpose": "exploit",
                "reward": mean,
                "dp_mean": mean,
                "dp_std": std,
            }
        return mean

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(
        seed=args.seed, n_startup_trials=args.n_startup_trials,
    )
    storage = None if args.no_study_db else f"sqlite:///{os.path.join(args.out, 'study.db')}"
    study = optuna.create_study(
        direction="maximize", sampler=sampler,
        study_name=f"bo_{os.path.basename(args.out)}",
        storage=storage, load_if_exists=True,
    )

    # n_trials=None + a stop callback lets us keep sampling past pruned (invalid)
    # trials until `budget` valid ones have been scored, rather than letting dead
    # ends eat into the budget. `_safety_cap` bounds the loop if a config makes
    # almost every trajectory infeasible.
    _safety_cap = max(budget * 20, budget + 1000)

    def _stop_when_budget_reached(study: "optuna.Study", _trial: "optuna.trial.FrozenTrial") -> None:
        if valid_count[0] >= budget or (valid_count[0] + invalid_count[0]) >= _safety_cap:
            study.stop()

    study.optimize(objective, n_trials=None, callbacks=[_stop_when_budget_reached],
                    show_progress_bar=False)
    if valid_count[0] < budget:
        print(f"[BO] WARNING: hit safety cap of {_safety_cap} total trials with only "
              f"{valid_count[0]}/{budget} valid — search space may be too constrained.")

    rows = list(results.values())
    out_csv = C.write_outputs(
        args.out, rows,
        run_config={
            "method": "bo", "backend": "optuna_tpe",
            "config": args.config, "seed": args.seed,
            "budget": budget, "n_startup_trials": args.n_startup_trials,
            "valid_trials": valid_count[0],
            "invalid_trials_skipped": invalid_count[0],
            "total_evaluations": eval_count[0],
            "n_unique_candidates": len(rows),
        },
    )
    best = max(rows, key=lambda r: r["reward"]) if rows else None
    print(f"[BO] Done. {valid_count[0]} valid trials ({invalid_count[0]} invalid skipped), "
          f"{eval_count[0]} unique predictor evals, {len(rows)} unique candidates.")
    if best:
        print(f"[BO] Best: {best['formula']} reward={best['reward']:.4f}")
    print(f"[BO] Results -> {out_csv}")


if __name__ == "__main__":
    main()
