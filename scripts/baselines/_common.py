#!/usr/bin/env python3
"""Shared helpers for the baseline optimizers (BO, GA).

The goal is that "a valid candidate" means *exactly* the same thing for the RL
agent and for the classical baselines.  Rather than re-deriving constraint rules
(charge neutrality, Pauling-EN, "O reserved for the last step", distinct cations,
sum-to-total feasibility, OOH phase rules), the baselines drive the **same env**
that ``run_experiment.py`` builds and validate candidates by *replaying* an action
trajectory through ``env.allowed_actions()`` / ``env.step()``.

Candidate representation
------------------------
A candidate is a list of ``n_components`` integer *choices*, one per env step.
At step ``k`` the env exposes a (constraint-filtered, feasibility-guaranteed) list
of allowed actions ``L_k``; choice ``c_k`` selects ``L_k[c_k % len(L_k)]``.  This
is uniform across env types:

* oxides (:class:`IntegerRatioEnv`): each action is (element, integer-digit);
  repeats are allowed and merged, O is forced into the last slot by the filter.
* OOH (:class:`CompositionEnv`): each action is (distinct element, fraction) and
  the env prunes amounts so the fractions can still sum to ``total_units``.

Because every selected action comes from ``allowed_actions()``, every decoded
candidate is valid by construction — no candidate is ever scored that the RL
agent could not also have produced.

Scoring
-------
The env's ``reward_fn`` is neutralised after construction (so replay is free of
predictor calls), and scoring is done explicitly via ``score_composition`` with a
cache keyed on the canonical composition, mirroring the RL ``reward`` column:
``reward = predictor.predict(comp)[0]`` where ``comp = env.terminal_cation_fractions()``.
"""

from __future__ import annotations

import csv
import json
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_SCRIPTS_DIR = os.path.join(_REPO_ROOT, "scripts")

# Reused by replay: we never read the stored material features, so swap in a
# trivial featurizer to skip the (Magpie) featurization cost on every step.
_ZERO_FEATS = np.zeros(1, dtype=float)


def _ensure_import_path() -> None:
    """Make ``run_experiment`` importable (mirrors run_ga.py)."""
    if _SCRIPTS_DIR not in os.sys.path:
        os.sys.path.insert(0, _SCRIPTS_DIR)


# ---------------------------------------------------------------------------
# Scenario / env construction
# ---------------------------------------------------------------------------

def load_scenario(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def make_predictor_and_env(cfg: dict, seed: Optional[int] = None):
    """Build the same predictor + env (with constraints) used by run_experiment.

    Returns ``(predictor, env, env_type)``.  The env's ``reward_fn`` is replaced
    with a no-op so replaying trajectories does not trigger predictor calls;
    callers score explicitly via :func:`score_composition`.
    """
    _ensure_import_path()
    from run_experiment import build_predictor, build_env

    predictor = build_predictor(cfg, seed=seed)
    env, env_type, _flat_cfg = build_env(cfg, predictor)

    # Neutralise scoring during replay; speed up replay by skipping featurization.
    env.reward_fn = lambda *_args, **_kw: 0.0
    env.state_featurizer = lambda _s: _ZERO_FEATS
    # Baselines always search the constrained space (generation semantics).
    env.constraints_enabled = True
    return predictor, env, env_type


def n_steps(env) -> int:
    return int(env.n_components)


def action_upper_bound(env) -> int:
    """Upper bound on the per-step action count (before constraint pruning).

    Used as the fixed integer range for BO's per-step choice variables; the
    decoder folds the suggested value into the live allowed set via modulo.
    """
    return max(1, len(env.species_set) * len(env.fraction_set))


# ---------------------------------------------------------------------------
# Trajectory decode / validity / random sampling
# ---------------------------------------------------------------------------

def decode_choices(env, choices: List[int]) -> Optional[Dict[str, object]]:
    """Replay a list of per-step choices through the env.

    Returns ``{"formula", "comp", "key"}`` for the terminal composition, or
    ``None`` if a dead-end (no allowed actions) is hit before termination.
    ``comp`` is the ``{element: fraction}`` dict the predictor expects (anion
    stripped for OOH; O included for oxides), matching ``run_experiment``'s
    ``reward_fn`` input.
    """
    env.initialize()
    n = n_steps(env)
    for k in range(n):
        allowed = env.allowed_actions()
        if not allowed:
            return None
        idx = int(choices[k]) % len(allowed)
        env.step(allowed[idx])
    if env.counter != n:
        return None
    return {
        "formula": env.terminal_formula,
        "comp": dict(env.terminal_cation_fractions()),
        "key": env.terminal_comp_key(),
    }


def random_choices(env, rng: random.Random) -> Tuple[List[int], Dict[str, object]]:
    """Walk the env choosing a random allowed action each step.

    Returns ``(choices, decoded)`` where ``choices`` decode back to the same
    valid terminal composition.  Guaranteed valid (the env only offers feasible,
    constraint-passing actions).
    """
    env.initialize()
    n = n_steps(env)
    choices: List[int] = []
    for _ in range(n):
        allowed = env.allowed_actions()
        if not allowed:
            # Extremely defensive: env guarantees completability, but a custom
            # filter could empty the set. Retry from scratch.
            return random_choices(env, rng)
        idx = rng.randrange(len(allowed))
        choices.append(idx)
        env.step(allowed[idx])
    return choices, {
        "formula": env.terminal_formula,
        "comp": dict(env.terminal_cation_fractions()),
        "key": env.terminal_comp_key(),
    }


# ---------------------------------------------------------------------------
# Scoring (cached)
# ---------------------------------------------------------------------------

def _comp_cache_key(comp) -> object:
    """Canonical hashable key for a (possibly nested) composition mapping.

    Single-group envs (OOH, oxides) pass ``{element: fraction}``. Multi-group
    envs (e.g. the perovskite ABO3 scenario) pass the *structured*
    ``{group_name: {element: fraction}}`` mapping straight through to
    ``predictor.predict`` (see ``env_multigroup.py:498``), so the key builder
    must recurse rather than assume every value is a float.
    """
    if isinstance(comp, dict):
        return tuple(sorted((str(k), _comp_cache_key(v)) for k, v in comp.items()))
    return round(float(comp), 6)


def score_composition(predictor, comp, cache: dict,
                      counter: Optional[List[int]] = None) -> Tuple[float, float]:
    """Cached ``predictor.predict``.

    ``reward = predict(comp)[0]`` (same scalar the RL ``generated.csv`` stores).
    ``cache`` is keyed on the canonical composition so duplicates don't consume
    the evaluation budget; ``counter[0]`` (if given) counts unique predictor calls.
    """
    key = _comp_cache_key(comp)
    if key not in cache:
        mean, std = predictor.predict(comp)
        cache[key] = (float(mean), float(std) if std is not None else float("nan"))
        if counter is not None:
            counter[0] += 1
    return cache[key]


# ---------------------------------------------------------------------------
# Output (RL-compatible generated.csv)
# ---------------------------------------------------------------------------

_GENERATED_FIELDS = ["formula", "purpose", "reward", "dp_mean", "dp_std"]


def write_outputs(out_dir: str, rows: List[dict], run_config: dict) -> str:
    """Write ``generated.csv`` (RL-compatible schema) + ``run_config.json``.

    ``rows`` must contain ``formula, reward, dp_mean, dp_std`` (``purpose``
    defaults to ``"exploit"``).  Rows are sorted by ``reward`` descending.
    Returns the path to ``generated.csv``.
    """
    os.makedirs(out_dir, exist_ok=True)
    for r in rows:
        r.setdefault("purpose", "exploit")
    rows = sorted(rows, key=lambda r: -r["reward"])

    out_csv = os.path.join(out_dir, "generated.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_GENERATED_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in _GENERATED_FIELDS})

    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)
    return out_csv


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

def default_budget(cfg: dict) -> int:
    """Predictor-evaluation budget that approximates the RL run's episode count.

    DQN: ``dqn_warmup_eps + dqn_num_train_eps + num_gen_eps``.
    PG (reinforce/a2c): ``pg_warmup_eps + pg_num_iters*pg_batch_eps + num_gen_eps``.

    This is a starting point; for a strictly fair comparison the caller should
    pass ``--budget`` matching the predictor-call count of the *actual* RL run.
    """
    gen = int(cfg.get("num_gen_eps", 200))
    method = str(cfg.get("method", "a2c"))
    if method == "dqn":
        train = int(cfg.get("dqn_warmup_eps", 0)) + int(cfg.get("dqn_num_train_eps", 0))
    else:
        train = (int(cfg.get("pg_warmup_eps", 0))
                 + int(cfg.get("pg_num_iters", 0)) * int(cfg.get("pg_batch_eps", 0)))
    return (train or 1000) + gen
