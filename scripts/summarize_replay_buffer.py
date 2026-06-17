from __future__ import annotations

"""Summarize a DQN replay buffer into per-episode formulas and rewards.

The classical online DQN (``train_dqn_online`` in ``rl_matdesign/training.py``)
keeps its replay buffer in memory as a ``collections.deque`` of per-step
transition rows. The buffer is persisted to disk only inside the periodic
mid-run checkpoint (``checkpoint.pt``), under the ``"buffer"`` key. There is no
longer a standalone ``random_dataset.npz``.

Each buffer row is a dict with these keys (see ``add_episode_to_buffer``):

    s_mat_raw         np.ndarray  unscaled material features of the state
    s_step            np.ndarray  one-hot of the step counter
    a_elem_idx        int         index into species_set of the chosen cation
    a_comp_val        float       chosen fraction value (e.g. 0.20)
    reward            float       immediate reward (nonzero only at terminal)
    s_mat_next_raw    np.ndarray  next-state material features (zeros if done)
    s_step_next       np.ndarray  next-state step one-hot (zeros if done)
    next_allowed_idx  list        (elem_idx, comp_idx) pairs for the next state
    done              bool        terminal-step flag

This script reconstructs episodes by splitting the row sequence on
``done=True`` boundaries (rows are appended in episode/step order), decodes the
composition from ``a_elem_idx`` + ``a_comp_val``, and reports the terminal
reward. Optionally it recomputes the predictor mean/std for each unique
composition using the run's configured predictor.
"""

import argparse
import csv
import json
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Make the package importable without installation (mirrors run_experiment.py).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Avoid noisy matminer warnings triggered by importing the featurizer.
for _msg in (
    r"^MagpieData\(impute_nan=False\):.*",
    r"^PymatgenData\(impute_nan=False\):.*",
    r"^ValenceOrbital\(impute_nan=False\):.*",
    r"^IonProperty\(impute_nan=False\):.*",
):
    warnings.filterwarnings("ignore", message=_msg)


def _load_run_config(run_dir: str) -> dict:
    cfg_path = os.path.join(run_dir, "run_config.json")
    if not os.path.exists(cfg_path):
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_buffer(checkpoint_path: str) -> List[dict]:
    """Load the ``buffer`` list from a DQN mid-run checkpoint."""
    import torch

    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict) or raw.get("type") != "dqn" or "buffer" not in raw:
        _type = raw.get("type") if isinstance(raw, dict) else type(raw).__name__
        raise SystemExit(
            f"{checkpoint_path} is not a DQN mid-checkpoint (type={_type!r}). "
            "Pass a checkpoint.pt written by a --method dqn run."
        )
    return list(raw["buffer"])


def _split_episodes(buffer: List[dict]) -> List[List[dict]]:
    """Group consecutive rows into episodes, splitting after each ``done`` row.

    Rows are appended in episode/step order, so a ``done=True`` row closes the
    current episode. A FIFO eviction can leave a partial leading episode (its
    first steps dropped); such a trailing-incomplete group at the *end* (no
    closing ``done``) is discarded.
    """
    episodes: List[List[dict]] = []
    current: List[dict] = []
    for row in buffer:
        current.append(row)
        if bool(row.get("done", False)):
            episodes.append(current)
            current = []
    return episodes


def _episode_composition(episode: List[dict], species_set: List[str]) -> Dict[str, float]:
    """Reconstruct the {symbol: fraction} mapping from one episode's rows."""
    comp: Dict[str, float] = {}
    for row in episode:
        idx = int(row["a_elem_idx"])
        if idx < 0 or idx >= len(species_set):
            raise ValueError(
                f"a_elem_idx={idx} out of range for species_set of size {len(species_set)}"
            )
        el = species_set[idx]
        comp[el] = comp.get(el, 0.0) + float(row["a_comp_val"])
    return comp


def _canonical_formula(
    comp: Dict[str, float],
    *,
    anion_formula: str,
    total_units: int,
) -> str:
    """Canonical formula string (major-first, alphabetical tie-break).

    Mirrors ``CompositionEnv.terminal_formula`` so formulas match generated.csv.
    """
    from rl_matdesign.env import _format_fraction

    items: List[Tuple[str, int]] = []
    for el, frac in comp.items():
        units = int(round(float(frac) * total_units))
        if units <= 0:
            continue
        items.append((el, units))
    items.sort(key=lambda t: (-t[1], t[0]))
    body = "".join(f"{el}{_format_fraction(units, total_units)}" for el, units in items)
    return f"{body}{anion_formula}"


@dataclass(frozen=True)
class Episode:
    formula: str
    comp: Dict[str, float]
    terminal_reward: float
    n_rows: int


def _decode_episodes(
    buffer: List[dict],
    *,
    species_set: List[str],
    anion_formula: str,
    total_units: int,
) -> List[Episode]:
    out: List[Episode] = []
    for ep_rows in _split_episodes(buffer):
        comp = _episode_composition(ep_rows, species_set)
        formula = _canonical_formula(
            comp, anion_formula=anion_formula, total_units=total_units
        )
        # Reward is nonzero only on the terminal (done) row; sum is robust either way.
        terminal_reward = float(sum(float(r.get("reward", 0.0)) for r in ep_rows))
        out.append(Episode(formula, comp, terminal_reward, len(ep_rows)))
    return out


def _maybe_build_predictor(cfg: dict, seed: int):
    """Build the run's configured predictor for optional reward recompute."""
    from rl_matdesign.registry import build_reward

    if not cfg.get("properties") and not cfg.get("predictor"):
        raise SystemExit(
            "--recompute requires a 'properties' list or a 'predictor' entry in "
            "run_config.json (or rerun without --recompute)."
        )
    return build_reward(cfg, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize a DQN replay buffer (from checkpoint.pt) into per-episode "
            "formulas and terminal rewards. Optionally recompute predictor "
            "mean/std for each unique composition."
        )
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run directory containing checkpoint.pt and run_config.json",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path (default: <run-dir>/checkpoint.pt)",
    )
    parser.add_argument(
        "--out-csv",
        default=None,
        help="Output CSV path (default: <run-dir>/replay_buffer_summary.csv)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on number of episodes written (for quick inspection)",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute predictor mean/std for each unique composition using "
             "the predictor configured in run_config.json.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for the recompute predictor (default: run_config 'dp_seed' or 0).",
    )

    args = parser.parse_args()

    run_cfg = _load_run_config(args.run_dir)
    species_set = run_cfg.get("species_set")
    if not species_set:
        raise SystemExit(
            "run_config.json missing 'species_set' — cannot decode a_elem_idx. "
            f"Looked in {os.path.join(args.run_dir, 'run_config.json')}"
        )
    anion_formula = run_cfg.get("anion_formula", "")
    total_units = int(run_cfg.get("total_units", 20))

    ckpt_path = args.checkpoint or os.path.join(args.run_dir, "checkpoint.pt")
    if not os.path.exists(ckpt_path):
        raise SystemExit(
            f"Not found: {ckpt_path}. The DQN buffer is saved inside the periodic "
            "checkpoint; ensure the run was DQN and checkpointing was enabled."
        )

    buffer = _load_buffer(ckpt_path)
    episodes = _decode_episodes(
        buffer,
        species_set=species_set,
        anion_formula=anion_formula,
        total_units=total_units,
    )
    if not episodes:
        raise SystemExit("No complete episodes found in buffer (no 'done' rows).")

    predictor = None
    if args.recompute:
        seed = args.seed if args.seed is not None else int(run_cfg.get("dp_seed", 0))
        predictor = _maybe_build_predictor(run_cfg, seed)

    out_csv = args.out_csv or os.path.join(args.run_dir, "replay_buffer_summary.csv")

    # Cache predictor calls by canonical composition key.
    pred_cache: Dict[Tuple[Tuple[str, float], ...], Tuple[float, float]] = {}

    n = len(episodes)
    limit = min(n, int(args.max_rows)) if args.max_rows is not None else n

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["formula", "terminal_reward", "n_buffer_rows"]
        if predictor is not None:
            fieldnames += ["pred_mean", "pred_std"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(limit):
            ep = episodes[i]
            row = {
                "formula": ep.formula,
                "terminal_reward": ep.terminal_reward,
                "n_buffer_rows": ep.n_rows,
            }
            if predictor is not None:
                key = tuple(sorted((k, float(v)) for k, v in ep.comp.items()))
                if key in pred_cache:
                    mean, std = pred_cache[key]
                else:
                    mean, std = predictor.predict(ep.comp)
                    mean, std = float(mean), float(std)
                    pred_cache[key] = (mean, std)
                row["pred_mean"] = mean
                row["pred_std"] = std
            writer.writerow(row)

    rewards = np.asarray([ep.terminal_reward for ep in episodes[:limit]], dtype=float)
    print(f"Wrote {limit} episodes ({len(buffer)} buffer rows) -> {out_csv}")
    print(
        "terminal_reward stats:",
        {
            "min": float(np.min(rewards)),
            "p10": float(np.quantile(rewards, 0.10)),
            "median": float(np.median(rewards)),
            "mean": float(np.mean(rewards)),
            "p90": float(np.quantile(rewards, 0.90)),
            "max": float(np.max(rewards)),
        },
    )


if __name__ == "__main__":
    main()
