#!/usr/bin/env python3
"""Generic Optuna-based hyperparameter-optimization driver for rl-matdesign.

Two-stage protocol:

  Stage 1 — cheap screen. N trials at reduced training budget. Each trial
            runs `seeds_per_trial_stage1` independent seeds; the objective is
            the mean across seeds of `top_k_mean(generated.csv)`.
  Stage 2 — confirm. Re-run the top-`n_top_to_confirm` trials at full base
            budget with `seeds_per_trial_stage2` seeds and re-rank.

Per-trial state lives in an Optuna SQLite study so the driver is crash-safe
and resumable: relaunch against the same `--out` to continue from where it
stopped. Stage progression is tracked via a state.json sentinel.

Usage
-----
    python scripts/hpo.py \\
        --hpo-config configs/hpo/ooh_dqn.yaml \\
        --out runs/hpo/ooh_dqn_v1

See `configs/hpo/*.yaml` for the search-space schema.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import queue
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure src/ is on the path (matches scripts/run_experiment.py and run_seeds.py).
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np
import optuna
import yaml

from rl_matdesign.hpo import (
    TrialRunError,
    run_trial_subprocess,
    sample_from_search_space,
    top_k_mean_from_csv,
    validate_search_space,
)


logging.basicConfig(level=logging.INFO, format="[hpo] %(message)s")
log = logging.getLogger("hpo")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# HPO config loading
# ---------------------------------------------------------------------------


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _resolve_base_config(hpo_config_path: str, base_config_ref: str) -> str:
    """Resolve `base_config:` relative to the HPO YAML if not absolute."""
    if os.path.isabs(base_config_ref):
        return base_config_ref
    cand1 = os.path.join(os.path.dirname(hpo_config_path), base_config_ref)
    if os.path.exists(cand1):
        return os.path.abspath(cand1)
    cand2 = _REPO_ROOT / base_config_ref
    if cand2.exists():
        return str(cand2.resolve())
    raise FileNotFoundError(f"base_config not found: {base_config_ref}")


def _validate_hpo_config(hpo_cfg: dict, hpo_config_path: str) -> dict:
    required = ["base_config", "study_name", "method", "search_space"]
    missing = [k for k in required if k not in hpo_cfg]
    if missing:
        raise SystemExit(f"hpo_config missing required keys: {missing}")
    if hpo_cfg["method"] not in ("dqn", "reinforce", "a2c"):
        raise SystemExit(f"method must be dqn|reinforce|a2c, got {hpo_cfg['method']!r}")

    # Defaults.
    hpo_cfg["base_config"] = _resolve_base_config(hpo_config_path, hpo_cfg["base_config"])
    hpo_cfg.setdefault("fixed_overrides", {})
    hpo_cfg.setdefault("metric", {"kind": "top_k_mean", "k": 10})
    hpo_cfg.setdefault("optuna", {"seed": 0})
    hpo_cfg.setdefault(
        "stage1",
        {
            "n_trials": 30,
            "n_parallel_trials": 1,
            "seeds_per_trial": 1,
            "budget_scaling": {
                "dqn_num_train_eps": 0.25,
                "dqn_warmup_eps": 0.25,
                "dqn_eps_anneal_eps": 0.25,
                "pg_num_iters": 0.25,
                "pg_warmup_eps": 0.25,
            },
        },
    )
    hpo_cfg.setdefault(
        "stage2",
        {"n_top_to_confirm": 3, "seeds_per_trial": 3},
    )

    # Validate the search space against base config + fixed overrides so
    # frac_of references resolve cleanly.
    base_cfg = _load_yaml(hpo_cfg["base_config"])
    validate_search_space(
        hpo_cfg["search_space"],
        fixed_overrides=hpo_cfg["fixed_overrides"],
        base_values=base_cfg,
    )

    if hpo_cfg["metric"].get("kind", "top_k_mean") != "top_k_mean":
        raise SystemExit("only metric.kind=top_k_mean is supported in v1")
    return hpo_cfg


# ---------------------------------------------------------------------------
# Per-trial YAML writer
# ---------------------------------------------------------------------------


def _build_trial_yaml(
    *,
    base_cfg: dict,
    fixed_overrides: dict,
    sampled: dict,
    budget_scaling: dict,
    method: str,
) -> dict:
    """Merge base + fixed + sampled, apply budget scaling, force method.

    Budget scaling: multiplicative on integer-typed budget knobs. Applied
    AFTER sampled values are merged (so a sampled budget knob is also scaled).
    """
    cfg = dict(base_cfg)
    cfg.update(fixed_overrides)
    cfg.update(sampled)
    cfg["method"] = method

    # Apply budget scaling. Scale relative to the resolved value (which may
    # have come from base_cfg, fixed_overrides, or the sampled dict).
    for key, mult in (budget_scaling or {}).items():
        if key in cfg and isinstance(cfg[key], (int, float)):
            cfg[key] = max(1, int(round(cfg[key] * float(mult))))

    # Defensive clamp: anneal eps cannot exceed train eps.
    if "dqn_eps_anneal_eps" in cfg and "dqn_num_train_eps" in cfg:
        cfg["dqn_eps_anneal_eps"] = min(int(cfg["dqn_eps_anneal_eps"]), int(cfg["dqn_num_train_eps"]))

    return cfg


def _dump_trial_yaml(path: str, cfg: dict) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


# ---------------------------------------------------------------------------
# GPU pool for parallel trials
# ---------------------------------------------------------------------------


def _build_gpu_pool(n_parallel: int, gpu_ids: Optional[List[int]]) -> Optional["queue.Queue[int]"]:
    if not gpu_ids or n_parallel <= 1:
        return None
    q: "queue.Queue[int]" = queue.Queue()
    for gid in gpu_ids:
        q.put(gid)
    return q


# ---------------------------------------------------------------------------
# Per-trial objective
# ---------------------------------------------------------------------------


def _run_one_seed(
    *,
    trial_yaml: str,
    seed_out: str,
    method: str,
    seed_idx: int,
    gpu_pool: Optional["queue.Queue[int]"],
    timeout_s: Optional[int],
) -> Tuple[float, int]:
    """Run one seed, return (top_k_mean, n_used). Raises TrialRunError on failure."""
    env_extra = None
    gid = None
    if gpu_pool is not None:
        gid = gpu_pool.get()
        env_extra = {"CUDA_VISIBLE_DEVICES": str(gid)}
    try:
        result = run_trial_subprocess(
            config_yaml=trial_yaml,
            out_dir=seed_out,
            method=method,
            train_seed=seed_idx,
            dp_seed=seed_idx + 10_000,
            gen_seed=seed_idx + 20_000,
            timeout_s=timeout_s,
            env_extra=env_extra,
        )
    finally:
        if gid is not None and gpu_pool is not None:
            gpu_pool.put(gid)

    # Always persist subprocess tails so failures are debuggable after the run.
    try:
        with open(os.path.join(seed_out, "stdout.log"), "w") as f:
            f.write(result.stdout_tail)
        with open(os.path.join(seed_out, "stderr.log"), "w") as f:
            f.write(result.stderr_tail)
    except OSError:
        pass  # never let log-writing kill a trial

    if result.returncode != 0:
        # Log to console too — otherwise Optuna's `catch=` swallows the message
        # and the user only sees "0 completed trials" at the end.
        log.warning(
            "trial subprocess failed (rc=%d) in %s; stderr tail:\n%s",
            result.returncode, seed_out, result.stderr_tail[-1500:] or "<empty>",
        )
        raise TrialRunError(
            f"run_experiment exited with code {result.returncode}; "
            f"see {seed_out}/stderr.log"
        )
    if not os.path.exists(result.generated_csv):
        log.error(
            "run_experiment exited 0 but %s is missing; stderr tail:\n%s",
            result.generated_csv, result.stderr_tail[-1500:] or "<empty>",
        )
        raise RuntimeError(
            f"run_experiment exited 0 but {result.generated_csv} is missing; "
            f"see {seed_out}/stderr.log"
        )
    return top_k_mean_from_csv(result.generated_csv, k=int(_METRIC_K))


def _make_objective(
    *,
    hpo_cfg: dict,
    out_root: str,
    stage_name: str,
    base_cfg: dict,
    budget_scaling: dict,
    seeds_per_trial: int,
    gpu_pool: Optional["queue.Queue[int]"],
    timeout_s: Optional[int],
):
    method = hpo_cfg["method"]
    fixed = hpo_cfg["fixed_overrides"]
    spec = hpo_cfg["search_space"]
    metric_k = int(hpo_cfg["metric"].get("k", 10))

    def objective(trial: optuna.Trial) -> float:
        # Resolve values for frac_of bases (fixed + base config).
        resolved = dict(base_cfg)
        resolved.update(fixed)
        sampled = sample_from_search_space(trial, spec, resolved_values=resolved)

        trial_dir = os.path.join(out_root, stage_name, f"trial_{trial.number:04d}")
        os.makedirs(trial_dir, exist_ok=True)
        cfg = _build_trial_yaml(
            base_cfg=base_cfg,
            fixed_overrides=fixed,
            sampled=sampled,
            budget_scaling=budget_scaling,
            method=method,
        )
        trial_yaml = os.path.join(trial_dir, "config.yaml")
        _dump_trial_yaml(trial_yaml, cfg)
        trial.set_user_attr("trial_dir", trial_dir)
        trial.set_user_attr("config_path", trial_yaml)

        # Stash the effective metric k on the module global so the per-seed
        # helper doesn't need it passed every time (closure works too; this
        # keeps the helper signature symmetric across uses).
        global _METRIC_K
        _METRIC_K = metric_k

        seed_scores: List[float] = []
        n_useds: List[int] = []
        for i in range(seeds_per_trial):
            seed_out = os.path.join(trial_dir, f"seed_{i}")
            score, n_used = _run_one_seed(
                trial_yaml=trial_yaml,
                seed_out=seed_out,
                method=method,
                seed_idx=i,
                gpu_pool=gpu_pool,
                timeout_s=timeout_s,
            )
            seed_scores.append(float(score))
            n_useds.append(int(n_used))

        trial.set_user_attr("seed_scores", seed_scores)
        trial.set_user_attr("seed_std", float(np.std(seed_scores)) if len(seed_scores) > 1 else 0.0)
        trial.set_user_attr("n_generated_used", n_useds)
        return float(np.mean(seed_scores))

    return objective


# module-global set by _make_objective; consumed by _run_one_seed
_METRIC_K = 10


# ---------------------------------------------------------------------------
# Stage summaries + final report
# ---------------------------------------------------------------------------


def _write_stage_summary(study: optuna.Study, csv_path: str) -> None:
    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        row: Dict[str, Any] = {"trial_id": t.number, "mean_score": t.value}
        seed_scores = t.user_attrs.get("seed_scores", [])
        row["seed_scores"] = json.dumps(seed_scores)
        row["seed_std"] = t.user_attrs.get("seed_std", 0.0)
        row["n_generated_used"] = json.dumps(t.user_attrs.get("n_generated_used", []))
        for k, v in t.params.items():
            row[f"param.{k}"] = v
        rows.append(row)
    if not rows:
        log.warning("no COMPLETE trials to summarize at %s", csv_path)
        return
    rows.sort(key=lambda r: r["mean_score"], reverse=True)
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _stage1_variance_warning(study: optuna.Study, k_confirm: int) -> Optional[str]:
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) < k_confirm:
        return None
    completed.sort(key=lambda t: t.value, reverse=True)
    rank1 = completed[0]
    rankK = completed[k_confirm - 1]
    spread = rank1.value - rankK.value
    rank1_std = float(rank1.user_attrs.get("seed_std", 0.0))
    if spread > 0 and rank1_std > 0.5 * spread:
        return (
            f"Stage-1 rank-1 trial #{rank1.number} has seed_std={rank1_std:.4g} which is "
            f">50% of the spread to rank-{k_confirm} ({spread:.4g}). Treat the ranking as noisy."
        )
    return None


def _write_final_report(
    *,
    out_root: str,
    hpo_cfg: dict,
    stage1_study: optuna.Study,
    stage2_records: List[Dict[str, Any]],
) -> None:
    """Stage-2 records: list of {trial_id, mean_score, std, config_path, seed_scores}."""
    report_path = os.path.join(out_root, "final_report.md")
    if not stage2_records:
        with open(report_path, "w") as f:
            f.write("# HPO Final Report\n\nStage 2 produced no completed trials.\n")
        return
    stage2_records_sorted = sorted(stage2_records, key=lambda r: r["mean_score"], reverse=True)
    winner = stage2_records_sorted[0]
    base_config_rel = os.path.relpath(hpo_cfg["base_config"], _REPO_ROOT)

    lines = []
    lines.append("# HPO Final Report\n")
    lines.append(f"- study: `{hpo_cfg['study_name']}`")
    lines.append(f"- method: `{hpo_cfg['method']}`")
    lines.append(f"- base config: `{base_config_rel}`\n")
    lines.append("## Recommended configuration\n")
    lines.append(f"Stage-2 rank-1 = stage-1 trial **#{winner['stage1_trial_id']}**.")
    lines.append(f"Stage-2 mean score: **{winner['mean_score']:.6g}**  (std across "
                 f"{len(winner['seed_scores'])} seeds: {winner['std']:.4g})\n")
    lines.append("Reproduce with:\n")
    cfg_rel = os.path.relpath(winner["config_path"], _REPO_ROOT)
    lines.append("```bash")
    lines.append(f"python scripts/run_experiment.py \\")
    lines.append(f"    --config {cfg_rel} \\")
    lines.append(f"    --method {hpo_cfg['method']} \\")
    lines.append(f"    --out runs/<your_name> \\")
    lines.append(f"    --train-seed 0 --dp-seed 10000 --gen-seed 20000")
    lines.append("```\n")
    lines.append("## Stage-2 ranking\n")
    lines.append("| rank | stage-1 trial id | mean | std | seed_scores |")
    lines.append("|------|------------------|------|-----|-------------|")
    for rank, r in enumerate(stage2_records_sorted, start=1):
        lines.append(
            f"| {rank} | {r['stage1_trial_id']} | {r['mean_score']:.6g} | "
            f"{r['std']:.4g} | {r['seed_scores']} |"
        )
    warn = _stage1_variance_warning(stage1_study, hpo_cfg["stage2"]["n_top_to_confirm"])
    if warn:
        lines.append(f"\n> **Note**: {warn}\n")

    analysis_dir = os.path.join(out_root, "analysis")
    if os.path.isdir(analysis_dir) and os.listdir(analysis_dir):
        lines.append("\n## Analysis artifacts\n")
        lines.append(f"See `{os.path.relpath(analysis_dir, out_root)}/`:")
        for name, blurb in (
            ("param_importances.csv", "fANOVA-style importance per hyperparameter (higher = matters more)"),
            ("param_importances.png", "bar chart of the above"),
            ("optimization_history.png", "per-trial score; sanity-check that the search actually improved"),
            ("parallel_coordinate.png", "trial trajectories through the search space; spot clusters"),
            ("slice.png", "per-param score scatter; spot which knobs drive the score"),
        ):
            p = os.path.join(analysis_dir, name)
            if os.path.exists(p):
                lines.append(f"- `{name}` — {blurb}")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Analysis outputs: param importances + matplotlib plots
# ---------------------------------------------------------------------------


def _write_analysis_outputs(
    *,
    study: optuna.Study,
    out_root: str,
    do_plots: bool,
) -> None:
    """Best-effort: write param_importances.csv and matplotlib PNGs to <out>/analysis/.

    All failures are logged as warnings — analysis is decorative; a failure
    here must not kill the HPO run.
    """
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) < 2:
        log.warning("analysis: need >=2 completed trials for importances/plots, have %d; skipping",
                    len(completed))
        return

    analysis_dir = os.path.join(out_root, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    try:
        importances = optuna.importance.get_param_importances(study)
        with open(os.path.join(analysis_dir, "param_importances.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["param", "importance"])
            for k, v in importances.items():
                w.writerow([k, v])
        log.info("analysis: wrote param_importances.csv (%d params)", len(importances))
    except Exception as e:
        log.warning("analysis: param importances failed: %s", e)

    if not do_plots:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from optuna.visualization import matplotlib as ovm
    except ImportError as e:
        log.warning("analysis: matplotlib not installed; skipping PNG plots (%s)", e)
        return

    def _save(fig_or_ax, name):
        fig = fig_or_ax.figure if hasattr(fig_or_ax, "figure") else fig_or_ax[0].figure
        fig.tight_layout()
        fig.savefig(os.path.join(analysis_dir, name), dpi=120, bbox_inches="tight")
        plt.close(fig)

    for plot_name, fn in [
        ("param_importances.png", ovm.plot_param_importances),
        ("optimization_history.png", ovm.plot_optimization_history),
        ("parallel_coordinate.png", ovm.plot_parallel_coordinate),
        ("slice.png", ovm.plot_slice),
    ]:
        try:
            _save(fn(study), plot_name)
            log.info("analysis: wrote %s", plot_name)
        except Exception as e:
            log.warning("analysis: %s failed: %s", plot_name, e)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _state_path(out_root: str) -> str:
    return os.path.join(out_root, "state.json")


def _read_state(out_root: str) -> dict:
    p = _state_path(out_root)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {"stage1_done": False}


def _write_state(out_root: str, state: dict) -> None:
    with open(_state_path(out_root), "w") as f:
        json.dump(state, f, indent=2)


def _run_stage2(
    *,
    out_root: str,
    hpo_cfg: dict,
    stage1_study: optuna.Study,
    base_cfg: dict,
    gpu_pool: Optional["queue.Queue[int]"],
    timeout_s: Optional[int],
) -> List[Dict[str, Any]]:
    """Re-run top-K stage-1 trials at full budget. Returns stage2 record list."""
    completed = [
        t for t in stage1_study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed:
        log.warning("stage 2: no completed stage-1 trials; skipping")
        return []
    completed.sort(key=lambda t: t.value, reverse=True)
    n_top = int(hpo_cfg["stage2"]["n_top_to_confirm"])
    seeds = int(hpo_cfg["stage2"]["seeds_per_trial"])
    top = completed[:n_top]

    records: List[Dict[str, Any]] = []
    metric_k = int(hpo_cfg["metric"].get("k", 10))
    global _METRIC_K
    _METRIC_K = metric_k

    for rank, t in enumerate(top):
        log.info("stage2: confirming stage-1 trial #%d (mean=%.6g)", t.number, t.value)
        # Stage 2 reuses the *sampled* params but with NO budget scaling.
        # Reload the stage-1 trial YAML to extract the sampled merge.
        s1_yaml = t.user_attrs.get("config_path")
        if not s1_yaml or not os.path.exists(s1_yaml):
            log.warning("stage2: cannot find stage-1 config for trial #%d, skipping", t.number)
            continue
        # Build stage-2 cfg = base + fixed_overrides + sampled (no scaling).
        cfg = _build_trial_yaml(
            base_cfg=base_cfg,
            fixed_overrides=hpo_cfg["fixed_overrides"],
            sampled=t.params,  # includes frac_of *fractions*; need to re-resolve them
            budget_scaling=None,
            method=hpo_cfg["method"],
        )
        # If any params were `frac_of`, t.params stored the raw fraction.
        # Re-multiply by the (now-unscaled) base value.
        for name, entry in hpo_cfg["search_space"].items():
            if entry.get("dist") == "frac_of" and name in t.params:
                base_key = entry["base"]
                base_val = cfg.get(base_key)
                if base_val is None:
                    continue
                frac = float(t.params[name])
                resolved_val = frac * float(base_val)
                from rl_matdesign.hpo.search_space import _INT_TYPED_KEYS
                cfg[name] = int(round(resolved_val)) if base_key in _INT_TYPED_KEYS else resolved_val
        # Re-clamp.
        if "dqn_eps_anneal_eps" in cfg and "dqn_num_train_eps" in cfg:
            cfg["dqn_eps_anneal_eps"] = min(int(cfg["dqn_eps_anneal_eps"]), int(cfg["dqn_num_train_eps"]))

        trial_dir = os.path.join(out_root, "stage2", f"top{rank}_from_s1_{t.number:04d}")
        os.makedirs(trial_dir, exist_ok=True)
        trial_yaml = os.path.join(trial_dir, "config.yaml")
        _dump_trial_yaml(trial_yaml, cfg)

        seed_scores: List[float] = []
        try:
            for i in range(seeds):
                seed_out = os.path.join(trial_dir, f"seed_{i}")
                score, _n = _run_one_seed(
                    trial_yaml=trial_yaml,
                    seed_out=seed_out,
                    method=hpo_cfg["method"],
                    seed_idx=i,
                    gpu_pool=gpu_pool,
                    timeout_s=timeout_s,
                )
                seed_scores.append(score)
        except (TrialRunError, FileNotFoundError) as e:
            log.warning("stage2: trial top%d (from s1 #%d) failed: %s", rank, t.number, e)
            continue

        records.append({
            "stage1_trial_id": t.number,
            "mean_score": float(np.mean(seed_scores)),
            "std": float(np.std(seed_scores)) if len(seed_scores) > 1 else 0.0,
            "seed_scores": [float(s) for s in seed_scores],
            "config_path": trial_yaml,
        })

    # Write stage2_summary.csv.
    summary_path = os.path.join(out_root, "stage2_summary.csv")
    if records:
        fieldnames = ["stage1_trial_id", "mean_score", "std", "seed_scores", "config_path"]
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in sorted(records, key=lambda r: r["mean_score"], reverse=True):
                w.writerow({**r, "seed_scores": json.dumps(r["seed_scores"])})
    return records


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hpo-config", required=True, help="Path to HPO search-space YAML")
    p.add_argument("--out", required=True, help="Output directory (study.db + per-trial subdirs)")
    p.add_argument("--gpu-ids", nargs="*", type=int, default=None,
                   help="GPU IDs to round-robin across parallel trials (sets CUDA_VISIBLE_DEVICES).")
    p.add_argument("--timeout-s", type=int, default=None,
                   help="Per-trial subprocess timeout in seconds. None = no timeout.")
    p.add_argument("--skip-stage2", action="store_true",
                   help="Run only stage 1, then exit. Useful for iterating on the search space.")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip matplotlib PNG generation in <out>/analysis/ "
                        "(param_importances.csv is still written).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = os.path.abspath(args.out)
    os.makedirs(out_root, exist_ok=True)

    hpo_cfg = _load_yaml(args.hpo_config)
    hpo_cfg = _validate_hpo_config(hpo_cfg, args.hpo_config)
    # Echo to disk for reproducibility.
    shutil.copy(args.hpo_config, os.path.join(out_root, "hpo_config.yaml"))

    base_cfg = _load_yaml(hpo_cfg["base_config"])

    n_parallel = int(hpo_cfg["stage1"].get("n_parallel_trials", 1))
    gpu_pool = _build_gpu_pool(n_parallel, args.gpu_ids)
    if n_parallel > 1 and gpu_pool is None and not args.gpu_ids:
        log.warning("n_parallel_trials=%d but no --gpu-ids given; trials will share the same GPU(s)",
                    n_parallel)

    storage_url = f"sqlite:///{os.path.join(out_root, 'study.db')}"
    sampler = optuna.samplers.TPESampler(seed=int(hpo_cfg["optuna"].get("seed", 0)))
    study = optuna.create_study(
        study_name=hpo_cfg["study_name"],
        storage=storage_url,
        direction="maximize",
        load_if_exists=True,
        sampler=sampler,
    )

    state = _read_state(out_root)
    if not state["stage1_done"]:
        objective_s1 = _make_objective(
            hpo_cfg=hpo_cfg,
            out_root=out_root,
            stage_name="stage1",
            base_cfg=base_cfg,
            budget_scaling=hpo_cfg["stage1"].get("budget_scaling", {}),
            seeds_per_trial=int(hpo_cfg["stage1"]["seeds_per_trial"]),
            gpu_pool=gpu_pool,
            timeout_s=args.timeout_s,
        )
        n_trials = int(hpo_cfg["stage1"]["n_trials"])
        # Optuna catches the listed exceptions per trial and marks the trial
        # FAIL — exactly what we want for deterministic config errors.
        try:
            study.optimize(
                objective_s1,
                n_trials=n_trials,
                n_jobs=n_parallel,
                catch=(TrialRunError, RuntimeError, FileNotFoundError, ValueError),
                show_progress_bar=False,
            )
        except KeyboardInterrupt:
            log.warning("interrupted during stage 1; state preserved in %s", storage_url)
            raise

        _write_stage_summary(study, os.path.join(out_root, "stage1_summary.csv"))
        # Surface stage-1 outcome counts so a "0 completed" run is loud, not silent.
        n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        n_fail = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
        log.info("stage 1 done: %d completed, %d failed (out of %d total trials)",
                 n_complete, n_fail, len(study.trials))
        if n_complete == 0 and n_fail > 0:
            sample_dir = os.path.join(out_root, "stage1", "trial_0000", "seed_0")
            log.error(
                "all %d stage-1 trials failed. Reproduce one manually to see the error:\n"
                "    python scripts/run_experiment.py \\\n"
                "        --config %s/stage1/trial_0000/config.yaml \\\n"
                "        --method %s --out /tmp/repro \\\n"
                "        --train-seed 0 --dp-seed 10000 --gen-seed 20000\n"
                "Or read %s/stderr.log directly.",
                n_fail, out_root, hpo_cfg["method"], sample_dir,
            )
        warn = _stage1_variance_warning(study, hpo_cfg["stage2"]["n_top_to_confirm"])
        if warn:
            log.warning(warn)
        state["stage1_done"] = True
        _write_state(out_root, state)

    # Analysis (param importances + PNGs) runs against the stage-1 study;
    # safe to call whether or not we're about to do stage 2.
    _write_analysis_outputs(study=study, out_root=out_root, do_plots=not args.no_plots)

    if args.skip_stage2:
        log.info("stage 1 done; --skip-stage2 set, exiting")
        return

    stage2_records = _run_stage2(
        out_root=out_root,
        hpo_cfg=hpo_cfg,
        stage1_study=study,
        base_cfg=base_cfg,
        gpu_pool=gpu_pool,
        timeout_s=args.timeout_s,
    )
    _write_final_report(
        out_root=out_root,
        hpo_cfg=hpo_cfg,
        stage1_study=study,
        stage2_records=stage2_records,
    )
    log.info("HPO complete. Report: %s/final_report.md", out_root)


if __name__ == "__main__":
    main()
