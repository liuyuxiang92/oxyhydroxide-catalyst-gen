#!/usr/bin/env python3
"""Verify the sampling-budget flags of finished runs against what actually ran.

Reads each run's ``run_config.json`` for what was *configured* and its
``training_log.csv`` / ``generated.csv`` for what was *executed*, then flags the
traps that silently invalidate a budget sweep:

* **Budget is predictor calls, not episodes.** DQN pays a real predictor call per
  warmup episode (``_rollout_random_episode`` -> ``env.step`` -> ``reward_fn``);
  PG's warmup pays nothing (``reward_fn`` is neutralised). So DQN's paid budget is
  ``dqn_warmup_eps + dqn_num_train_eps`` while A2C's is
  ``pg_num_iters * pg_batch_eps``. Matching the two on *episodes* hands one method
  more paid evaluations than the other.

* **epsilon must actually reach dqn_eps_min.** ``eps = max(eps_min, 1 - ep/anneal)``.
  Reuse a 50,000-episode ``dqn_eps_anneal_eps`` at a 2,500-episode budget and
  epsilon never drops below ~0.92 — the arm is random search, not DQN. Measured on
  the archived sweep: the 2,500-episode runs ended at epsilon=0.950.

* **pg_batch_eps must be constant across the sweep.** Varying it changes gradient
  noise *and* updates-per-episode at once, confounding budget with collapse rate.

* **gen_epsilon=0 can collapse generation.** At 45k a trained Q-net produced three
  unique candidates from 1,000 generation episodes (calcine: one), so the budget
  comparison is read through a 3-sample readout.

Usage
-----
    python scripts/check_budget_config.py calc_time/*_eps2500_*
    python scripts/check_budget_config.py calc_time          # recurses one level
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
from typing import List, Optional

_EPISODE_PHASES = ("dqn_warmup", "dqn_train", "pg_episode", "pg_train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+",
                   help="Run directories, globs, or a parent holding them")
    p.add_argument("--quiet", action="store_true",
                   help="Only print the problems, not the per-run table")
    return p.parse_args()


def _expand(paths: List[str]) -> List[str]:
    out: List[str] = []
    for p in paths:
        for hit in sorted(glob.glob(p)) or [p]:
            if os.path.isfile(os.path.join(hit, "run_config.json")):
                out.append(hit)
            elif os.path.isdir(hit):
                for sub in sorted(os.listdir(hit)):
                    d = os.path.join(hit, sub)
                    if os.path.isfile(os.path.join(d, "run_config.json")):
                        out.append(d)
    return out


def _phase_counts(run: str) -> collections.Counter:
    path = os.path.join(run, "training_log.csv")
    counts: collections.Counter = collections.Counter()
    if not os.path.exists(path):
        return counts
    with open(path) as f:
        for row in csv.DictReader(f):
            counts[row.get("phase", "")] += 1
    return counts


def _n_candidates(run: str) -> Optional[int]:
    path = os.path.join(run, "generated.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return sum(1 for _ in csv.DictReader(f))


def inspect(run: str) -> dict:
    with open(os.path.join(run, "run_config.json")) as f:
        cfg = json.load(f)
    counts = _phase_counts(run)
    cfg_method = cfg.get("method") or cfg.get("rl_method") or "?"
    target = cfg.get("dqn_target_mode") or ""

    # timing.json records the method actually dispatched. run_config.json used to be
    # written with **cfg spread last, so a YAML `method: dqn` clobbered the real value
    # whenever --method chose a different arm. Trust timing.json, and say so.
    method, method_conflict = cfg_method, None
    timing_path = os.path.join(run, "timing.json")
    if os.path.exists(timing_path):
        try:
            with open(timing_path) as f:
                t_method = json.load(f).get("method")
            if t_method:
                method = t_method
                if t_method != cfg_method:
                    method_conflict = (cfg_method, t_method)
        except (ValueError, OSError):
            pass
    # The presence of these files is the ground truth for which trainer ran.
    if os.path.exists(os.path.join(run, "policy.pt")) and method == "dqn":
        method = "a2c"
    elif os.path.exists(os.path.join(run, "qnet.pt")) and method in ("a2c", "reinforce"):
        method = "dqn"

    info = {
        "run": os.path.basename(run.rstrip("/")),
        "arm": f"{method}{'/' + target if target else ''}",
        "method": method,
        "problems": [],
        "n_candidates": _n_candidates(run),
        "num_gen_eps": cfg.get("num_gen_eps"),
        "gen_epsilon": cfg.get("gen_epsilon"),
        "gen_temperature": cfg.get("gen_temperature"),
        "pg_batch_eps": cfg.get("pg_batch_eps"),
    }

    if method == "dqn":
        warm = int(cfg.get("dqn_warmup_eps") or 0)
        train = int(cfg.get("dqn_num_train_eps") or 0)
        anneal = float(cfg.get("dqn_eps_anneal_eps") or 0)
        eps_min = float(cfg.get("dqn_eps_min") or 0.0)
        info["configured"] = warm + train
        info["detail"] = f"warmup {warm:,} + train {train:,}"
        info["actual"] = counts["dqn_warmup"] + counts["dqn_train"]
        # Runs from before dqn_warmup logging existed have only dqn_train rows, so
        # the log legitimately falls short of the paid budget by exactly the warmup.
        # Compare against train-only there rather than reporting a phantom shortfall.
        if not counts["dqn_warmup"] and counts["dqn_train"]:
            info["expect"] = train
            info["note"] = f"pre-dates dqn_warmup logging; {warm:,} warmup calls unlogged"
        else:
            info["expect"] = warm + train

        # eps = max(eps_min, 1 - ep/anneal); it reaches the floor at
        # ep = anneal * (1 - eps_min).
        if anneal > 0:
            eps_end = max(eps_min, 1.0 - train / anneal)
            hit_at = anneal * (1.0 - eps_min)
            info["eps_end"] = eps_end
            info["anneal_frac"] = anneal / train if train else float("nan")
            info["at_floor_frac"] = max(0.0, (train - hit_at) / train) if train else 0.0
            if eps_end > eps_min + 1e-9:
                info["problems"].append(
                    f"epsilon never reaches dqn_eps_min: ends at {eps_end:.3f} "
                    f"(min {eps_min}). The agent is {100 * eps_end:.0f}% random for "
                    f"the whole run -- this arm is random search. Set "
                    f"dqn_eps_anneal_eps ~= {round(0.6 * train):,} (0.6 x train)."
                )
        else:
            info["problems"].append("dqn_eps_anneal_eps missing or zero")
    else:
        iters = int(cfg.get("pg_num_iters") or 0)
        batch = int(cfg.get("pg_batch_eps") or 0)
        warm = int(cfg.get("pg_warmup_eps") or 0)
        info["configured"] = iters * batch
        info["expect"] = iters * batch
        info["detail"] = f"{iters:,} iters x {batch} batch (warmup {warm:,}, free)"
        info["actual"] = counts["pg_episode"] or counts["pg_train"] * batch
        if counts["pg_train"] and not counts["pg_episode"]:
            info["problems"].append(
                "no pg_episode rows -- this run predates per-episode PG logging, so "
                "only batch means are available (spread compressed ~1/sqrt(batch))."
            )

    expect = info.get("expect") or info["configured"]
    if info["actual"] and expect:
        drift = info["actual"] - expect
        if abs(drift) > max(5, 0.02 * expect):
            info["problems"].append(
                f"expected {expect:,} logged episodes but found {info['actual']:,} "
                f"({drift:+,}) -- config and log disagree"
            )
    if not info["actual"]:
        info["problems"].append("no training_log.csv rows -- run may be incomplete")

    if method_conflict:
        info["problems"].append(
            f"run_config.json says method={method_conflict[0]!r} but timing.json says "
            f"{method_conflict[1]!r}. The run itself was fine; the recorded metadata "
            f"is wrong because run_config.json was written with the YAML spread last, "
            f"so a config-file `method:` overwrote the --method that actually ran. "
            f"Fixed in run_experiment.py; re-runs will record it correctly."
        )

    gen_eps = info["gen_epsilon"]
    n_cand = info["n_candidates"]
    if gen_eps is not None and float(gen_eps) == 0.0 and n_cand is not None \
            and info["num_gen_eps"] and n_cand < 0.1 * int(info["num_gen_eps"]):
        info["problems"].append(
            f"gen_epsilon=0 and only {n_cand} unique candidates from "
            f"{info['num_gen_eps']} generation episodes -- the budget comparison is "
            f"being read through a {n_cand}-sample readout. Set gen_epsilon=0.2."
        )
    return info


def main() -> None:
    args = parse_args()
    runs = _expand(args.paths)
    if not runs:
        raise SystemExit("no run directories with run_config.json found")

    infos = [inspect(r) for r in runs]

    if not args.quiet:
        print(f"{'run':<44} {'arm':<14} {'paid calls':>11} {'logged':>9} "
              f"{'eps_end':>8} {'anneal':>7} {'cands':>6}")
        print("-" * 104)
        for i in infos:
            eps_end = f"{i['eps_end']:.3f}" if "eps_end" in i else "-"
            frac = f"{i['anneal_frac']:.2f}" if "anneal_frac" in i else "-"
            cands = i["n_candidates"] if i["n_candidates"] is not None else "-"
            flag = " !" if i["problems"] else ""
            print(f"{i['run'][:44]:<44} {i['arm']:<14} {i['configured']:>11,} "
                  f"{i['actual']:>9,} {eps_end:>8} {frac:>7} {cands:>6}{flag}")
        print()
        for i in infos:
            note = f"   [{i['note']}]" if i.get("note") else ""
            print(f"  {i['run']}: {i['detail']}{note}")
        print()

    # --- cross-run consistency ---------------------------------------------
    cross: List[str] = []
    batches = {i["pg_batch_eps"] for i in infos
               if i["method"] != "dqn" and i["pg_batch_eps"] is not None}
    if len(batches) > 1:
        cross.append(
            f"pg_batch_eps differs across runs: {sorted(batches)}. Hold it FIXED and "
            f"vary only pg_num_iters, or budget is confounded with collapse rate."
        )
    gen = {(i["gen_epsilon"], i["gen_temperature"]) for i in infos}
    if len(gen) > 1:
        cross.append(
            f"generation settings differ across runs: "
            f"{sorted(gen, key=str)} as (gen_epsilon, gen_temperature). Generation is "
            f"the readout -- keep it identical so it cannot confound the budget axis."
        )

    n_bad = sum(1 for i in infos if i["problems"])
    if n_bad or cross:
        print("PROBLEMS")
        print("=" * 104)
        for i in infos:
            for p in i["problems"]:
                print(f"  [{i['run']}]\n      {p}")
        for c in cross:
            print(f"  [across runs]\n      {c}")
        print()
        print(f"{n_bad}/{len(infos)} run(s) flagged"
              + (f", {len(cross)} cross-run issue(s)" if cross else ""))
        raise SystemExit(1)

    print(f"OK: {len(infos)} run(s), budgets consistent and epsilon anneals fully.")


if __name__ == "__main__":
    main()
