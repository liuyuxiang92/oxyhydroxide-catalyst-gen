#!/usr/bin/env python3
"""Genetic Algorithm baseline for material composition design.

Uses DEAP over the **same constrained search space** the RL agent searches, with
the **same** ``PropertyPredictor`` (``build_predictor`` -> ``registry.build_reward``)
and the **same** env constraints.  Validity is enforced by replaying each genome
through the real env's ``allowed_actions()`` (see ``scripts/baselines/_common.py``),
so the GA never scores a composition the RL agent could not have produced.

This replaces the previous "distinct cations + partition of 20" chromosome, which
could not represent the integer-ratio oxide scenarios (repeated picks, O reserved
for the last step, charge-neutrality) and so searched a different, easier space.

Genome
------
Each individual is a list of ``n_components`` integers (one per env step).  At
step ``k`` the env offers a constraint-filtered, feasibility-guaranteed list of
allowed actions ``L_k``; gene ``g_k`` selects ``L_k[g_k % len(L_k)]``.  All
crossover/mutation operators stay within ``[0, UB)`` and any genome decodes to a
valid candidate (or a rare dead-end, which is penalised in selection).

Usage
-----
    python scripts/baselines/run_ga.py --config configs/oxides_sinter.yaml \\
        --out runs/ga_sinter_seed0 --seed 0 --budget 1000

Requirements
------------
    pip install deap pyyaml
"""

from __future__ import annotations

import argparse
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import _common as C  # noqa: E402

# Fitness assigned to a genome that decodes to a dead-end (no valid terminal).
# Large-negative so selection drops it; the GA is not a surrogate model, so this
# does not poison anything (unlike returning -inf to a BO model). Only used as a
# defensive fallback inside evaluate() — the generational loop below repairs
# dead-end offspring before evaluate() is ever called on them.
_DEAD_END_FITNESS = -1e12


def _run_ga_with_repair(env, population, toolbox, repair_fn, cxpb, mutpb, ngen,
                         stats=None, halloffame=None, verbose=True):
    """Generational GA loop, mirroring ``deap.algorithms.eaSimple`` but repairing
    dead-end offspring before they occupy a scored budget slot.

    Crossover/mutation act on raw integer genes and can produce a trajectory
    that dead-ends against the live constraint set. The RL env this baseline is
    compared against guarantees every episode completes, so an RL agent never
    "spends" an episode on an infeasible trajectory the way an unrepaired GA
    offspring would. ``repair_fn`` replaces any dead-end offspring with a fresh
    resample (cheap: replaying a trajectory costs no predictor call) so that
    every one of the ``pop_size * n_gen`` evaluate() calls scores a genuinely
    valid — possibly duplicate — composition, matching how episode budget is
    spent on the RL side.
    """
    from deap import tools, algorithms

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    for i, ind in enumerate(population):
        if C.decode_choices(env, list(ind)) is None:
            population[i] = toolbox.individual()
    fitnesses = map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    if halloffame is not None:
        halloffame.update(population)
    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(population), **record)
    if verbose:
        print(logbook.stream)

    for gen in range(1, ngen + 1):
        offspring = toolbox.select(population, len(population))
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)
        offspring = repair_fn(offspring)

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(offspring)
        population[:] = offspring

        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GA baseline for composition design")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pop-size", type=int, default=None,
                   help="GA population size (default: 50)")
    p.add_argument("--n-gen", type=int, default=None,
                   help="Number of generations (default: budget / pop_size)")
    p.add_argument("--budget", type=int, default=None,
                   help="Total predictor-evaluation budget (default: from config; "
                        "set to match your RL run)")
    return p.parse_args()


def main() -> None:
    try:
        from deap import base, creator, tools
    except ImportError as exc:  # pragma: no cover
        raise ImportError("GA baseline requires DEAP: pip install deap") from exc

    args = parse_args()

    from rl_matdesign.utils.seeding import set_global_seed
    set_global_seed(args.seed)
    rng = random.Random(args.seed)
    random.seed(args.seed)

    cfg = C.load_scenario(args.config)
    os.makedirs(args.out, exist_ok=True)

    predictor, env, env_type = C.make_predictor_and_env(cfg, seed=args.seed)
    n = C.n_steps(env)
    ub = C.action_upper_bound(env)

    pop_size = args.pop_size or 50
    budget = args.budget or C.default_budget(cfg)
    n_gen = args.n_gen or max(1, budget // pop_size)
    print(f"[GA] env_type={env_type}, n_components={n}, action_ub={ub}, "
          f"pop_size={pop_size}, n_gen={n_gen}, budget(valid evals)={pop_size * n_gen}")

    cache: dict = {}
    eval_count = [0]
    results: dict = {}  # canonical comp key -> best row
    repaired_count = [0]

    # DEAP setup (avoid re-creating classes if already defined).
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    def init_individual():
        # Seed with a valid trajectory (random walk over allowed actions).
        choices, _decoded = C.random_choices(env, rng)
        return creator.Individual(choices)

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(ind):
        decoded = C.decode_choices(env, list(ind))
        if decoded is None:
            return (_DEAD_END_FITNESS,)
        mean, std = C.score_composition(predictor, decoded["comp"], cache, eval_count)
        key = decoded["key"]
        if key not in results or mean > results[key]["reward"]:
            results[key] = {
                "formula": decoded["formula"], "purpose": "exploit",
                "reward": mean, "dp_mean": mean, "dp_std": std,
            }
        return (mean,)

    def mutate(ind):
        # Replace one gene with a fresh in-range integer.
        pos = random.randrange(len(ind))
        ind[pos] = random.randint(0, ub - 1)
        del ind.fitness.values
        return (ind,)

    def repair(offspring):
        # Replace dead-end genomes with a fresh valid resample so they never
        # occupy one of the pop_size*n_gen scored slots (see
        # _run_ga_with_repair docstring). A duplicate of an already-scored
        # composition is still valid and is left alone — it consumes a slot
        # like any other, matching how a repeat episode costs RL a slot too.
        for i, ind in enumerate(offspring):
            if C.decode_choices(env, list(ind)) is None:
                # toolbox.individual() always returns a fresh, unset Fitness,
                # so the replacement is automatically picked up by the
                # invalid_ind filter below and gets evaluated.
                offspring[i] = toolbox.individual()
                repaired_count[0] += 1
        return offspring

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(pop_size)
    import numpy as np
    stats = tools.Statistics(lambda ind: ind.fitness.values[0] if ind.fitness.valid else 0.0)
    stats.register("max", np.max)
    stats.register("mean", np.mean)

    pop, _logbook = _run_ga_with_repair(
        env, pop, toolbox, repair,
        cxpb=0.6, mutpb=0.3, ngen=n_gen,
        stats=stats, halloffame=hof, verbose=True,
    )

    rows = list(results.values())
    out_csv = C.write_outputs(
        args.out, rows,
        run_config={
            "method": "ga", "config": args.config, "seed": args.seed,
            "pop_size": pop_size, "n_gen": n_gen,
            "budget": pop_size * n_gen,
            "dead_ends_repaired": repaired_count[0],
            "total_evaluations": eval_count[0],
            "n_unique_candidates": len(rows),
        },
    )
    best = max(rows, key=lambda r: r["reward"]) if rows else None
    print(f"[GA] Done. {pop_size * n_gen} valid evals ({repaired_count[0]} dead ends "
          f"repaired, not counted), {eval_count[0]} unique predictor evals, "
          f"{len(rows)} unique candidates.")
    if best:
        print(f"[GA] Best: {best['formula']} reward={best['reward']:.4f}")
    print(f"[GA] Results -> {out_csv}")


if __name__ == "__main__":
    main()
