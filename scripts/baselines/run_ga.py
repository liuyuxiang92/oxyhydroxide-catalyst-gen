#!/usr/bin/env python3
"""Genetic Algorithm baseline for material composition design.

Uses DEAP with constraint-preserving crossover and mutation operators that
guarantee all generated compositions are valid (fractions sum to 1.0, all
elements distinct).  Uses the same YAML config and PropertyPredictor interface
as run_experiment.py for a fair evaluation-budget comparison.

Chromosome representation
-------------------------
Each individual is a pair ``(element_indices, unit_allocation)`` stored flat
as a list of 2*N integers where N = n_components:
  - ``element_indices[i]`` : index into cation_set (all distinct by construction)
  - ``unit_allocation[i]``  : integer units ≥ 1, sum = total_units (default 20)

This representation allows constraint-preserving operators:
- Crossover: swap (element, unit) pairs between parents as matched pairs.
- Mutation — element swap: replace one element index with an unused one.
- Mutation — unit transfer: subtract 1 from one component, add 1 to another
  (preserves the total sum invariant).

Usage
-----
    python scripts/baselines/run_ga.py --config configs/hea.yaml \\
        --out runs/hea_ga_seed0 --seed 0

Requirements
------------
    pip install deap pyyaml
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from typing import Dict, List, Tuple

import numpy as np
import yaml

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GA baseline for composition design")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pop-size", type=int, default=None,
                   help="GA population size (default: 50)")
    p.add_argument("--n-gen", type=int, default=None,
                   help="Number of GA generations (default: from config budget / pop_size)")
    p.add_argument("--budget", type=int, default=None,
                   help="Total predictor evaluation budget (overrides n-gen)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Chromosome encoding / decoding
# ---------------------------------------------------------------------------

def encode_individual(
    element_indices: List[int], unit_allocation: List[int]
) -> List[int]:
    """Flatten (element_indices, unit_allocation) into a single list."""
    n = len(element_indices)
    assert len(unit_allocation) == n
    return list(element_indices) + list(unit_allocation)


def decode_individual(
    ind: List[int], n_components: int
) -> Tuple[List[int], List[int]]:
    return list(ind[:n_components]), list(ind[n_components:])


def ind_to_composition(
    ind: List[int], cation_set: List[str], n_components: int, total_units: int
) -> Dict[str, float]:
    elem_ids, units = decode_individual(ind, n_components)
    comp = {}
    for eid, u in zip(elem_ids, units):
        el = cation_set[eid]
        comp[el] = u / total_units
    return comp


def is_valid(ind: List[int], n_components: int, total_units: int) -> bool:
    elem_ids, units = decode_individual(ind, n_components)
    if len(set(elem_ids)) != n_components:
        return False
    if any(u < 1 for u in units):
        return False
    if sum(units) != total_units:
        return False
    return True


# ---------------------------------------------------------------------------
# Constraint-preserving operators
# ---------------------------------------------------------------------------

def random_valid_individual(
    n_components: int, cation_set: List[str], total_units: int
) -> List[int]:
    """Generate a random valid individual."""
    elem_ids = random.sample(range(len(cation_set)), n_components)
    # Distribute total_units among n_components with each ≥ 1.
    units = _random_partition(total_units, n_components)
    return encode_individual(elem_ids, units)


def _random_partition(total: int, k: int) -> List[int]:
    """Random partition of *total* into *k* positive integers summing to *total*."""
    # Use stars-and-bars: place k-1 dividers in [1, total-1].
    if k == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), k - 1))
    parts = [cuts[0]] + [cuts[i] - cuts[i-1] for i in range(1, len(cuts))] + [total - cuts[-1]]
    return parts


def cx_paired_swap(
    ind1: List[int], ind2: List[int], n_components: int
) -> Tuple[List[int], List[int]]:
    """Crossover: swap a random subset of (element, unit) pairs between parents.

    Preserves validity: both parents have distinct elements and sum = total_units.
    After swapping pairs, we may end up with duplicate element indices in one
    child; a repair step replaces duplicates with elements from the other parent.
    """
    e1, u1 = decode_individual(ind1, n_components)
    e2, u2 = decode_individual(ind2, n_components)

    # Choose crossover points.
    cx_point = random.randint(1, n_components - 1)
    c1_e = e1[:cx_point] + e2[cx_point:]
    c1_u = u1[:cx_point] + u2[cx_point:]
    c2_e = e2[:cx_point] + e1[cx_point:]
    c2_u = u2[:cx_point] + u1[cx_point:]

    _repair_elements(c1_e, n_components)
    _repair_elements(c2_e, n_components)
    _repair_units(c1_u)
    _repair_units(c2_u)

    ind1[:] = encode_individual(c1_e, c1_u)
    ind2[:] = encode_individual(c2_e, c2_u)
    return ind1, ind2


def _repair_elements(elem_ids: List[int], n_cation: int) -> None:
    """In-place: replace duplicate element indices with random unused ones."""
    seen = {}
    for i, eid in enumerate(elem_ids):
        if eid in seen:
            # Find an unused index.
            used = set(elem_ids)
            candidates = [x for x in range(n_cation) if x not in used]
            if candidates:
                new_eid = random.choice(candidates)
                elem_ids[i] = new_eid
        else:
            seen[eid] = i


def _repair_units(units: List[int]) -> None:
    """In-place: ensure all units >= 1."""
    for i in range(len(units)):
        if units[i] < 1:
            units[i] = 1


def mut_element_swap(ind: List[int], n_components: int, n_cation: int) -> List[int]:
    """Mutation: replace one element with a random unused one."""
    e, u = decode_individual(ind, n_components)
    used = set(e)
    candidates = [x for x in range(n_cation) if x not in used]
    if not candidates:
        return ind
    pos = random.randrange(n_components)
    e[pos] = random.choice(candidates)
    ind[:] = encode_individual(e, u)
    return ind


def mut_unit_transfer(ind: List[int], n_components: int) -> List[int]:
    """Mutation: subtract 1 unit from one component, add 1 to another.

    Preserves sum invariant.  No-op if all components already have 1 unit.
    """
    e, u = decode_individual(ind, n_components)
    donors = [i for i in range(n_components) if u[i] > 1]
    if not donors:
        return ind
    donor = random.choice(donors)
    receiver = random.choice([i for i in range(n_components) if i != donor])
    u[donor] -= 1
    u[receiver] += 1
    ind[:] = encode_individual(e, u)
    return ind


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        from deap import base, creator, tools, algorithms
    except ImportError as exc:
        raise ImportError("GA baseline requires DEAP: pip install deap") from exc

    args = parse_args()

    from rl_matdesign.utils.seeding import set_global_seed
    set_global_seed(args.seed)
    random.seed(args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.out, exist_ok=True)

    # Load predictor (same factory as run_experiment.py).
    sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))
    from run_experiment import build_predictor
    predictor = build_predictor(cfg)

    cation_set = cfg["cation_set"]
    n_components = int(cfg.get("n_components", 5))
    total_units = 20
    pop_size = args.pop_size or 50

    # Budget: total predictor calls.
    budget = args.budget or int(cfg.get("num_gen_eps", 200)) + int(cfg.get("pg_train_eps", 1000))
    n_gen = args.n_gen or max(1, budget // pop_size)
    print(f"[GA] pop_size={pop_size}, n_gen={n_gen}, budget≈{pop_size * n_gen}")

    # DEAP setup (avoid re-creating classes if already defined).
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register(
        "individual", tools.initIterate, creator.Individual,
        lambda: random_valid_individual(n_components, cation_set, total_units)
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Evaluation function.
    eval_count = [0]
    eval_cache: Dict[tuple, float] = {}

    def evaluate(ind):
        comp = ind_to_composition(ind, cation_set, n_components, total_units)
        key = tuple(sorted((k, int(round(v * total_units))) for k, v in comp.items()))
        if key not in eval_cache:
            mean, _ = predictor.predict(comp)
            eval_cache[key] = mean
            eval_count[0] += 1
        return (eval_cache[key],)

    toolbox.register("evaluate", evaluate)
    toolbox.register(
        "mate", cx_paired_swap, n_components=n_components
    )
    toolbox.register(
        "mutate_elem", mut_element_swap,
        n_components=n_components, n_cation=len(cation_set)
    )
    toolbox.register(
        "mutate_unit", mut_unit_transfer, n_components=n_components
    )
    toolbox.register("select", tools.selTournament, tournsize=3)

    def combined_mutate(ind):
        if random.random() < 0.5:
            toolbox.mutate_elem(ind)
        else:
            toolbox.mutate_unit(ind)
        del ind.fitness.values
        return (ind,)

    toolbox.register("mutate", combined_mutate)

    # Run GA.
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(50)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0] if ind.fitness.valid else 0.0)
    stats.register("max", np.max)
    stats.register("mean", np.mean)

    pop, logbook = algorithms.eaSimple(
        pop, toolbox,
        cxpb=0.6, mutpb=0.3, ngen=n_gen,
        stats=stats, halloffame=hof, verbose=True,
    )

    # Write results.
    rows = []
    for ind in hof:
        comp = ind_to_composition(ind, cation_set, n_components, total_units)
        mean, std = predictor.predict(comp)
        # Build formula string.
        items = sorted(comp.items(), key=lambda x: -x[1])
        formula = "".join(f"{el}{v:.2f}" for el, v in items)
        formula += cfg.get("anion_formula", "")
        rows.append({
            "formula": formula,
            "purpose": "exploit",
            "reward": float(ind.fitness.values[0]),
            "dp_mean": mean,
            "dp_std": std,
        })

    rows.sort(key=lambda r: -r["dp_mean"])

    out_csv = os.path.join(args.out, "generated.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["formula", "purpose", "reward", "dp_mean", "dp_std"])
        writer.writeheader()
        writer.writerows(rows)

    run_cfg = {"config": args.config, "seed": args.seed, "pop_size": pop_size,
               "n_gen": n_gen, "total_evaluations": eval_count[0]}
    with open(os.path.join(args.out, "run_config.json"), "w") as f:
        json.dump(run_cfg, f, indent=2)

    print(f"[GA] Done. {eval_count[0]} evaluations. Best: {rows[0]['dp_mean']:.4f}")
    print(f"[GA] Results → {out_csv}")


if __name__ == "__main__":
    main()
