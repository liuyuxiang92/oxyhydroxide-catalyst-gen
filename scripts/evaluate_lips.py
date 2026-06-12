#!/usr/bin/env python3
"""Evaluate one or more doped-Li6PS6 (SSE) compositions with the real predictor.

The LiPS analogue of ``scripts/evaluate_formulas_dp.py``: you describe a doped
composition, and the script builds the supercell with the **same** ``sse``
builder + ``structure_score`` predictor the RL run uses (loaded straight from the
scenario YAML), so the printed conductivity / stability / reward match
``generated.csv`` exactly. Optionally it writes the built (and relaxed) POSCAR.

Why a dedicated script: the SSE predictor takes a *structured* candidate
``{P_site: {metal: level}, S_site: {O: form, Cl: count}}`` — not a flat formula —
and it optimises the operating temperature per composition. This wraps that
plumbing and prints a per-temperature breakdown so you can see how each property
trades off (conductivity ↑ with T, stability ↓) and which T was selected.

Examples
--------
    # By explicit picks (unambiguous):
    python scripts/evaluate_lips.py --config configs/lips_sse.yaml \
        --metal Sn --level 0.06 --o-form 1 --cl 1.2 \
        --save-poscar built_Sn0.06.vasp

    # By formula (metal/level/O/Cl are parsed; Li/S/Br are recomputed by the
    # builder's charge balance, so any values you give for them are ignored):
    python scripts/evaluate_lips.py --config configs/lips_sse.yaml \
        --formula "Cl1.2O1P0.94Sn0.06"

    # Several at once, from a file (one spec per line, same syntax as --formula):
    python scripts/evaluate_lips.py --config configs/lips_sse.yaml \
        --formulas-file candidates.txt --out-csv lips_eval.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Mirror run_experiment.py: add src/ + scripts/ to sys.path so we can run from a
# fresh checkout without installing.
_ROOT = Path(__file__).resolve().parent.parent
for _sub in ("src", "scripts"):
    _p = str(_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Mitigate macOS BLAS/OpenMP segfaults, matching the main pipeline.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import yaml  # noqa: E402


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _group_meta(cfg: dict) -> Tuple[str, str, str, List[str]]:
    """Return (p_site_group, s_site_group, host_P, p_site_cation_set)."""
    groups = cfg.get("groups", [])
    p_group = s_group = None
    host_p = str(cfg.get("host", {}).get("P", "P"))
    cation_set: List[str] = []
    for g in groups:
        if str(g.get("kind", "composition")) == "categorical":
            s_group = g.get("name")
        else:
            p_group = g.get("name")
            host_p = str(g.get("host", host_p))
            cation_set = list(g.get("cation_set", []))
    # Fall back to the builder's defaults / config keys.
    p_group = p_group or cfg.get("p_site_group", "P_site")
    s_group = s_group or cfg.get("s_site_group", "S_site")
    return p_group, s_group, host_p, cation_set


def parse_spec(
    spec: str, *, host_p: str, cation_set: List[str], s_o_element: str = "O",
) -> Tuple[str, float, float, float]:
    """Parse a formula-like spec into (metal, level, o_form, cl_count).

    Only the four builder inputs are extracted; host/derived elements (Li, S, Br,
    and the host metal P) in the spec are ignored — the builder recomputes them.
    The metal is the parsed element that appears in the P-site ``cation_set``.
    """
    pairs = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]+)", spec)
    if not pairs:
        raise ValueError(f"Could not parse any (element, amount) pairs from {spec!r}.")
    amounts: Dict[str, float] = {}
    for el, num in pairs:
        amounts[el] = amounts.get(el, 0.0) + float(num)

    cset = set(cation_set)
    metals = [el for el in amounts if el in cset]
    if len(metals) != 1:
        raise ValueError(
            f"Expected exactly one P-site dopant metal from the cation_set in "
            f"{spec!r}; found {metals or 'none'}. Pass --metal/--level explicitly."
        )
    metal = metals[0]
    level = amounts[metal]
    o_form = 1.0 if amounts.get(s_o_element, 0.0) > 0 else 0.0
    cl_count = amounts.get("Cl", 0.0)
    return metal, level, o_form, cl_count


def build_candidate(
    p_group: str, s_group: str, host_p: str,
    metal: str, level: float, o_form: float, cl_count: float,
) -> Dict[str, Dict[str, float]]:
    """Assemble the structured candidate the SSE builder expects."""
    return {
        p_group: {metal: float(level), host_p: 1.0 - float(level)},
        s_group: {"O": float(o_form), "Cl": float(cl_count)},
    }


def iter_specs(args: argparse.Namespace) -> Iterable[str]:
    for s in args.formula:
        if s.strip():
            yield s.strip()
    if args.formulas_file:
        with open(args.formulas_file, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    yield line.split()[0]


def _fmt(x: float) -> str:
    return f"{x:.6g}"


def evaluate_one(
    predictor: Any, candidate: Dict[str, Dict[str, float]], sweep_name: str,
) -> Dict[str, Any]:
    """Run the predictor and collect reward, dp_mean, per-property + breakdown."""
    breakdown = predictor.score_breakdown(candidate)
    best = breakdown["best"]
    stats = best["stats"]                              # {prop: (mean, std)} at best T
    reward, _std = predictor.predict(candidate)        # full objective (with std penalty)
    raw_mean, _ = predictor.predict_raw(candidate)     # dp_mean column = Σ direction·mean
    formula = predictor.composition_formula(candidate) or "?"
    return {
        "formula": formula,
        "reward": float(reward),
        "dp_mean": float(raw_mean),
        f"{sweep_name}": best["value"],
        "stats": stats,
        "rows": breakdown["rows"],
    }


def print_result(res: Dict[str, Any], sweep_name: str) -> None:
    print(f"\n=== {res['formula']} ===")
    print(f"  reward (objective, incl. std penalty) : {_fmt(res['reward'])}")
    print(f"  dp_mean (Σ direction·mean, raw)        : {_fmt(res['dp_mean'])}")
    if res[sweep_name] is not None:
        print(f"  selected {sweep_name:<29}: {_fmt(res[sweep_name])}")
    print(f"  per-property at selected {sweep_name}:")
    for name, (m, s) in res["stats"].items():
        print(f"      {name:<22} mean={_fmt(m)}  std={_fmt(s)}")
    rows = res["rows"]
    if len(rows) > 1:
        prop_names = list(res["stats"].keys())
        header = f"  {sweep_name:>10} | " + " | ".join(f"{n:>18}" for n in prop_names) + " |   reward"
        print("\n  per-" + sweep_name + " breakdown:")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for r in rows:
            cells = " | ".join(f"{_fmt(r['stats'][n][0]):>18}" for n in prop_names)
            star = " *" if r["value"] == res[sweep_name] else "  "
            print(f"  {_fmt(r['value']):>10} | {cells} | {_fmt(r['reward']):>8}{star}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate doped-Li6PS6 (SSE) compositions with the scenario's "
        "real structure_score predictor; optionally save the built POSCAR.",
    )
    ap.add_argument("--config", required=True, help="Scenario YAML (e.g. configs/lips_sse.yaml).")
    ap.add_argument("--formula", action="append", default=[],
                    help="Formula-like spec, e.g. 'Cl1.2O1P0.94Sn0.06'. Repeatable. "
                         "Only metal/level/O/Cl are read; Li/S/Br are recomputed.")
    ap.add_argument("--formulas-file", default=None,
                    help="Text file with one spec per line (# comments allowed).")
    # Explicit picks (override / alternative to --formula). Apply to a single eval.
    ap.add_argument("--metal", default=None, help="P-site dopant metal symbol.")
    ap.add_argument("--level", type=float, default=None, help="Dopant fraction of P replaced.")
    ap.add_argument("--o-form", type=float, default=None,
                    help="S-site oxygen form flag: 0 = sulfide, 1 = oxide.")
    ap.add_argument("--cl", type=float, default=None, help="Cl atoms per formula unit.")
    ap.add_argument("--seed", type=int, default=0, help="Builder/predictor RNG seed.")
    geo = ap.add_mutually_exclusive_group()
    geo.add_argument("--geo-opt", dest="geo_opt", action="store_true", default=None,
                     help="Force geometry optimization ON before scoring (overrides config).")
    geo.add_argument("--no-geo-opt", dest="geo_opt", action="store_false", default=None,
                     help="Force geometry optimization OFF (score the as-built cell; "
                          "overrides config). Faster, but properties are unrelaxed.")
    ap.add_argument("--save-poscar", default=None,
                    help="Write the built structure to this path (single eval only). "
                         "Relaxed iff geometry optimization is active for this run.")
    ap.add_argument("--out-csv", default=None, help="Write a summary CSV here.")
    args = ap.parse_args()

    cfg = _load_config(args.config)
    p_group, s_group, host_p, cation_set = _group_meta(cfg)
    sweep_name = str((cfg.get("sweep") or {}).get("name", "sweep"))

    from rl_matdesign.registry import resolve_predictor
    predictor = resolve_predictor(cfg.get("predictor", "structure_score"), cfg, seed=args.seed)

    # Geometry-optimization override. Default (None) keeps the config's setting;
    # --geo-opt / --no-geo-opt force it for both scoring and the saved POSCAR.
    if args.geo_opt is not None:
        if args.geo_opt and not getattr(predictor, "geo_opt", None):
            print("[WARN] --geo-opt requested but the config has no 'geo_opt' block; "
                  "relaxation will use built-in defaults (model 'models/DPA-3.1-3M.pt').",
                  flush=True)
        predictor.geo_opt_enabled = bool(args.geo_opt)
    geo_state = "ON" if getattr(predictor, "geo_opt_enabled", False) else "OFF"
    print(f"[info] geometry optimization: {geo_state}")

    # Collect (label, candidate) work items.
    items: List[Tuple[str, Dict[str, Dict[str, float]]]] = []
    if args.metal is not None or args.level is not None:
        if args.metal is None or args.level is None:
            ap.error("--metal and --level must be given together.")
        cand = build_candidate(
            p_group, s_group, host_p, args.metal, args.level,
            args.o_form if args.o_form is not None else 0.0,
            args.cl if args.cl is not None else 0.0,
        )
        items.append((f"{args.metal}{args.level:g}", cand))

    skipped: List[Tuple[str, str]] = []
    for spec in iter_specs(args):
        try:
            metal, level, o_form, cl = parse_spec(spec, host_p=host_p, cation_set=cation_set)
            items.append((spec, build_candidate(p_group, s_group, host_p, metal, level, o_form, cl)))
        except Exception as exc:  # noqa: BLE001
            skipped.append((spec, str(exc)))
            print(f"[WARN] skipping {spec!r}: {exc}", flush=True)

    if not items:
        ap.error("No compositions to evaluate. Pass --formula/--formulas-file or --metal/--level.")

    if args.save_poscar and len(items) != 1:
        ap.error("--save-poscar supports exactly one composition; narrow the input.")

    rows: List[Dict[str, Any]] = []
    for label, cand in items:
        try:
            res = evaluate_one(predictor, cand, sweep_name)
        except Exception as exc:  # noqa: BLE001
            skipped.append((label, str(exc)))
            print(f"[WARN] evaluation failed for {label!r}: {exc}", flush=True)
            continue
        print_result(res, sweep_name)
        rows.append(res)

        if args.save_poscar:
            _save_poscar(predictor, cand, args)

    if args.out_csv and rows:
        _write_csv(args.out_csv, rows, sweep_name)
        print(f"\nWrote {len(rows)} row(s) -> {args.out_csv}")

    if skipped:
        print(f"\n[WARN] {len(skipped)} spec(s) skipped:")
        for label, reason in skipped:
            print(f"  {label}: {reason}")


def _save_poscar(predictor: Any, candidate: Dict[str, Dict[str, float]], args: argparse.Namespace) -> None:
    from ase.io import write as ase_write

    builder = getattr(predictor, "_shared_builder", None)
    if builder is None:
        builders = getattr(predictor, "_builders", None)
        builder = builders[0] if builders else None
    if builder is None:
        print("[WARN] predictor exposes no builder; cannot save POSCAR.", flush=True)
        return
    structures = builder.build(candidate, n_configs=1)
    atoms = structures[0]
    relaxed = getattr(predictor, "geo_opt_enabled", False)
    if relaxed:
        atoms = predictor._relax(atoms)
    ase_write(args.save_poscar, atoms, format="vasp")
    tag = "relaxed" if relaxed else "unrelaxed"
    print(f"  saved {tag} structure -> {args.save_poscar} ({len(atoms)} atoms)")


def _write_csv(path: str, rows: List[Dict[str, Any]], sweep_name: str) -> None:
    prop_names = list(rows[0]["stats"].keys())
    cols = ["formula", "reward", "dp_mean", sweep_name]
    for n in prop_names:
        cols += [f"{n}_mean", f"{n}_std"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            out = {"formula": r["formula"], "reward": r["reward"],
                   "dp_mean": r["dp_mean"], sweep_name: r[sweep_name]}
            for n in prop_names:
                m, s = r["stats"][n]
                out[f"{n}_mean"] = m
                out[f"{n}_std"] = s
            w.writerow(out)


if __name__ == "__main__":
    main()
