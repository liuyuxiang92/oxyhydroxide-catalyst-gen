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

Two dopant metals (P-site ``kind: independent``) are supported: pass one
``--dopant METAL:LEVEL:OFORM`` per metal (repeat for two; same element merges),
or a ``--formula`` from which the dopant metals/levels are parsed. Per-metal
O-form is paired to the **sorted** metals just as the builder/filter do it.

Examples
--------
    # Two metals, explicit O-forms (unambiguous), save the built POSCAR:
    python scripts/evaluate_lips.py --config configs/lips_sse.yaml \
        --dopant Sn:0.06:1 --dopant Mn:0.04:0 --cl 1.2 \
        --save-poscar built.vasp

    # Same element twice (merges to Mn 0.10, shares one O-form):
    python scripts/evaluate_lips.py --config configs/lips_sse.yaml \
        --dopant Mn:0.06:1 --dopant Mn:0.04:1 --cl 1.0

    # By formula (dopant metals/levels + Cl parsed; Li/S/Br/O recomputed by the
    # builder's charge balance; per-metal O-form inferred by category):
    python scripts/evaluate_lips.py --config configs/lips_sse.yaml \
        --formula "Mn0.06Sn0.04Cl1.2"

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


class SceneMeta:
    """Group names + S-site slot wiring discovered from the scenario YAML."""

    def __init__(self, cfg: dict) -> None:
        self.p_group = self.s_group = None
        self.host_p = str(cfg.get("host", {}).get("P", "P"))
        self.cation_set: List[str] = []
        self.o_element = "O"
        self.cl_element = "Cl"
        self.o_form_slots: List[str] = []
        self.cl_slot = "Cl"
        self.metal_only: set = set()
        self.oxide_only: set = set()
        for g in cfg.get("groups", []) or []:
            if str(g.get("kind", "composition")) == "categorical":
                self.s_group = g.get("name")
                self.o_element = str(g.get("o_element", "O"))
                self.metal_only = set(g.get("metal_only", []))
                self.oxide_only = set(g.get("oxide_only", []))
                for c in g.get("choices", []) or []:
                    name, el = str(c.get("name", c["element"])), str(c["element"])
                    if el == self.o_element:
                        self.o_form_slots.append(name)
                    elif el == self.cl_element:
                        self.cl_slot = name
            else:
                self.p_group = g.get("name")
                self.host_p = str(g.get("host", self.host_p))
                self.cation_set = list(g.get("cation_set", []))
        self.p_group = self.p_group or cfg.get("p_site_group", "P_site")
        self.s_group = self.s_group or cfg.get("s_site_group", "S_site")
        if not self.o_form_slots:
            self.o_form_slots = [self.o_element]

    def infer_o_form(self, metal: str) -> Tuple[float, bool]:
        """O-form a metal must take by category; (value, ambiguous?) for 'both' metals."""
        if metal in self.metal_only:
            return 0.0, False
        if metal in self.oxide_only:
            return 1.0, False
        return 0.0, True            # 'both' — formula can't disambiguate; default sulfide


def parse_formula(spec: str, meta: SceneMeta) -> Tuple[List[Tuple[str, float, float]], float, bool]:
    """Parse a formula into ([(metal, level, o_form), ...], cl_count, ambiguous?).

    Dopant metals are the parsed elements found in the P-site ``cation_set``;
    host/derived elements (Li, S, Br, P, O) are ignored — the builder recomputes
    them. Per-metal O-form can't be recovered from the single merged O count, so
    it is inferred by category (returns ``ambiguous=True`` if any 'both' metal is
    present, since its form is a guess).
    """
    pairs = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]+)", spec)
    if not pairs:
        raise ValueError(f"Could not parse any (element, amount) pairs from {spec!r}.")
    amounts: Dict[str, float] = {}
    for el, num in pairs:
        amounts[el] = amounts.get(el, 0.0) + float(num)
    cset = set(meta.cation_set)
    metals = [el for el in amounts if el in cset]
    if not metals:
        raise ValueError(
            f"No P-site dopant metal from the cation_set found in {spec!r}. "
            "Pass --dopant METAL:LEVEL:OFORM explicitly."
        )
    dopants, ambiguous = [], False
    for m in metals:
        of, amb = meta.infer_o_form(m)
        ambiguous = ambiguous or amb
        dopants.append((m, amounts[m], of))
    return dopants, amounts.get(meta.cl_element, 0.0), ambiguous


def parse_dopant_flag(spec: str) -> Tuple[str, float, float]:
    """Parse ``METAL:LEVEL[:OFORM]`` (e.g. ``Mn:0.06:1``) -> (metal, level, o_form)."""
    parts = spec.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"--dopant expects METAL:LEVEL[:OFORM], got {spec!r}.")
    metal = parts[0].strip()
    level = float(parts[1])
    o_form = float(parts[2]) if len(parts) == 3 else 0.0
    return metal, level, o_form


def build_candidate(
    meta: SceneMeta, dopants: List[Tuple[str, float, float]], cl_count: float,
) -> Dict[str, Dict[str, float]]:
    """Assemble the structured candidate the SSE builder expects (two-metal aware).

    Same-element picks merge (levels add); per Option A they share one O-form (the
    first given wins). Each metal's form is placed in the O-slot that the builder
    pairs with the sorted metal order.
    """
    p_site: Dict[str, float] = {}
    forms: Dict[str, float] = {}
    for m, lv, of in dopants:
        p_site[m] = p_site.get(m, 0.0) + float(lv)
        forms.setdefault(m, float(of))
    s_site: Dict[str, float] = {meta.cl_slot: float(cl_count)}
    for i, m in enumerate(sorted(p_site)):
        if i < len(meta.o_form_slots):
            s_site[meta.o_form_slots[i]] = forms[m]
    return {meta.p_group: p_site, meta.s_group: s_site}


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
                    help="Formula-like spec, e.g. 'Mn0.06Sn0.04Cl1.2'. Repeatable. Dopant "
                         "metals/levels + Cl are read; per-metal O-form is inferred by "
                         "category (use --dopant to set O-form exactly).")
    ap.add_argument("--formulas-file", default=None,
                    help="Text file with one spec per line (# comments allowed).")
    # Explicit picks (alternative to --formula). One --dopant per metal; repeat
    # for two metals. Same element repeated merges (levels add, Option A).
    ap.add_argument("--dopant", action="append", default=[], metavar="METAL:LEVEL[:OFORM]",
                    help="A dopant pick, e.g. 'Mn:0.06:1' (metal:level:o_form). OFORM is "
                         "0=sulfide / 1=oxide (default 0). Repeat for two metals.")
    ap.add_argument("--cl", type=float, default=None, help="Cl atoms per formula unit (with --dopant).")
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
    meta = SceneMeta(cfg)
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
    skipped: List[Tuple[str, str]] = []

    if args.dopant:
        if args.cl is None:
            ap.error("--cl is required when using --dopant.")
        try:
            dopants = [parse_dopant_flag(d) for d in args.dopant]
            label = "+".join(f"{m}{lv:g}{'ox' if of > 0 else ''}" for m, lv, of in dopants)
            items.append((label, build_candidate(meta, dopants, args.cl)))
        except Exception as exc:  # noqa: BLE001
            ap.error(str(exc))

    for spec in iter_specs(args):
        try:
            dopants, cl, ambiguous = parse_formula(spec, meta)
            if ambiguous:
                print(f"[WARN] {spec!r}: a 'both'-category metal's O-form can't be read "
                      "from the formula; defaulting to sulfide (0). Use --dopant to set it.",
                      flush=True)
            items.append((spec, build_candidate(meta, dopants, cl)))
        except Exception as exc:  # noqa: BLE001
            skipped.append((spec, str(exc)))
            print(f"[WARN] skipping {spec!r}: {exc}", flush=True)

    if not items:
        ap.error("No compositions to evaluate. Pass --formula/--formulas-file or --dopant.")

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
    # sort=True groups atoms by element so the POSCAR species/count lines are one
    # clean run per element (Li In P S Cl O Br) instead of hundreds of site-order
    # fragments. Scoring is order-invariant (mixed_type), so this is display-only.
    ase_write(args.save_poscar, atoms, format="vasp", sort=True, vasp5=True)
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
