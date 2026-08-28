#!/usr/bin/env python
"""screen_band.py — filter a run's generated.csv to candidates inside a target band.

The reward already steers the agent toward the target; this is the *readout*: which
generated candidates actually landed inside the band, and how they rank on the other
objectives once they are there.

It reads only ``generated.csv``, which already carries one ``obj_<name>_mean`` /
``obj_<name>_std`` column pair per property (written by ``training.py``'s generation
loop), so nothing has to be recomputed and no model is loaded.

Usage::

    python scripts/screen_band.py runs/cs2agbicl6_l3/generated.csv
    python scripts/screen_band.py runs/.../generated.csv --target 1.34 --band 0.10 \\
        --property bandgap --sort ehull --top 25
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Optional


def _to_float(row: Dict[str, str], key: str) -> Optional[float]:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv_path", help="Path to a run's generated.csv")
    p.add_argument("--property", default="bandgap",
                   help="Property name to band-filter on (default: bandgap).")
    p.add_argument("--target", type=float, default=1.34,
                   help="Target value in the property's real units (default: 1.34 eV).")
    p.add_argument("--band", type=float, default=0.10,
                   help="Half-width of the accepted band (default: 0.10).")
    p.add_argument("--sort", default=None,
                   help="Property to rank survivors by, ascending (e.g. ehull). "
                        "Default: by distance to the target.")
    p.add_argument("--top", type=int, default=20, help="Rows to print (default: 20).")
    p.add_argument("--out", default=None, help="Optional path to write the survivors as CSV.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if not os.path.exists(args.csv_path):
        raise SystemExit(f"no such file: {args.csv_path}")

    with open(args.csv_path, newline="") as f:
        rows: List[Dict[str, str]] = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{args.csv_path} has no rows.")

    col = f"obj_{args.property}_mean"
    if col not in rows[0]:
        objs = sorted(c[4:-5] for c in rows[0] if c.startswith("obj_") and c.endswith("_mean"))
        raise SystemExit(
            f"{args.csv_path} has no column {col!r}. Available objectives: {objs or '(none)'}. "
            "Pass --property with one of those."
        )

    lo, hi = args.target - args.band, args.target + args.band
    keep: List[Dict[str, Any]] = []
    n_missing = 0
    for r in rows:
        v = _to_float(r, col)
        if v is None:
            n_missing += 1
            continue
        if lo <= v <= hi:
            r = dict(r)
            r["_dist"] = abs(v - args.target)
            keep.append(r)

    sort_col = f"obj_{args.sort}_mean" if args.sort else None
    if sort_col and sort_col not in rows[0]:
        raise SystemExit(f"--sort {args.sort!r}: no column {sort_col!r} in {args.csv_path}.")
    keep.sort(key=(lambda r: (_to_float(r, sort_col) if sort_col else None, r["_dist"]))
              if sort_col else (lambda r: r["_dist"]))

    obj_cols = [c for c in rows[0] if c.startswith("obj_") and c.endswith("_mean")]
    print(f"{args.csv_path}: {len(rows)} generated, {len(keep)} inside "
          f"{args.property} in [{lo:.4g}, {hi:.4g}]"
          + (f", {n_missing} rows missing {col}" if n_missing else ""))
    if args.sort:
        print(f"ranked by {args.sort} (ascending); ties broken by distance to target")
    print()
    head = ["formula"] + obj_cols
    widths = [max(len(h), 22 if h == "formula" else 14) for h in head]
    print("  ".join(h.ljust(w) for h, w in zip(head, widths)))
    print("  ".join("-" * w for w in widths))
    for r in keep[: args.top]:
        cells = [str(r.get("formula", ""))[:widths[0]]]
        cells += [f"{_to_float(r, c):.4f}" if _to_float(r, c) is not None else "-"
                  for c in obj_cols]
        print("  ".join(c.ljust(w) for c, w in zip(cells, widths)))
    if len(keep) > args.top:
        print(f"... {len(keep) - args.top} more")

    if args.out:
        fieldnames = [c for c in rows[0]]
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in keep:
                w.writerow({k: r.get(k, "") for k in fieldnames})
        print(f"\nwrote {len(keep)} rows to {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
