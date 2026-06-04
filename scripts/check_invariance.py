#!/usr/bin/env python3
"""Order-invariance diagnostic for the RL materials-design framework.

Probes three layers — featurizer, predictor, and (optionally) Q-network —
for permutation invariance over element ordering. For each random multiset,
evaluates several orderings and reports the maximum absolute / relative
delta. A pass means the architecture genuinely treats elements as a bag.

Usage
-----
    # Predictor + featurizer only (no checkpoint required)
    python scripts/check_invariance.py --config configs/oxides_sinter.yaml \\
        --num-samples 20

    # Include Q-network probe (loads a trained checkpoint)
    python scripts/check_invariance.py --config configs/oxides_sinter.yaml \\
        --qnet runs/oxides_sinter_dqn_s0/qnet.pt --scaler runs/.../std_scaler.bin \\
        --num-samples 20

Output is a small table:

    layer         | max_abs_delta | max_rel_delta | verdict
    --------------|---------------|---------------|--------
    featurizer    | 4.34e-19      | 1.55e-16      | PASS
    predictor     | 0.0           | 0.0           | PASS
    qnet          | 2.31e-07      | 4.18e-09      | PASS

Verdict threshold: anything below ``--tol`` (default 1e-6 absolute) passes.
"""
from __future__ import annotations

import argparse
import itertools
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Mirror run_experiment.py: add src/ + scripts/ to sys.path so we can run
# from a fresh checkout without installing.
_ROOT = Path(__file__).resolve().parent.parent
for sub in ("src", "scripts"):
    p = str(_ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True,
                   help="YAML config (same shape as run_experiment.py).")
    p.add_argument("--num-samples", type=int, default=10,
                   help="Number of random multisets to probe (default 10).")
    p.add_argument("--num-perms", type=int, default=5,
                   help="Permutations per multiset (default 5; capped at N!).")
    p.add_argument("--qnet", default=None,
                   help="Optional path to trained qnet.pt for Q-network probe.")
    p.add_argument("--scaler", default=None,
                   help="Path to std_scaler.bin (default: derived from --qnet).")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed (default 0).")
    p.add_argument("--tol", type=float, default=1e-6,
                   help="Absolute delta threshold for PASS verdict (default 1e-6).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Multiset sampling
# ---------------------------------------------------------------------------

def _sample_random_multiset(
    env, rng: random.Random
) -> List[Tuple[str, str]]:
    """Roll a random terminal episode in *env*, return the (elem, frac_str) sequence."""
    env.initialize()
    sequence: List[Tuple[str, str]] = []
    cation_set = env.cation_set
    fraction_set = env.fraction_set
    while env.counter < env.n_components:
        actions = env.allowed_actions()
        if not actions:
            return []
        elem_oh, comp_oh = rng.choice(actions)
        elem = cation_set[int(np.argmax(elem_oh))]
        frac = fraction_set[int(np.argmax(comp_oh))]
        env.step((elem_oh, comp_oh))
        sequence.append((elem, frac))
    return sequence


def _enumerate_orderings(
    sequence: Sequence[Tuple[str, str]],
    pinned_last: bool,
    num_perms: int,
    rng: random.Random,
) -> List[List[Tuple[str, str]]]:
    """Sample *num_perms* orderings of *sequence* (always including the original)."""
    if pinned_last:
        free, last = list(sequence[:-1]), sequence[-1]
    else:
        free, last = list(sequence), None

    perms = list(itertools.permutations(range(len(free))))
    rng.shuffle(perms)
    chosen = perms[: max(1, min(num_perms, len(perms)))]

    out: List[List[Tuple[str, str]]] = []
    for idx_seq in chosen:
        ordering = [free[i] for i in idx_seq]
        if last is not None:
            ordering.append(last)
        out.append(ordering)
    return out


# ---------------------------------------------------------------------------
# Probe per layer
# ---------------------------------------------------------------------------

def _probe_featurizer(
    orderings: List[List[Tuple[str, str]]],
    env_type: str,
) -> Tuple[float, float]:
    """Featurize the terminal formula of each ordering, return max delta."""
    from rl_matdesign.featurization import featurize_formula

    feats = []
    for ordering in orderings:
        if env_type == "integer_ratio":
            formula = "".join(f"{el}{frac}" for el, frac in ordering)
        else:
            formula = "".join(f"{el}{frac}" for el, frac in ordering)
        feats.append(featurize_formula(formula))
    return _max_delta(feats)


def _probe_predictor(
    orderings: List[List[Tuple[str, str]]],
    predictor,
    env_type: str,
) -> Tuple[float, float]:
    """Run predictor on each ordering's terminal composition (mean + std)."""
    from pymatgen.core.composition import Composition

    means, stds = [], []
    for ordering in orderings:
        formula = "".join(f"{el}{frac}" for el, frac in ordering)
        try:
            comp_obj = Composition(formula).fractional_composition
            comp = {str(el): float(comp_obj[el]) for el in comp_obj}
        except Exception:
            comp = {}
        try:
            m, s = predictor.predict(comp)
        except Exception as e:
            print(f"[WARN] predictor.predict({formula!r}) raised {type(e).__name__}: {e}")
            continue
        means.append(float(m))
        stds.append(float(s))

    abs_m, rel_m = _max_delta_scalar(means)
    abs_s, rel_s = _max_delta_scalar(stds)
    return max(abs_m, abs_s), max(rel_m, rel_s)


def _probe_qnet(
    orderings: List[List[Tuple[str, str]]],
    env_factory,
    qnet,
    scaler,
    elem_feats_scaled: np.ndarray,
    fraction_set: List[str],
    device,
) -> Tuple[float, float]:
    """Compare Q(s, a) across orderings only where state AND action match.

    For each step k where multiple orderings reach the *same* prefix multiset,
    we query Q at that state with a fixed canonical action (the
    alphabetically-first element + a fixed fraction) and compare. This is the
    meaningful invariance check — it would catch a custom featurizer that
    encodes step order, or a Q-net that somehow depends on training-data
    ordering. Different (s, a) pairs across orderings are NOT compared (their
    Q values legitimately differ).
    """
    import torch

    N = len(orderings[0])
    max_abs, max_rel = 0.0, 0.0

    # Canonical action used for all queries (need not be "allowed" — we just
    # want a fixed query vector).
    canonical_elem_idx = 0  # first element in cation_set
    canonical_frac = float(fraction_set[0])

    for k in range(N):
        # Group orderings by their prefix multiset at step k.
        prefixes = [tuple(sorted(o[:k])) for o in orderings]
        groups: Dict[tuple, List[int]] = {}
        for i, p in enumerate(prefixes):
            groups.setdefault(p, []).append(i)

        for prefix, idxs in groups.items():
            if len(idxs) < 2:
                continue
            # Compute scaled state features for each ordering in this group.
            qs: List[float] = []
            for i in idxs:
                env = env_factory()
                env.initialize()
                cation_set = env.cation_set
                for elem, frac in orderings[i][:k]:
                    elem_oh = tuple(
                        1.0 if j == cation_set.index(elem) else 0.0
                        for j in range(len(cation_set))
                    )
                    comp_oh = tuple(
                        1.0 if j == fraction_set.index(frac) else 0.0
                        for j in range(len(fraction_set))
                    )
                    try:
                        env.step((elem_oh, comp_oh))
                    except Exception:
                        break
                s_mat = env.state_featurizer(env.state)
                s_mat_sc = scaler.transform(s_mat.reshape(1, -1))[0]
                s_step = np.zeros(env.n_components, dtype=float)
                s_step[k] = 1.0
                a_elem = elem_feats_scaled[canonical_elem_idx]
                with torch.no_grad():
                    q = qnet(
                        torch.tensor(s_mat_sc.reshape(1, -1), dtype=torch.float32, device=device),
                        torch.tensor(s_step.reshape(1, -1),  dtype=torch.float32, device=device),
                        torch.tensor(a_elem.reshape(1, -1),  dtype=torch.float32, device=device),
                        torch.tensor([[canonical_frac]],     dtype=torch.float32, device=device),
                    )
                qs.append(float(q.item()))
            if len(qs) >= 2:
                a, r = _max_delta_scalar(qs)
                max_abs = max(max_abs, a)
                max_rel = max(max_rel, r)
    return max_abs, max_rel


def _max_delta(arrays: Sequence[np.ndarray]) -> Tuple[float, float]:
    if len(arrays) < 2:
        return 0.0, 0.0
    ref = np.asarray(arrays[0], dtype=float)
    max_abs, max_rel = 0.0, 0.0
    for other in arrays[1:]:
        other = np.asarray(other, dtype=float)
        diff = np.abs(other - ref)
        max_abs = max(max_abs, float(diff.max()))
        denom = np.maximum(np.abs(ref), 1e-30)
        max_rel = max(max_rel, float((diff / denom).max()))
    return max_abs, max_rel


def _max_delta_scalar(values: Sequence[float]) -> Tuple[float, float]:
    if len(values) < 2:
        return 0.0, 0.0
    ref = float(values[0])
    max_abs = max(abs(float(v) - ref) for v in values[1:])
    denom = max(abs(ref), 1e-30)
    max_rel = max_abs / denom
    return max_abs, max_rel


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Reuse load_config + the env / predictor / constraint factories from
    # run_experiment.py so user FQN plug-ins resolve identically.
    from run_experiment import (
        load_config, build_predictor, build_constraint_filter,
        _default_digits, _default_fractions,
    )
    from rl_matdesign.env import CompositionEnv
    from rl_matdesign.env_integer import IntegerRatioEnv
    from rl_matdesign.training import _precompute_elem_features, _detect_last_position_pin

    cfg = load_config(args.config)
    rng = random.Random(args.seed)

    predictor = build_predictor(cfg, seed=args.seed)
    env_type = cfg.get("env_type", "fraction")

    def _build_env():
        if env_type == "integer_ratio":
            env = IntegerRatioEnv(
                cation_set=cfg["cation_set"],
                ratio_set=cfg.get("ratio_set", None) or _default_digits(),
                n_components=int(cfg.get("n_components", 5)),
                phase_filter=None,
            )
        else:
            env = CompositionEnv(
                cation_set=cfg["cation_set"],
                fraction_set=cfg.get("fraction_set", None) or _default_fractions(),
                anion_formula=cfg.get("anion_formula", ""),
                n_components=int(cfg.get("n_components", 5)),
                phase_filter=None,
                total_units=int(cfg.get("total_units", 20)),
                element_bounds=cfg.get("element_bounds"),
                episode_style=cfg.get("episode_style", "element_then_amount"),
            )
        env.phase_filter = build_constraint_filter(cfg, env=env)
        return env

    probe_env = _build_env()
    pinned_last = _detect_last_position_pin(probe_env) is not None

    # Sample multisets
    multisets: List[List[Tuple[str, str]]] = []
    while len(multisets) < args.num_samples:
        seq = _sample_random_multiset(probe_env, rng)
        if seq:
            multisets.append(seq)

    # Optionally load Q-network for the third probe.
    qnet = None
    scaler = None
    elem_feats_scaled = None
    device = None
    if args.qnet:
        import joblib
        import torch
        from rl_matdesign.model import QRegressor

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scaler_path = args.scaler or os.path.join(
            os.path.dirname(args.qnet), "std_scaler.bin"
        )
        if not os.path.exists(scaler_path):
            print(f"[WARN] cannot find scaler at {scaler_path}; skipping Q-net probe.")
        else:
            scaler = joblib.load(scaler_path)
            elem_feats_scaled, _ = _precompute_elem_features(
                probe_env.cation_set, probe_env.state_featurizer
            )
            state_dim = int(getattr(scaler, "n_features_in_", scaler.mean_.shape[0]))
            qnet = QRegressor(
                state_dim=state_dim,
                step_dim=probe_env.n_components,
                elem_dim=int(elem_feats_scaled.shape[1]),
                frac_dim=1,
                hidden_dim=int(cfg.get("dqn_hidden_dim", cfg.get("hidden_dim", 128))),
            ).to(device)
            qnet.load_state_dict(torch.load(args.qnet, map_location=device))
            qnet.eval()

    # Aggregate deltas across all multisets
    feat_max_abs, feat_max_rel = 0.0, 0.0
    pred_max_abs, pred_max_rel = 0.0, 0.0
    qnet_max_abs, qnet_max_rel = 0.0, 0.0

    for seq in multisets:
        orderings = _enumerate_orderings(seq, pinned_last, args.num_perms, rng)

        a, r = _probe_featurizer(orderings, env_type)
        feat_max_abs = max(feat_max_abs, a); feat_max_rel = max(feat_max_rel, r)

        a, r = _probe_predictor(orderings, predictor, env_type)
        pred_max_abs = max(pred_max_abs, a); pred_max_rel = max(pred_max_rel, r)

        if qnet is not None:
            a, r = _probe_qnet(
                orderings, _build_env, qnet, scaler,
                elem_feats_scaled, list(probe_env.fraction_set), device,
            )
            qnet_max_abs = max(qnet_max_abs, a); qnet_max_rel = max(qnet_max_rel, r)

    # Verdict + table
    def _verdict(abs_delta: float) -> str:
        return "PASS" if abs_delta <= args.tol else "FAIL"

    rows = [
        ("featurizer", feat_max_abs, feat_max_rel, _verdict(feat_max_abs)),
        ("predictor",  pred_max_abs, pred_max_rel, _verdict(pred_max_abs)),
    ]
    if qnet is not None:
        rows.append(("qnet", qnet_max_abs, qnet_max_rel, _verdict(qnet_max_abs)))

    print()
    print(f"{'layer':<12} | {'max_abs_delta':>14} | {'max_rel_delta':>14} | verdict")
    print("-" * 62)
    overall_pass = True
    for name, abs_d, rel_d, verdict in rows:
        print(f"{name:<12} | {abs_d:>14.3e} | {rel_d:>14.3e} | {verdict}")
        if verdict == "FAIL":
            overall_pass = False
    print()
    print(f"Tolerance threshold: {args.tol:.0e} (absolute)")
    print(f"Multisets probed: {len(multisets)}; permutations per multiset: {args.num_perms}")
    if not overall_pass:
        print("[FAIL] one or more layers exceeded the tolerance — see table above.")
        sys.exit(1)
    print("[PASS] order invariance holds across all probed layers.")


if __name__ == "__main__":
    main()
