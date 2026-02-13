# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Does

Offline DQN-style composition generator for discovering novel oxyhydroxide catalysts (ABCDE-OOH). A 5-step RL environment sequentially picks 5 distinct cations and their fractions (summing to 1.0) from 28 candidate elements, then a Q-network trained on Monte Carlo returns generates promising candidates.

## Commands

### Run the main experiment pipeline
```bash
python scripts/run_ABCDEOOH_experiment.py --out runs/demo --reward-mode none --num-random-eps 200 --dqn-epochs 5 --num-gen-eps 50
```

### Iterative buffer mode (Option B)
```bash
python scripts/run_ABCDEOOH_experiment.py --out runs/iter --reward-mode none --buffer-mode iterative --num-random-eps 200 --num-online-eps 200 --dqn-epochs 5 --num-gen-eps 50
```

### DeepMD reward mode (requires `ase` and `deepmd-kit`)
```bash
python scripts/run_ABCDEOOH_experiment.py --out runs/dp --reward-mode dp --dp-poscar PATH/TO/POSCAR --dp-model model_1.ckpt.pt --dp-model model_2.ckpt.pt --dp-objective mean_minus_kstd
```

### Summarize a replay buffer
```bash
python scripts/summarize_replay_buffer.py <path-to-random_dataset.npz>
```

### Install dependencies
```bash
pip install -r requirements.txt
```

There is no test suite, linter configuration, or build step.

## Architecture

### Pipeline flow (all in `scripts/run_ABCDEOOH_experiment.py`)

1. **Random buffer** — Roll out random episodes in `ABCDEOOHEnv`, compute MC returns as Q targets, save to `random_dataset.npz`
2. **Q-network training** — `StandardScaler` on state features, then train `QRegressor` (3-layer MLP) with MSE loss
3. **Iterative buffer** (optional) — Collect more episodes using learned policy (epsilon-greedy or stochastic top-k), append to buffer, retrain
4. **Candidate generation** — Greedy (or epsilon-greedy) action selection via trained Q-network, deduplicate, optionally filter by primary phase, write `generated.csv`

### Key modules (`src/abcde_ooh/`)

- **`env.py`** — `ABCDEOOHEnv`: 5-step constrained environment. Fractions are internally tracked as integer units out of 20 (i.e., 0.05 = 1 unit). Uses `_possible_sums_by_k` to precompute feasibility so that every action guarantees a valid terminal state. The `terminal_formula` property canonicalizes cation order (major-first, then alphabetical).
- **`model.py`** — `QRegressor`: input = concat(scaled_material_features, step_onehot, elem_onehot, frac_onehot) → scalar Q value.
- **`featurization.py`** — Wraps matminer/pymatgen Magpie composite features (Stoichiometry, ElementProperty, ValenceOrbital, IonProperty). Falls back to zlib-hash-based lightweight features for empty/invalid formulas.
- **`encoding.py`** — One-hot encode/decode for cation and fraction choices.
- **`dp_predictor.py`** — Optional DeepMD ensemble predictor for overpotential with uncertainty. Lazily imported only when `--reward-mode dp`.
- **`constraints/primary_phase.py`** — `check_primary_phase(comp)` validates against 5 oxyhydroxide phase types (NiFeCo, NiFe, CoFe, Ni-only, Co-only) with dopant fraction and ratio rules.

### Key design details

- Scripts add `src/` to `sys.path` at runtime, so the package works without installation.
- The environment enforces that each episode picks exactly 5 **distinct** cations with fractions on a 0.05-step grid summing to exactly 1.0. Feasibility is checked dynamically at each step.
- The scaler is fit once on the initial random buffer and reused for all subsequent phases (iterative collection and generation) for consistency.
- DeepMD predictions are cached by a canonicalized composition key to avoid redundant expensive calls.
- OpenMP thread environment variables are set at the top of the main script to mitigate macOS segfaults from conflicting BLAS/OpenMP libraries.
