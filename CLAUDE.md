# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Does

RL-based composition generator for discovering novel oxyhydroxide catalysts (ABCDE-OOH). A 5-step RL environment sequentially picks 5 distinct cations and their fractions (summing to 1.0) from 28 candidate elements. Three algorithms are supported: offline DQN (Q-network trained on Monte Carlo returns), REINFORCE, and A2C.

## Commands

### Run the main experiment pipeline
```bash
python scripts/run_ABCDEOOH_experiment.py --out runs/demo --num-random-eps 200 --dqn-epochs 5 --num-gen-eps 50 --dp-poscar POSCAR --dp-model model_1.ckpt.pt
```

### Iterative buffer mode (Option B)
```bash
python scripts/run_ABCDEOOH_experiment.py --out runs/iter --buffer-mode iterative --num-random-eps 200 --num-online-eps 200 --dqn-epochs 5 --num-gen-eps 50 --dp-poscar POSCAR --dp-model model_1.ckpt.pt
```

### Skip retraining; generate from a saved checkpoint
```bash
python scripts/run_ABCDEOOH_experiment.py --out runs/demo --only-generate --num-gen-eps 100 --dp-poscar POSCAR --dp-model model_1.ckpt.pt
```
`--only-generate` loads `<out>/std_scaler.bin` and `<out>/qnet.pt` and skips buffer building and Q training. Use `--load-qnet` / `--load-scaler` to override those paths.

### Reuse a previously collected random buffer
```bash
python scripts/run_ABCDEOOH_experiment.py --out runs/demo --use-saved-random-dataset --dqn-epochs 5 --num-gen-eps 50 --dp-poscar POSCAR --dp-model model_1.ckpt.pt
```

### REINFORCE
```bash
python scripts/run_ABCDEOOH_experiment.py --out runs/reinforce --rl-method reinforce \
  --pg-warmup-eps 200 --pg-train-eps 1000 --num-gen-eps 200 --pg-gen-stochastic \
  --dp-poscar POSCAR --dp-model model_1.ckpt.pt
```
Use `--pg-gen-stochastic` when the policy hasn't learned strong preferences to avoid duplicate compositions in generation (greedy argmax on a near-uniform policy is deterministic).

### A2C
```bash
python scripts/run_ABCDEOOH_experiment.py --out runs/a2c --rl-method a2c \
  --pg-warmup-eps 200 --pg-train-eps 1000 --num-gen-eps 200 --pg-gen-stochastic \
  --dp-poscar POSCAR --dp-model model_1.ckpt.pt
```
A2C additionally writes `value_net.pt`. Key extra flags: `--pg-lr-actor`, `--pg-lr-critic`, `--entropy-coef`.

### Fair comparison across all three methods
```bash
# DQN
python scripts/run_ABCDEOOH_experiment.py --out runs/dqn --rl-method dqn \
  --num-random-eps 1000 --dqn-epochs 50 --num-gen-eps 200 \
  --dp-poscar POSCAR --dp-model model_1.ckpt.pt --dp-model model_2.ckpt.pt

# REINFORCE
python scripts/run_ABCDEOOH_experiment.py --out runs/reinforce --rl-method reinforce \
  --pg-warmup-eps 200 --pg-train-eps 1000 --num-gen-eps 200 \
  --dp-poscar POSCAR --dp-model model_1.ckpt.pt --dp-model model_2.ckpt.pt

# A2C
python scripts/run_ABCDEOOH_experiment.py --out runs/a2c --rl-method a2c \
  --pg-warmup-eps 200 --pg-train-eps 1000 --num-gen-eps 200 \
  --dp-poscar POSCAR --dp-model model_1.ckpt.pt --dp-model model_2.ckpt.pt
```
Compare `generated.csv` reward distributions across runs.

### DeepMD reward (requires `ase` and `deepmd-kit`)
```bash
python scripts/run_ABCDEOOH_experiment.py --out runs/dp --dp-poscar PATH/TO/POSCAR --dp-model model_1.ckpt.pt --dp-model model_2.ckpt.pt --dp-objective mean_minus_kstd
```

### Evaluate specific formulas with DeepMD (standalone)
```bash
python scripts/evaluate_formulas_dp.py --formula "Ni0.70Fe0.15Ce0.05Er0.05Tm0.05O2H1" --dp-model model.pt --dp-poscar POSCAR
# Or batch from a file (one formula per line):
python scripts/evaluate_formulas_dp.py --formulas-file candidates.txt --dp-model model.pt --out-csv results.csv
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

**DQN path (`--rl-method dqn`, default):**
1. **Random buffer** — Roll out random episodes in `ABCDEOOHEnv`, compute MC returns as Q targets, save to `random_dataset.npz`
2. **Q-network training** — `StandardScaler` on state features, then train `QRegressor` (3-layer MLP) with MSE loss
3. **Iterative buffer** (optional) — Collect more episodes using learned policy (epsilon-greedy or stochastic top-k), append to buffer, retrain
4. **Candidate generation** — Greedy (or epsilon-greedy) action selection via trained Q-network, deduplicate, optionally filter by primary phase, write `generated.csv`

**PG path (`--rl-method reinforce` or `a2c`):**
1. **Warmup** — Roll out `--pg-warmup-eps` random episodes to fit `StandardScaler`
2. **Online training** — Each episode: roll out using `PolicyNet` (softmax sampling), compute MC returns, update actor with REINFORCE gradient; A2C additionally updates `ValueNet` critic to reduce variance
3. **Candidate generation** — Greedy argmax (default) or stochastic sampling (`--pg-gen-stochastic`) via trained `PolicyNet`

### Output files (under `--out`)

| File | DQN | REINFORCE | A2C | Description |
|---|---|---|---|---|
| `random_dataset.npz` | ✓ | — | — | Replay buffer arrays: `s_mat`, `s_step`, `a_elem`, `a_comp`, `y` |
| `std_scaler.bin` | ✓ | ✓ | ✓ | Serialized `StandardScaler` (joblib) |
| `qnet.pt` | ✓ | — | — | Q-network state dict (PyTorch) |
| `policy.pt` | — | ✓ | ✓ | PolicyNet state dict (PyTorch) |
| `value_net.pt` | — | — | ✓ | ValueNet (critic) state dict (PyTorch) |
| `generated.csv` | ✓ | ✓ | ✓ | Deduplicated candidates with `formula`, `reward`, `dp_mean/std`, `primary_ok/label` |
| `run_config.json` | ✓ | ✓ | ✓ | Full `argparse` namespace for reproducibility |

### Key modules (`src/abcde_ooh/`)

- **`env.py`** — `ABCDEOOHEnv`: 5-step constrained environment. Fractions are internally tracked as integer units out of 20 (i.e., 0.05 = 1 unit). Uses `_possible_sums_by_k` to precompute feasibility so that every action guarantees a valid terminal state. The `terminal_formula` property canonicalizes cation order (major-first, then alphabetical). Each action is a pair of one-hot vectors `(elem_oh, comp_oh)`; `EpisodeStep` stores per-step state/action/reward/allowed_actions.
- **`model.py`** — `QRegressor`: input = concat(scaled_material_features, step_onehot, elem_onehot, frac_onehot) → scalar Q value. `PolicyNet`: identical architecture, outputs scalar logit (softmax over allowed actions gives π(a|s)). `ValueNet`: state-only input (material + step), outputs scalar baseline for A2C. All three use three linear layers with LeakyReLU and `hidden_dim=128`.
- **`featurization.py`** — Wraps matminer/pymatgen Magpie composite features (Stoichiometry, ElementProperty, ValenceOrbital, IonProperty). Falls back to zlib-hash-based lightweight features for empty/invalid formulas. Feature dimension is inferred at import time via `feature_labels()`.
- **`encoding.py`** — One-hot encode/decode for cation and fraction choices.
- **`dp_predictor.py`** — DeepMD ensemble predictor for overpotential with uncertainty. Imported at startup; `--dp-model` is required.
- **`constraints/primary_phase.py`** — `check_primary_phase(comp)` validates against 5 oxyhydroxide phase types (NiFeCo, NiFe, CoFe, Ni-only, Co-only) with dopant fraction and ratio rules. Used by `--primary-phase-filter {none,buffer,generated,both}`.

### Key design details

- Scripts add `src/` to `sys.path` at runtime, so the package works without installation.
- The environment enforces that each episode picks exactly 5 **distinct** cations with fractions on a 0.05-step grid summing to exactly 1.0. Feasibility is checked dynamically at each step using the precomputed `_possible_sums_by_k` table.
- The scaler is fit once on the initial random buffer and reused for all subsequent phases (iterative collection and generation) for consistency.
- DeepMD predictions are cached by a canonicalized composition key to avoid redundant expensive calls.
- OpenMP thread environment variables are set at the top of the main script to mitigate macOS segfaults from conflicting BLAS/OpenMP libraries.
- Iterative buffer schedule: `--iter-num-iters` controls rounds of collect→retrain; `--iter-online-eps-per-iter` overrides episodes per round; `--iter-train-epochs` overrides training epochs per round (defaults to `--dqn-epochs`).
- PG methods are on-policy: each episode uses the current policy, no replay buffer. The `EpisodeStep.allowed_actions` field (populated by `env.step` before each transition) is used in `train_pg` to reconstruct the full action distribution for the log-probability computation.
- Use `--pg-gen-stochastic` when the policy hasn't yet learned strong preferences; greedy argmax on a near-uniform policy will generate the same composition every episode.

## Order invariance

The framework's state and reward are **permutation-invariant over element choices** by construction. Picking elements in any order yields the same per-step state features (whenever the partial multisets match), the same Q / π estimates, and the same terminal reward.

### What's guaranteed

- **State at step k** is a function of `(partial Composition multiset, step counter)` — never of the trajectory's action order. Two episodes that have picked the same multiset of `(element, fraction)` pairs by step k get bit-identical (up to floating-point round-off) `state_material_features`.
- **Action at step k** is `(elem_identity, fraction)` — the element label itself, not a positional slot. So `Q(s, "add Co at 0.2")` doesn't depend on whether Co is being added at step 2 or step 4 from the same partial bag.
- **Terminal reward** is `predictor(terminal Composition)` — the predictor receives a `Composition` (unordered mapping), so the order in which the episode built the composition is invisible to the predictor.
- **Generation dedup** in `generated.csv` collapses permutation-equivalent compositions to a single row via `env.terminal_comp_key()` (canonical sorted multiset).

### Where invariance is enforced

- `featurize_formula` (`src/rl_matdesign/featurization.py:43`) routes the partial-state string through `pymatgen.Composition`, which is an unordered element→amount mapping. All downstream Magpie features are statistics over that mapping (mean / std / sum / max) — operations that don't care about order.
- `env.terminal_comp_key` (`env.py:258`, `env_integer.py:148`) canonicalizes the terminal composition to a sorted tuple for cross-episode dedup.

### Predictor contract for FQN plug-ins

A user-authored predictor (`predictor: pkg.module:ClassName` in YAML) **MUST** treat the input dict as an unordered mapping. Specifically:

- Don't iterate `composition.items()` assuming a meaningful order (e.g. "the first key is the major cation"). Use `sorted(composition.items())` if you need a stable iteration order.
- Don't extract `next(iter(composition))` and treat it as semantically distinguished.
- If you build derived features, drive them off `sorted(composition.items())` or pass through `pymatgen.Composition` like the built-in predictors do.

Violations are silent — the agent's value function fragments across orderings, training slows down, and generated.csv contains "duplicate" compositions that the predictor scored differently.

### Test guardrail

`tests/test_order_invariance.py` pins this property at three layers (featurizer, env, predictor contract). If you add a custom predictor or modify the featurizer, add a parametrized test there asserting `predict(composition_A) == predict(composition_B)` for two differently-ordered dicts with the same content. The existing `test_contract_example_predictor_is_order_invariant` and `test_contract_violation_is_detected` document the expected shape.

### Sample-efficiency knob — `dqn_augment_permutations`

YAML key: `dqn_augment_permutations: K` (CLI: `--dqn-augment-permutations K`, default `0`).

When set on a DQN run, each completed episode is re-inserted into the replay buffer K additional times under random permutations of the action sequence. The terminal reward is reused (no extra predictor call), and within-episode duplicates are skipped. This gives DQN more `(state, action)` coverage per expensive lab call — useful when the predictor is the bottleneck (DeepMD ensembles, OOH overpotential) or when `cation_set` is large.

**DQN only.** PG / A2C are on-policy: their gradient direction is tied to the action *actually sampled by the current policy*, so permuting trajectories breaks the policy-gradient theorem. The flag is silently ignored (with a one-line warning) for non-DQN methods.

Validity rules: with `LastStepElementFilter.reserve_for_last=True` (sinter/calcine/OOH configs), only the first N-1 positions are permuted to keep the required element in the last slot. With `episode_style: fixed_order_amount`, augmentation is a no-op (cation order is forced by config).

Recommended starting value: `K=3`. For N=5 with a fixed last position, up to 23 alternative permutations exist; the helper caps `K` at the available count and logs a warning if you set it higher.

### Generation diversity caveat

Even with a perfectly trained order-invariant Q-network, **greedy argmax decoding** deterministically picks the same first cation every episode — the symmetry says many starting elements would give the same value, but `argmax` arbitrarily picks one. The framework's existing mitigations:

- DQN generation: `--gen-temperature 1.0` (default) fires Boltzmann sampling, which spreads mass across tied actions.
- PG generation: pass `--pg-gen-stochastic` to sample from `π(a|s)` rather than argmax.
- Generation dedup in `generated.csv` collapses any permutation-equivalent duplicates that slip through.

If you see "all candidates start with the same element" in `generated.csv`, drop temperature toward 0 *only* if you've confirmed via `scripts/check_invariance.py` that the Q-network has actually learned the symmetry; otherwise raise temperature instead.

### Diagnostic — `scripts/check_invariance.py`

Standalone CLI that loads a config and (optionally) a trained checkpoint, samples random N-element multisets, and evaluates each under permutations to report the max delta in:

- Featurizer output (should be 0 modulo float round-off — catches broken featurizers).
- Predictor `predict()` mean / std (catches contract-violating user predictors).
- Q-network outputs (catches custom architectures that inadvertently encode order).

Usage: `python scripts/check_invariance.py --config configs/oxides_sinter.yaml --num-samples 20`. Pass `--qnet runs/<my-run>/qnet.pt` to also probe a trained checkpoint.
