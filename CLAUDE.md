# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Does

RL-based composition generator for discovering novel oxyhydroxide catalysts (ABCDE-OOH). A 5-step RL environment sequentially picks 5 distinct cations and their fractions (summing to 1.0) from 28 candidate elements. Three algorithms are supported: classical online DQN (Q-network trained on one-step TD targets from a replay buffer), REINFORCE, and A2C.

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

### Cost accounting — comparing methods on time, not just reward quality

Every run writes `<out>/timing.json` and adds cumulative cost columns to
`training_log.csv`. Nothing needs to be enabled; the instrumentation is always on.

**How it works.** `run_experiment.py` wraps the predictor in a `PredictorTimer`
(`src/rl_matdesign/utils/timing.py`) immediately after `build_predictor`, *before*
`build_env`. Every training reward flows through the `reward_fn` / `mg_reward_fn`
closures in `build_env`, and the envs only call them at an episode's terminal step
— so one wrap accounts for every predictor call in every phase, for all three env
types and all three methods. The proxy delegates unknown attributes to the real
predictor, so `_cache` checkpointing and `predict_raw` keep working.

`timing.json` holds `phases_s` (`setup` / `warmup` / `train` / `generate`),
`total_s`, `overhead_s` (= total minus predictor time, i.e. the RL machinery), and
a `predictor` block with `t_predict_s`, `n_calls`, `n_unique`, `cache_hit_rate`,
`mean_s_per_unique` and `best_reward`. `n_unique` is counted against the timer's
own key set, not the predictor's internal cache — the three predictors cache under
different attribute names (`_cache` vs `_stats_cache`) and `dummy` doesn't cache
at all.

`training_log.csv` gains `t_wall`, `t_predict_cum`, `n_predict_calls`,
`n_predict_unique` and `best_reward_so_far` on each `dqn_train` / `pg_train` row.
`generated.csv` is deliberately unchanged.

**Making the figures** — reads only saved text files, so the expensive runs can
happen on a GPU box and the plots can be made later on a laptop:

```bash
python scripts/baselines/compare_timing.py \
    --run "DQN(bootstrap):runs/ooh_dqn_boot" \
    --run "DQN(mc):runs/ooh_dqn_mc" \
    --run "A2C:runs/ooh_a2c" \
    --out runs/compare/cost --title "OOH: cost to best candidate"
```

Four panels: best-reward-so-far vs wall-clock, best-reward-so-far vs cumulative
unique predictor calls, a predictor-vs-overhead wall-clock breakdown, and cost per
real evaluation with the cache-hit rate overlaid. Each `PATH` is a run directory,
or a parent holding `seed_*/` dirs (from `run_seeds.py`), in which case curves are
drawn as a median line with a min–max band. Like `compare_methods.py`, panel order
follows the `--run` order unless you pass `--sort-by-best`.

**Two things to keep straight when interpreting the result:**

- `--dqn-target-mode mc` is a DQN *ablation*, not a fourth method — same rollout
  loop, same buffer, same episode count, only the regression target changes. Label
  the arms `DQN(bootstrap)` / `DQN(mc)` / `A2C`.
- Configured episode budgets are asymmetric across methods (e.g. `oxides_sinter.yaml`:
  DQN 1000+50000 episodes vs A2C 500×15 = 7500). That is *why* the comparison is
  plotted against time and predictor calls rather than episode index — those curves
  stay comparable no matter how long each arm runs.
- A run resumed with `--resume-training` restarts its counters with a warm cache,
  so its hit rate reads artificially high. `compare_timing.py` flags such runs;
  benchmark without `--resume-training` for clean numbers.

### DeepMD reward (requires `ase` and `deepmd-kit`)
```bash
python scripts/run_ABCDEOOH_experiment.py --out runs/dp --dp-poscar PATH/TO/POSCAR --dp-model model_1.ckpt.pt --dp-model model_2.ckpt.pt --dp-objective mean_minus_kstd
```

### Choosing which adsorbates the OOH predictor builds — `adsorbates`

`predictor: ooh` places adsorbate intermediates on each randomly-doped slab before
scoring it. The YAML key `adsorbates` selects which ones, in frame order:

```yaml
predictor: ooh
adsorbates: [O, OH, OOH]   # default — historical behaviour
# adsorbates: [O]          # only the frame that is actually read (see below)
# adsorbates: []           # bare parent slab: no adsorbate atoms at all
```

**The empty list means the bare parent slab** — one frame per random config instead
of three, so ~3× fewer DeepMD evaluations and ~3× fewer relaxations under
`geo_opt: true`. `ads_height` / `ads_dz` are unused in that regime.

Bare is the *empty selection* rather than a member of the list on purpose: every
frame in one DeepMD batch must have the same atom count (an adsorbate frame has
`nat_slab + 3` atoms, a bare one `nat_slab`), so the two are mutually exclusive by
construction and the equal-`natoms` check in `_build_dp_inputs_for_one_doped_slab`
can never be violated from YAML.

**How `adsorbates` interacts with `output_index`.** All frames go into a single
`dp.eval` batch, and `pick_scalar` (`utils/dp_eval.py:81-89`) flattens the whole
batch before indexing. The flattening is frame-major, so with the default list and
`output_index: 0` **only the O\* frame's value is read** — OH\* and OOH\* are built,
optionally relaxed, evaluated, and discarded. DeepMD scores frames independently,
so they do not affect the O\* number. If you are running with the default
`output_index`, `adsorbates: [O]` is therefore ~3× cheaper for identical rewards.

**Cache invalidation.** `adsorbates` and `output_index` are folded into
`OOHCatalystPredictor._comp_key`, because both change which structure the cached
number describes. Changing either means a saved `dp_cache` (carried in the DQN
`checkpoint.pt`) no longer matches, so a resumed run recomputes once rather than
returning values for a different structure.

There is no ΔG / Sabatier arithmetic in this repo — no 1.23 V, no ZPE or entropy
corrections. The DeepProperty head emits the number directly, so changing
`adsorbates` changes *which structure the model sees*, not a thermodynamic cycle.
A bare slab may be out of distribution for a head trained on adsorbate-bearing
slabs; use `--dp-debug-dir` to dump and inspect the exact frames.

### Evaluate specific formulas with DeepMD (standalone)
```bash
python scripts/evaluate_formulas_dp.py --formula "Ni0.70Fe0.15Ce0.05Er0.05Tm0.05O2H1" --dp-model model.pt --dp-poscar POSCAR
# Or batch from a file (one formula per line):
python scripts/evaluate_formulas_dp.py --formulas-file candidates.txt --dp-model model.pt --out-csv results.csv
# Bare parent slab (flag passed with no names), dumping the structures for inspection:
python scripts/evaluate_formulas_dp.py --formula "Ni0.70Fe0.15Ce0.05Er0.05Tm0.05O2H1" \
  --dp-model model.pt --dp-poscar POSCAR --adsorbates --dp-debug-dir /tmp/ooh_bare
```
Omit `--adsorbates` for the default three; pass it bare for the clean slab; pass
names (`--adsorbates O OH`) for a subset. `--output-index` is also exposed here —
it previously defaulted to 0 with no way to change it from this script.

### Summarize a replay buffer
The DQN replay buffer is persisted inside the periodic `checkpoint.pt` (key `"buffer"`), not as a standalone file. Point the summarizer at the run directory:
```bash
python scripts/summarize_replay_buffer.py --run-dir runs/dqn
# Recompute predictor mean/std for each unique composition:
python scripts/summarize_replay_buffer.py --run-dir runs/dqn --recompute
```
It reconstructs episodes (splitting buffer rows on `done=True`), decodes each composition via `species_set` from `run_config.json`, and writes `replay_buffer_summary.csv` (`formula`, `terminal_reward`, `n_buffer_rows`, plus `pred_mean/std` when `--recompute`).

### Install dependencies
```bash
pip install -r requirements.txt
```

There is no test suite, linter configuration, or build step.

## Architecture

### Pipeline flow (all in `scripts/run_ABCDEOOH_experiment.py`)

**DQN path (`--rl-method dqn`, default), `train_dqn_online`:**
1. **Warmup** — Roll out `--dqn-warmup-eps` random episodes with real rewards to pre-fill an in-memory FIFO replay buffer (`collections.deque(maxlen=dqn_buffer_size)`) and fit the `StandardScaler` on raw state features. Each env step becomes one transition row: `{s_mat_raw, s_step, a_elem_idx, a_comp_val, reward, s_mat_next_raw, s_step_next, next_allowed_idx, done}` (see `add_episode_to_buffer`).
2. **Online training** — For `--dqn-num-train-eps` episodes: ε-greedy rollout → new rows, then `--dqn-grad-steps-per-ep` minibatch SGD steps minimizing SmoothL1 on the one-step TD target `r + γ·max_{a'∈next_allowed} Q_target(s',a')`. A target network is hard-copied every `--dqn-target-update-freq` episodes; ε anneals linearly. The buffer is checkpointed to `checkpoint.pt` (key `"buffer"`) for exact-state resume — there is no `random_dataset.npz`.
3. **Candidate generation** — Boltzmann-sampled (or greedy) action selection via trained Q-network, deduplicate, optionally filter by phase, write `generated.csv`

**PG path (`--rl-method reinforce` or `a2c`):**
1. **Warmup** — Roll out `--pg-warmup-eps` random episodes to fit `StandardScaler`
2. **Online training** — Each episode: roll out using `PolicyNet` (softmax sampling), compute MC returns, update actor with REINFORCE gradient; A2C additionally updates `ValueNet` critic to reduce variance
3. **Candidate generation** — Greedy argmax (default) or stochastic sampling (`--pg-gen-stochastic`) via trained `PolicyNet`

**Advantage standardisation and the entropy floor (PG only).** Advantages are
standardised across each batch in `train_pg` — unconditionally, no flag. Without
it the actor term scales with the raw reward (sintering temperatures are 400–700)
while the entropy bonus is at most `pg_entropy_coef · ln|A| ≈ 0.5`, so the entropy
term cannot influence the update and the policy collapses to a single composition.
The observed failure: A2C got *worse* with more episodes (best 632 → 670 → 649 as
the budget went 2.7k → 7.7k → 45.2k), because mean return kept improving while the
best candidate froze ~20% in and `unique_comps_seen` stopped growing.

Consequences to keep straight:

- `pg_entropy_coef` and `pg_repeat_penalty_coef` are in **σ of batch return**, not
  the property's units, as is the `repeat_penalty` log column. `return_shaped`
  converts the penalty back to the property's units so it stays comparable against
  `return_raw`. Logs from before this change match neither convention.
- `pg_entropy_min` (default 0.3) floors **normalised** entropy `H / ln|A|`, not
  nats, so it ports between the 80-element oxide env and OOH. `entropy_norm` and
  `entropy_coef_eff` in `training_log.csv` are the diagnostic columns.
- DQN is unaffected — ε-greedy is exogenous exploration and the replay buffer keeps
  diverse transitions alive, so it does not have this failure mode.
- When sweeping budgets, hold `pg_batch_eps` fixed and vary only `pg_num_iters`.
  Batch size changes gradient noise *and* updates-per-episode, so varying both
  confounds budget with collapse rate.

### Output files (under `--out`)

| File | DQN | REINFORCE | A2C | Description |
|---|---|---|---|---|
| `checkpoint.pt` | ✓ | ✓ | ✓ | Periodic mid-run checkpoint. For DQN (`type=="dqn"`) it holds the replay `buffer` (list of transition-row dicts), `qnet_state`, `target_net_state`, `opt_state`, `eps`, `episodes_completed`, `dp_cache`. Enables `--resume-training`. (Replaces the old `random_dataset.npz`.) |
| `std_scaler.bin` | ✓ | ✓ | ✓ | Serialized `StandardScaler` (joblib) |
| `qnet.pt` | ✓ | — | — | Q-network state dict (PyTorch) |
| `policy.pt` | — | ✓ | ✓ | PolicyNet state dict (PyTorch) |
| `value_net.pt` | — | — | ✓ | ValueNet (critic) state dict (PyTorch) |
| `generated.csv` | ✓ | ✓ | ✓ | Deduplicated candidates with `formula`, `reward`, `dp_mean/std`, `primary_ok/label` |
| `run_config.json` | ✓ | ✓ | ✓ | Full `argparse` namespace for reproducibility |
| `training_log.csv` | ✓ | ✓ | ✓ | Per-episode (`phase="dqn_train"`) / per-iteration (`phase="pg_train"`) metrics, plus the cumulative cost columns below |
| `timing.json` | ✓ | ✓ | ✓ | Wall-clock + predictor-call accounting for the run (see "Cost accounting") |

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

When set on a DQN run, each completed episode is re-inserted into the replay buffer K additional times under random permutations of the action sequence. The terminal reward is reused (no extra predictor call), and within-episode duplicates are skipped. This gives DQN more `(state, action)` coverage per expensive lab call — useful when the predictor is the bottleneck (DeepMD ensembles, OOH overpotential) or when `species_set` is large.

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

## Hyperparameter optimization

`scripts/hpo.py` is a generic Optuna-based HPO driver that works for any scenario (`configs/ooh.yaml`, `oxides_sinter.yaml`, `ti_alloy.yaml`, `hea.yaml`, etc.) and any method (`dqn`, `reinforce`, `a2c`) without driver edits — you only write a search-space YAML in `configs/hpo/`.

### Quick start

```bash
pip install "optuna>=3.0"
python scripts/hpo.py \
    --hpo-config configs/hpo/ooh_dqn.yaml \
    --out runs/hpo/ooh_dqn_v1
```

Templates ship for `ooh_dqn`, `ooh_a2c`, `oxides_sinter_dqn`, `oxides_sinter_a2c`. Copy + edit the `base_config:` and `search_space:` keys for any other scenario.

### What it does

1. **Stage 1 (cheap screen)**: runs `n_trials` trials (default 30) at reduced training budget (default 25%); each trial averages over `seeds_per_trial` independent seeds. The objective is the top-K mean of `reward` in `generated.csv` (default K=10). `num_gen_eps` is **not** scaled — metric variance is ~1/√n_gen, and shrinking generation makes scores noisier without much cost saving.
2. **Stage 2 (confirm)**: re-runs the top `n_top_to_confirm` trials (default 3) at full base-config budget with `seeds_per_trial_stage2` seeds (default 3).
3. **Final report**: `<out>/final_report.md` shows the rank-1 config + a copy-paste reproduction command.

### State / resume

- Optuna SQLite study at `<out>/study.db`. Relaunching against the same `--out` resumes from the last completed trial.
- Stage 1 → 2 progression tracked via `<out>/state.json`.
- `--skip-stage2` ends the run after stage 1 (useful while iterating on the search space).
- Crashed subprocess trials are marked `FAIL` and excluded from the surrogate — TPE handles `FAIL` gracefully (don't return `-inf`; that poisons the model).

### Parallelism

`stage1.n_parallel_trials > 1` enables Optuna's threading-mode `n_jobs`. For multi-GPU, pass `--gpu-ids 0 1 2 3` — the driver round-robins `CUDA_VISIBLE_DEVICES` across the trial workers. Alternative: run one driver per GPU with the same `study_name` (SQLite handles concurrent writes).

### Seeds and reproducibility

Each "seed" of a trial passes three decorrelated integers to `run_experiment.py`: `--train-seed i`, `--dp-seed i+10_000`, `--gen-seed i+20_000`. This decouples the training RNG, predictor random-config sampling, and generation sampling, giving a cleaner variance signal than passing the same int three times.

### Analysis outputs

After stage 1 finishes the driver writes `<out>/analysis/` (best-effort, all failures are soft warnings):

- `param_importances.csv` — fANOVA-style importance score per hyperparameter (higher = matters more).
- `param_importances.png` — bar chart of the above.
- `optimization_history.png` — score per trial; flat line = the search isn't improving (widen the space or raise n_trials).
- `parallel_coordinate.png` — trial trajectories through the search space; useful for spotting clusters.
- `slice.png` — per-param score scatter; the clearest way to see *which* knob is driving the score.

Pass `--no-plots` to skip PNG generation (CSV still written). Requires `matplotlib`; importance/plots silently skip with a warning if you have <2 completed trials or the score surface is flat (e.g. constant predictor).

For deeper analysis, the SQLite study is just `optuna.load_study(study_name=..., storage="sqlite:///<out>/study.db")` away — every Optuna analysis API works against it.
