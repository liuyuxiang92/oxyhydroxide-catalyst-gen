# Trace: oxyhydroxide-catalyst-gen

## 2026-05-28 — greedy argmax fallback + DQN algorithm audit

### EARS — Progress (2026-05-28 10:26)
<!-- concepts: reinforcement-learning, dqn, reproducibility -->
Full audit of DQN execution path between `feat/classical-dqn` and `general-framework`. Root cause of `_choose_action_dqn` / `choose_action` raising `ValueError`: the "greedy" fallback listed in every docstring was never implemented. This crashed DQN training whenever the greedy branch of ε-greedy was selected (which happens ~6% of episodes by ep=1000 with eps_anneal_eps=15000). Fixed on both branches: replace `raise ValueError` with `argmax(Q)` (DQN) and `argmax(logits)` (PG generation).

Key insight: generation is unaffected — CLI default `--gen-temperature 1.0` fires the Boltzmann branch before reaching the fallback. The argmax fallback only matters for DQN training rollouts (called with no gen args).

Audit confirmed all other aspects match: seeding order, element features, buffer fill, training loop, epsilon schedule, gradient steps, target net copy, generation seeding, generation RNG order, and CSV sort order (ascending dp_mean_minus_std ≡ descending reward since reward = -dp_mean_minus_std).

## 2026-05-28 — general-framework DQN algorithm bug fix

### EARS — Progress (2026-05-28 09:45)
<!-- concepts: reinforcement-learning, dqn, reproducibility -->
Found a critical bug in `general-framework`'s classical DQN implementation (`src/rl_matdesign/training.py:train_dqn_online`).

**Root cause**: `choose_action()` had no pure greedy/argmax fallback — it raised `ValueError` if all three strategies (epsilon, temperature, top_frac) were zero. The training loop worked around this by passing `gen_temperature=1.0`, which accidentally triggered **Boltzmann sampling** during ε-greedy training rollouts instead of pure greedy argmax. This made the greedy branch of DQN training stochastic, breaking reproducibility vs `feat/classical-dqn`.

**Fix (two-part)**:
1. `choose_action()`: replaced the `raise ValueError` fallback with `return allowed_actions[int(torch.argmax(q).item())]` — pure greedy argmax when no strategy is specified.
2. `train_dqn_online()` training loop: removed `gen_temperature=1.0` argument so the greedy branch now correctly does argmax.

**Architecture clarification from user**: `general-framework` should have NO `run_ABCDEOOH_experiment.py` — only `run_experiment.py` is the intended interface. The DQN/PG algorithms live in `src/rl_matdesign/training.py` and must match `feat/classical-dqn`'s inline implementations exactly (only difference: general-framework works for any system, classical-dqn is OOH-specific). Earlier mistakenly overwrote `run_ABCDEOOH_experiment.py` from classical-dqn — reverted.

## 2026-05-27 — general-framework vs classical-dqn reproducibility audit

### EARS — Progress (2026-05-27 14:15)
<!-- concepts: reproducibility, reinforcement-learning, ooh-catalyst -->
Audited all divergences between `general-framework` and `feat/classical-dqn` that break reproducibility when running the same seeds. Found and fixed 9 bugs across 5 files:

**Training bugs (affect which structures are generated):**
1. `ooh.yaml` `objective: mean_minus_kstd` → `mean_plus_kstd` — std sign was flipped vs classical-dqn's `-(mean-std)`
2. `ooh.yaml` `pg_num_iters: 1000`/`num_gen_eps: 2000` → `50`/`50` — YAML values silently override CLI since run_experiment.py has no those flags
3. `run_experiment.py:285-287` probe episode burned 5 RNG draws before warmup — removed probe, infer state_dim from scaler after warmup
4. `dp_predictor.py:450` `hash()` (PYTHONHASHSEED-randomized) → `hashlib.md5` for stable composition seed
5. `constraint_filter: null` → `ooh_phase` + `target_phases: [any]` — classical-dqn uses PhaseActionFilter even with `--target-phase any`

**Generation bugs (affect CSV output values):**
6. `generate_candidates` applied objective twice to reward; fixed to use `env.path[-1].reward` directly
7. No shared cache between training and generation; fixed via internal cache in `OOHCatalystPredictor.predict_raw()`
8. Double DeepMD call per episode (env.step + generate_candidates); fixed via same cache
9. Missing `primary_ok`/`primary_label` columns; added via `predictor.check_phase()` hook

Key architectural decision: added `predict_raw()` and `check_phase()` to `OOHCatalystPredictor` to expose raw overpotential values and phase labels, keeping the generic `PropertyPredictor.predict()` interface intact for other systems.

### EARS — Stuck (2026-05-27 14:17)
<!-- concepts: reproducibility, reinforcement-learning, ooh-catalyst -->
Not stuck — applying a large multi-file patch (9 bugs across 5 files) sequentially. run_experiment.py requires 4 edits because the changes are in separate non-adjacent sections: imports, build_constraint_filter function, env creation block, and the fresh-PG state_dim block. Each edit is a distinct logical fix; no thrashing.

## 2026-05-27 — environment-gpu.yml fix + editable install

### EARS — Progress (2026-05-27 14:05)
<!-- concepts: python-packaging, conda, pytorch-cuda -->
Fixed `environment-gpu.yml` for HPC clusters. Two root causes of the conda solve failure:
1. `pytorch::pytorch` (pytorch channel) requires MKL BLAS — conflicts with `blas=*=openblas` pin.
2. HPC cluster provides CUDA system-wide; conda `pytorch-cuda=12.1` needs `libcublas`/`cuda-cudart` as conda packages, which don't exist on this cluster.
Fix: remove pytorch from conda section entirely, remove `pytorch` channel, install via pip wheel (`torch --index-url https://download.pytorch.org/whl/cu124`). Cluster CUDA confirmed 12.4. Same pattern applied on both branches alongside the editable install (`-e .`) migration.

## 2026-05-26 — general-framework seeding fix

### EARS — Session Start (2026-05-26 15:35)
<!-- concepts: reinforcement-learning, structure-generation, reproducibility -->
- Task: Wire `--seed` through `build_predictor` → predictors → `substitute_sites` → `_sqs_config` so both `random` and `sqs` structure modes are reproducible
- Why: `rng_seed` was accepted by `HEAPropertyPredictor`/`PerovskitePropertyPredictor` but never passed from `build_predictor`; SQS mode also ignored the `rng` entirely

### EARS — Progress (2026-05-26 15:40)
<!-- concepts: structure-generation, reproducibility, sqsgenerator -->
Wiring `--seed` through the structure generation stack. Two gaps found:
1. `substitute_sites()` passed `rng` to `_random_configs` but dropped it for `_sqs_config` — fixed by deriving `sqs_seed = int(rng.integers(0, 2**31))` and passing to `_sqs_config`
2. `_sqs_config` had no seed param — added `rng_seed`, sets `np.random.seed(rng_seed)` as numpy fallback, then tries `sqsgenerator.run_sqs_iterations(..., random_seed=rng_seed)` with `TypeError` fallback for older sqsgenerator versions that don't accept it
3. `build_predictor()` never forwarded `rng_seed` to HEA/perovskite predictors — fixing next by adding `seed` param and wiring to call site in `run_experiment.py`

### EARS — Stuck (2026-05-26 16:20)
<!-- concepts: reproducibility, seeding, argparse -->
Not stuck — renaming --seed to --dp-seed and aligning seeding behaviour to match feat/classical-dqn exactly. Multiple edits to run_experiment.py are intentional: (1) rename argparse flag, (2) update main() seed references and startup call, (3) remove pre-training set_global_seed calls, (4) remaining generation seed fixes.

### EARS — Stuck (2026-05-26 16:07)
<!-- concepts: reproducibility, seeding -->
Not stuck — aligning seeding behaviour across 3 files in one session. Multiple edits to run_experiment.py are intentional: (1) add `import random`, (2) fix startup seed, (3) fix DQN gen seed, (4) fix PG gen seed.

### EARS — Progress (2026-05-26 16:06)
<!-- concepts: reproducibility, seeding, general-framework -->
Aligning `--seed`/`--train-seed`/`--gen-seed` in general-framework to match `--dp-seed`/`--train-seed`/`--gen-seed` behaviour in feat/classical-dqn. Three gaps identified and fixed:
1. `seeding.py`: added `deterministic=False` param to `set_global_seed` — enables cudnn.deterministic + use_deterministic_algorithms when `--train-seed` is explicitly set
2. `run_experiment.py` startup: was `set_global_seed(args.seed)`, changed to `set_global_seed(train_seed, deterministic=(args.train_seed is not None))` — matches classical-dqn which seeds with `train_seed or dp_seed` at startup
3. `run_experiment.py` generation: was `set_global_seed(gen_seed)` (reseeds torch), changed to `np.random.seed + random.seed` only — matches classical-dqn which only reseeds numpy+python random before generation

### EARS — Stuck (2026-05-26 15:54)
<!-- concepts: structure-generation, refactoring -->
Not stuck — removing SQS mode across 5 files in one session. Multiple edits to structure.py are intentional: (1) docstring, (2) function signature, (3) body, (4) drop _sqs_config. Remaining: hea.py, perovskite.py, run_experiment.py, configs.

### EARS — Progress (2026-05-26 15:54)
<!-- concepts: structure-generation, reproducibility -->
User decided to remove SQS mode entirely and keep only random structure generation. Removes sqsgenerator dependency and all seeding complexity around it. Removing from: structure.py (drop `_sqs_config`, `sqs_iterations` param, SQS branch), hea.py + perovskite.py (drop `structure_mode` param, hardcode `mode="random"`), run_experiment.py (drop `structure_mode` from build_predictor calls), configs (drop `structure_mode` key).

### EARS — Progress (2026-05-26 18:20)
<!-- concepts: resume-training, only-generate, skip-generation, general-framework -->
Porting Tier-1 features from classical-dqn to general-framework. Added: --only-generate, --resume-training, --skip-generation, --load-qnet/policy/scaler/value-net flags to run_experiment.py. Full branching logic for DQN (only-generate loads qnet, resume warns + fresh start) and PG (only-generate loads policy, resume loads checkpoint.pt or policy.pt, restores optimizer + visit_counts via checkpoint_cfg). training_log.csv appends when resuming. Also updated train_pg in training.py to restore visit_counts + optimizer states from checkpoint_cfg on resume. Still need: add mode param to RunMetrics.to_csv(), syntax check, commit.

### EARS — Stuck (2026-05-27 09:40)
<!-- concepts: deterministic-algorithms, cuda, general-framework -->
Not stuck — fixing runtime crash: `torch.use_deterministic_algorithms(True)` (triggered by --train-seed) causes DeepMD CUDA matmul to fail on CUDA >= 10.2. Fix: set CUBLAS_WORKSPACE_CONFIG=:4096:8 at startup, matching classical-dqn (line 24 of run_ABCDEOOH_experiment.py). One-line change to run_experiment.py.

### EARS — Stuck (2026-05-26 18:18)
<!-- concepts: resume-training, only-generate, skip-generation, general-framework -->
Not stuck — porting three Tier-1 features from classical-dqn to general-framework in sequence: (1) training.py: restore opt states + visit_counts in train_pg when checkpoint_cfg has resume data, (2) run_experiment.py: add --only-generate, --resume-training, --skip-generation, --load-* flags + full PG/DQN branching logic. Multiple edits to training.py are intentional sequential steps.

### EARS — Stuck (2026-05-26 18:05)
<!-- concepts: checkpoint-saving, general-framework -->
Not stuck — adding checkpoint support requires multiple sequential edits to training.py: (1) _save_checkpoint helper, (2) checkpoint_cfg param on train_dqn_online, (3) checkpoint saving inside DQN loop, (4) checkpoint_cfg param on train_pg, (5) checkpoint saving inside PG loop. Then run_experiment.py needs --save-checkpoint-freq arg and wiring. All planned, not thrashing.

### EARS — Progress (2026-05-26 18:05)
<!-- concepts: checkpoint-saving, general-framework, classical-dqn-parity -->
Porting `--save-checkpoint-freq` from `feat/classical-dqn` to general-framework. Discovered this flag was never ported: classical-dqn has `_save_checkpoint()` (atomic write via tmp+symlink), periodic saves in both `train_pg` (per-iter) and `train_dqn_online` (per-episode). Adding: (1) `_save_checkpoint` helper to `training.py`, (2) `checkpoint_cfg` param to both training functions, (3) `--save-checkpoint-freq` CLI arg to `run_experiment.py`. Training log (`training_log.csv`) was already correct in both branches — written once at end via `RunMetrics.to_csv()`. Also confirmed `feat/classical-dqn` was 3 commits ahead of remote; pushed it.

### EARS — Progress (2026-05-26 17:08)
<!-- concepts: pg-training, general-framework, classical-dqn-parity -->
Discovered that general-framework's `train_pg` was the old per-episode version, not the batch version from feat/classical-dqn. Classical-dqn uses `num_iters × batch_eps` (collect batch_eps episodes, one gradient step per batch). Ported the batch `train_pg` to `src/rl_matdesign/training.py`: added `_episode_pg_terms` helper, replaced `n_episodes` loop with `num_iters × batch_eps` outer loop. Updated `run_experiment.py` to read `pg_num_iters` and `pg_batch_eps` from YAML. Also removed `normalise_returns` (not in classical-dqn) and corrected gamma default to 0.9 (was 0.99). Now updating all YAML configs to replace `pg_train_eps` with `pg_num_iters` + `pg_batch_eps`.

### EARS — Stuck (2026-05-26 16:54)
<!-- concepts: general-framework, hidden-dim, pg-training -->
Not stuck — making multiple edits to run_experiment.py in one session. Removing `hidden_dim` from YAML/config (it's hardcoded at 128 in both classical-dqn branches, never a user flag). Also clarified batch PG vs per-episode PG: user confirmed general-framework should match feat/classical-dqn (per-episode), which it already does.

### EARS — Progress (2026-05-26 16:34)
<!-- concepts: ooh-catalyst, predictor-factory, general-framework -->
Adding OOH catalyst support to general-framework so experiments can run via run_experiment.py + YAML instead of the OOH-specific script. Three changes:
1. Created `src/rl_matdesign/predictors/ooh.py` — `OOHCatalystPredictor` wraps `abcde_ooh.dp_predictor.DeepMDOverpotentialPredictor`. Negates mean overpotential before `objective_from_mean_std` (same sign convention as HEA/perovskite predictors). Exposes `uncertainty` mode ("models"/"configs"/"total").
2. Updated `build_predictor()` in `run_experiment.py` — added `"ooh"` branch reading `base_poscar`, `dp_models`, `ads_height`, `ads_dz`, `geo_opt`, `uncertainty` from YAML cfg.
3. Created `configs/ooh.yaml` — 28-cation set (matching DEFAULT_CATION_SET from abcde_ooh), 16 fractions (0.05–0.80), `anion_formula: "O2H1"`, full DQN-online + PG hyperparameters.

### EARS — Stuck (2026-05-27 11:43)
<!-- concepts: python-packaging, sys.path, editable-install -->
Not stuck — removing `sys.path.insert(0, src/)` from 5 scripts in one session as part of migrating to `pip install -e .` + `pyproject.toml`. Multiple edits to `run_ABCDEOOH_experiment.py` are intentional sequential steps: (1) remove `_REPO_ROOT` + `sys.path.insert` block, (2) remove `# noqa: E402` comments on the imports that followed the hack. Not thrashing.
