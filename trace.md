# Trace: oxyhydroxide-catalyst-gen

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
