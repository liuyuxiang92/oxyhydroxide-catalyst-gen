# Trace: oxyhydroxide-catalyst-gen

## 2026-05-28 — QRegressor hidden_dim mismatch (reproducibility root cause)

### EARS — Progress (2026-05-28 12:06)
<!-- concepts: reinforcement-learning, dqn, reproducibility -->
Found root cause of classical-dqn vs general-framework output divergence: `QRegressor` default `hidden_dim` is 256 in `feat/classical-dqn` but 128 in `general-framework`. `train_dqn_online` never received `hidden_dim` from the YAML, so it always built a smaller network. Different parameter count → different PyTorch random weight initialization with same seed → different Q-values from episode 1 → completely different training trajectories. Fix: add `dqn_hidden_dim: 256` to `ooh_dqn.yaml` and pass it through `run_experiment.py` → `train_dqn_online`.

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

### EARS — Progress (2026-05-29 11:59)
<!-- concepts: DQN resume on general-framework, predictor cache wiring -->
- Now on general-framework branch (stashed classical-dqn changes). Extended `train_dqn_online` with a `resume_state` kwarg that bypasses warmup; checkpoint dict now includes buffer + buffer_size + dp_cache (sourced from `checkpoint_cfg["dp_cache"]`).
- Discovery: on general-framework the DP cache lives inside `predictor._cache` (OOH predictor) rather than as a free-standing dict. Wired via `getattr(predictor, "_cache", None)` so other predictors (HEA, perovskite, dummy) gracefully skip the cache snapshot.
- Wired run_experiment.py DQN branch to mirror PG resume pattern: load scaler, build qnet/target/optimizer/buffer, look for checkpoint.pt (type="dqn" with "buffer"), fall back to qnet.pt with reset.

### EARS — Progress (2026-05-29 14:13)
<!-- concepts: dead-code audit, subagent false positives, verification before deletion -->
- Audit pass on both branches. Subagent reports were unreliable — general-framework agent flagged 6 "dead" functions (`_make_loss_fn`, `objective_from_mean_std`, `choose_action`, `_rollout_random_episode`, `_episode_pg_terms`, `_comp_key`) and 1 "unused" module (`env_integer.py`) — only `_comp_key` (the top-level one in training.py:84) is actually unused. Verified each claim with grep before acting.
- Lesson: Explore-style subagents are good at locating code but not at proving non-existence of callers. Always re-verify their "dead code" claims directly.
- Concrete cleanup (in progress): general-framework removes `_comp_key` from training.py and dead YAML keys (`name`, `description`, `num_random_eps`, `dqn_epochs`, `structure_mode`). classical-dqn removes two unused imports (`Sequence`, `feature_calculators`).

### EARS — Progress (2026-05-29 14:26)
<!-- concepts: cross-branch debris, egg-info hygiene, gitignore -->
- On `feat/classical-dqn`, found `src/rl_matdesign/` (bytecode-only — no `.py` tracked on this branch) and `src/rl_matdesign.egg-info/` left behind from a prior `general-framework` checkout + `pip install -e .`. Both untracked. Deleted.
- Root cause: switching branches removes tracked `.py` files but `__pycache__/*.pyc` and `*.egg-info/` from `pip install -e .` persist as untracked debris and confuse `git status`.
- Added `*.egg-info/` to `.gitignore` to suppress future regeneration noise. Other dirs (`abcde_ooh/__pycache__`, `data/`, `rf_models/`, `models/sinter_calcine/`) are live or intentional.

### EARS — Progress (2026-05-29 16:30)
<!-- concepts: yaml-first refactor, flag standardization, back-compat aliases -->
- Context: Executing approved plan `feat/yaml-first-systems` — section F (flag-naming standardization) first.
- Goal: Add prefix to DQN-only (`dqn_*`) and PG-only (`pg_*`) YAML keys; keep old names as deprecated aliases.
- Status: Not stuck. Recent edits to `run_experiment.py` are sequential parts of the same plan step (alias map → load_config wiring → call-site rename). Continuing.

### EARS — Progress (2026-05-29 16:33)
<!-- concepts: yaml-first refactor, training.py kwarg rename, ε-anneal schedule -->
- Renaming `train_dqn_online` kwargs to prefixed forms: `buffer_size → dqn_buffer_size`, `num_train_eps → dqn_num_train_eps`, `grad_steps_per_ep → dqn_grad_steps_per_ep`, `target_update_freq → dqn_target_update_freq`, `eps_anneal_eps → dqn_eps_anneal_eps`, `eps_min → dqn_eps_min`.
- Note: the saved checkpoint dict key `"buffer_size"` at training.py:428 will be renamed to `"dqn_buffer_size"`. Need to also update the read path at run_experiment.py:415 with a fallback so old checkpoints still resume.
- Next: rename remaining call sites in training.py (lines 395, 400, 428), then PG signature, then run_experiment.py call sites, then YAML configs.

### EARS — Progress (2026-05-29 16:46)
<!-- concepts: registry dispatch, DPStructurePredictor, dp_eval extractor -->
- Task #16 (flag rename + back-compat aliases) DONE — smoke-tested: all 8 configs load with zero deprecation warnings; old keys still work with warning.
- Task #17 (predictor/constraint registry with FQN dispatch) DONE — `src/rl_matdesign/registry.py` written; `scripts/run_experiment.py` `build_predictor`/`build_constraint_filter` collapsed from ~120 LOC to thin 2-line wrappers. Smoke-tested: built-ins resolve; FQN dispatch works; unknown names give helpful errors listing built-ins.
- Now on Task #18: just wrote `src/rl_matdesign/utils/dp_eval.py` with shared helpers (`load_ase_calculators`, `eval_energy_ase`, `pick_scalar`). Next: write `predictors/dp_structure.py` (cfg-dict constructor), then collapse hea.py and perovskite.py to subclasses, then wire `output_index` through OOH `DPConfig`.
- Discovery: HEA and perovskite predict() bodies are byte-identical except `site_symbol` default. Confirmed by reading both. The "merge into one parameterized class" is genuinely a no-op for behavior.

### EARS — Progress (2026-05-29 16:50)
<!-- concepts: DPStructurePredictor merger, output_index plumbing, cache key invariance -->
- Multiple registry.py edits are intentional sequential steps of Task #18: (1) swap _make_hea/_make_perovskite to cfg-dict constructors, (2) add _make_dp_structure, (3) register "dp_structure" key, (4) thread output_index through OOH factory. All sequential, not thrashing.
- DPStructurePredictor written; HEA/perovskite collapsed to 5-line subclasses; OOH _eval_models_on_prepared_inputs now uses shared pick_scalar() helper.
- Decision: did NOT include output_index in OOH cache key — that would break on-disk dp_cache from existing checkpoints. Added docstring caveat instead. Acceptable trade-off since output_index changes are rare.
- Next: smoke-test registry resolves dp_structure / hea / perovskite cleanly, then move to Task #19 (env per-element bounds).

### EARS — Progress (2026-05-29 16:58)
<!-- concepts: env per-element bounds, total_units parameterization, latent bug -->
- Task #19 in progress: extended `CompositionEnv` with `element_bounds` dict + `episode_style="fixed_order_amount"` mode. Default behavior byte-identical when both are omitted (the "for_element_idx is None" branch of `_allowed_fraction_units_now` is the unchanged original).
- DISCOVERED LATENT BUG: `_format_fraction(units) -> f"{units/20:.2f}"` and `_fractions_to_units(fractions) -> int(round(val*20))` both hardcoded 20. Worked fine for OOH (total_units=20) but would silently produce wrong fractions for any env with total_units != 20 (e.g. Ti alloy needs total_units=100, step=1). Fixed by parameterizing both helpers with `total_units` and threading from self._total_units at all 3 call sites.
- Validation in __init__: element_bounds with element_then_amount → NotImplementedError; sum(min) <= total <= sum(max) check; per-element lo<=hi; unknown keys warned.
- step() now also validates per-element bounds AND that elem matches cation_set[counter] in fixed_order_amount mode.
- Next: smoke-test env (OOH default still works byte-for-byte; Ti-alloy-style config completes a valid episode), then move to Task #20 (phase_pattern).

### EARS — Progress (2026-05-29 17:08)
<!-- concepts: phase_pattern declarative filter, Tier-1 YAML example, scope choices -->
- Task #20 (declarative phase_pattern constraint) DONE. PhasePatternFilter supports: required (per-element bounds), forbidden, primary_sum, dopant_sum (with ALL_OTHERS sentinel), ratios. Multi-pattern OR-of-ANDs. Registered as "phase_pattern" in CONSTRAINTS.
- Scope choice: action filter is THOROUGH at terminal step only (steps_left==0) — at intermediate steps it passes actions through. Reasoning: intermediate-step soundness would require per-subset interval arithmetic (complex), and unmatched trajectories simply yield post-hoc-filtered terminals. Documented in module docstring. The user explicitly accepted that the OOH-style cross-element inequalities (Fe<=max(Ni,Co)) keep their dedicated Python filter.
- Smoke-tested all rule kinds + multi-pattern OR + terminal-step action filter. All pass.
- Task #21 in progress: wrote `configs/ti_alloy.yaml` for the user's 11-element Ti alloy case (45-90 Ti, 0-25 Al, …, 0-5 Ta; sum=100, step=1). Uses `predictor: dp_structure`, `episode_style: fixed_order_amount`, `total_units: 100`, `element_bounds`. Zero Python needed.
- Next: env-loading smoke test against the new ti_alloy.yaml (no DP eval — just confirm env constructs and produces valid actions), then Task #22 verification suite.

### EARS — Session End (2026-05-29 19:20)
<!-- concepts: yaml-first refactor, registry plug-in pattern, ff-merge to general-framework -->
- Accomplished: YAML-first refactor (`feat/yaml-first-systems` @ fc8a186) merged fast-forward into `general-framework`. Tagged `v2.0-yaml-first` locally. Confirmed OOH bit-for-bit parity with `v1.0-parity-with-classical` (user verified for both DQN and REINFORCE). Plan section A in progress.
- Next session: (1) push `general-framework` + both tags (`v1.0-parity-with-classical`, `v2.0-yaml-first`) to origin to finish Phase A; (2) Phase B delete `src/abcde_ooh/{featurization,model}.py` (zero-import dead code); (3) Phase C README "Custom material systems" section; (4) Phase D pytest suite under `tests/`; (5) Phase E real-DP-model Ti-alloy run.
- Open issues: branch rename `general-framework → main` deferred per plan G.2 (needs audit of main's 10 unique commits). `abcde_ooh/env.py` and `encoding.py` still imported by `summarize_replay_buffer.py` and `constraints/phase_sampler.py` — deeper unification deferred.

### EARS — Progress (2026-05-29 19:24)
<!-- concepts: README tier-based docs, flag-naming convention, registry surface reference -->
- Phase A DONE: pushed `general-framework` + tags `v1.0-parity-with-classical` and `v2.0-yaml-first` to origin.
- Phase B DONE (commit 1b3ac5f): removed `src/abcde_ooh/{featurization,model}.py` (zero imports). Smoke a2c run passed.
- Phase C in progress: 5 sequential edits to README.md — (1) added ti_alloy.yaml and ooh_dqn.yaml to configs table, (2) Quickstart prose `entropy_coef → pg_entropy_coef`, (3) REINFORCE YAML example renamed to prefixed names, (4) DQN YAML example renamed to prefixed names, (5) replaced 6-line "Adding a new material system" stub with ~150-line tier-based section (Tier 1/2/3, plug-in surface reference, per-element bounds, flag-naming + alias table). Not thrashing; this is one section rewrite split across 5 atomic edits.
- Next: verify README renders, commit Phase C, then Phase D (pytest), then Phase E (real Ti DP run).

### EARS — Progress (2026-05-29 19:34)
<!-- concepts: ti_alloy_user.yaml, multi-output DP indexing, GPU handoff -->
- Phase D DONE (commit 85798d8): 56 pytest tests, all green. CI-friendly (no DeepMD/ASE needed).
- Phase E adapted to GPU handoff: user has 5 DP models + POSCAR on a remote GPU machine, will run themselves. Wrote `configs/ti_alloy_user.yaml` as a fill-in-the-blanks template with REQUIRED USER EDITS section for base_poscar/site_symbol/dp_models paths.
- Decision: multi-output DP (4 properties per structure) handled via the `output_index` YAML key (default 0). User can pick any of {0,1,2,3} by editing the YAML. Alternative `output_aggregator: mean|max` documented inline. No code change needed — `pick_scalar` already supports both modes.
- Next: validate ti_alloy_user.yaml loads cleanly through `load_config`, smoke-run with `predictor: dummy` override to confirm env builds and rollouts work; then commit + push.

### EARS — Session Start (2026-06-01 11:08)
<!-- concepts: yaml-config-design, dp-predictor-knobs, framework-onboarding -->
- Task: Answer follow-up questions about the YAML-first framework — branch to use, YAML-vs-CLI split, ti_alloy.yaml vs ti_alloy_user.yaml, and the meaning of `energy_per_atom`.
- Why: User is preparing to launch the Ti-alloy GPU run and is internalizing the new config schema; clear answers reduce misconfiguration risk on the remote machine.

### EARS — Progress (2026-06-01 11:27)
<!-- concepts: deepmd-multitask-models, dp_structure-head-support, ti-alloy-gpu-handoff -->
- Discovery: User's Ti-alloy DP checkpoints are **multi-task** models with heads ['Omat24', 'property']. DeepMD's `DPCalculator(model=p)` asserts `head is not None` for multi-task checkpoints → AssertionError on first reward call during PG warmup.
- Root cause: `load_ase_calculators` in src/rl_matdesign/utils/dp_eval.py hard-coded `DPCalculator(model=p)` with no head argument. `DPStructurePredictor` had no cfg key for it either.
- Fix in progress: (1) added `head: Optional[str] = None` kwarg to `load_ase_calculators`, only passes through when set so single-task models still work. Next: wire `dp_head` config key through `DPStructurePredictor`, document in ti_alloy_user.yaml.
- Caveat to flag to user: `dp_structure` uses ASE `atoms.get_potential_energy()` → scalar energy only. Works with energy-style head (`Omat24`). The 4-property `property` head returns a vector and would need the DeepProperty path (the one OOH uses), NOT this fix. So `output_index` in ti_alloy_user.yaml is currently a no-op for `dp_structure` — only meaningful for the OOH-style predictor.

### EARS — Progress (2026-06-01 11:44)
<!-- concepts: dp_property-predictor, multi-head-DP-models, vector-output-reduction -->
- User confirmed Ti-alloy DP model: use the `property` head, which returns a 4-component vector per structure (not energy). The `dp_structure` ASE path can't read that — ASE only surfaces scalar potential energy.
- Building new `DPPropertyPredictor` (src/rl_matdesign/predictors/dp_property.py) that mirrors `dp_structure` but uses `deepmd.pt.infer.deep_eval.DeepProperty` (same backend the OOH overpotential predictor uses).
- Key design decisions: (1) `dp_head` defaults to "property" (vs `dp_structure`'s default of unset); (2) `output_index` / `output_aggregator` now do real work here (they were no-ops in `dp_structure`); (3) added `maximize: bool` cfg knob to flip optimization direction (default False = lower property is better, matches formation-energy convention); (4) derive `elem_to_type` from `dp.get_type_map()` rather than assuming periodic-table order like the OOH predictor does — more robust if the Ti checkpoint uses a custom type ordering.
- Next: register `dp_property` short name in registry.py, update ti_alloy_user.yaml to switch from `dp_structure` to `dp_property`, run pytest to ensure nothing else broke.

### EARS — Progress (2026-06-01 16:33)
<!-- concepts: multi-objective-rl, predictor-registry, deepmd-multi-task -->
Approved a multi-objective design via composite predictor (plan: `~/.claude/plans/harmonic-wiggling-storm.md`). Three design decisions worth capturing:
- **Cross-backend by construction.** `CompositePredictor` wraps N children, each resolved via the existing registry — so one objective can come from `dp_structure` (ASE/energy head) while another comes from `dp_property` (vector head). No new code path inside the inner predictors.
- **Per-property std stays in native units.** Reward = `Σ w_k · dir_k · v_k / scale_k` where `v_k = objective_from_mean_std(m_k, s_k, ...)` is folded *per child with the child's own std*, then weighted-summed. No joint `dp_std` is computed (mixing eV/atom with GPa is meaningless). Composite emits `obj_<name>_mean` / `obj_<name>_std` columns instead of `dp_std` / `dp_mean_minus_std` in `generated.csv`.
- **Deterministic child seeding without a new knob.** Composite seed → child seed = `composite_seed + child_index`. Order of `objectives:` in YAML fixes child indices, so `--dp-seed 321 --train-seed 123 --gen-seed 213` reproduces byte-for-byte. No per-child `seed:` YAML knob.
Started implementation: refactored `DPPropertyPredictor.predict()` to delegate to a new public `raw_mean_std(composition) -> (mean, std)` method (behavior-preserving). Next: same refactor on `DPStructurePredictor`, then write `predictors/composite.py`.

### EARS — Progress (2026-06-01 17:02)
<!-- concepts: multi-objective-rl, exploit-vs-explore, uncertainty-penalty -->
**Bug in composite predictor as shipped in `d7bb9aa`.** User asked what `objective: mean_minus_kstd` does given that `direction` is per-child, which led me to spot that the formula `reward += w * dir * objective_from_mean_std(m, s, ...) / scale` multiplies `dir` through BOTH the mean and std terms. For `mean_minus_kstd` that gives `dir*(m - k*s) = dir*m - dir*k*s`, so the std term **flips sign with direction** — exploit-min objectives accidentally become explore-min (std bonus instead of penalty). Maximize side stayed correct.
Convention from single predictors (`DPStructurePredictor.predict`, `DPPropertyPredictor.predict`): apply sign flip to mean BEFORE the objective fold — `objective_from_mean_std(sign * mean, std, ...)` — so `mean_minus_kstd` is always `sign*m - k*s` regardless of direction. Fixed composite to match: `v = objective_from_mean_std(d * m, s, ...)` then `reward += w * v / sc`.
Net algebra change only affects rows where `objective ∈ {mean_minus_kstd, mean_plus_kstd}` AND `direction: min` AND `std > 0`. `objective: mean` and `predict_raw` paths are bit-identical. One unit test (`test_weighted_sum_math_objective_mean_minus_kstd`) had locked in the wrong expected value — needs updating to `-1.675` (was `-1.575`).
Lesson worth remembering: when wrapping an `f(x, y)` that mixes sign-bearing terms, multiplying the wrapper's output by `dir` is NOT equivalent to passing `dir * x` into the wrapper. The std term has no opinion on direction.

### EARS — Progress (2026-06-01 17:23)
<!-- concepts: schema-design, multi-objective-rl -->
Following the std-sign discussion, user asked whether a composite-level `objective:` makes sense when `direction`/`weight`/`scale` are already per-child. Answer: no — moving it per-child unlocks mixed exploit/explore (e.g. exploit a well-validated energy head + explore an under-trained property head in the same run). `k` stays global as the single uncertainty knob; per-child `k` would over-parameterize and can be absorbed into `scale_k` when really needed. No backward-compat burden — composite predictor landed earlier today, no external consumers. Implementing now: drop `self.objective`, store per-child, raise on legacy top-level key with migration hint.

### EARS — Progress (2026-06-02 17:46)
<!-- concepts: plugin-architecture, documentation -->
Walkthrough session with user on the general-framework branch: explained the env layer (CompositionEnv fractional grid + sum-to-1; IntegerRatioEnv integer digits + repeats allowed), the predictor protocol (`predict(comp) -> (mean,std)` Protocol + registry FQN dispatch), and the constraint filter contract (`filter_actions` per-step pruning + "return original if empty" safety fallback). Two user-facing clarifications worth keeping: (1) `IntegerRatioEnv` has zero element-identity rules — "step N = O" is enforced by `LastStepElementFilter`, not by the env, and the filter is a *prospective action mask* (O is removed from the available choices pre-last-step), not a post-hoc reorder; (2) `SMACTChargeFilter` is anion-agnostic (any `[{symbol, charge, stoich}, …]`) and `n_components`-agnostic — only triggers at `steps_left == 0`. Saved a markdown summary to `general_framework_summary.md` at repo root for the user to convert to PDF (pandoc + weasyprint both present). PDF conversion step not yet executed.

### EARS — Progress (2026-06-03 10:56)
<!-- concepts: plugin-architecture, constraint-composition -->
User asked whether two constraint filters could be used together. Current framework has a single `env.phase_filter` slot + scalar `constraint_filter:` YAML key, so out of the box: only one. Implementing a `ChainConstraintFilter` (new `src/rl_matdesign/constraints/chain.py`) that takes a list of child filter configs, resolves each through the same `resolve_constraint` dispatcher (so built-in short names and FQN plug-ins both work, including nested chains), and applies them in list order. Each child must carry its own `constraint_filter:` key + kwargs in a self-contained sub-dict. Safety: tracks pre-chain input and returns it if any intermediate step or the final output is empty — defense-in-depth over each child's own "if empty return original" fallback, which protects against user filters that don't honor that convention. Ordering matters for performance only (cheap identity filters first, expensive SMACT-like screens second), not correctness. Registering as built-in short name `chain` so YAML can write `constraint_filter: chain`. Registry edit still pending after this trace nudge.

### EARS — Progress (2026-06-03 11:06)
<!-- concepts: testing, constraint-composition -->
Following the `ChainConstraintFilter` implementation + sinter/calcine config edits, added `tests/test_chain.py` with 11 focused tests covering: registry short-name dispatch, two-child composition over real `LastStepElementFilter` instances (verifies (O, nonzero-digit) survival at the final step), explicit ordering proof via stub children where the second observes the first's output, two safety-fallback paths (any child emptying the action list → return pre-chain input), and four input-validation errors (empty `filters`, missing `filters`, non-list `filters`, non-dict child, child missing `constraint_filter:` key). Final test rounds-trips both shipped oxide YAMLs through the registry and asserts the resolved child types are `[LastStepElementFilter, SMACTChargeFilter]` — gated with `importorskip("smact")` so it skips cleanly on machines without the SMACT package. Style mirrors `test_phase_pattern.py` (in-function imports, local `_oh` helper, direct `filter_actions` calls with kwargs matching the env contract). Construction-time stubs use `__new__` + manual `.children` assignment to bypass the registry round-trip and isolate the safety-fallback / ordering behaviors. Tests not yet executed.

## 2026-06-03 — Order-invariance contract + DQN trajectory-permutation augmentation

### EARS — Progress (2026-06-03 19:18)
<!-- concepts: reinforcement-learning, dqn, order-invariance, symmetry-augmentation -->
Plan approved (`~/.claude/plans/synchronous-petting-liskov.md`): pin the existing
order-invariance property of the framework via test + docs + DQN augmentation
knob + diagnostic script.

Discovery during planning: order-invariance is already structurally enforced —
state = `(Magpie features of partial Composition, step counter)`; `featurize_formula`
routes through `pymatgen.Composition` so element order is normalized away. No
correctness fix needed. The plan adds *guardrails* (test) and a *sample-efficiency*
knob (`dqn_augment_permutations: K` for DQN only — PG/A2C is on-policy so
trajectory permutation breaks the log-prob attribution).

Started Component 1: wrote `tests/test_order_invariance.py` covering featurizer
(fractional + integer), CompositionEnv state-features-match-when-multisets-match,
IntegerRatioEnv same, predictor contract via example + anti-pattern + real
SinterCalcineRF check. Test asserts `state_material_features` are equal across
permuted episodes at every step k where the partial multiset matches — NOT at
indices where they don't (e.g., step 1 of Fe-first vs Ni-first paths differs by
design).

Key design choice in the augmentation helper (Component 2): re-drive the env via
`env.step()` with a temporarily-swapped `env.reward_fn` rather than manually
rebuilding state features. Keeps featurization, constraint filtering, and
`allowed_actions` consistent with the production code path; only cost is the
predictor swap and a deepcopy/initialize of env state. Validity rules detect
`LastStepElementFilter.reserve_for_last=True` (including via `ChainConstraintFilter`)
and keep last position fixed; `fixed_order_amount` short-circuits.

Within-episode row deduplication added after walking through the duplicate-row
pattern at late steps (when last position is fixed, the terminal row is identical
across all permutations). Dedup is scoped to the current episode's augmentation
pass — cross-episode coincidences are legitimate independent observations and
bootstrapping needs them.

### EARS — Progress (2026-06-03 19:45)
<!-- concepts: reinforcement-learning, dqn, symmetry-augmentation -->
Components 1 + 3 done (invariance test green; CLAUDE.md "Order invariance"
section added). Augmentation helper (`_augment_episode_in_buffer` in
training.py) wired up: `_detect_last_position_pin` walks ChainConstraintFilter
children to find LastStepElementFilter.reserve_for_last and pins position N-1;
fingerprint-based dedup uses np.round(s_mat_raw, 6) + s_step + a_elem_idx +
a_comp_val + done; permutations are enumerated then random.shuffle()d and
top-K taken. Re-drive wraps env.reward_fn = lambda _f: original_R inside
try/finally so swap reverts even on exception. Threaded through CLI
(`--dqn-augment-permutations K`) + YAML (`dqn_augment_permutations: K`)
into `train_dqn_online(augment_permutations=K)`. Module-level
_AUG_WARNED_* flags rate-limit no-op / capping warnings to once per run.
Tolerance fix on test_order_invariance.py: assert_allclose with rtol=1e-10,
atol=1e-12 instead of bit-exact assert_array_equal — matminer aggregations
have benign FP round-off from operand reorder; the architectural invariance
is "equal up to FP precision" which is what we now assert.
