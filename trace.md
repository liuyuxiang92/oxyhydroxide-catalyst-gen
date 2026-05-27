# Trace: oxyhydroxide-catalyst-gen

## 2026-05-22 — Classical DQN upgrade

Starting implementation of classical DQN on branch `feat/classical-dqn`. Replacing offline supervised regression (MC returns → MSE) with proper online DQN: target network, TD Bellman targets, replay buffer with `next_allowed_idx`, Magpie element features + scalar fraction encoding.

Key design decisions confirmed with user:
- No Double DQN or gradient clipping in this phase
- `a_elem`: one-hot(28) → Magpie features(~130), separately scaled
- `a_comp`: one-hot(16) → scalar float
- `next_allowed_idx` stored as `List[Tuple[int,int]]` to allow fresh TD target computation
- 5 grad steps per episode (1:1 ratio with env steps), each sampling from full buffer
- Linear epsilon annealing over `--eps-anneal-eps` episodes

### EARS — Progress (2026-05-22 18:28)
<!-- concepts: reinforcement-learning, dqn, replay-buffer -->
Not stuck. Iterative refinements to the classical DQN warmup design:
1. Initially disabled DeepMD during warmup (reward=0) to save cost
2. User correctly identified that reward=0 on terminal steps corrupts TD targets,
   and that bootstrapping propagates the error backward to all non-terminal steps
3. Considered buffer.clear() after warmup — but this wastes collected data
4. Final decision: keep DeepMD enabled during warmup (real rewards). Total DeepMD
   cost is identical; buffer is correctly pre-filled from episode 1 of training.
   This matches standard DQN practice.

### EARS — Stuck (2026-05-22 17:37)
<!-- concepts: reinforcement-learning, dqn, script-refactoring -->
Not stuck — multiple edits to run_ABCDEOOH_experiment.py are intentional: implementing
the classical DQN upgrade in one session (imports, helper functions, argparse flags,
main loop replacement). Each edit targets a distinct section of the 2200-line file.

<!-- concepts: reinforcement-learning, dqn, magpie-features -->

### EARS — Progress (2026-05-25 11:07)
<!-- concepts: reinforcement-learning, dqn, candidate-generation -->
Unifying generation exploration flags across DQN, REINFORCE, A2C. Root cause: DQN generation was pure greedy → only 1 unique candidate. Two existing flags (`--gen-epsilon`, `--stochastic-top-frac`) were dead code. Decision: replace all method-specific generation flags with three unified ones (`--gen-epsilon`, `--gen-top-frac`, `--gen-temperature`) with consistent priority (ε-greedy > Boltzmann > top-k > greedy). Removed `--pg-gen-stochastic` + `--pg-gen-temperature` + `--stochastic-top-frac` + `--online-epsilon`.

### EARS — Stuck (2026-05-25 11:06)
<!-- concepts: reinforcement-learning, dqn, candidate-generation -->
Not stuck — implementing unified generation exploration flags across DQN, REINFORCE, A2C in one session. Edits so far: `_choose_action_dqn` upgraded with epsilon/top-frac/temperature params. Remaining: wire flags into DQN generation call, upgrade `generate_pg`, update both call sites, update argparse.

### EARS — Progress (2026-05-22 21:02)
<!-- concepts: reinforcement-learning, policy-gradient, encoding -->
Implementing three REINFORCE/A2C improvements in one session:
1. Removed `repeats_per_iter` — was a redundant inner loop (default=1, no-op). Flattened
   `train_pg` loop: `for it in range(num_iters)` directly collects batch and steps.
2. Removed `pg_epsilon` — epsilon-greedy is off-policy contamination in an on-policy method.
   REINFORCE/A2C already explore via stochastic softmax sampling.
3. Upgraded `a_elem` one-hot(28) → Magpie features, `a_comp` one-hot(16) → scalar float,
   matching what DQN already uses. `_precompute_elem_features` reused from DQN path.
All three `PolicyNet` instantiations updated to `elem_dim=elem_dim, frac_dim=1`.
Remaining: fix second `generate_pg` call site (post-training generation).

### EARS — Stuck (2026-05-22 21:01)
<!-- concepts: reinforcement-learning, policy-gradient, encoding -->
Not stuck — continuing same three-change PG upgrade in one session. Edits so far:
_rollout_pg_episode, _episode_pg_terms, train_pg (signature+loop), generate_pg, argparse.
Remaining: remove --pg-repeats-per-iter argparse, fix three PolicyNet calls, add
_precompute_elem_features in all PG entry points, fix train_pg/generate_pg call sites.

### EARS — Session Start (2026-05-26 11:04)
<!-- concepts: reinforcement-learning, actor-critic, policy-gradient -->
- Task: Explain how A2C works in detail and compare with general actor-critic
- Why: User wants to understand the A2C algorithm mechanics and how our implementation differs from the textbook version

### EARS — Stuck (2026-05-26 15:28)
<!-- concepts: reinforcement-learning, dqn, argparse -->
Not stuck — renaming three DQN-only argparse flags to add `--dqn-` prefix (`--batch-size` → `--dqn-batch-size`, `--lr` → `--dqn-lr`, `--warmup-eps` → `--dqn-warmup-eps`) plus their three usage sites. Multiple edits to run_ABCDEOOH_experiment.py are expected for this rename.

### EARS — Commit Digest (2026-05-26 14:24)
<!-- concepts: git-branching, repo-organisation -->
Moved all general-framework files (run_experiment.py, src/rl_matdesign/, configs/, scripts/baselines/, scripts/run_seeds.py) off feat/classical-dqn and onto the dedicated general-framework branch. The framework was introduced in one commit (32813b0) so cherry-pick onto general-framework was clean with zero conflicts. feat/classical-dqn is now OOH-only; general-framework owns the config-driven multi-system runner. Any future work on the general framework (new configs, rl_matdesign changes) should happen on general-framework, not here.

### EARS — Progress (2026-05-27 11:49)
<!-- concepts: python-packaging, editable-install, sys-path -->
Migrating `feat/classical-dqn` branch to editable install (`pyproject.toml` + `pip install -e .`). Removing all `sys.path.insert(0, src/)` runtime hacks from three scripts: `run_ABCDEOOH_experiment.py`, `summarize_replay_buffer.py`, `evaluate_formulas_dp.py`. Also adding `- -e .` to both environment files. Same changes were already committed on `general-framework` branch — applying identically here.

### EARS — Stuck (2026-05-27 14:06)
<!-- concepts: git-branching, python-packaging, conda -->
Not stuck — README.md edits across two branches look like thrashing but are intentional. The same `pytorch-cuda=12.1` stale mention needed fixing on both `general-framework` and `feat/classical-dqn` independently (git branch checkout resets files to their branch state). Each edit is a clean targeted fix.

### EARS — Progress (2026-05-27 11:36)
<!-- concepts: environment-management, openmp, documentation -->
Two sessions of work captured here:

1. **OpenMP runtime fix**: The `os.environ.setdefault(OMP_NUM_THREADS=1, ...)` block in both scripts was a band-aid that throttled all threads to 1. Root cause is MKL's `libiomp5` (from pip numpy/sklearn) conflicting with PyTorch's bundled `libiomp5`. Proper fix: `environment.yml` + `environment-gpu.yml` both pin `blas=*=openblas` so numpy/sklearn use OpenBLAS (pthreads-based, loads zero OpenMP runtime). PyTorch's OpenMP then becomes the sole runtime — conflict eliminated at the binary level. Removed the thread-throttle env vars from `run_ABCDEOOH_experiment.py` and `run_experiment.py`; kept only `CUBLAS_WORKSPACE_CONFIG` (unrelated — needed for `torch.use_deterministic_algorithms`).

2. **README rewrite**: Both branches had identical, severely outdated READMEs (wrong flags, broken code blocks, no installation section). Rewrote both: `feat/classical-dqn` README covers installation, all three RL methods with accurate flags, checkpoint/resume, generation diversity flags, DeepMD ensemble, phase constraints, output file table. `general-framework` README (in progress) covers YAML-driven multi-system usage.
