# oxyhydroxide-catalyst-gen — ABCDEOOH Branch

RL-based composition generator for 5-cation oxyhydroxide (ABCDEOOH) catalyst discovery.
A 5-step environment sequentially picks cations and fractions (summing to 1.0) from 28 candidates.
Three RL algorithms are supported: **DQN** (online, target-network), **REINFORCE**, and **A2C**.

---

## Installation

Two conda environments are provided. Both pin `blas=*=openblas` so numpy/sklearn link against
OpenBLAS (pthreads-based) rather than MKL — eliminating the OpenMP runtime conflict with PyTorch
that otherwise causes segfaults on macOS and Linux.

**macOS / CPU (development):**
```bash
conda env create -f environment.yml
conda activate ooh-catalyst
```

**Linux + CUDA 12.x (HPC production):**
```bash
conda env create -f environment-gpu.yml
conda activate ooh-catalyst-gpu
```

DeepMD-kit is installed automatically via the `pip:` section in each environment file.
For a different CUDA version, change `pytorch-cuda=12.1` → `pytorch-cuda=11.8` and
`deepmd-kit[torch,cu12]` → `deepmd-kit[torch,cu11]` in `environment-gpu.yml` before creating
the environment.

---

## Seeds

Three independent seeds give full control over reproducibility:

| Flag | Default | Purpose |
|---|---|---|
| `--dp-seed N` | 0 | DeepMD predictor: controls random alloy configs and adsorbate placement. Acts as fallback for the other two seeds if not set. |
| `--train-seed N` | *(uses dp-seed)* | Training RNG. Also enables GPU deterministic mode (`cudnn.deterministic`, `use_deterministic_algorithms`). |
| `--gen-seed N` | *(uses dp-seed)* | Generation phase only — makes stochastic sampling reproducible independently of training. |

Fix all three for fully reproducible results. `--dp-seed` alone is sufficient for a quick sweep.

---

## REINFORCE (recommended)

```bash
python scripts/run_ABCDEOOH_experiment.py \
    --out runs/reinforce_seed0 \
    --rl-method reinforce \
    --dp-seed 321 --train-seed 123 --gen-seed 213 \
    --pg-warmup-eps 1000 \
    --pg-num-iters 1000 --pg-batch-eps 21 \
    --entropy-coef 0.15 \
    --repeat-penalty-coef 10 --repeat-penalty-shape log \
    --num-gen-eps 2000 --gen-temperature 3.0 \
    --save-checkpoint-freq 50 \
    --dp-poscar POSCAR \
    --dp-model model_1.ckpt.pt \
    --dp-model model_2.ckpt.pt
```

Key PG flags:

| Flag | Default | Meaning |
|---|---|---|
| `--pg-warmup-eps N` | 200 | Random episodes to fit the `StandardScaler` before training. |
| `--pg-num-iters N` | 500 | Outer training iterations. |
| `--pg-batch-eps N` | 15 | Episodes per iteration; one gradient step per batch. |
| `--entropy-coef α` | 0.01 | Entropy bonus weight — prevents premature policy collapse. |
| `--repeat-penalty-coef α` | 0.0 | Penalises revisiting the same composition: `α·log(1+visits)`. |

---

## A2C

```bash
python scripts/run_ABCDEOOH_experiment.py \
    --out runs/a2c_seed0 \
    --rl-method a2c \
    --dp-seed 0 --train-seed 42 --gen-seed 99 \
    --pg-warmup-eps 1000 \
    --pg-num-iters 1000 --pg-batch-eps 21 \
    --entropy-coef 0.15 \
    --num-gen-eps 2000 --gen-temperature 3.0 \
    --dp-poscar POSCAR \
    --dp-model model_1.ckpt.pt
```

A2C trains an additional critic (`value_net.pt`) to reduce gradient variance. Use
`--pg-lr-actor` and `--pg-lr-critic` to tune the two learning rates independently.

---

## DQN

```bash
python scripts/run_ABCDEOOH_experiment.py \
    --out runs/dqn_seed0 \
    --rl-method dqn \
    --dp-seed 0 --train-seed 42 --gen-seed 99 \
    --dqn-warmup-eps 500 \
    --num-train-eps 20000 \
    --buffer-size 50000 \
    --grad-steps-per-ep 5 \
    --target-update-freq 100 \
    --eps-anneal-eps 10000 --eps-min 0.05 \
    --num-gen-eps 2000 \
    --save-checkpoint-freq 200 \
    --dp-poscar POSCAR \
    --dp-model model_1.ckpt.pt
```

Key DQN flags:

| Flag | Default | Meaning |
|---|---|---|
| `--dqn-warmup-eps N` | 500 | Random episodes to populate the initial replay buffer and fit the scaler. |
| `--num-train-eps N` | 20000 | Total online training episodes. |
| `--buffer-size N` | 50000 | Replay buffer capacity (FIFO eviction). |
| `--grad-steps-per-ep N` | 5 | Gradient updates per training episode. |
| `--target-update-freq N` | 100 | Hard-copy Q-net → target-net every N episodes. |
| `--eps-anneal-eps N` | 10000 | Linear ε annealing: reaches `--eps-min` after N episodes. |
| `--dqn-loss` | smoothl1 | Loss function: `mse` or `smoothl1` (Huber). |

---

## Checkpointing and resume

```bash
# 1. Save checkpoints during training (every 50 PG iterations or DQN episodes)
python scripts/run_ABCDEOOH_experiment.py \
    --out runs/reinforce_seed0 --rl-method reinforce \
    --pg-num-iters 1000 --save-checkpoint-freq 50 ...

# 2. Resume from checkpoint after interruption
python scripts/run_ABCDEOOH_experiment.py \
    --out runs/reinforce_seed0 --rl-method reinforce \
    --pg-num-iters 1000 --resume-training ...

# 3. Generate candidates from a saved model without re-training
python scripts/run_ABCDEOOH_experiment.py \
    --out runs/reinforce_seed0 --rl-method reinforce \
    --only-generate --num-gen-eps 5000 --gen-temperature 3.0

# 4. Extend training without generating
python scripts/run_ABCDEOOH_experiment.py \
    --out runs/reinforce_seed0 --rl-method reinforce \
    --pg-num-iters 500 --resume-training --skip-generation
```

Use `--load-policy`, `--load-qnet`, `--load-scaler`, `--load-value-net` to load from a
non-default path.

---

## Generation diversity flags

All three RL methods share the same generation policy flags:

| Flag | Default | Effect |
|---|---|---|
| `--gen-temperature T` | 1.0 | Boltzmann: `softmax(logits / T)`. T > 1 → more diverse; T < 1 → sharper. |
| `--gen-epsilon ε` | 0.0 | ε-greedy: random action with prob ε. Overrides temperature. |
| `--gen-top-frac f` | 0.0 | Uniformly sample from the top-f% of actions by Q/logit value. |

For REINFORCE/A2C, use `--gen-temperature 3.0` to avoid duplicate compositions after training convergence.

---

## DeepMD ensemble

```bash
# Five-model ensemble (recommended)
--dp-model model_1.ckpt.pt \
--dp-model model_2.ckpt.pt \
--dp-model model_3.ckpt.pt \
--dp-model model_4.ckpt.pt \
--dp-model model_5.ckpt.pt

# Per-composition structure randomisation
--dp-n-random-configs 10      # random alloy configs per composition (default: 10)
--dp-ads-height 1.9           # adsorbate height above surface in Å (default: 1.9)
--dp-ads-dz 1.0               # vertical spacing between OOH atoms in Å (default: 1.0)
```

Reward = `-(mean_overpotential - std)` — lower overpotential and higher ensemble disagreement
are both rewarded (LCB-style exploration-exploitation).

---

## Phase constraints

Constrain every generated composition to a valid catalyst phase:

```bash
--target-phase NiFeCo          # Ni+Fe+Co ≥ 75%, Fe not dominant
--target-phase NiFe            # Ni+Fe ≥ 75%, Ni:Fe ≈ 2:1–3:1
--target-phase Ni Co           # Ni-majority OR Co-majority (logical OR)
--target-phase any             # any of the five valid phase types
```

Phase constraints are applied directly to `allowed_actions()` at every step — 100% acceptance rate
at generation time, no wasted DeepMD calls.

---

## Output files

| File | DQN | PG | Description |
|---|---|---|---|
| `std_scaler.bin` | ✓ | ✓ | Fitted `StandardScaler` (joblib) |
| `qnet.pt` | ✓ | — | DQN Q-network weights |
| `policy.pt` | — | ✓ | Actor (PolicyNet) weights |
| `value_net.pt` | — | A2C | Critic (ValueNet) weights |
| `checkpoint.pt` | ✓ | ✓ | Latest mid-training checkpoint (symlink to numbered file) |
| `training_log.csv` | ✓ | ✓ | Per-episode: return, actor_loss, entropy, critic_loss, … |
| `generated.csv` | ✓ | ✓ | Candidates: formula, reward, dp_mean, dp_std, primary_ok |
| `run_config.json` | ✓ | ✓ | Full argparse namespace for reproducibility |
