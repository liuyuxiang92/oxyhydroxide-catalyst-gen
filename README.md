# oxyhydroxide-catalyst-gen — General Framework

Config-driven RL runner for materials composition discovery.
Supports multiple material systems (OOH catalyst, HEA, perovskite, oxides) via YAML configs.
Three RL algorithms: **DQN** (online, target-network), **REINFORCE**, and **A2C**.

For the OOH-specific CLI-driven branch see `feat/classical-dqn`.

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

## How it works

All hyperparameters, material system settings, and RL method defaults are stored in a **YAML config
file** under `configs/`. The CLI only takes the config path, output directory, method override, and
operational flags (seeds, resume, checkpoint). This means:

- Switching material systems = swap `--config`
- Reproducing a run = keep the same config + seeds
- Tuning = edit the YAML, re-run

---

## Available configs

| Config | Material system | Default method |
|---|---|---|
| `configs/ooh.yaml` | 5-cation ABCDEOOH oxyhydroxide (28 cation set) | reinforce |
| `configs/hea.yaml` | High-entropy alloy formation energy | reinforce |
| `configs/perovskite.yaml` | Perovskite oxide stability | reinforce |
| `configs/oxides_sinter.yaml` | Oxide sintering temperature | reinforce |
| `configs/oxides_calcine.yaml` | Oxide calcination temperature | reinforce |
| `configs/test_dummy.yaml` | Fast smoke test (no real predictor) | dqn |

---

## Quickstart

```bash
python scripts/run_experiment.py \
    --config configs/ooh.yaml \
    --method reinforce \
    --out runs/ooh_reinforce_seed0 \
    --dp-seed 321 --train-seed 123 --gen-seed 213
```

All hyperparameters (`pg_num_iters`, `pg_batch_eps`, `entropy_coef`, `num_gen_eps`, etc.)
come from the YAML. Override them by editing the config or creating a copy for each experiment.

---

## Seeds

| Flag | Default | Purpose |
|---|---|---|
| `--dp-seed N` | 0 | Predictor seed: controls random structure generation and alloy configs. Acts as fallback for the other two seeds. |
| `--train-seed N` | *(uses dp-seed)* | Training RNG. Also enables GPU deterministic mode (`cudnn.deterministic`, `use_deterministic_algorithms`). |
| `--gen-seed N` | *(uses dp-seed)* | Generation phase only — makes stochastic sampling reproducible independently of training. |

Fix all three for fully reproducible results. `--dp-seed` alone is sufficient for a quick sweep.

---

## REINFORCE (default for OOH)

```bash
python scripts/run_experiment.py \
    --config configs/ooh.yaml \
    --method reinforce \
    --out runs/ooh_reinforce_seed0 \
    --dp-seed 321 --train-seed 123 --gen-seed 213 \
    --save-checkpoint-freq 50
```

Key YAML fields (edit `configs/ooh.yaml`):

```yaml
pg_warmup_eps: 1000       # random episodes to fit the StandardScaler
pg_num_iters: 1000        # outer training iterations
pg_batch_eps: 21          # episodes per batch; one gradient step per batch
entropy_coef: 0.15        # entropy bonus weight
repeat_penalty_coef: 10   # penalises revisiting the same composition
gen_temperature: 3.0      # Boltzmann T for generation diversity
num_gen_eps: 2000         # unique compositions to generate
```

---

## A2C

```bash
python scripts/run_experiment.py \
    --config configs/ooh.yaml \
    --method a2c \
    --out runs/ooh_a2c_seed0 \
    --dp-seed 0 --train-seed 42 --gen-seed 99
```

A2C trains an additional critic (`value_net.pt`) to reduce gradient variance.
Set `pg_lr_actor` and `pg_lr_critic` independently in the YAML.

---

## DQN

```bash
python scripts/run_experiment.py \
    --config configs/ooh.yaml \
    --method dqn \
    --out runs/ooh_dqn_seed0 \
    --dp-seed 0 --train-seed 42 --gen-seed 99
```

Key YAML fields:

```yaml
dqn_warmup_eps: 500       # warmup episodes to populate the initial buffer
num_train_eps: 20000      # total online training episodes
buffer_size: 50000        # replay buffer capacity (FIFO)
grad_steps_per_ep: 5      # gradient updates per training episode
target_update_freq: 100   # hard-copy Q-net → target-net every N episodes
eps_anneal_eps: 10000     # linear ε annealing: reaches eps_min after N episodes
eps_min: 0.05
dqn_lr: 0.001
dqn_batch_size: 256
```

Override the loss function from the CLI: `--dqn-loss smoothl1` (default) or `--dqn-loss mse`.

---

## Checkpointing and resume

```bash
# 1. Save checkpoints every 50 PG iterations (or DQN episodes)
python scripts/run_experiment.py \
    --config configs/ooh.yaml --method reinforce \
    --out runs/ooh_reinforce_seed0 \
    --dp-seed 321 --train-seed 123 \
    --save-checkpoint-freq 50

# 2. Resume after interruption (loads checkpoint.pt, appends training_log.csv)
python scripts/run_experiment.py \
    --config configs/ooh.yaml --method reinforce \
    --out runs/ooh_reinforce_seed0 \
    --dp-seed 321 --train-seed 123 \
    --resume-training

# 3. Generate candidates from a saved model without re-training
python scripts/run_experiment.py \
    --config configs/ooh.yaml --method reinforce \
    --out runs/ooh_reinforce_seed0 \
    --only-generate

# 4. Extend training without generating
python scripts/run_experiment.py \
    --config configs/ooh.yaml --method reinforce \
    --out runs/ooh_reinforce_seed0 \
    --resume-training --skip-generation
```

Use `--load-policy`, `--load-qnet`, `--load-scaler`, `--load-value-net` to load
model files from a non-default path.

---

## Adding a new material system

1. Create a new config under `configs/` (copy the closest existing one).
2. If the material system is new, implement a predictor in `src/rl_matdesign/predictors/`
   following the existing `hea.py` / `ooh.py` pattern.
3. Register the predictor in `build_predictor()` inside `scripts/run_experiment.py`.
4. Run with `--config configs/your_system.yaml`.

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
| `generated.csv` | ✓ | ✓ | Candidates: formula, reward, dp_mean, dp_std |
| `run_config.json` | ✓ | ✓ | Full config + argparse namespace for reproducibility |
