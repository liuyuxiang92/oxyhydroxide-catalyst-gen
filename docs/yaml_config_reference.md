---
title: "rl-matdesign — YAML Configuration Reference"
subtitle: "Every config flag, what it does, and its choices"
date: "2026-06-10"
geometry: "margin=1.8cm"
fontsize: 10pt
mainfont: "Helvetica Neue"
monofont: "Menlo"
colorlinks: true
---

# Overview

A run is configured by one YAML file plus a few CLI flags. The three pluggable
layers are chosen by **`env_type`** (the search space), **`predictor`** (the
reward), and **`constraint_filter`** (action masking); each then reads its own
keys. The RL and generation knobs are shared across scenarios.

- The YAML is the base; a few CLI flags override matching keys.
- **Seeds are CLI-only**: `--dp-seed` (predictor RNG), `--train-seed` (training;
  defaults to `--dp-seed`), `--gen-seed` (generation; defaults to `--dp-seed`).
- The full merged config is written to `<out>/run_config.json`.

**CLI flags:** `--config` (required), `--out` (required), `--method`,
`--dp-seed`, `--train-seed`, `--gen-seed`, `--device` (`cpu`/`cuda`),
`--dqn-loss`, `--dqn-augment-permutations`, `--save-checkpoint-freq`,
`--resume-training`, `--only-generate`, `--skip-generation`, `--max-gen-attempts`,
`--load-qnet` / `--load-policy` / `--load-value-net` / `--load-scaler`.

\newpage

# 1. Environment

| Key | What it does | Choices / default |
|---|---|---|
| `env_type` | Selects the environment | `fraction` (default, `CompositionEnv`); `integer_ratio` (`IntegerRatioEnv`); `multi_group` (`MultiGroupEnv`) |

## `fraction` env keys

| Key | What it does | Choices / default |
|---|---|---|
| `cation_set` | Candidate element symbols | **required** (list) |
| `fraction_set` | Allowed fraction strings (grid) | default = built-in `0.05 … 0.80` |
| `total_units` | Grid resolution; step = `1/total_units` | `20` (->0.05); use `100` for 0.01 |
| `n_components` | Distinct elements per episode | `5` |
| `anion_formula` | Fixed suffix appended to the formula | `""` (e.g. `O2H1`, `O3`) |
| `episode_style` | How each step picks | `element_then_amount` (default: choose element + amount); `fixed_order_amount` (order pinned to `cation_set`, choose only amounts) |
| `element_bounds` | Per-element `{el: [min,max]}` fraction caps | none; **only with `fixed_order_amount`** |
| `constraint_filter` | Action-masking filter (see §3) | `null` |

## `integer_ratio` env keys

| Key | What it does | Choices / default |
|---|---|---|
| `cation_set` | Candidate elements | **required** |
| `ratio_set` | Allowed integer-ratio digits | default `"0" … "9"` |
| `n_components` | Elements per episode | `5` |
| `constraint_filter` | Action-masking filter | `null` |

No sum-to-1 constraint; the composition is normalized afterward.

\newpage

# 2. `multi_group` env — `groups:`

`groups:` is an **ordered list** of sublattice **groups**, filled in order. A group
is the atomic unit; a single `fraction`/`integer_ratio` env is the one-group case.
The predictor receives the structured `{group_name: {element: value}}`.

**Common group keys**

| Key | What it does | Choices / default |
|---|---|---|
| `name` | Group label (appears in the structured terminal) | string |
| `kind` | Group type | `composition` (default) or `categorical` |
| `sites` | Sublattice size (atoms per formula unit) — bridges fraction<->count and assembles the chemical formula (`amount × sites`) | `1` |
| `constraint_filter` + its kwargs | Per-group filter (sees `prior_groups` for cross-group coupling) | (see §3) |

**`kind: composition`** — pick N elements with amounts summing to 1. Reuses the
`fraction`-env keys (`cation_set`, `fraction_set`, `total_units`, `n_components`,
`episode_style`, `element_bounds`) plus friendly knobs:

| Key | What it does | Choices / default |
|---|---|---|
| `amount` | Generate the value grid instead of writing `fraction_set` | `{min, max, step}` or a list |
| `host` | A host element auto-takes the leftover (list only the dopants; wires a `host_complement` filter) | element symbol |

**`kind: categorical`** — pick discrete **real values**; no sum-to-1. Each slot is
one element with its own value list; the terminal returns the chosen values
unchanged (so a builder reads `Cl = 1.0` / `O = 1` directly).

| Key | What it does |
|---|---|
| `choices` | `[{element, values: [...]}, …]` — one slot per element |

A categorical filter may mask a slot by an earlier group (e.g. `sse_doping` masks
the O slot by the P-site metal's category).

# 3. Constraint filters (`constraint_filter:`)

| Value | What it does | Its keys |
|---|---|---|
| `null` | No constraint (default) | — |
| `smact_charge` | Keep only compositions that can be charge-neutral (SMACT oxidation states) | `smact_anions: [{symbol, charge, stoich}]` (or `smact_anion` / `smact_anion_charge` / `smact_anion_stoich`) |
| `last_step_element` | Force the last step to a required element | `required_elements`; `nonzero_ratio_at_last` (true); `reserve_for_last` (true) |
| `phase_pattern` | Keep compositions matching phase patterns | `phase_patterns` |
| `ooh_phase` | OOH oxyhydroxide phase screen | `target_phases` (default `[any]`) |
| `chain` | Apply several filters in sequence | `filters: [ {constraint_filter: …, …}, … ]` |
| `host_complement` | Dopants at a level, host takes the rest (wired automatically by a composition group's `host` knob) | `host_element`, `levels` |
| `sse_doping` | Mask the LiPS S-site O-form slot by the metal's category | `o_element` (`O`), `host_P` (`P`), `metal_only`, `oxide_only` |
| `"pkg.mod:Class"` | Your own filter (FQN) | your class's keys |

\newpage

# 4. Predictors (`predictor:`)

| Value | What it does |
|---|---|
| `dummy` | Random reward (default; no models — smoke tests) |
| `dp_structure` | Substitute placeholder sites -> DP **energy** ensemble |
| `dp_property` | Substitute -> DP **property-vector** ensemble (`DeepProperty`) |
| `composite` | Weighted combo of child predictors (shares the *composition*) |
| `structure_pipeline` | Build once -> relax once -> N property ensembles (shares the *structure*) |
| `ooh` | OOH overpotential (adsorbate slabs) |
| `hea` / `perovskite` | `dp_structure` subclasses (different default `site_symbol`) |
| `sinter_calcine` | RandomForest sintering/calcine temperature |
| `"pkg.mod:Class"` | Your own predictor (FQN) |

## `dp_structure` / `dp_property` keys

| Key | What it does | Default |
|---|---|---|
| `base_poscar` (or `poscar` / `poscar_template`) | Template with placeholder sites | required |
| `dp_models` | List of `.pt` checkpoints | required |
| `site_symbol` | Placeholder element | `X` (`hea`/`perovskite` differ) |
| `dp_head` | Model head | none / `property` (dp_property) |
| `objective` | `(mean,std)` -> scalar | `mean_minus_kstd`; also `mean`, `mean_plus_kstd` |
| `k` | Uncertainty coefficient | `1.0` |
| `n_random_configs` | Random placements averaged | `5` |
| `energy_per_atom` *(dp_structure)* | Normalize by atom count | `true` |
| `output_index` *(dp_property)* | Which vector component | `0` |
| `output_aggregator` *(dp_property)* | Collapse vector -> scalar | `index` (`index`/`mean`/`max`) |
| `maximize` *(dp_property)* | Larger raw value = better | `false` |

## `composite` keys

`objectives: [ {name, predictor, direction (min/max), weight, scale, objective}, … ]`,
plus `k`. Shared `base_poscar` / `site_symbol` / `n_random_configs` inherit into
children unless a child overrides them.

## `structure_pipeline` keys

| Key | What it does | Default |
|---|---|---|
| `builder` | Builder name/FQN (composition/groups -> ASE Atoms) | required (e.g. `sse`) |
| `n_random_configs` | Placements per candidate | `1` |
| `geo_opt` | Relaxation stage (see below); omit / `enabled: false` to skip | optional |
| `properties` | List of property specs (see below) | required |
| `k` | Uncertainty coefficient | `1.0` |

`geo_opt:` sub-keys — `model` (default `models/DPA-3.1-3M.pt`), `head` (user-defined),
`fmax` (`0.001`), `steps` (`1000`), `relax_cell` (`true`), `enabled` (`true`).

`properties[]:` sub-keys — `name`, `models` (ensemble list), `head`,
`direction` (`max`/`min`), `weight` (`1.0`), `scale` (`1.0`),
`objective` (`mean_minus_kstd`/`mean`/`mean_plus_kstd`),
`output_index` (`0`), `output_aggregator` (`index`).

Reward = `sum weight * (direction*mean - k*std) / scale` over the properties.

## `ooh` keys

`base_poscar`, `dp_models`, `objective`, `k`, `n_random_configs`, `ads_height`,
`ads_dz`, `geo_opt` (bool), `geo_opt_model`, `uncertainty`
(`models`/`configs`/`total`), `output_index`, `target_phases`.

## `sinter_calcine` keys

`rf_model` (joblib path), `mode` (`sinter` / `calcine`).

\newpage

# 5. Builder (`builder: sse`) keys

| Key | What it does | Default |
|---|---|---|
| `base_poscar` | Base supercell | required |
| `valences` | Charge table `{el: int \| {sulfide, oxide}}` (drives the Li-vacancy solve **and** oxide O count) | required |
| `halide_total` | Cl + Br per formula unit | `1.7` |
| `eligible_region` | Substitutable S region | `{symbol: S, take: last, count: 1000}` |
| `formula_units` | Formula units in the supercell — **inferred from the POSCAR** (host-P count / `p_site_per_fu`) unless set | inferred |
| `p_site_group` / `s_site_group` | Group names to read | `P_site` / `S_site` |
| `host` | `{P, S, Li}` host symbols | `{P:P, S:S, Li:Li}` |
| `p_site_per_fu` / `s_site_per_fu` / `li_per_fu` | Sites per formula unit | `1` / `6` / `6` |

The S-site reads the categorical values directly — O is a form flag (`0` metal /
`>0` oxide), Cl is a real per-f.u. count. `Br = halide_total − Cl` and the
charge-neutral Li vacancy are derived. (No `cl_map` / `o_off`.)

# 6. RL method & hyperparameters

| Key | What it does | Choices / default |
|---|---|---|
| `method` | Algorithm (CLI `--method` overrides) | `a2c` (default), `reinforce`, `dqn` |

## Policy gradient (`a2c` / `reinforce`)

| Key | Default | Key | Default |
|---|---|---|---|
| `pg_warmup_eps` | 200 | `pg_entropy_coef` | 0.01 |
| `pg_num_iters` | 1000 | `pg_repeat_penalty_coef` | 0.0 |
| `pg_batch_eps` | 15 | `pg_repeat_penalty_shape` | `log` (also `sqrt`) |
| `pg_lr_actor` | 0.001 | `gamma` (or `pg_gamma`) | 0.9 |
| `pg_lr_critic` | 0.001 (a2c only) | | |

## DQN

| Key | Default | Key | Default |
|---|---|---|---|
| `dqn_lr` | 0.001 | `dqn_target_update_freq` | 100 |
| `dqn_hidden_dim` | 256 | `dqn_eps_anneal_eps` | 10000 |
| `dqn_batch_size` | 256 | `dqn_eps_min` | 0.05 |
| `dqn_warmup_eps` | ->`pg_warmup_eps`/500 | `dqn_gamma` (or `gamma`) | 0.9 |
| `dqn_num_train_eps` | 20000 | `dqn_loss` | `smoothl1` (also `mse`) |
| `dqn_buffer_size` | 50000 | `dqn_augment_permutations` | 0 |
| `dqn_grad_steps_per_ep` | 5 | | |

## Generation (all methods)

| Key | What it does | Default |
|---|---|---|
| `num_gen_eps` | Final candidates to generate | 200 |
| `exploration_gen_eps` | Extra high-temperature candidates | 0 |
| `gen_temperature` | Sampling temperature (DQN Boltzmann / PG) | 1.0 |
| `gen_top_frac` | Restrict sampling to top fraction of actions | 0.0 |
| `gen_epsilon` | ε-random during generation | 0.0 |
| `k` | Uncertainty coefficient for reward folding | 1.0 |
