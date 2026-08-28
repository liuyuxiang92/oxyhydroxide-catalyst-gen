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
| `species_set` | Candidate element symbols | **required** (list) |
| `fraction_set` | Allowed fraction strings (grid) | default = built-in `0.05 … 0.80` |
| `total_units` | Grid resolution; step = `1/total_units` | `20` (->0.05); use `100` for 0.01 |
| `n_components` | Distinct elements per episode | `5` |
| `anion_formula` | Fixed suffix appended to the formula | `""` (e.g. `O2H1`, `O3`) |
| `episode_style` | How each step picks | `element_then_amount` (default: choose element + amount); `fixed_order_amount` (order pinned to `species_set`, choose only amounts) |
| `element_bounds` | Per-element `{el: [min,max]}` fraction caps | none; **only with `fixed_order_amount`** |
| `constraint_filter` | Action-masking filter (see §3) | `null` |

## `integer_ratio` env keys

| Key | What it does | Choices / default |
|---|---|---|
| `species_set` | Candidate elements | **required** |
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
`fraction`-env keys (`species_set`, `fraction_set`, `total_units`, `n_components`,
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
| `structure_score` | **The one structure-based predictor** — build -> [relax] -> score N properties -> combine (see below) |
| `ooh` | OOH overpotential (adsorbate slabs) |
| `sinter_calcine` | RandomForest sintering/calcine temperature |
| `"pkg.mod:Class"` | Your own predictor (FQN) |

## `structure_score` keys

One predictor steered by dials. The pipeline is
`builder.build -> [relax once, optional] -> score each property -> combine`.
It replaces the former `dp_structure` / `dp_property` / `composite` /
`structure_pipeline` (and the `hea` / `perovskite` aliases): each is now just a
particular setting of the dials below.

| Key | What it does | Default |
|---|---|---|
| `builder` | Builder name/FQN (`substitute` = fixed-lattice element swap; `sse` = doped supercell; or `pkg.mod:Class`) | `substitute` |
| `share_structure` | `true` = build+relax **one** cell, score all properties on it; `false` = each property builds its **own** | `true` |
| `n_random_configs` | Random placements per candidate | `1` |
| `geo_opt` | Relaxation stage (see below); omit / `enabled: false` to skip | optional |
| `properties` | Non-empty list of objective specs (see below) | required |
| `k` | Uncertainty coefficient | `1.0` |
| `sweep` | Optional `{name, values}` — optimize a shared operating condition (e.g. temperature) per composition (see below) | optional |
| `base_poscar` / `site_symbol` | Builder knobs (for `substitute`); inherited by per-objective builders when `share_structure: false` | — |

`geo_opt:` sub-keys — `model` (default `models/DPA-3.1-3M.pt`), `head` (user-defined),
`fmax` (`0.001`), `steps` (`1000`), `relax_cell` (`true`), `enabled` (`true`).

`properties[]:` sub-keys:

| Key | What it does | Default |
|---|---|---|
| `name` | Unique label (CSV column prefix) | `prop{i}` |
| `backend` | `energy` (DP potential energy via ASE) or `property` (DP `DeepProperty` head) | `property` |
| `models` (or legacy `dp_models`) | Ensemble checkpoint list | required |
| `head` | DP head (energy: calculator head; property: `DeepProperty` head) | none |
| `direction` | `max` / `min` / `target` (energy: use `min` for "lower is better") | **required** |
| `target_value` | Value to hit, in the property's real units. Required with `direction: target`. **Not** the leaf's `model:`/checkpoint key. | — |
| `target_tolerance` | Deadband half-width; inside it the term is at its maximum (0) and stops competing with other objectives | `0.0` |
| `weight` / `scale` | Combine weight / unit scale | `1.0` / `1.0` |
| `objective` | `mean_minus_kstd` / `mean` / `mean_plus_kstd` | `mean_minus_kstd` |
| `energy_per_atom` *(energy backend)* | Normalize by atom count | `true` |
| `output_index` / `output_aggregator` *(property backend)* | Which vector component / collapse mode | `0` / `index` (`index`/`mean`/`max`) |
| `fparam` / `aparam` *(property backend)* | Frame / atomic parameters for heads trained with them. A `null` in `fparam` marks the slot filled by the `sweep` value (requires a top-level `sweep`) | none |

Reward = `sum weight * value(prop) / scale`, where `value` is

- `max`/`min`: `objective_from_mean_std(direction*mean, std, objective, k)`
- `target`:    `objective_from_mean_std(-e, std, objective, k)` with
  `e = max(0, |mean - target_value| - target_tolerance)`

i.e. a target objective scores the (deadbanded) **distance** to `target_value`, negated so
closer is better. Routing it through the same helper keeps uncertainty handling identical:
`mean_minus_kstd` reads as "effectively farther from the target when less certain".

### What `mean` and `std` actually are

Two axes, folded separately and deliberately not pooled:

- **structures** — the `n_random_configs` random decorations of one composition. Their
  scatter is *configurational*: it is how the property is defined for a disordered
  composition, so it is **averaged away**.
- **models** — independent checkpoints of the same property. Their scatter is *epistemic
  uncertainty*, and is what `k*std` penalises.

So `mean` is the mean of the per-model averages and `std` is the spread **across models**.
A single model therefore gives `std == 0` exactly, whatever `n_random_configs` is, and
`mean_minus_kstd` is then identical to `mean` — no flag needed.
over the properties (identical formula in both `share_structure` regimes).

### `sweep:` — per-composition operating-condition optimization

Some property heads depend on an external operating condition (e.g. temperature)
passed as a frame parameter. `sweep: {name, values}` turns that condition into an
**inner optimization**: for each composition the predictor scores every property
at each value and keeps the single shared value maximizing the **combined**
reward. Because the structure is built + relaxed **once** and reused, sweeping is
cheap (only the property inference repeats).

```yaml
sweep:
  name: temperature
  values: [460, 470, 480, 490]
properties:
  - name: conductivity
    backend: property
    fparam: [3.9568e-05, null, 6]      # null = the swept temperature slot
    direction: max
    ...
  - name: stability
    backend: property
    fparam: [3.9568e-05, null, 6, 4]
    direction: max
    ...
```

The chosen value is logged per candidate as `obj_<name>_mean` (e.g.
`obj_temperature_mean`) in `generated.csv`. A property with no `null` in its
`fparam` (or an `energy` backend) is condition-independent — scored once and
contributing equally at every sweep value.

**Migration cheatsheet** (old -> `structure_score`):

| Old predictor | Now |
|---|---|
| `dp_structure` | `builder: substitute`, one property `backend: energy`, `direction: min` |
| `dp_property` | `builder: substitute`, one property `backend: property` |
| `hea` / `perovskite` | as `dp_structure`, with `site_symbol: X` / `Fe` |
| `structure_pipeline` | `share_structure: true` (default), N `backend: property` objectives |
| `composite` | `share_structure: false`, the `objectives:` list becomes `properties:` |

## `ooh` keys

`base_poscar`, `dp_models`, `objective`, `k`, `n_random_configs`, `ads_height`,
`ads_dz`, `geo_opt` (bool), `geo_opt_model`, `uncertainty`
(`models`/`configs`/`total`), `output_index`, `adsorbates`, `target_phases`.

`adsorbates` selects which intermediates are placed on each doped slab, in frame
order — default `[O, OH, OOH]`. An **empty list** means the bare parent slab (no
adsorbate atoms at all): one frame per random config instead of three, so ~3x
fewer DeepMD evaluations and ~3x fewer relaxations under `geo_opt: true`.
`ads_height` / `ads_dz` are unused in that case.

Bare is expressed as the empty list rather than as a member of the list because
all frames in one batch must have the same atom count (an adsorbate frame has
`nat_slab + 3` atoms, a bare one `nat_slab`).

The list order matters: the batch is frame-major and `output_index` indexes the
flattened output, so with the default list `output_index: 0` reads the **O\***
frame. Changing `adsorbates` or `output_index` invalidates a saved `dp_cache`
(they are folded into the cache key), so a resumed run pays a one-time recompute
instead of returning values that describe a different structure.

## `sinter_calcine` keys

`rf_model` (joblib path), `mode` (`sinter` / `calcine`).

\newpage

## Writing your own predictor — the leaf contract

A **leaf represents exactly one model**; the reward engine owns all folding. That is what
makes "one model => no uncertainty" automatic instead of a per-scenario flag.

`model:` accepts one path **or a list of paths**. The engine builds one leaf instance per
path (each with a decorrelated seed), so an ensemble is a config change, never a code
change:

```yaml
model: ckpt/a.pt                    # 1 instance -> std == 0
model: [ckpt/a.pt, ckpt/b.pt]       # 2 instances -> std = their disagreement
```

Expose whichever of these fits; everything is auto-detected, with no YAML flag:

| method | returns | who folds | use when |
|---|---|---|---|
| `score_structures(atoms_list)` | one float **per structure** | engine | one model, geometry-dependent property |
| `score(composition)` | one float | engine | one model, stoichiometry-only property |
| `predict_structures(atoms_list)` | `(mean, std)` | leaf | leaf has a genuine *internal* ensemble |
| `predict(composition)` | `(mean, std)` | leaf | same, composition-based |

- **structure vs composition** — a leaf exposing either `*_structures` method receives the
  built (and optionally relaxed) cells; otherwise it receives the candidate composition.
- **engine-folded vs self-folding** — the `score*` form wins when both are present. The
  self-folding form is correct only when the leaf really does hold its own ensemble (e.g.
  `rf_magpie`'s random-forest tree variance); its `(mean, std)` is passed through untouched.

Prefer `score*` for anything wrapping a single checkpoint: fold your own structures and you
will report *configurational* scatter as if it were model uncertainty, which
`mean_minus_kstd` will then silently penalise.

Constructor is `__init__(self, cfg, *, seed=None)`, where `cfg` is the whole `properties[]`
entry. Add an optional `close()` to release external resources (subprocesses, file handles);
the engine calls it on every instance at the end of a run.

## `mgtransformer` keys

Bridge to an external MGTransformer checkout (`../MGTransformer`), one persistent
`serve.py` subprocess per instance.

| Key | Meaning | Default |
|---|---|---|
| `model` | Path to a finetuned checkpoint, forwarded as `serve.py --ckpt`. MGTransformer derives the target — and hence which calibration constants apply — from the path, so it is never named twice. | **required** |
| `mgt_repo` | Path to the MGTransformer checkout | **required** |
| `mgt_python` | Interpreter of MGTransformer's own env (its deps conflict with this repo's) | **required** |
| `device` | Forwarded to `serve.py --device` | `cpu` |
| `max_neighbors`, `cutoff`, `atom_features`, `triplet_endpoint`, `triplet_pad_mode` | Featurizer overrides forwarded to `serve.py` | see that repo |

Scores arrive in **real units** (eV, eV/atom): MGTransformer un-normalizes with the
train-split constants in its `mgt_calibration.json`. A target with no entry there falls back
to a raw z-score and warns once — ranking is unaffected either way, but a `direction: target`
objective would then be meaningless.

# 5. Builders (`builder:`)

A builder turns the agent's pick (a flat `{element: fraction}` or a structured
`{group: {…}}`) into ASE structures. Built-ins:

| `builder:` | What it does |
|---|---|
| `substitute` | Fixed-lattice element swap (no vacancies). **Single-sublattice**: fills `site_symbol` placeholder sites from a flat `{element: fraction}`. **Multi-sublattice**: add `site_map: {group: symbol}` and it fills each sublattice from a structured `{group: {element: fraction}}` candidate, one op per group. Keys: `base_poscar` (or `poscar`), `site_symbol` (default `X`), `site_map`. |
| `site_pick` | One element per site across N sublattices via `site_map` — the *categorical* counterpart of `substitute`'s multi-sublattice mode (no fractions). |
| `defect_site` | A/B-site doping plus a signed vacancy/interstitial defect axis. |
| `sse` | Doped-supercell recipe (P->metal, S->O/Cl/Br, charge-neutral Li vacancies). Keys below. |
| `"pkg.mod:Class"` | Your own builder (FQN) with `build(candidate, *, n_configs, rng)`. |

## `sse` builder keys

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
| `pg_num_iters` | 1000 | `pg_entropy_min` | 0.3 |
| `pg_batch_eps` | 15 | `pg_repeat_penalty_coef` | 0.0 |
| `pg_lr_actor` | 0.001 | `pg_repeat_penalty_shape` | `log` (also `sqrt`) |
| `pg_lr_critic` | 0.001 (a2c only) | `gamma` (or `pg_gamma`) | 0.9 |

Advantages are **standardised across each batch** before the actor update. This is
unconditional and has no flag: without it the actor term scales with the raw
reward (sintering temperatures are 400–700, so |advantage| is O(hundreds)) while
the entropy bonus is at most `pg_entropy_coef · ln|A| ≈ 0.5`, which leaves the
entropy term ~2 orders of magnitude too small to matter. Two consequences:

- `pg_entropy_coef` and `pg_repeat_penalty_coef` are in **standard deviations of
  batch return**, so they mean the same thing in every scenario whatever units the
  property has. A `pg_repeat_penalty_coef` of 0.1 shifts the advantage by 0.1σ.
- In `training_log.csv`, `repeat_penalty` is now in σ, while `return_shaped`
  converts it back to the property's units so it stays comparable against
  `return_raw`. Logs written before this change match neither.

`pg_entropy_min` is a floor on **normalised** entropy `H / ln|A|` in `[0, 1]`, not
absolute nats — `|A|` is ~268 for the 80-element oxide env but far smaller for
OOH, so an absolute floor would not port. A proportional controller raises the
effective entropy weight while entropy sits below the floor and decays it back to
`pg_entropy_coef` above it; the base value is the lower clamp, so the controller
can only ever add exploration pressure. Set `pg_entropy_min: 0` to disable.

Watch `entropy_norm` and `entropy_coef_eff` in `training_log.csv`: `entropy_norm`
is directly comparable to `pg_entropy_min`, and a run whose `entropy_coef_eff` is
pinned at the ceiling is telling you the policy wants to collapse harder than the
controller can prevent — raise `pg_entropy_coef`.

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
