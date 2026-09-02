#!/usr/bin/env bash
#
# Seed sweep for ONE scenario config: pick an arm (or all), run N seeds.
#
# Generalised from the old submit_perovskite_sweep.sh — the scenario is now a
# --config argument rather than baked into the filename, so a new scenario needs
# a new YAML, not a new launcher. Its sibling scripts/submit_sweep.sh is the
# other shape: MANY scenarios (oxides_<scen>.yaml) at ONE budget label.
#
#     scripts/submit_seed_sweep.sh --config configs/cs2agbicl6_level1.yaml
#     scripts/submit_seed_sweep.sh --config configs/cs2agbicl6_level1.yaml --arm a2c
#     scripts/submit_seed_sweep.sh --config configs/perovskite_level1.yaml --budget 250 --arm bo
#
# Run it from the directory where `rl-matdesign --config <CONFIG> ...` already
# works: config-relative paths inside the YAML (base_poscar, mgt_repo, model,
# geo_opt.model) are resolved from the CWD, and this script does not chdir.
#
# Arms — the default set is the three RL arms, which is the benchmark you almost
# always want (same env, same budget, three learners):
#
#     dqn_bootstrap   --method dqn --dqn-target-mode bootstrap
#     dqn_mc          --method dqn --dqn-target-mode mc      (ablation, not a 4th method)
#     a2c             --method a2c
#     bo / ga         scripts/baselines/run_bo.py / run_ga.py   -- REQUIRE --budget
#
#   --arm a2c        one arm
#   --arm rl         the three RL arms (default)
#   --arm all        the three RL arms plus bo and ga
#
# DQN(bootstrap) and DQN(mc) are the SAME method with different regression
# targets; label them that way in plots, not as two independent methods.
#
# ---------------------------------------------------------------------------
# CONCURRENCY: one run is not one process. A structure_score config spawns one
# predictor subprocess PER MODEL PATH (e.g. cs2agbicl6_level3.yaml has three
# MGTransformer heads => three serve.py processes, each holding GPU memory). Ten
# seeds x three arms x three heads is 90 resident models if you let it all run at
# once. MAX_JOBS defaults to 4 for that reason — raise it only after checking
# `nvidia-smi` headroom for YOUR config's model count.
#
# BUDGET: predictor calls, not episodes.
#     DQN: dqn_warmup_eps + dqn_num_train_eps   (warmup pays a real call/episode)
#     PG:  pg_num_iters * pg_batch_eps          (PG warmup is free)
#     BO/GA: --budget directly
# The RL budgets live in the YAML by this repo's convention (a generator would
# hide the two traps below); --budget is passed through to BO/GA only, and is
# used as the default run-name label. Before changing budget, check:
#     dqn_eps_anneal_eps  ~60% of dqn_num_train_eps, or epsilon never anneals and
#                         the "DQN" arm is random search for the whole run.
#     pg_batch_eps        hold FIXED across budgets; vary only pg_num_iters, or
#                         budget is confounded with A2C's entropy-collapse rate.
# Both configs' budget tables sit next to those keys in the YAML.
#
# Env overrides:
#   OUT=runs/<config-stem>   output root (seeded run dirs are created inside)
#   SEEDS="1 2 3"            seed list (ignored when --seed is passed)
#   MAX_JOBS=4               concurrent runs
#   RL_BIN=rl-matdesign      launcher for the RL arms
#   FORCE=1                  redo runs that already finished
#   DRYRUN=1                 print the commands instead of running them
#
# Runs whose generated.csv exists are skipped, so a crashed sweep resumes.

set -euo pipefail

_RL_ARMS="dqn_bootstrap dqn_mc a2c"
_BASELINE_ARMS="bo ga"
_VALID_ARMS="$_RL_ARMS $_BASELINE_ARMS"

usage() {
    cat >&2 <<EOF
usage: $0 --config CONFIG [--arm ARM] [--seed N] [--budget B] [--label L]

  --config CONFIG   scenario YAML (required; env CONFIG also works)
  --arm ARM         one of {$_VALID_ARMS}, or 'rl' (default: the three RL arms),
                    or 'all' (RL arms + bo + ga)
  --seed N          pin to a single seed instead of the full SEEDS list
  --budget B        integer; REQUIRED for the bo/ga arms, otherwise only used to
                    label run directories
  --label L         run-name tag (default: eps<budget>, or 'run' with no budget)

e.g.  $0 --config configs/cs2agbicl6_level1.yaml --arm a2c
      $0 --config configs/perovskite_level1.yaml --budget 250 --arm bo --seed 7
EOF
    exit 1
}

CONFIG="${CONFIG:-}"
ARM="rl"
SEED_ONE=""
BUDGET=""
LABEL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)          CONFIG="${2:-}";   [[ -n "$CONFIG" ]]   || usage; shift 2 ;;
        --arm)             ARM="${2:-}";      [[ -n "$ARM" ]]      || usage; shift 2 ;;
        -seed|--seed)      SEED_ONE="${2:-}"; [[ -n "$SEED_ONE" ]] || usage; shift 2 ;;
        --budget)          BUDGET="${2:-}";   [[ -n "$BUDGET" ]]   || usage; shift 2 ;;
        --label)           LABEL="${2:-}";    [[ -n "$LABEL" ]]    || usage; shift 2 ;;
        -h|--help)         usage ;;
        *) echo "error: unrecognised argument '$1'" >&2; usage ;;
    esac
done

[[ -n "$CONFIG" ]] || { echo "error: --config is required" >&2; usage; }
[[ -f "$CONFIG" ]] || { echo "error: config '$CONFIG' not found (cwd: $PWD)" >&2; exit 1; }
if [[ -n "$BUDGET" && ! "$BUDGET" =~ ^[0-9]+$ ]]; then
    echo "error: --budget must be an integer, got '$BUDGET'" >&2
    exit 1
fi

case "$ARM" in
    rl)  ARMS_TO_RUN="$_RL_ARMS" ;;
    all) ARMS_TO_RUN="$_VALID_ARMS" ;;
    *)   case " $_VALID_ARMS " in
             *" $ARM "*) ARMS_TO_RUN="$ARM" ;;
             *) echo "error: unknown arm '$ARM'. Expected one of:" \
                     "$_VALID_ARMS, 'rl', or 'all'" >&2; exit 1 ;;
         esac ;;
esac

# bo/ga read the budget from their own CLI, so without it they would silently run
# at their built-in default and be incomparable to the RL arms at this budget.
if [[ -z "$BUDGET" ]]; then
    for a in $ARMS_TO_RUN; do
        case "$a" in
            bo|ga) echo "error: arm '$a' needs --budget (its trial count comes from" \
                        "the CLI, not the YAML)" >&2; exit 1 ;;
        esac
    done
fi

CONFIG_STEM="$(basename "$CONFIG")"; CONFIG_STEM="${CONFIG_STEM%.*}"
# With no --label and no --budget the name carries no tag at all: `a2c_seed7`
# rather than a filler word. LABEL is still used to scope the final FAIL scan,
# so an empty one there means "every log in this OUT".
[[ -n "$LABEL" ]] || LABEL="${BUDGET:+eps$BUDGET}"
NAME_TAG="${LABEL:+${LABEL}_}"

RL_BIN="${RL_BIN:-rl-matdesign}"
OUT="${OUT:-runs/$CONFIG_STEM}"
SEEDS="${SEEDS:-7 19 23 42 58 61 77 84 96 103}"
[[ -n "$SEED_ONE" ]] && SEEDS="$SEED_ONE"
MAX_JOBS="${MAX_JOBS:-4}"
FORCE="${FORCE:-0}"
DRYRUN="${DRYRUN:-0}"

n_seeds=$(printf '%s\n' $SEEDS | wc -l | tr -d ' ')
n_arms=$(printf '%s\n' $ARMS_TO_RUN | wc -l | tr -d ' ')
echo "config : $CONFIG" >&2
echo "arms   : $ARMS_TO_RUN" >&2
echo "seeds  : $SEEDS  (${n_seeds} x ${n_arms} arms = $((n_seeds * n_arms)) runs," \
     "$MAX_JOBS at a time)" >&2
echo "out    : $OUT" >&2
case "$ARMS_TO_RUN" in
    *dqn*|*a2c*)
        echo "Reminder: the RL episode budgets come from $CONFIG" \
             "(dqn_warmup_eps + dqn_num_train_eps, pg_num_iters * pg_batch_eps)," \
             "not from --budget — see this script's header." >&2 ;;
esac

LOG_DIR="$OUT/logs"
[[ "$DRYRUN" == "1" ]] || mkdir -p "$LOG_DIR"

launched=0
skipped=0

wait_for_slot() {
    while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
        wait -n 2>/dev/null || sleep 1
    done
}

# Deliberately NOT a bash 4+ associative array (`declare -A`) — macOS ships
# bash 3.2 by default, which doesn't support them; a case statement works
# identically everywhere this script might run.
arm_cmd() {
    local arm="$1" seed="$2" dp_seed="$3" gen_seed="$4"
    case "$arm" in
        dqn_bootstrap)
            echo "$RL_BIN --config $CONFIG --method dqn --dqn-target-mode bootstrap --train-seed $seed --dp-seed $dp_seed --gen-seed $gen_seed" ;;
        dqn_mc)
            echo "$RL_BIN --config $CONFIG --method dqn --dqn-target-mode mc --train-seed $seed --dp-seed $dp_seed --gen-seed $gen_seed" ;;
        a2c)
            echo "$RL_BIN --config $CONFIG --method a2c --train-seed $seed --dp-seed $dp_seed --gen-seed $gen_seed" ;;
        bo)
            echo "python scripts/baselines/run_bo.py --config $CONFIG --seed $seed --budget $BUDGET" ;;
        ga)
            echo "python scripts/baselines/run_ga.py --config $CONFIG --seed $seed --budget $BUDGET" ;;
        *)
            echo "error: unknown arm '$arm'" >&2; return 1 ;;
    esac
}

for seed in $SEEDS; do
    # Decorrelated seeds: the training RNG, the predictor's random-decoration
    # sampling and the generation sampling get independent streams, so seed-to-seed
    # spread is real variance rather than one shared stream reused three times.
    dp_seed=$((seed + 10000))
    gen_seed=$((seed + 20000))

    for arm in $ARMS_TO_RUN; do
        name="${arm}_${NAME_TAG}seed${seed}"
        run_out="$OUT/$name"

        # generated.csv is written last, so its presence means the run finished.
        if [[ "$FORCE" != "1" && -f "$run_out/generated.csv" ]]; then
            skipped=$((skipped + 1))
            continue
        fi

        cmd=($(arm_cmd "$arm" "$seed" "$dp_seed" "$gen_seed") --out "$run_out/")
        log="$LOG_DIR/${name}.log"

        if [[ "$DRYRUN" == "1" ]]; then
            echo "nohup ${cmd[*]} > $log 2>&1 &"
            launched=$((launched + 1))
            continue
        fi

        wait_for_slot
        echo "[run ] $name"
        nohup "${cmd[@]}" > "$log" 2>&1 &
        launched=$((launched + 1))
    done
done

if [[ "$DRYRUN" == "1" ]]; then
    echo "# $launched commands ($skipped already complete)" >&2
    exit 0
fi

wait
echo
echo "=== ${LABEL:-all}: $launched launched, $skipped already complete"
echo "    logs: $LOG_DIR/"
failed=0
for f in "$LOG_DIR"/*"${LABEL}"*.log; do
    [[ -f "$f" ]] || continue
    d="$OUT/$(basename "$f" .log)"
    [[ -f "$d/generated.csv" ]] || { echo "    [FAIL] $(basename "$f")"; failed=$((failed + 1)); }
done
(( failed == 0 )) && echo "    all runs produced generated.csv" || echo "    $failed run(s) failed — see logs above"
