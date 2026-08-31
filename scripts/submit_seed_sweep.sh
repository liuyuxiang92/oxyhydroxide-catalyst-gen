#!/usr/bin/env bash
#
# Submit one budget's worth of perovskite-level-1 runs: 5 arms x N seeds.
#
# Mirrors scripts/submit_sweep.sh's design (decorrelated seeds, skip-if-done,
# concurrency limiter, DRYRUN) for the single perovskite_level1.yaml scenario,
# extended with the two non-rl-matdesign-CLI baselines (BO, GA).
#
# Every episode/trial of every arm calls the REAL DPA4-relax -> MGTransformer
# pipeline (via configs/perovskite_level1.yaml's MGTransformerPredictor) — there
# is no lookup-table shortcut here. Run it from your working directory — the one
# where `rl-matdesign --config configs/perovskite_level1.yaml ...` already works
# and configs/perovskite.vasp / configs/perovskite_dpa4.ckpt.pt resolve.
#
#     ./submit_perovskite_sweep.sh 100
#     ./submit_perovskite_sweep.sh 250
#     ./submit_perovskite_sweep.sh 500
#     ./submit_perovskite_sweep.sh 1000
#
# An optional second positional argument restricts the sweep to ONE arm
# (default: all 5). Combine with -seed to pin both the method and the seed:
#
#     ./submit_perovskite_sweep.sh 100 bo             # bo, all seeds
#     ./submit_perovskite_sweep.sh 100 bo -seed 7      # bo, seed 7 only
#
# Valid arms: dqn_bootstrap dqn_mc a2c bo ga (or 'all', the default).
#
# Unlike submit_sweep.sh's free-text <label>, BUDGET here is a REQUIRED number:
# it is passed straight through as --budget to the BO/GA baselines (their own
# CLI reads it directly, no YAML edit needed for those two arms). It is NOT
# read back out of the YAML for the DQN/A2C arms — this repo's own established
# convention (see submit_sweep.sh's header) is that RL episode budgets are set
# by hand in the YAML before each call, not auto-generated, because a generator
# hides exactly the footguns below. So BEFORE each call, edit
# configs/perovskite_level1.yaml's dqn_num_train_eps / pg_num_iters to match
# BUDGET, and check:
#
#   dqn_eps_anneal_eps   ~60% of dqn_num_train_eps (dqn_num_train_eps = BUDGET -
#                        dqn_warmup_eps). Too high relative to the budget and
#                        the "DQN" arm is mostly random search the whole run.
#   pg_batch_eps         Keep FIXED across budgets; vary only pg_num_iters
#                        (= BUDGET / pg_batch_eps). Varying both confounds
#                        budget with A2C's entropy-collapse rate.
#
# Note budget means PREDICTOR CALLS, not episodes/trials:
#   DQN: dqn_warmup_eps + dqn_num_train_eps  (warmup pays a real call per episode)
#   PG:  pg_num_iters * pg_batch_eps         (PG warmup is free)
#   BO/GA: --budget directly (one predictor call per trial/individual eval)
#
# DRYRUN=1 prints the commands instead of running them.
# Runs whose generated.csv already exists are skipped, so a crashed sweep resumes.
#
#   CONFIG=configs/perovskite_level1.yaml   scenario config
#   OUT=runs/perovskite_l1                  output root
#   MAX_JOBS=5                              concurrent runs
#   SEEDS="1 2 3"                           override the seed list (ignored if -seed passed)
#   FORCE=1                                 redo completed runs

set -euo pipefail

_VALID_ARMS="dqn_bootstrap dqn_mc a2c bo ga"

BUDGET="${1:-}"
if [[ -z "$BUDGET" || ! "$BUDGET" =~ ^[0-9]+$ ]]; then
    echo "usage: $0 <budget:int> [arm] [-seed N]    e.g. $0 250 bo -seed 7" >&2
    echo "       arm: one of {$_VALID_ARMS} or 'all' (default)" >&2
    exit 1
fi
shift

ARM="all"
if [[ $# -gt 0 && "$1" != "-seed" && "$1" != "--seed" ]]; then
    ARM="$1"
    shift
    case " $_VALID_ARMS " in
        *" $ARM "*) ;;
        *) echo "error: unknown arm '$ARM'. Expected one of: $_VALID_ARMS (or 'all')" >&2
           exit 1 ;;
    esac
fi

SEED_ONE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -seed|--seed)
            SEED_ONE="${2:-}"
            if [[ -z "$SEED_ONE" ]]; then
                echo "error: -seed needs a value" >&2
                exit 1
            fi
            shift 2
            ;;
        *)
            echo "error: unrecognised argument '$1'" >&2
            exit 1
            ;;
    esac
done

RL_BIN="${RL_BIN:-rl-matdesign}"
CONFIG="${CONFIG:-configs/perovskite_level1.yaml}"
OUT="${OUT:-runs/perovskite_l1}"
SEEDS="${SEEDS:-7 19 23 42 58}"
[[ -n "$SEED_ONE" ]] && SEEDS="$SEED_ONE"
MAX_JOBS="${MAX_JOBS:-5}"
FORCE="${FORCE:-0}"
DRYRUN="${DRYRUN:-0}"

if [[ ! -f "$CONFIG" ]]; then
    echo "error: $CONFIG not found in $PWD" >&2
    exit 1
fi

ARMS_TO_RUN="$_VALID_ARMS"
[[ "$ARM" != "all" ]] && ARMS_TO_RUN="$ARM"

if [[ "$ARMS_TO_RUN" == *dqn* || "$ARMS_TO_RUN" == *a2c* ]]; then
    echo "Reminder: dqn_num_train_eps/pg_num_iters in $CONFIG should reflect budget" \
         "$BUDGET before this call — see the header comment in this script." >&2
fi
echo "Arm(s): $ARMS_TO_RUN" >&2

LOG_DIR="$OUT/logs"
mkdir -p "$LOG_DIR"

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
    dp_seed=$((seed + 10000))
    gen_seed=$((seed + 20000))

    for arm in $ARMS_TO_RUN; do
        name="${arm}_eps${BUDGET}_seed${seed}"
        run_out="$OUT/$name"

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
echo "=== budget $BUDGET: $launched launched, $skipped already complete"
echo "    logs: $LOG_DIR/"
failed=0
for f in "$LOG_DIR"/*"eps${BUDGET}"*.log; do
    [[ -f "$f" ]] || continue
    d="$OUT/$(basename "$f" .log)"
    [[ -f "$d/generated.csv" ]] || { echo "    [FAIL] $(basename "$f")"; failed=$((failed + 1)); }
done
(( failed == 0 )) && echo "    all runs produced generated.csv" || echo "    $failed run(s) failed — see logs above"
