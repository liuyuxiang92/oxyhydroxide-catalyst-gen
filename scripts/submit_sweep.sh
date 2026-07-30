#!/usr/bin/env bash
#
# Submit one budget's worth of runs: 3 scenarios x 3 arms x 10 seeds = 90 runs.
#
# Run it from your working directory — the one where this already works:
#
#     rl-matdesign --config oxides_sinter.yaml --out calc_time/... --method dqn ...
#
# Config paths stay relative (oxides_<scenario>.yaml) and no directory change is
# made, so the relative model paths inside the configs keep resolving.
#
#     ./submit_sweep.sh eps2500          # after setting the budget in the yaml
#     ./submit_sweep.sh eps7500
#     ./submit_sweep.sh eps22500
#     ./submit_sweep.sh eps45000
#
# Each call launches 3 scenarios x 3 arms x len(SEEDS) runs, all in parallel (up to
# MAX_JOBS). With the default 10 seeds that's 90 runs sharing one GPU at once, which
# slows every run down. Pin ONE seed per call with -seed to submit only 9 at a time
# (3 scenarios x 3 arms):
#
#     ./submit_sweep.sh eps2500 -seed 7
#
# The script blocks until its own runs finish (see `wait` below), so looping over
# seeds gives you a queue of 9-at-a-time batches with no extra tooling:
#
#     for s in 7 19 23 42 58 61 77 84 96 103; do
#         ./submit_sweep.sh eps2500 -seed "$s"
#     done
#
# Put that in its own nohup'd wrapper script if you want the whole queue to survive
# a logout; each ./submit_sweep.sh call already blocks until its batch is done, so
# the wrapper's job is only to not exit between iterations.
#
# DRYRUN=1 prints the commands instead of running them.
# Runs whose generated.csv already exists are skipped, so a crashed sweep resumes
# (and a seed already finished in an earlier -seed call is skipped, not rerun).
#
#   OUT=calc_time          output root
#   MAX_JOBS=30            concurrent runs (only matters if SEEDS has >1 value)
#   SCENARIOS="sinter"     narrow the set
#   SEEDS="1 2 3"          override the seed list (ignored if -seed is passed)
#   FORCE=1                redo completed runs
#
# ---------------------------------------------------------------------------
# CHECK THREE KEYS IN EACH YAML BEFORE THE FIRST BUDGET. All three silently
# invalidate the short-budget arms if left at their 50,000-episode defaults:
#
#   dqn_eps_anneal_eps   Ships at 30000. At budget 2500 epsilon only falls to
#                        1 - 2500/30000 = 0.92, so the agent is ~92% random for the
#                        whole run and the "DQN" arm is really random search. The
#                        archived 2,500-episode runs ended at epsilon=0.950. Use
#                        ~60% of dqn_num_train_eps (the reference ratio 30000/50000):
#                            budget   dqn_num_train_eps   dqn_eps_anneal_eps
#                             2500          1500                  900
#                             7500          6500                 3900
#                            22500         21500                12900
#                            45000         44000                26400
#                        (dqn_num_train_eps = budget - dqn_warmup_eps, warmup=1000)
#
#   gen_epsilon          Ships at 0.0. At 45k the trained Q-net produced THREE
#                        unique candidates from 1000 generation episodes (calcine:
#                        one). A budget comparison read through a 3-sample readout
#                        is noise. Set 0.2 in all three configs — same value at
#                        every budget, since a probability cannot confound the
#                        budget axis.
#
#   pg_batch_eps         Keep FIXED at 15; vary only pg_num_iters
#                        (= budget / 15 -> 167 / 500 / 1500 / 3000). Changing both
#                        confounds budget with collapse rate.
#
# Note budget means PREDICTOR CALLS, not episodes:
#   DQN: dqn_warmup_eps + dqn_num_train_eps  (warmup pays a real call per episode)
#   PG:  pg_num_iters * pg_batch_eps         (PG warmup is free)
# ---------------------------------------------------------------------------

set -euo pipefail

LABEL="${1:-}"
if [[ -z "$LABEL" ]]; then
    echo "usage: $0 <label> [-seed N]    e.g. $0 eps2500 -seed 7" >&2
    exit 1
fi
shift

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
OUT="${OUT:-calc_time}"
# -seed pins the sweep to one seed (9 jobs: 3 scenarios x 3 arms). Without it, all
# of SEEDS runs in this one call (up to MAX_JOBS at a time).
SEEDS="${SEEDS:-7 19 23 42 58 61 77 84 96 103}"
[[ -n "$SEED_ONE" ]] && SEEDS="$SEED_ONE"
MAX_JOBS="${MAX_JOBS:-30}"
SCENARIOS="${SCENARIOS:-sinter calcine sinter_calcine}"
FORCE="${FORCE:-0}"
DRYRUN="${DRYRUN:-0}"

# arm name : --method : --dqn-target-mode
ARMS=(
    "dqn_bootstrap:dqn:bootstrap"
    "dqn_mc:dqn:mc"
    "a2c:a2c:"
)

LOG_DIR="$OUT/logs"
mkdir -p "$LOG_DIR"

launched=0
skipped=0

wait_for_slot() {
    while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
        wait -n 2>/dev/null || sleep 1
    done
}

for scen in $SCENARIOS; do
    config="oxides_${scen}.yaml"
    if [[ ! -f "$config" ]]; then
        echo "[skip] $config not in $PWD — skipping scenario '$scen'" >&2
        continue
    fi

    for arm_spec in "${ARMS[@]}"; do
        IFS=':' read -r arm method target <<< "$arm_spec"

        for seed in $SEEDS; do
            name="${scen}_${arm}_${LABEL}_seed${seed}"
            run_out="$OUT/$name"

            # generated.csv is written last, so its presence means the run finished.
            if [[ "$FORCE" != "1" && -f "$run_out/generated.csv" ]]; then
                skipped=$((skipped + 1))
                continue
            fi

            # Decorrelated seeds: training RNG, predictor sampling and generation
            # sampling get independent streams, so seed-to-seed spread isn't one
            # shared stream re-used three times.
            cmd=("$RL_BIN"
                 --config "$config"
                 --method "$method"
                 --out "$run_out/"
                 --train-seed "$seed"
                 --dp-seed "$((seed + 10000))"
                 --gen-seed "$((seed + 20000))")
            [[ -n "$target" ]] && cmd+=(--dqn-target-mode "$target")

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
done

if [[ "$DRYRUN" == "1" ]]; then
    echo "# $launched commands ($skipped already complete)" >&2
    exit 0
fi

wait
echo
echo "=== '$LABEL': $launched launched, $skipped already complete"
echo "    logs: $LOG_DIR/"
failed=0
for f in "$LOG_DIR"/*"${LABEL}"*.log; do
    [[ -f "$f" ]] || continue
    d="$OUT/$(basename "$f" .log)"
    [[ -f "$d/generated.csv" ]] || { echo "    [FAIL] $(basename "$f")"; failed=$((failed + 1)); }
done
(( failed == 0 )) && echo "    all runs produced generated.csv" || echo "    $failed run(s) failed — see logs above"
