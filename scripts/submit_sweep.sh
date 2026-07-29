#!/usr/bin/env bash
#
# Submit one budget's worth of runs: 3 scenarios x 3 methods x 10 seeds.
#
# Edit the episode budget in the YAML configs yourself, then run this with a
# label naming that budget:
#
#     ./scripts/submit_sweep.sh eps2500
#     ./scripts/submit_sweep.sh eps7500
#     ./scripts/submit_sweep.sh eps15000
#     ./scripts/submit_sweep.sh eps30000
#
# Output lands in runs/sweep/<label>/<scenario>_<arm>/seed_<N>/.
# run_seeds.py also writes all_seeds.csv and all_seeds_timing.csv per group.
#
# Already-finished groups are skipped, so re-running after a crash resumes.
# Set FORCE=1 to redo everything.
#
# ---------------------------------------------------------------------------
# BEFORE THE FIRST RUN, check two keys in each YAML — both silently invalidate
# the short-budget arms if left at their 50,000-episode defaults:
#
#   dqn_eps_anneal_eps   Epsilon anneals linearly over this many episodes. The
#                        configs ship with 30000. At a 2,500-episode budget that
#                        leaves epsilon at 1 - 2500/30000 = 0.92, so the agent is
#                        ~92% random for the whole run and the "DQN" arm is really
#                        random search. This is not hypothetical: the archived
#                        2,500-episode runs ended at epsilon=0.950. Set it to
#                        roughly 60% of dqn_num_train_eps (the ratio the reference
#                        config uses: 30000/50000).
#
#   pg_batch_eps         Hold this FIXED at 15 across all budgets and vary only
#                        pg_num_iters. Changing both at once confounds budget with
#                        collapse rate — the archived sweep did this (25 at the
#                        small budgets, 15 at 45k) and the arms aren't comparable.
#
# Budget is counted in *predictor calls*, which is not the same as episodes:
#   DQN: dqn_warmup_eps + dqn_num_train_eps   (warmup pays a real call per episode)
#   PG:  pg_num_iters * pg_batch_eps          (PG warmup is free — reward_fn is
#                                              neutralised during it)
# ---------------------------------------------------------------------------

set -euo pipefail

LABEL="${1:-}"
if [[ -z "$LABEL" ]]; then
    echo "usage: $0 <label>    e.g. $0 eps2500" >&2
    exit 1
fi

cd "$(dirname "$0")/.."

# --- knobs -----------------------------------------------------------------
SEEDS="${SEEDS:-7 19 23 42 58 61 77 84 96 103}"
OUT_ROOT="${OUT_ROOT:-runs/sweep}"
# run_seeds.py --parallel runs all 10 seeds of a group concurrently.
PARALLEL="${PARALLEL:-1}"
# How many groups to run at once. 1 => 10 concurrent processes. Raise it if the
# GPU has headroom (you said it does); 9 runs every group of this sweep at once.
CONCURRENT_GROUPS="${CONCURRENT_GROUPS:-1}"
FORCE="${FORCE:-0}"
# DRYRUN=1 prints every command without launching anything.
DRYRUN="${DRYRUN:-0}"

SCENARIOS="${SCENARIOS:-sinter calcine sinter_calcine}"
# arm name : config suffix : --method : --dqn-target-mode
ARMS=(
    "dqn_bootstrap::dqn:bootstrap"
    "dqn_mc::dqn:mc"
    "a2c:_a2c:a2c:"
)

OUT_BASE="$OUT_ROOT/$LABEL"
LOG_DIR="$OUT_BASE/logs"
mkdir -p "$LOG_DIR"

echo "=== sweep '$LABEL'"
echo "    seeds:       $SEEDS"
echo "    scenarios:   $SCENARIOS"
echo "    out:         $OUT_BASE"
echo "    groups:      $CONCURRENT_GROUPS at a time, seeds $([[ $PARALLEL == 1 ]] && echo parallel || echo sequential)"
echo

submitted=0
skipped=0
pids=()

wait_for_slot() {
    while (( ${#pids[@]} >= CONCURRENT_GROUPS )); do
        wait -n 2>/dev/null || true
        local alive=()
        for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive+=("$p"); done
        pids=("${alive[@]}")
    done
}

for scen in $SCENARIOS; do
    for arm_spec in "${ARMS[@]}"; do
        IFS=':' read -r arm suffix method target <<< "$arm_spec"

        config="configs/oxides_${scen}${suffix}.yaml"
        if [[ ! -f "$config" ]]; then
            echo "[skip] $config not found — skipping ${scen}/${arm}" >&2
            continue
        fi

        out="$OUT_BASE/${scen}_${arm}"
        if [[ "$FORCE" != "1" && -f "$out/all_seeds.csv" ]]; then
            echo "[done] ${scen}/${arm} already complete — skipping (FORCE=1 to redo)"
            skipped=$((skipped + 1))
            continue
        fi

        cmd=(python scripts/run_seeds.py
             --config "$config"
             --method "$method"
             --out "$out"
             --seeds $SEEDS)
        [[ -n "$target" ]] && cmd+=(--dqn-target-mode "$target")
        [[ "$PARALLEL" == "1" ]] && cmd+=(--parallel)

        log="$LOG_DIR/${scen}_${arm}.log"
        echo "[run ] ${scen}/${arm}  -> $out"
        echo "       ${cmd[*]}"
        echo "       log: $log"

        if [[ "$DRYRUN" == "1" ]]; then
            submitted=$((submitted + 1))
            continue
        fi

        wait_for_slot
        ( "${cmd[@]}" >"$log" 2>&1 \
            && echo "[ok  ] ${scen}/${arm}" \
            || echo "[FAIL] ${scen}/${arm} — see $log" ) &
        pids+=($!)
        submitted=$((submitted + 1))
    done
done

wait
echo
echo "=== sweep '$LABEL' finished: $submitted submitted, $skipped skipped"
echo
echo "Next, once every budget label exists, build the comparison figures:"
echo "  python scripts/baselines/compare_timing.py \\"
for scen in $SCENARIOS; do
    echo "      --run \"DQN(bootstrap):$OUT_BASE/${scen}_dqn_bootstrap\" \\"
    echo "      --run \"DQN(mc):$OUT_BASE/${scen}_dqn_mc\" \\"
    echo "      --run \"A2C:$OUT_BASE/${scen}_a2c\" \\"
    break
done
echo "      --out $OUT_BASE/compare --title \"$LABEL\""
