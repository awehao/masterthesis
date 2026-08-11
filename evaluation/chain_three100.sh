#!/usr/bin/env bash
# The three-method comparison, re-run under unified hardware limits.
#
# Report S4.1 has carried "pending re-measurement" since the hardware analysis:
# the old table put GMPC on vx +-0.35 with acceleration 0.8/0.6/1.2 against
# MPPI's 1.5/1.0/2.0, so it compared configurations, not methods. All three now
# run the chassis's real box -- 0.2775 m/s per axis, 6.25 m/s^2, 1.1327 rad/s --
# on the same 100 routes.
#
#   gmpc   GMPC + horizon CBF + raw-scan shield (corrected fallback)
#   mppi   nav2 MPPI, no shield
#   rpp    nav2 RegulatedPurePursuit, no shield
#
# RPP keeps vy = 0. That is the algorithm, not a handicap -- pure pursuit never
# commands lateral motion -- but it has to be stated whenever its numbers are
# quoted, because on a holonomic base it is a real disadvantage.
#
# GMPC runs first so the headline number exists even if the chain is stopped
# early. Each arm is guarded on its first trial.
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
exec 2>&1

LOCKFILE=/tmp/omnibot_dynamic.pid
if [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE" 2>/dev/null)" 2>/dev/null; then
    echo "ERROR: a batch is already running."; exit 1
fi
N=100
POSES="$PWD/evaluation/results/bigarena_poses_big.csv"
say () { echo "[$(date +%H:%M:%S)] $*"; }

run_arm () {              # $1 outdir  $2 method  $3 csv-tag  $4 bag-prefix  $5.. env
    local out="$1" method="$2" tag="$3" prefix="$4"; shift 4
    local csv="evaluation/results/omnibot_dynamic_${tag}.csv"
    mkdir -p "evaluation/results/$out"; rm -rf "evaluation/bags/archive_$out"
    rm -f "$csv"
    say "=== $out ($method, $N routes) ==="
    env "$@" MASK_HW=10.0 POSES_CSV="$POSES" \
        BIGARENA=1 TRAJ=bigarena_traffic GUI=0 \
        MARGIN_MODE=fixed CBF_SAFE_MARGIN=0.60 \
        PLANNER_SCAN=/scan_filtered PLAN_BLEND=0.0 \
        CBF_ALPHA=0.5 CBF_PRUNE_RANGE=1.2 CBF_FAR_STRIDE=2 \
        POSE_SOURCE=odom HARD_K0_STATIC=0 \
        ./evaluation/run_omnibot_dynamic.sh "$N" 250 0 0 "$method" \
        > "evaluation/logs/${out}.log" 2>&1 &
    local bpid=$!
    ./evaluation/guard_first_trial.sh "$csv" || {
        wait $bpid 2>/dev/null; say "$out ABORTED"; return 1; }
    wait $bpid
    cp "$csv" "evaluation/results/$out/batch.csv"
    local ad="evaluation/bags/archive_$out"; mkdir -p "$ad"
    # Only this arm's own bags. `evaluation/bags/*seed*` sweeps in every batch
    # still sitting in that directory -- nocbf, truth, nosm and the other two
    # methods -- and any of them recorded in a different scenario is then scored
    # against the wrong occupancy grid, which reports -0.300 (the robot radius,
    # i.e. centre on an occupied cell) for a trial that touched nothing.
    for d in evaluation/bags/${prefix}seed*; do
        case "$d" in *__prev_*) continue;; esac
        [ -f "$d/metadata.yaml" ] || continue
        local s; s=$(basename "$d" | sed 's/.*seed//'); [ "$s" -le "$N" ] || continue
        cp -r "$d" "$ad/$(basename "$d")" 2>/dev/null
    done
    cp "$csv" "$ad/results.csv"
    local neg; neg=$(awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="min_clearance_m") k=i; next}
                              $k+0 < 0 {n++} END{print n+0}' "$csv")
    say "$out done: $(( $(wc -l < "$csv") - 1 )) trials, $neg negative"
}

run_arm gmpc100 gmpc_scan gmpc_scan 'gmpc_cbf__scan_' SHIELD=1
run_arm mppi100 mppi mppi 'mppi__mppi_'
run_arm rpp100  rpp  rpp  'rpp__rpp_'

say "analysing"
python3 evaluation/summarise_three100.py > evaluation/results/THREE100_SUMMARY.md 2>&1
say "ALL DONE -> evaluation/results/THREE100_SUMMARY.md"
