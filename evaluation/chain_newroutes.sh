#!/usr/bin/env bash
# Fixed vs derived margins, 80 trials each, on a fresh single draw of routes.
#
# The 90 routes used so far are three separate draws of 30, and a chi-square on
# their negative-clearance counts (8/30, 2/29, 13/30) gives p = 0.006 -- the
# sets are not equally hard, so comparing across them confounds route difficulty
# with configuration. These 80 come from one draw (bigarena_poses_big.csv).
#
# Both arms run the SAME 80 routes back to back: paired in route and in time.
#
# What 80 can and cannot show. At an 8% contact rate the 95% interval is about
# 3-16%, so this run answers "did anything get much worse" and compares the
# clearance distribution and arrival time -- both of which use every control
# cycle and are precise at this n. It cannot establish that one margin scheme is
# safer than the other; that needs several hundred trials per arm.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
LOCKFILE=/tmp/omnibot_dynamic.pid
if [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE" 2>/dev/null)" 2>/dev/null; then
    echo "ERROR: a batch is already running."; exit 1
fi
CSV=evaluation/results/omnibot_dynamic_gmpc_scan.csv
P=evaluation/results/bigarena_poses_big.csv

run_one () {                     # $1 outdir  $2.. env
    local out="$1"; shift
    mkdir -p "evaluation/results/$out"; rm -rf "evaluation/bags/archive_$out"
    echo "[$(date +%T)] === $out ==="
    env "$@" POSES_CSV="$PWD/$P" BIGARENA=1 TRAJ=bigarena_traffic GUI=0 \
        PLANNER_SCAN=/scan_filtered PLAN_BLEND=0.0 \
        CBF_ALPHA=0.5 CBF_PRUNE_RANGE=1.2 CBF_FAR_STRIDE=2 \
        ./evaluation/run_omnibot_dynamic.sh 80 250 0 0 gmpc_scan \
        > "evaluation/logs/${out}.log" 2>&1 &
    local bpid=$!
    ./evaluation/guard_first_trial.sh "$CSV" || { wait $bpid 2>/dev/null; return 1; }
    wait $bpid
    cp "$CSV" "evaluation/results/$out/batch.csv"
    local ad="evaluation/bags/archive_$out"; mkdir -p "$ad"
    for d in evaluation/bags/gmpc_cbf__scan_seed*; do
        case "$d" in *__prev_*) continue;; esac
        [ -f "$d/metadata.yaml" ] || continue
        cp -r "$d" "$ad/$(basename "$d")" 2>/dev/null
    done
    cp "$CSV" "$ad/results.csv"
    echo "[$(date +%T)] $out done ($(ls -d "$ad"/*_seed* 2>/dev/null|wc -l) bags)"
}

run_one new_fixed   MARGIN_MODE=fixed   CBF_SAFE_MARGIN=0.60
run_one new_derived MARGIN_MODE=derived
echo "[$(date +%T)] new-routes sweep complete"
