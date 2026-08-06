#!/usr/bin/env bash
# The locked configuration over 120 routes (pose sets A-D), for the thesis.
#
# Everything is now in the config files, not env vars: vx_min -0.35, accel
# 1.5/1.0/2.0, replan 1.0 s, EKF rejection gate off. The env vars below only
# restate values the files already hold, so a log line always shows what ran.
#
# What this batch is for: stage 4 gave 0 contacts in 30 trials, whose 95%
# interval is still 0-11%. 120 trials narrows that to 0-3% if the count holds,
# which is the difference between "we saw none" and a number worth printing.
#
# Nothing is tuned here. If contacts appear, they are the result -- the point of
# a confirmation run is that it can fail.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

LOCKFILE=/tmp/omnibot_dynamic.pid
if [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE" 2>/dev/null)" 2>/dev/null; then
    echo "ERROR: a batch is already running (pid $(cat "$LOCKFILE"))."; exit 1
fi
CSV=evaluation/results/omnibot_dynamic_gmpc_scan.csv

run_one () {                     # $1 outdir  $2 poses file
    local out="$1" poses="$2"
    mkdir -p "evaluation/results/$out"; rm -rf "evaluation/bags/archive_$out"
    echo "[$(date +%T)] === $out ==="
    env POSES_CSV="$PWD/$poses" BIGARENA=1 TRAJ=bigarena_traffic GUI=0 \
        PLANNER_SCAN=/scan_filtered PLAN_BLEND=0.0 \
        CBF_ALPHA=0.5 CBF_SAFE_MARGIN=0.60 \
        CBF_PRUNE_RANGE=1.2 CBF_FAR_STRIDE=2 \
        ./evaluation/run_omnibot_dynamic.sh 30 250 0 0 gmpc_scan \
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
    local n c
    n=$(( $(wc -l < "$CSV") - 1 ))
    c=$(awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="min_clearance_m") k=i; next}
                 $k+0 < 0 {n++} END{print n+0}' "$CSV")
    echo "[$(date +%T)] $out done: $n trials, $c with negative clearance"
}

run_one f120_A evaluation/results/bigarena_poses.csv
run_one f120_B evaluation/results/bigarena_poses_b.csv
run_one f120_C evaluation/results/bigarena_poses_c.csv
run_one f120_D evaluation/results/bigarena_poses_d.csv
echo "[$(date +%T)] 120-route confirmation complete"
