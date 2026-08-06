#!/usr/bin/env bash
# Derived margins over pose sets A and C, 15 trials each.
#
# MARGIN_MODE=derived replaces the tuned 0.38 / 0.60 with keep-outs sized from
# what each obstacle has to absorb:
#
#   static   loc_err + v^2/(2a) + v dt
#   dynamic  the same, plus v_obs * percep_lag + obs_pos_err
#
# At 0.35 m/s that is 0.139 static and 0.243 dynamic; at rest it collapses to
# the 0.08 floor. The fixed values were never derived, and the measurements say
# they are 2.7-4x larger than the physics needs.
#
# One watched trial on C seed22 is not evidence -- obstacle phase differs every
# run (the 1-5 s random goal delay), and the two fixed-margin runs of that same
# route differed by 0.29 m in dynamic clearance. This batch is the check.
#
# A and C because they are the extremes of the existing data: A had 8 negative
# clearances in 30, C had 13 in 30, B only 2. Anything that helps should show on
# C first.
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
    env MARGIN_MODE=derived \
        POSES_CSV="$PWD/$poses" BIGARENA=1 TRAJ=bigarena_traffic GUI=0 \
        PLANNER_SCAN=/scan_filtered PLAN_BLEND=0.0 \
        CBF_ALPHA=0.5 CBF_PRUNE_RANGE=1.2 CBF_FAR_STRIDE=2 \
        ./evaluation/run_omnibot_dynamic.sh 15 250 0 0 gmpc_scan \
        > "evaluation/logs/${out}.log" 2>&1 &
    local bpid=$!
    ./evaluation/guard_first_trial.sh "$CSV" || { wait $bpid 2>/dev/null; return 1; }
    wait $bpid
    cp "$CSV" "evaluation/results/$out/batch.csv"
    local ad="evaluation/bags/archive_$out"; mkdir -p "$ad"
    for d in evaluation/bags/gmpc_cbf__scan_seed*; do
        case "$d" in *__prev_*) continue;; esac
        [ -f "$d/metadata.yaml" ] || continue
        s=$(basename "$d" | sed 's/.*seed//'); [ "$s" -le 15 ] || continue
        cp -r "$d" "$ad/$(basename "$d")" 2>/dev/null
    done
    cp "$CSV" "$ad/results.csv"
    echo "[$(date +%T)] $out done ($(ls -d "$ad"/*_seed* 2>/dev/null | wc -l) bags)"
}

run_one derm_A evaluation/results/bigarena_poses.csv
run_one derm_C evaluation/results/bigarena_poses_c.csv
echo "[$(date +%T)] derived-margin sweep complete"
