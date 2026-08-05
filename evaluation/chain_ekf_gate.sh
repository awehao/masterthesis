#!/usr/bin/env bash
# Three EKF rejection-gate settings, 10 trials each.
#
# The EKF, not AMCL, is what loses the robot. On the GUI trial AMCL stayed
# within 0.118 m for the whole run and kept publishing 13-20 poses per 10 s,
# while the EKF wandered to 2.488 m and snapped back 25 s later. Across 89
# trials the EKF exceeded 0.5 m of error in 13 of them and 8 m in three.
#
# pose0_rejection_threshold is a Mahalanobis gate on /amcl_pose. At 2.5 it locks
# the filter out of its own corrections: drift makes each correct AMCL pose look
# like an outlier, so it is discarded and the drift grows. The gate was added to
# reject AMCL jumps near unknown obstacles -- a problem the measurements do not
# show (AMCL's worst error is 0.118 m).
#
# Judged on EKF ERROR against gz ground truth, not on contacts: every control
# cycle is a sample, so 10 trials give tens of thousands of them, while contacts
# are far too rare to move at this n.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

LOCKFILE=/tmp/omnibot_dynamic.pid
if [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE" 2>/dev/null)" 2>/dev/null; then
    echo "ERROR: a batch is already running (pid $(cat "$LOCKFILE"))."; exit 1
fi
CSV=evaluation/results/omnibot_dynamic_gmpc_scan.csv

run_one () {                     # $1 label  $2 outdir  $3 EKF_REJECT ('' = file default)
    local label="$1" out="$2" rej="$3"
    mkdir -p "evaluation/results/$out"
    rm -rf "evaluation/bags/archive_$out"
    echo "[$(date +%T)] === $label ==="
    env ${rej:+EKF_REJECT=$rej} \
        POSES_CSV="$PWD/evaluation/results/bigarena_poses.csv" \
        BIGARENA=1 TRAJ=bigarena_traffic GUI=0 \
        PLANNER_SCAN=/scan_filtered PLAN_BLEND=0.0 \
        CBF_ALPHA=0.5 CBF_SAFE_MARGIN=0.60 \
        CBF_PRUNE_RANGE=1.2 CBF_FAR_STRIDE=2 \
        ./evaluation/run_omnibot_dynamic.sh 10 250 0 0 gmpc_scan \
        > "evaluation/logs/${out}.log" 2>&1 &
    local bpid=$!
    ./evaluation/guard_first_trial.sh "$CSV" || { wait $bpid 2>/dev/null; return 1; }
    wait $bpid
    cp "$CSV" "evaluation/results/$out/batch.csv"
    local ad="evaluation/bags/archive_$out"; mkdir -p "$ad"
    for d in evaluation/bags/gmpc_cbf__scan_seed*; do
        case "$d" in *__prev_*) continue;; esac
        [ -f "$d/metadata.yaml" ] || continue
        s=$(basename "$d" | sed 's/.*seed//')
        [ "$s" -le 10 ] || continue          # this batch only ran seeds 1-10
        cp -r "$d" "$ad/$(basename "$d")" 2>/dev/null
    done
    cp "$CSV" "$ad/results.csv"
    echo "[$(date +%T)] $label done (bags $(ls -d "$ad"/*_seed* 2>/dev/null | wc -l))"
}

run_one "gate 2.5 (current)" ekf_g25  2.5
run_one "gate 10.0 (loose)"  ekf_g10  10.0
run_one "gate off"           ekf_goff 1e9
echo "[$(date +%T)] ekf gate sweep complete"
