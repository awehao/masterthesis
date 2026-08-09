#!/usr/bin/env bash
# Minimal ablation: the instantaneous-speed gate, and nothing else.
#
# T0-T2 showed the fragmentation hypothesis was wrong. A track near the mover
# existed in 100% of cycles with a median age of 218 and a KF speed matching
# ground truth (0.096 vs 0.097 m/s) -- nothing was lost. It was only published
# in 29% of them, because dyn_obs_5's configured speed IS 0.10 m/s and
# min_track_speed is 0.10, so measurement noise decided each cycle whether the
# object counted as a mover at all. dyn_obs_2 sits on the same threshold.
#
# The net-displacement gate already does this job and does it better: 0.10 m/s
# over the 2 s window is 0.20 m of travel against its 0.05 m/s threshold, and
# unlike an instantaneous reading it cannot be fooled by centroid jitter. The
# instantaneous gate is the redundant one.
#
#   S0  min_track_speed 0.10, everything else stock  (repeat of T0, control)
#   S1  min_track_speed 0.05, everything else stock
#
# Deliberately NOT combined with the association/fragment/coast work: those are
# switched off here, so anything this moves is attributable to the threshold.
# 0.05 rather than something just under 0.10 because the estimate straddles the
# threshold, and a bound that close would still let noise decide.
#
# Primary metric is the publish rate for the mover, NOT the contact count. If
# the publish rate does not jump, ten clean runs would prove nothing.
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
exec 2>&1

LOCKFILE=/tmp/omnibot_dynamic.pid
if [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE" 2>/dev/null)" 2>/dev/null; then
    echo "ERROR: a batch is already running."; exit 1
fi
CSV=evaluation/results/omnibot_dynamic_gmpc_scan.csv
REPS=10
say () { echo "[$(date +%H:%M:%S)] $*"; }

P27=evaluation/results/poses_seed27.csv
head -1 evaluation/results/bigarena_poses_big.csv > "$P27"
awk -F, -v OFS=, 'NR>1 && $1==27 {$1=1; print}' \
    evaluation/results/bigarena_poses_big.csv >> "$P27"
grep -q '^1,' "$P27" || { say "FATAL: route row not relabelled"; exit 1; }
say "route: $(sed -n 2p "$P27")"

run_arm () {
    local out="$1"; shift
    mkdir -p "evaluation/results/$out"; rm -rf "evaluation/bags/archive_$out"
    rm -f "$CSV" "evaluation/logs/${out}.log"
    say "=== $out ==="
    for i in $(seq 1 "$REPS"); do
        env "$@" MASK_HW=10.0 POSES_CSV="$PWD/$P27" \
            BIGARENA=1 TRAJ=bigarena_traffic GUI=0 \
            MARGIN_MODE=fixed CBF_SAFE_MARGIN=0.60 \
            PLANNER_SCAN=/scan_filtered PLAN_BLEND=0.0 \
            CBF_ALPHA=0.5 CBF_PRUNE_RANGE=1.2 CBF_FAR_STRIDE=2 \
            POSE_SOURCE=odom \
            ./evaluation/run_omnibot_dynamic.sh 1 250 0 0 gmpc_scan \
            >> "evaluation/logs/${out}.log" 2>&1
        sleep 15
        local ad="evaluation/bags/archive_$out"; mkdir -p "$ad"
        [ -f evaluation/bags/gmpc_cbf__scan_seed1/metadata.yaml ] && {
            rm -rf "$ad/rep$i"
            cp -r evaluation/bags/gmpc_cbf__scan_seed1 "$ad/rep$i" 2>/dev/null; }
    done
    say "$out done"
}

run_arm S0 MIN_TRACK_SPEED=0.10 RELEASE_TRACK_SPEED=0.10
run_arm S1 MIN_TRACK_SPEED=0.05 RELEASE_TRACK_SPEED=0.05

say "analysing"
python3 evaluation/summarise_speedgate.py > evaluation/results/SPEEDGATE_SUMMARY.md 2>&1
say "ALL DONE -> evaluation/results/SPEEDGATE_SUMMARY.md"
