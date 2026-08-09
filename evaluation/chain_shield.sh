#!/usr/bin/env bash
# U2: speed-gate fix PLUS the raw-scan shield, on seed27.
#
# U0 and U1 already exist as the speed-gate ablation's S0 and S1 -- same route,
# same ten replays, same POSE_SOURCE=odom, no shield, differing only in
# min_track_speed -- so only the third arm has to run. Re-running the first two
# would cost 1.3 h and change nothing.
#
#   U0 = S0   min_track_speed 0.10, no shield   (baseline)
#   U1 = S1   min_track_speed 0.05, no shield   (classification-chain repair)
#   U2        min_track_speed 0.05, SHIELD=1    (independent safety layer)
#
# Splitting it this way is what lets the two contributions be attributed
# separately. S0->S1 showed the gate was binding but that lowering it only
# takes the publish rate from 23% to 32%, because the KF underestimates this
# body's speed by half (0.049 estimated against 0.097 true) -- the centroid of a
# 1.6 m box slides along its visible face as the bearing changes. No threshold
# on that estimate can be reliable, which is the argument for a layer that does
# not need to know whether anything is moving.
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

out=U2
mkdir -p "evaluation/results/$out"; rm -rf "evaluation/bags/archive_$out"
rm -f "$CSV" "evaluation/logs/${out}.log"
say "=== $out : speed gate 0.05 + raw-scan shield ==="
for i in $(seq 1 "$REPS"); do
    env SHIELD=1 MIN_TRACK_SPEED=0.05 RELEASE_TRACK_SPEED=0.05 \
        MASK_HW=10.0 POSES_CSV="$PWD/$P27" \
        BIGARENA=1 TRAJ=bigarena_traffic GUI=0 \
        MARGIN_MODE=fixed CBF_SAFE_MARGIN=0.60 \
        PLANNER_SCAN=/scan_filtered PLAN_BLEND=0.0 \
        CBF_ALPHA=0.5 CBF_PRUNE_RANGE=1.2 CBF_FAR_STRIDE=2 \
        POSE_SOURCE=odom \
        ./evaluation/run_omnibot_dynamic.sh 1 250 0 0 gmpc_scan \
        >> "evaluation/logs/${out}.log" 2>&1
    sleep 15
    ad="evaluation/bags/archive_$out"; mkdir -p "$ad"
    [ -f evaluation/bags/gmpc_cbf__scan_seed1/metadata.yaml ] && {
        rm -rf "$ad/rep$i"
        cp -r evaluation/bags/gmpc_cbf__scan_seed1 "$ad/rep$i" 2>/dev/null; }
done
say "$out done"

say "analysing"
python3 evaluation/summarise_shield.py > evaluation/results/SHIELD_SUMMARY.md 2>&1
say "ALL DONE -> evaluation/results/SHIELD_SUMMARY.md"
