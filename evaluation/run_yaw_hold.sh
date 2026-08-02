#!/usr/bin/env bash
# Batch M -- no heading reference at all (YAW_HOLD=1).
#
# Identical to batch K apart from YAW_HOLD, so the comparison isolates the
# heading reference. K ran with Q_yaw=0 already and still span a median of
# 2387 deg per trial, because the solver returns u = xi_ref[0] + delta and
# xi_ref carries whatever angular rate the path direction implies. YAW_HOLD
# pins every reference sample to the robot's current heading, so that term is
# exactly zero. Rotation stays available to the QP; it is never demanded.
#
# YAW_LOOKAHEAD is left at K's 1.2 deliberately: with YAW_HOLD on it has no
# effect (the look-ahead chord is never consulted), and keeping the line
# identical to K makes the diff between the two batches a single variable.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

# Use the lock run_omnibot_dynamic.sh maintains, NOT pgrep -f: any shell whose
# command line merely mentions the script name matches pgrep and looks like a
# running batch.
LOCKFILE=/tmp/omnibot_dynamic.pid
if [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE" 2>/dev/null)" 2>/dev/null; then
    echo "ERROR: a batch is already running (pid $(cat "$LOCKFILE"))." >&2
    exit 1
fi

OUT=evaluation/results/yawholdM
mkdir -p "$OUT"

# run_omnibot_dynamic.sh does `rm -f "$OUT_CSV"` on start, so archive first.
PREV=evaluation/results/omnibot_dynamic_gmpc_scan.csv
if [ -s "$PREV" ]; then
    cp "$PREV" "evaluation/results/archived_$(date +%Y%m%d_%H%M%S).csv"
fi

POSES_CSV="$HERE/results/bigarena_poses.csv" \
BIGARENA=1 TRAJ=bigarena_traffic GUI=0 \
PLANNER_SCAN=/scan_filtered PLAN_BLEND=1.0 \
CBF_ALPHA=0.5 \
YAW_HOLD=1 \
YAW_LOOKAHEAD=1.2 CBF_SAFE_MARGIN=0.60 \
CBF_PRUNE_RANGE=1.2 CBF_FAR_STRIDE=2 \
    ./evaluation/run_omnibot_dynamic.sh 30 250 0 0 gmpc_scan

cp evaluation/results/omnibot_dynamic_gmpc_scan.csv "$OUT/batch.csv"
echo "batch M done -> $OUT/batch.csv"
