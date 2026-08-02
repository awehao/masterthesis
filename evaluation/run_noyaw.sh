#!/usr/bin/env bash
# Batch M -- first batch with no heading reference anywhere in the stack.
#
# The yaw parameters are gone, not switched off: build_reference_window pins
# every horizon sample to the robot's current heading, so xi_ref's yaw rate is
# identically zero and nothing rotational is fed forward. Q_yaw is a hard zero.
# Batch K ran with the heading WEIGHT at zero and still span a median 2387 deg
# per trial, because the solver returns u = xi_ref[0] + delta and the reference
# carried the path direction's yaw rate regardless of what it cost.
#
# Otherwise identical to K (same poses, same alpha, same margin, same pruning,
# PLAN_BLEND still 1.0), so the difference between K and M is the heading
# reference alone. PLAN_BLEND=0 is batch N's question, not this one's.
#
# Rotation is not forbidden: wz is still a free input the QP may use.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

# The lock run_omnibot_dynamic.sh maintains -- NOT pgrep -f, which also matches
# any shell whose command line merely mentions the script name.
LOCKFILE=/tmp/omnibot_dynamic.pid
if [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE" 2>/dev/null)" 2>/dev/null; then
    echo "ERROR: a batch is already running (pid $(cat "$LOCKFILE"))." >&2
    exit 1
fi

OUT=evaluation/results/noyawM
mkdir -p "$OUT"

# run_omnibot_dynamic.sh does `rm -f "$OUT_CSV"` on start; archive first.
PREV=evaluation/results/omnibot_dynamic_gmpc_scan.csv
if [ -s "$PREV" ]; then
    cp "$PREV" "evaluation/results/archived_$(date +%Y%m%d_%H%M%S).csv"
fi

POSES_CSV="$HERE/results/bigarena_poses.csv" \
BIGARENA=1 TRAJ=bigarena_traffic GUI=0 \
PLANNER_SCAN=/scan_filtered PLAN_BLEND=1.0 \
CBF_ALPHA=0.5 CBF_SAFE_MARGIN=0.60 \
CBF_PRUNE_RANGE=1.2 CBF_FAR_STRIDE=2 \
    ./evaluation/run_omnibot_dynamic.sh 30 250 0 0 gmpc_scan

cp evaluation/results/omnibot_dynamic_gmpc_scan.csv "$OUT/batch.csv"
echo "batch M done -> $OUT/batch.csv"
