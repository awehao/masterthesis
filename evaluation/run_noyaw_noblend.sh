#!/usr/bin/env bash
# Batch N -- batch M plus PLAN_BLEND=0 (adopt each new plan instantly).
#
# The 1 s cross-fade was added because a new /plan can step the reference in
# front of the robot, and the controller then chases a step at 0.96 of a_max.
# Half of what motivated it is now structurally impossible: the measurement
# behind it was position jumps (median 0.05 m, p90 0.44 m, max 1.82 m) AND
# reference-heading jumps (p90 26 deg, max 63 deg). With no heading reference
# the second term is gone; the first is not. Hence: measure, do not assume.
#
# Otherwise identical to M, so the difference between M and N is the blend
# alone.
#
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

OUT=evaluation/results/noblendN
mkdir -p "$OUT"

# run_omnibot_dynamic.sh does `rm -f "$OUT_CSV"` on start; archive first.
PREV=evaluation/results/omnibot_dynamic_gmpc_scan.csv
if [ -s "$PREV" ]; then
    cp "$PREV" "evaluation/results/archived_$(date +%Y%m%d_%H%M%S).csv"
fi

POSES_CSV="$HERE/results/bigarena_poses.csv" \
BIGARENA=1 TRAJ=bigarena_traffic GUI=0 \
PLANNER_SCAN=/scan_filtered PLAN_BLEND=0.0 \
CBF_ALPHA=0.5 CBF_SAFE_MARGIN=0.60 \
CBF_PRUNE_RANGE=1.2 CBF_FAR_STRIDE=2 \
    ./evaluation/run_omnibot_dynamic.sh 30 250 0 0 gmpc_scan

cp evaluation/results/omnibot_dynamic_gmpc_scan.csv "$OUT/batch.csv"
echo "batch N done -> $OUT/batch.csv"
