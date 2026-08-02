#!/usr/bin/env bash
# Batch L -- alpha control for batch K.
#
# K changed two things at once: it capped mover speeds at 0.18 m/s AND ran
# alpha=0.5. Any improvement K shows cannot be attributed to either. L is K with
# ONE difference, alpha=1.5, on the same slowed scenario and the same 30
# start/goal pairs, so the alpha effect is isolated.
#
# Everything else below is byte-identical to the K command line. Do not "tidy"
# it -- the point is that only CBF_ALPHA differs.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

# Use the lock run_omnibot_dynamic.sh already maintains, NOT pgrep -f. Any shell
# whose command line merely mentions the script name -- a monitor, an editor, an
# agent's own check -- matches pgrep -f and looks like a running batch. This
# blocked two legitimate launches before the lockfile was used instead.
LOCKFILE=/tmp/omnibot_dynamic.pid
if [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE" 2>/dev/null)" 2>/dev/null; then
    echo "ERROR: a batch is already running. Refusing to start a second one." >&2
    echo "       (Two concurrent batches have contaminated results before.)" >&2
    exit 1
fi

OUT=evaluation/results/slowL
mkdir -p "$OUT"

# run_omnibot_dynamic.sh line 73 does `rm -f "$OUT_CSV"` -- it wipes the shared
# results file on start. Batch K writes to that same file, so archive whatever
# is there before it is destroyed. A previous batch has been lost this way.
PREV=evaluation/results/omnibot_dynamic_gmpc_scan.csv
if [ -s "$PREV" ]; then
    cp "$PREV" "evaluation/results/archived_$(date +%Y%m%d_%H%M%S).csv"
    echo "archived previous results -> evaluation/results/archived_*.csv"
fi

POSES_CSV="$HERE/results/bigarena_poses.csv" \
BIGARENA=1 TRAJ=bigarena_traffic GUI=0 \
PLANNER_SCAN=/scan_filtered PLAN_BLEND=1.0 \
CBF_ALPHA=1.5 \
YAW_LOOKAHEAD=1.2 CBF_SAFE_MARGIN=0.60 \
CBF_PRUNE_RANGE=1.2 CBF_FAR_STRIDE=2 \
    ./evaluation/run_omnibot_dynamic.sh 30 250 0 0 gmpc_scan

cp evaluation/results/omnibot_dynamic_gmpc_scan.csv "$OUT/batch.csv"
echo "batch L done -> $OUT/batch.csv"
