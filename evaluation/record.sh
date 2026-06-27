#!/usr/bin/env bash
# Standardised rosbag recording wrapper for the GMPC vs MPPI vs RPP benchmark.
# Records only the topics analyze.py needs, so bags stay small.
#
# Usage:
#   ./record.sh METHOD RUN_TAG [DURATION_S]
#
# Example:
#   ./record.sh gmpc seed42_run1 60
#
# Output:
#   bags/<METHOD>__<RUN_TAG>/   (rosbag2 directory format)
set -e

if [[ $# -lt 2 ]]; then
    echo "usage: $0 METHOD RUN_TAG [DURATION_S]" >&2
    echo "  METHOD = rpp | mppi | gmpc | gmpc_cbf" >&2
    exit 1
fi

METHOD="$1"
RUN_TAG="$2"
DURATION="${3:-60}"

case "$METHOD" in
    rpp|mppi|gmpc|gmpc_cbf) ;;
    *) echo "ERROR: METHOD must be one of rpp | mppi | gmpc | gmpc_cbf"; exit 1 ;;
esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${HERE}/bags/${METHOD}__${RUN_TAG}"

if [[ -e "$OUT_DIR" ]]; then
    echo "ERROR: $OUT_DIR exists already — choose a different RUN_TAG or remove it"
    exit 1
fi

echo "Recording for $DURATION s into $OUT_DIR"
echo "Topics: /odom /cmd_vel /cmd_vel_nav /plan /goal_pose /tf /tf_static \\"
echo "        /gmpc/solve_time_ms /gmpc/obstacles /gmpc/min_h \\"
echo "        /model/dyn_obs_{0,1,2}/pose  (ground-truth obstacle poses, all methods)"
echo "Press Ctrl+C earlier if the robot reaches goal sooner."

# Forward SIGINT (Ctrl+C) to ros2 bag record so it closes the bag cleanly;
# escalate to SIGKILL 5s after the duration if it refuses to die.
#   --foreground   : pass SIGINT from this shell straight through
#   --signal=INT   : after DURATION, send SIGINT (not SIGTERM) — lets rosbag2 flush
#   --kill-after=5 : if rosbag2 still alive 5s later, SIGKILL it
timeout --foreground --signal=INT --kill-after=5 "${DURATION}s" \
    ros2 bag record \
        -o "$OUT_DIR" \
        --topics /odom /cmd_vel /cmd_vel_nav /plan /goal_pose /tf /tf_static \
                 /gmpc/solve_time_ms /gmpc/obstacles /gmpc/min_h \
                 /model/dyn_obs_0/pose /model/dyn_obs_1/pose /model/dyn_obs_2/pose \
                 /model/dyn_obs_5/pose \
    < /dev/null || true

echo "Done. Now analyse with:"
echo "  python3 ${HERE}/analyze.py $OUT_DIR --method $METHOD --run $RUN_TAG"
