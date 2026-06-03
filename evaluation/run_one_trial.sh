#!/usr/bin/env bash
# Run a single benchmark trial (one method, one seed) end-to-end.
#
# Steps (all in this single script, no manual intervention):
#   1.  Launch Gazebo dynamic world headless         (gz sim -s)
#   2.  Launch the method-specific Nav2 stack
#   3.  Reset AMCL pose to (0,0,0) via /initialpose
#   4.  Start rosbag2 recording                       (record.sh)
#   5.  Publish goal pose                             (17,17)
#   6.  Wait DURATION seconds (or until record exits)
#   7.  Tear down every ROS / Gazebo process so the next trial starts clean
#
# Usage:
#   ./run_one_trial.sh METHOD SEED [DURATION_S]
#
# Example:
#   ./run_one_trial.sh gmpc_cbf 0 250
#
# Output:
#   bags/<METHOD>__seed<SEED>/
#   logs/<METHOD>__seed<SEED>.log
#
# Notes:
#   - SEED is currently only used for the bag tag (deterministic phase
#     randomisation of obstacles is TODO).  Different seeds still yield
#     different trajectories because Gazebo + MPPI have stochastic timing.
#   - The script is designed so multiple invocations DO NOT bleed state
#     into one another: every ROS node is killed before exit.

set -u   # don't  set -e  -- we want cleanup to run even after failures.

# --------------------------------------------------------------------- args
METHOD="${1:-}"
SEED="${2:-}"
DURATION="${3:-250}"

if [[ -z "$METHOD" || -z "$SEED" ]]; then
    echo "usage: $0 METHOD SEED [DURATION_S]"
    echo "  METHOD = rpp | mppi | gmpc | gmpc_cbf"
    exit 1
fi

case "$METHOD" in
    rpp)      STACK_LAUNCH="ammr_navigation nav2.launch.py"               ;;
    mppi)     STACK_LAUNCH="ammr_navigation nav2_omni_mppi.launch.py"     ;;
    gmpc)     STACK_LAUNCH="ammr_wholebody_mpc gmpc_nav2.launch.py"       ;;
    gmpc_cbf) STACK_LAUNCH="ammr_wholebody_mpc gmpc_nav2_cbf.launch.py"   ;;
    *) echo "ERROR: METHOD must be rpp | mppi | gmpc | gmpc_cbf"; exit 1  ;;
esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_TAG="seed${SEED}"
LOG_DIR="${HERE}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${METHOD}__${RUN_TAG}.log"

# Goal location must match the analyze.py default plan goal.
GOAL_X="17.0"
GOAL_Y="17.0"

# Aggregated PID list for the cleanup trap.
PIDS=()

# --------------------------------------------------------------------- cleanup
cleanup() {
    echo "[$(date +%T)] [trial] cleanup ..."
    for pid in "${PIDS[@]}"; do
        kill -INT  "$pid" 2>/dev/null || true
    done
    sleep 3
    for pid in "${PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 2
    # Belt-and-braces:  ros2 launch spawns many children that don't always
    # die when the parent gets SIGTERM, so we also pattern-kill.
    pkill -INT  -f 'gz sim'        2>/dev/null || true
    pkill -INT  -f 'ros2 launch'   2>/dev/null || true
    pkill -INT  -f 'ros2 bag'      2>/dev/null || true
    sleep 3
    pkill -KILL -f 'gz sim'        2>/dev/null || true
    pkill -KILL -f 'ros2 launch'   2>/dev/null || true
    pkill -KILL -f 'ros2 bag'      2>/dev/null || true
    pkill -KILL -f 'parameter_bridge' 2>/dev/null || true
    pkill -KILL -f 'gmpc_node'     2>/dev/null || true
    pkill -KILL -f 'controller_server' 2>/dev/null || true
    sleep 2
    echo "[$(date +%T)] [trial] cleanup done"
}
trap cleanup EXIT INT TERM

echo "[$(date +%T)] === trial start: METHOD=$METHOD SEED=$SEED DURATION=${DURATION}s ==="
echo "[$(date +%T)] log: $LOG_FILE"

# --------------------------------------------------------------------- 1. Gazebo
echo "[$(date +%T)] [1/6] launching Gazebo headless ..."
ros2 launch ammr_bringup gazebo_dynamic.launch.py gui:=false \
    >> "$LOG_FILE" 2>&1 < /dev/null &
PIDS+=( $! )
sleep 20  # Gazebo + bridges + spawn robot take ~15s; extra cushion

# --------------------------------------------------------------------- 2. Controller stack
echo "[$(date +%T)] [2/6] launching $METHOD stack ($STACK_LAUNCH) ..."
ros2 launch $STACK_LAUNCH \
    >> "$LOG_FILE" 2>&1 < /dev/null &
PIDS+=( $! )
sleep 12

# --------------------------------------------------------------------- 3. Reset AMCL
echo "[$(date +%T)] [3/6] resetting AMCL pose to (0,0,0) ..."
ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped \
    "{ header: { frame_id: 'map' }, pose: { pose: { position: { x: 0.0, y: 0.0, z: 0.0 }, orientation: { w: 1.0 } } } }" \
    >> "$LOG_FILE" 2>&1 || true
sleep 3

# --------------------------------------------------------------------- 4. Recording
echo "[$(date +%T)] [4/6] starting rosbag2 recording for ${DURATION}s ..."
"${HERE}/record.sh" "$METHOD" "$RUN_TAG" "$DURATION" \
    >> "$LOG_FILE" 2>&1 < /dev/null &
REC_PID=$!
PIDS+=( $REC_PID )
sleep 3

# --------------------------------------------------------------------- 5. Publish goal
echo "[$(date +%T)] [5/6] publishing goal (${GOAL_X}, ${GOAL_Y}) ..."
ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped \
    "{ header: { frame_id: 'map' }, pose: { position: { x: $GOAL_X, y: $GOAL_Y, z: 0.0 }, orientation: { w: 1.0 } } }" \
    >> "$LOG_FILE" 2>&1 || true

# --------------------------------------------------------------------- 6. Wait
echo "[$(date +%T)] [6/6] waiting for recording to finish ..."
wait $REC_PID 2>/dev/null || true

echo "[$(date +%T)] === trial done: $METHOD seed=$SEED ==="
echo "[$(date +%T)]     bag : ${HERE}/bags/${METHOD}__${RUN_TAG}"
echo "[$(date +%T)]     log : $LOG_FILE"
# cleanup runs from trap
