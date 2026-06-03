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

# --------------------------------------------------------------------- source ROS
# MUST happen *before* `set -u` because /opt/ros/jazzy/setup.bash references
# AMENT_TRACE_SETUP_FILES without first defaulting it, which trips nounset.
#
# Sub-shells launched by nohup / cron don't inherit a sourced workspace; we
# must explicitly bring in /opt/ros + this project's install tree, otherwise
# `ros2 launch ammr_bringup ...` cannot locate the packages and Gazebo
# silently never starts (root cause of the first smoke-test failure).
WS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1091
source "${WS_ROOT}/install/setup.bash"

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
echo "[$(date +%T)] [1/7] launching Gazebo headless ..."
ros2 launch ammr_bringup gazebo_dynamic.launch.py gui:=false \
    >> "$LOG_FILE" 2>&1 < /dev/null &
PIDS+=( $! )
sleep 20  # Gazebo + bridges + spawn robot take ~15s; extra cushion

# --------------------------------------------------------------------- 2. Controller stack
echo "[$(date +%T)] [2/7] launching $METHOD stack ($STACK_LAUNCH) ..."
ros2 launch $STACK_LAUNCH \
    >> "$LOG_FILE" 2>&1 < /dev/null &
PIDS+=( $! )
# Give AMCL ≥20s of /scan to settle before sending the goal — early goals
# trigger plans from a pre-convergence pose, then AMCL jumps mid-run and
# breaks the global plan. (See smoke test where AMCL jumped 12 m at t=17s.)
sleep 20

# --------------------------------------------------------------------- 3. Reset AMCL (best-effort)
# AMCL may not have any subscriber on /initialpose yet (lifecycle takes a
# while). Wrap in timeout so the trial doesn't hang forever; if AMCL is
# late, robot still starts at (0,0) in Gazebo and AMCL will self-converge.
echo "[$(date +%T)] [3/7] resetting AMCL pose (best-effort, 8s timeout) ..."
timeout 8 ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped \
    "{ header: { frame_id: 'map' }, pose: { pose: { position: { x: 0.0, y: 0.0, z: 0.0 }, orientation: { w: 1.0 } } } }" \
    >> "$LOG_FILE" 2>&1 || echo "[$(date +%T)] [3/7] initialpose pub timed out (continuing)"
sleep 3

# --------------------------------------------------------------------- 4. Recording
echo "[$(date +%T)] [4/7] starting rosbag2 recording (timeout ${DURATION}s) ..."
"${HERE}/record.sh" "$METHOD" "$RUN_TAG" "$DURATION" \
    >> "$LOG_FILE" 2>&1 < /dev/null &
REC_PID=$!
PIDS+=( $REC_PID )
sleep 3

# --------------------------------------------------------------------- 5. Publish goal
echo "[$(date +%T)] [5/7] publishing goal (${GOAL_X}, ${GOAL_Y}) — 15s timeout ..."
timeout 15 ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped \
    "{ header: { frame_id: 'map' }, pose: { position: { x: $GOAL_X, y: $GOAL_Y, z: 0.0 }, orientation: { w: 1.0 } } }" \
    >> "$LOG_FILE" 2>&1 || echo "[$(date +%T)] [5/7] goal_pose pub timed out (continuing)"

# --------------------------------------------------------------------- 6. Goal watcher (race)
# Watcher exits 0 the moment robot enters goal tolerance in MAP frame; we
# then SIGINT the recorder so the bag is flushed cleanly instead of waiting
# out the full DURATION budget.
echo "[$(date +%T)] [6/7] starting goal watcher (tol=0.25 m) ..."
python3 "${HERE}/goal_watcher.py" \
    --goal-x "$GOAL_X" --goal-y "$GOAL_Y" \
    --tol 0.25 --timeout "$DURATION" \
    >> "$LOG_FILE" 2>&1 < /dev/null &
WATCH_PID=$!
PIDS+=( $WATCH_PID )

# --------------------------------------------------------------------- 7. Wait — whichever ends first
echo "[$(date +%T)] [7/7] racing record vs goal_watcher ..."
# Bash 4.3+: -n waits for ANY listed PID.
wait -n $REC_PID $WATCH_PID 2>/dev/null || true

# If watcher won (goal reached), tell recorder to flush and exit early.
if kill -0 $WATCH_PID 2>/dev/null; then
    # watcher still alive → recorder must have ended first (timeout / crash)
    echo "[$(date +%T)] recorder ended first (timeout or error) — stopping watcher"
    kill -INT $WATCH_PID 2>/dev/null || true
else
    echo "[$(date +%T)] watcher signalled GOAL — flushing recorder"
    kill -INT $REC_PID 2>/dev/null || true
fi

# Make sure both have actually terminated before we tear the stack down.
wait $REC_PID   2>/dev/null || true
wait $WATCH_PID 2>/dev/null || true

echo "[$(date +%T)] === trial done: $METHOD seed=$SEED ==="
echo "[$(date +%T)]     bag : ${HERE}/bags/${METHOD}__${RUN_TAG}"
echo "[$(date +%T)]     log : $LOG_FILE"
# cleanup runs from trap
