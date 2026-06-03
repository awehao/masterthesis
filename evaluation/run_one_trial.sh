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
# The orphan-node problem is real: when we kill `ros2 launch`, its launched
# executables (amcl, map_server, gz sim, parameter_bridge, ...) usually do
# NOT die because they are spawned without forming a process group with the
# launcher.  So we must enumerate every known executable and pattern-kill it.
# This list must cover *all four* baseline stacks (RPP / MPPI / GMPC / CBF)
# plus the gazebo_dynamic.launch.py children.
NODE_PAT='gz sim'
NODE_PAT+='|ros2 launch|ros2 bag|ros2 topic pub'
NODE_PAT+='|nav2_map_server|nav2_amcl|nav2_planner|nav2_controller'
NODE_PAT+='|nav2_smoother|nav2_behaviors|nav2_bt_navigator|nav2_waypoint_follower'
NODE_PAT+='|nav2_velocity_smoother|nav2_lifecycle_manager'
NODE_PAT+='|map_server|map_publisher|amcl'
NODE_PAT+='|planner_server|controller_server|behavior_server|bt_navigator'
NODE_PAT+='|waypoint_follower|velocity_smoother|smoother_server|lifecycle_manager'
NODE_PAT+='|goal_to_plan_relay|obstacle_aggregator|dynamic_obstacle_driver'
NODE_PAT+='|gmpc_node|scan_relay|omni_drive_controller'
NODE_PAT+='|parameter_bridge|robot_state_publisher|static_transform_publisher'
NODE_PAT+='|goal_watcher.py|record.sh'

cleanup() {
    echo "[$(date +%T)] [trial] cleanup ..."
    # Polite first: SIGINT the tracked PIDs and their direct children.
    for pid in "${PIDS[@]}"; do
        pkill -INT -P "$pid" 2>/dev/null || true   # kill children too
        kill  -INT     "$pid" 2>/dev/null || true
    done
    sleep 3
    pkill -INT -f "$NODE_PAT" 2>/dev/null || true
    sleep 3
    # Hard kill anything still standing.
    pkill -KILL -f "$NODE_PAT" 2>/dev/null || true
    sleep 1
    # The ros2 daemon caches stale node registrations; without this restart
    # the next trial sees ghost nodes from the previous run.
    ros2 daemon stop  > /dev/null 2>&1 || true
    sleep 1
    ros2 daemon start > /dev/null 2>&1 || true
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
# Race fix (was the dominant apparatus failure in batch_cbf_400s.log):
# `ros2 topic pub --once` publishes once and exits with default QoS
# DURABILITY=VOLATILE, so any subscriber that hasn't completed DDS
# discovery yet silently misses the message. We saw seeds 7/8/9 fail this
# way -- goal was published but goal_to_plan_relay never received it.
# Wait until /goal_pose has >= 2 subscribers (rosbag2_recorder + relay)
# before publishing, with a 30-second cap so we still degrade gracefully.
echo "[$(date +%T)] [5/7] waiting for /goal_pose subscribers (rosbag + relay) ..."
for i in $(seq 1 30); do
    count=$(ros2 topic info /goal_pose 2>/dev/null \
            | awk '/[Ss]ubscri.*[Cc]ount/ {print $NF; exit}')
    count=${count:-0}
    if [ "$count" -ge 2 ]; then
        echo "[$(date +%T)] [5/7]   /goal_pose subscribers=$count after ${i}s, ok to publish"
        break
    fi
    sleep 1
done
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
# Killing only $REC_PID (the bash wrapping record.sh) doesn't reliably
# propagate SIGINT to the inner `timeout` + `ros2 bag record` grandchild,
# which is why the previous smoke test's bag ran the full 200 s after the
# watcher fired at t=71 s. Pkill the descendants too.
if kill -0 $WATCH_PID 2>/dev/null; then
    # watcher still alive → recorder must have ended first (timeout / crash)
    echo "[$(date +%T)] recorder ended first (timeout or error) — stopping watcher"
    pkill -INT -P $WATCH_PID 2>/dev/null || true
    kill  -INT    $WATCH_PID 2>/dev/null || true
else
    echo "[$(date +%T)] watcher signalled GOAL — flushing recorder"
    pkill -INT -P $REC_PID 2>/dev/null || true
    kill  -INT    $REC_PID 2>/dev/null || true
fi

# Make sure both have actually terminated before we tear the stack down.
wait $REC_PID   2>/dev/null || true
wait $WATCH_PID 2>/dev/null || true

echo "[$(date +%T)] === trial done: $METHOD seed=$SEED ==="
echo "[$(date +%T)]     bag : ${HERE}/bags/${METHOD}__${RUN_TAG}"
echo "[$(date +%T)]     log : $LOG_FILE"
# cleanup runs from trap
