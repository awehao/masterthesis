#!/usr/bin/env bash
# Run a single benchmark trial in GAZEBO with ground-truth pose recorded.
#
# Same as run_one_trial.sh plus step 1b, which bridges the scene broadcaster's
# true model poses and relays the robot's onto /model/ammr_base/pose -- the
# same topic isaac_bench_sim.py publishes, so one analysis reads both. Nothing
# else differs; the bag name is prefixed gz_ so it never collides.
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
#   bags/gz_<METHOD>__seed<SEED>/
#   logs/gz_<METHOD>__seed<SEED>.log
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
METHOD_TAG="gz_${METHOD}"
# --------------------------------------------------------------------- scenario
# SCENARIO=v2 selects the reproducible obstacle schedule: position computed
# from simulation time with /case_start as phase zero, so the encounter no
# longer depends on process start order. v1 (default) is the historical
# feedback ping-pong. Bags are tagged with the version so the two are never
# mixed into one group.
SCENARIO="${SCENARIO:-v1}"
if [ "$SCENARIO" = "v2" ]; then
    export AMMR_TRAJ_FILE="${WS_ROOT}/src/ammr_bringup/config/dynamic_trajectories_v2.yaml"
    export AMMR_OBSTACLE_MODE="scheduled"
    SCEN_TAG="_v2"
else
    SCEN_TAG=""
fi
RUN_TAG="seed${SEED}${SCEN_TAG}"

LOG_DIR="${HERE}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${METHOD_TAG}__${RUN_TAG}.log"

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
NODE_PAT+='|goal_watcher.py|record.sh|gz_truth_relay'

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

# --------------------------------------------------------------------- 1b. Ground truth
# Gazebo never published a robot ground-truth pose, so /odom -- which is
# integrated from /cmd_vel -- had nothing to be checked against. The scene
# broadcaster already emits it; bridging one extra topic and relaying one model
# out of it changes no world, model, launch or controller.
# The scene broadcaster's dynamic_pose/info was tried first and does not work
# for this: ros_gz's Pose_V -> TFMessage converter takes frame names from the
# per-pose header data, which the broadcaster does not set, so every transform
# arrives with empty frame ids even though the raw gz message carries `name`.
# The model's own PosePublisher (added to ammr_base.urdf.xacro, output only)
# publishes gz.msgs.Pose on /model/ammr_base/pose, exactly as the dynamic
# obstacles already do, so one plain bridge line is enough.
echo "[$(date +%T)] [1b] bridging Gazebo robot ground truth ..."
ros2 run ros_gz_bridge parameter_bridge \
    "/model/ammr_base/pose@geometry_msgs/msg/PoseStamped[gz.msgs.Pose" \
    >> "$LOG_FILE" 2>&1 < /dev/null &
PIDS+=( $! )
sleep 3

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
"${HERE}/record.sh" "$METHOD_TAG" "$RUN_TAG" "$DURATION" \
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

# Phase zero for the scheduled obstacles. Published at the SAME point in both
# runners, immediately before the goal, so the scenario starts from the same
# instant regardless of how long the stack took to come up. Latched-style
# repetition covers a late subscriber; the driver resets phase on each message,
# so the last one before the goal is the one that counts.
if [ "$SCENARIO" = "v2" ]; then
    # Same DDS race the goal publication already documents: a fresh publisher
    # that sends a burst and exits can finish before the subscriber has been
    # discovered. The first attempt did exactly that -- rosbag recorded three
    # /case_start messages, the driver received none, and every obstacle sat at
    # its start pose for the whole run. Wait for the subscribers, then publish
    # over five seconds like the goal does.
    echo "[$(date +%T)] [5/7] waiting for /case_start subscribers ..."
    for i in $(seq 1 30); do
        cs=$(ros2 topic info /case_start 2>/dev/null \
             | awk '/[Ss]ubscri.*[Cc]ount/ {print $NF; exit}')
        cs=${cs:-0}
        if [ "$cs" -ge 2 ]; then
            echo "[$(date +%T)] [5/7]   /case_start subscribers=$cs after ${i}s"
            break
        fi
        sleep 1
    done
    # Publish ONCE. The driver resets phase on every /case_start, which is the
    # behaviour we want for a reset but makes the epoch ambiguous if the
    # message is repeated: a five-message burst reset the phase five times and
    # the effective zero was the last one, ~4 s after the first. Publish a
    # single message and then verify the driver actually adopted it, retrying
    # only if it did not.
    CS_OK=0
    for attempt in 1 2 3; do
        echo "[$(date +%T)] [5/7] publishing /case_start (attempt ${attempt}) ..."
        timeout 8 ros2 topic pub -t 1 /case_start std_msgs/msg/Empty "{}" \
            >> "$LOG_FILE" 2>&1 || true
        for j in 1 2 3 4 5 6 7 8; do
            ep=$(timeout 3 ros2 topic echo --once --field data \
                 /dynamic_obstacles/phase_epoch 2>/dev/null | head -1)
            case "$ep" in
                ""|nan|NaN|.nan) ;;
                *) echo "[$(date +%T)] [5/7]   相位零點已設定：${ep}"; CS_OK=1; break ;;
            esac
            sleep 1
        done
        [ "$CS_OK" -eq 1 ] && break
    done
    if [ "$CS_OK" -ne 1 ]; then
        echo "[$(date +%T)] [5/7] ERROR: driver 未接受 /case_start — 障礙物不會移動，中止"
        exit 1
    fi
fi
echo "[$(date +%T)] [5/7] publishing goal (${GOAL_X}, ${GOAL_Y}) — 5 times @ 1 Hz ..."
# Multi-publish: -t 5 -r 1 emits the message 5 times at 1 Hz over 5 seconds,
# which gives goal_to_plan_relay and rosbag2_recorder a 5-second window to
# complete their DDS subscription discovery and catch at least one of the
# emissions. Fixes the goal-drop race observed in seeds 4/5/7/8/9 where the
# single --once publish raced ahead of the late subscriber.
timeout 15 ros2 topic pub -t 5 -r 1 /goal_pose geometry_msgs/msg/PoseStamped \
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
echo "[$(date +%T)]     bag : ${HERE}/bags/${METHOD_TAG}__${RUN_TAG}"
echo "[$(date +%T)]     log : $LOG_FILE"
# cleanup runs from trap
