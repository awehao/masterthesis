#!/usr/bin/env bash
# Headless batch of DYNAMIC-obstacle trials on the ported omni_bot, GMPC + CBF.
# Obstacles are detected from /scan (obstacle_source:=scan, the Sprint-B real
# perception) or read from ground truth (obstacle_source:=truth, baseline).
# Runs in the background (gz sim -s); results land on disk.
#
# Per trial: launch omni_bot_dynamic (headless) -> reset AMCL -> record.sh
# (captures /gmpc/obstacles + /gmpc/solve_time_ms + /gmpc/min_h) -> publish goal
# -> goal_watcher races recorder -> clean finalize -> analyze.py.
#
# Usage:
#   ./run_omnibot_dynamic.sh [N_TRIALS] [DURATION_S] [GOAL_X] [GOAL_Y] [METHOD]
#     METHOD = gmpc_scan (default) | gmpc_scan_nosm | gmpc_truth | mppi | rpp
#       gmpc_scan      = our GMPC+CBF, obstacles from /scan (real perception)
#       gmpc_scan_nosm = gmpc_scan but velocity_smoother OFF (smoother ablation)
#       gmpc_truth     = our GMPC+CBF, obstacles from ground truth (ablation)
#       mppi           = Nav2 controller_server + MPPIController baseline
#       rpp            = Nav2 controller_server + RegulatedPurePursuit baseline
# Examples:
#   ./run_omnibot_dynamic.sh 15 200 17 17 gmpc_scan
#   ./run_omnibot_dynamic.sh 15 200 17 17 mppi
#
# Output (METHOD kept separate so runs don't overwrite):
#   bags/<analyze_method>__<tag>_seed<i>/   logs/<analyze_method>__<tag>_seed<i>.log
#   results/omnibot_dynamic_<METHOD>.csv

WS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1091
source "${WS_ROOT}/install/setup.bash"

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LOCKFILE="/tmp/omnibot_dynamic.pid"
if [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE" 2>/dev/null)" 2>/dev/null; then
    echo "ERROR: another run_omnibot_dynamic.sh is already running (pid $(cat "$LOCKFILE"))."
    echo "       Stop it first:  pkill -TERM -f run_omnibot_dynamic.sh"
    exit 1
fi
echo $$ > "$LOCKFILE"

N_TRIALS="${1:-5}"
DURATION="${2:-200}"
GOAL_X="${3:-17.0}"
GOAL_Y="${4:-17.0}"
METHOD="${5:-gmpc_scan}"     # gmpc_scan | gmpc_truth | mppi | rpp

# GUI=1 runs the very same sequence with the gz window open, so a run can be
# WATCHED under exactly the benchmark conditions (same waits, same AMCL reset,
# same goal handshake). Default stays headless: that is how the N=40 numbers
# were recorded, and rendering steals time from the gpu_lidar.
GUI="${GUI:-0}"
if [ "$GUI" = "1" ]; then GUI_ARG="gui:=true"; else GUI_ARG="gui:=false"; fi

# ARM=1 mounts the Lite 6 for the run (Phase 2 check: does carrying the arm
# change navigation?). Default 0 = the exact robot the N=40 numbers came from.
ARM="${ARM:-0}"
if [ "$ARM" = "1" ]; then GUI_ARG="$GUI_ARG use_arm:=true"; fi

case "$METHOD" in
  gmpc_scan)      LAUNCH_ARGS="omni_bot_dynamic.launch.py $GUI_ARG obstacle_source:=scan";                    AMETHOD="gmpc_cbf"; TAG="scan"      ;;
  gmpc_scan_nosm) LAUNCH_ARGS="omni_bot_dynamic.launch.py $GUI_ARG obstacle_source:=scan use_smoother:=false"; AMETHOD="gmpc_cbf"; TAG="scan_nosm" ;;
  gmpc_truth)     LAUNCH_ARGS="omni_bot_dynamic.launch.py $GUI_ARG obstacle_source:=truth";                   AMETHOD="gmpc_cbf"; TAG="truth"     ;;
  mppi)           LAUNCH_ARGS="omni_bot_baseline.launch.py $GUI_ARG method:=mppi";                            AMETHOD="mppi";     TAG="mppi"      ;;
  rpp)            LAUNCH_ARGS="omni_bot_baseline.launch.py $GUI_ARG method:=rpp";                             AMETHOD="rpp";      TAG="rpp"       ;;
  *) echo "ERROR: METHOD (arg 5) = gmpc_scan | gmpc_scan_nosm | gmpc_truth | mppi | rpp"; exit 1 ;;
esac

OUT_CSV="${HERE}/results/omnibot_dynamic_${METHOD}.csv"
mkdir -p "${HERE}/bags" "${HERE}/logs" "${HERE}/results"
rm -f "$OUT_CSV"

NODE_PAT='gz sim|ros2 launch|ros2 bag|ros2 topic pub|ros_gz_sim'
NODE_PAT+='|nav2_map_server|nav2_amcl|nav2_planner|nav2_lifecycle_manager'
NODE_PAT+='|map_server|amcl|planner_server|lifecycle_manager'
NODE_PAT+='|goal_to_plan_relay|gmpc_node|scan_relay|odom_tf_broadcaster'
NODE_PAT+='|scan_obstacle_tracker|obstacle_aggregator|dynamic_obstacle_driver'
NODE_PAT+='|parameter_bridge|robot_state_publisher|foxglove_bridge'
NODE_PAT+='|ekf_node|ekf_global|robot_localization'
NODE_PAT+='|controller_server|smoother_server|behavior_server'
NODE_PAT+='|bt_navigator|waypoint_follower|velocity_smoother'

PIDS=()
cleanup() {
    echo "[$(date +%T)] [trial] cleanup ..."
    for pid in "${PIDS[@]}"; do
        pkill -INT -P "$pid" 2>/dev/null || true
        kill  -INT     "$pid" 2>/dev/null || true
    done
    sleep 3
    pkill -INT -f "$NODE_PAT" 2>/dev/null || true
    sleep 2
    pkill -KILL -f "$NODE_PAT" 2>/dev/null || true
    sleep 1
    ros2 daemon stop  > /dev/null 2>&1 || true
    sleep 1
    ros2 daemon start > /dev/null 2>&1 || true
    sleep 2
    PIDS=()
}
trap 'echo "[$(date +%T)] interrupted -> stopping batch"; cleanup; exit 130' INT TERM
trap 'cleanup; rm -f "$LOCKFILE"' EXIT

run_trial() {
    local seed="$1"
    local run_tag="${TAG}_seed${seed}"
    local log_file="${HERE}/logs/${AMETHOD}__${run_tag}.log"
    local bag_dir="${HERE}/bags/${AMETHOD}__${run_tag}"
    rm -rf "$bag_dir"
    rm -f  "$log_file"

    echo "=========================================================="
    echo "[$(date +%T)] TRIAL ${seed}/${N_TRIALS}  method=${METHOD}  goal=(${GOAL_X},${GOAL_Y})  dur=${DURATION}s"
    echo "=========================================================="

    # 1. headless dynamic world + omni_bot + (GMPC-CBF | MPPI | RPP) + perception
    echo "[$(date +%T)] [1/5] launch (headless, method=${METHOD}) ..."
    ros2 launch my_omnibot_description $LAUNCH_ARGS \
        >> "$log_file" 2>&1 < /dev/null &
    PIDS+=( $! )
    sleep 32   # gz dynamic world spawn + obstacle driver + Nav2 lifecycle + amcl

    # 2. reset AMCL (best effort)
    echo "[$(date +%T)] [2/5] reset AMCL pose ..."
    timeout 8 ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped \
        "{ header: { frame_id: 'map' }, pose: { pose: { position: { x: 0.0, y: 0.0, z: 0.0 }, orientation: { w: 1.0 } } } }" \
        >> "$log_file" 2>&1 || echo "    initialpose timed out (continuing)"
    sleep 3

    # 3. record (record.sh already captures /gmpc/obstacles + diagnostics)
    echo "[$(date +%T)] [3/5] start recording -> ${bag_dir}"
    "${HERE}/record.sh" "$AMETHOD" "$run_tag" "$DURATION" \
        >> "$log_file" 2>&1 < /dev/null &
    REC_PID=$!
    PIDS+=( $REC_PID )
    sleep 3

    # 4. publish goal once >=2 subscribers
    echo "[$(date +%T)] [4/5] wait for /goal_pose subscribers, then publish ..."
    for i in $(seq 1 30); do
        count=$(ros2 topic info /goal_pose 2>/dev/null | awk '/[Ss]ubscri.*[Cc]ount/ {print $NF; exit}')
        count=${count:-0}
        [ "$count" -ge 2 ] && break
        sleep 1
    done
    # publish the goal 5x at 1 Hz (not --once): a single VOLATILE publish is
    # easily missed by goal_to_plan_relay if its subscription isn't ready ->
    # plan_requests=0 -> robot never moves. Repeating fixes the DDS race.
    timeout 15 ros2 topic pub -t 5 -r 1 /goal_pose geometry_msgs/msg/PoseStamped \
        "{ header: { frame_id: 'map' }, pose: { position: { x: ${GOAL_X}, y: ${GOAL_Y}, z: 0.0 }, orientation: { w: 1.0 } } }" \
        >> "$log_file" 2>&1 || true

    # 5. goal_watcher races the recorder
    echo "[$(date +%T)] [5/5] waiting for goal (tol=0.30 m, cap ${DURATION}s) ..."
    python3 "${HERE}/goal_watcher.py" \
        --goal-x "$GOAL_X" --goal-y "$GOAL_Y" --tol 0.30 --timeout "$DURATION" \
        >> "$log_file" 2>&1 < /dev/null &
    WATCH_PID=$!
    PIDS+=( $WATCH_PID )
    wait -n "$REC_PID" "$WATCH_PID" 2>/dev/null || true

    if kill -0 "$WATCH_PID" 2>/dev/null; then
        echo "[$(date +%T)]     ${DURATION}s cap hit without reaching goal"
        pkill -INT -P "$WATCH_PID" 2>/dev/null || true
        kill  -INT    "$WATCH_PID" 2>/dev/null || true
    else
        echo "[$(date +%T)]     goal reached -> flushing recorder"
        sleep 2
        pkill -INT -P "$REC_PID" 2>/dev/null || true
        kill  -INT    "$REC_PID" 2>/dev/null || true
    fi
    for _ in $(seq 1 15); do kill -0 "$REC_PID" 2>/dev/null || break; sleep 1; done

    cleanup

    echo "[$(date +%T)] analyze ${run_tag} ..."
    python3 "${HERE}/analyze.py" "$bag_dir" --method "$AMETHOD" --run "$run_tag" --out "$OUT_CSV" \
        || echo "    analyze.py failed for ${run_tag}"
}

echo "[$(date +%T)] === omni_bot DYNAMIC batch: method=${METHOD} N=${N_TRIALS}, dur=${DURATION}s, goal=(${GOAL_X},${GOAL_Y}) ==="
for s in $(seq 1 "$N_TRIALS"); do
    run_trial "$s"
done

echo
echo "[$(date +%T)] === DONE. results: ${OUT_CSV} ==="
column -s, -t "$OUT_CSV" 2>/dev/null | cut -c1-170 || cat "$OUT_CSV"
echo "next: paste ${OUT_CSV} here"
