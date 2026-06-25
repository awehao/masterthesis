#!/usr/bin/env bash
# Headless batch of STATIC navigation trials on the ported omni_bot, using the
# SE(2) GMPC controller. Runs entirely in the background (gz sim -s, no GUI), so
# you can kick it off and walk away; results land on disk for later analysis.
#
# Per trial it:
#   1. launches omni_bot_nav.launch.py gui:=false   (headless gz + robot + Nav2 + GMPC)
#   2. resets AMCL to (0,0,0)
#   3. records a rosbag (incl. /gmpc/solve_time_ms, /gmpc/min_h)
#   4. publishes the goal once >=2 subscribers exist (DDS race fix)
#   5. waits DURATION, then tears every ROS/gz process down cleanly
#   6. runs analyze.py -> results/omnibot_static.csv
#
# Usage:
#   ./run_omnibot_static.sh [N_TRIALS] [DURATION_S] [GOAL_X] [GOAL_Y]
# Example (default 3 trials, 180 s each, goal (17,17) — same as the benchmark):
#   ./run_omnibot_static.sh
#   ./run_omnibot_static.sh 5 200 17.0 17.0
#
# Output:
#   bags/gmpc__seed<i>/                 (rosbag per trial)
#   logs/gmpc__seed<i>.log              (launch log, used by analyze.py)
#   results/omnibot_static.csv          (one row per trial; NOT the benchmark runs.csv)

WS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1091
source "${WS_ROOT}/install/setup.bash"

set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Single-instance guard (pidfile). Two overlapping batches share the cleanup
# pkill pattern and would kill each other's gz/record processes (corrupting
# bags). Pidfile (not flock) because an flock fd is inherited by every child
# process (gz, nav nodes, recorder), so a single orphan would hold the lock
# forever after a hard kill. A pidfile is robust to kill -9 (stale pid → dead).
LOCKFILE="/tmp/omnibot_static.pid"
if [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE" 2>/dev/null)" 2>/dev/null; then
    echo "ERROR: another run_omnibot_static.sh is already running (pid $(cat "$LOCKFILE"))."
    echo "       Stop it first:  pkill -TERM -f run_omnibot_static.sh"
    exit 1
fi
echo $$ > "$LOCKFILE"

N_TRIALS="${1:-3}"
DURATION="${2:-180}"
GOAL_X="${3:-17.0}"
GOAL_Y="${4:-17.0}"
METHOD="${5:-gmpc}"          # gmpc | gmpc_cbf

case "$METHOD" in
    gmpc)     CBF_FLAG="false" ;;
    gmpc_cbf) CBF_FLAG="true"  ;;
    *) echo "ERROR: METHOD (arg 5) must be gmpc | gmpc_cbf"; exit 1 ;;
esac

OUT_CSV="${HERE}/results/omnibot_static_${METHOD}.csv"
mkdir -p "${HERE}/bags" "${HERE}/logs" "${HERE}/results"
rm -f "$OUT_CSV"   # fresh batch

NODE_PAT='gz sim|ros2 launch|ros2 bag|ros2 topic pub'
NODE_PAT+='|nav2_map_server|nav2_amcl|nav2_planner|nav2_lifecycle_manager'
NODE_PAT+='|map_server|amcl|planner_server|lifecycle_manager'
NODE_PAT+='|goal_to_plan_relay|gmpc_node|scan_relay|omni_drive_controller'
NODE_PAT+='|parameter_bridge|robot_state_publisher|foxglove_bridge'

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
# On Ctrl-C / SIGTERM: clean up AND actually exit (previously the handler ran
# cleanup but fell through, so `pkill -TERM` could not stop the batch and two
# runs could overlap and kill each other's processes).
trap 'echo "[$(date +%T)] interrupted -> stopping batch"; cleanup; exit 130' INT TERM
trap 'cleanup; rm -f "$LOCKFILE"' EXIT

run_trial() {
    local seed="$1"
    local run_tag="seed${seed}"
    local log_file="${HERE}/logs/${METHOD}__${run_tag}.log"
    local bag_dir="${HERE}/bags/${METHOD}__${run_tag}"
    rm -rf "$bag_dir"
    rm -f  "$log_file"   # fresh log per trial (analyze.py parses it for goal-reached)

    echo "=========================================================="
    echo "[$(date +%T)] TRIAL ${seed}/${N_TRIALS}  goal=(${GOAL_X},${GOAL_Y})  dur=${DURATION}s"
    echo "=========================================================="

    # 1. headless gz + robot + Nav2 + GMPC (single launch)
    echo "[$(date +%T)] [1/5] launch omni_bot_nav (headless) ..."
    ros2 launch my_omnibot_description omni_bot_nav.launch.py gui:=false cbf:="$CBF_FLAG" \
        >> "$log_file" 2>&1 < /dev/null &
    PIDS+=( $! )
    sleep 28   # gz spawn (~15s) + Nav2 lifecycle + amcl

    # 2. reset AMCL (best effort)
    echo "[$(date +%T)] [2/5] reset AMCL pose ..."
    timeout 8 ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped \
        "{ header: { frame_id: 'map' }, pose: { pose: { position: { x: 0.0, y: 0.0, z: 0.0 }, orientation: { w: 1.0 } } } }" \
        >> "$log_file" 2>&1 || echo "    initialpose timed out (continuing)"
    sleep 3

    # 3. record via record.sh — proven `timeout --signal=INT --kill-after=5`
    #    finalize (background `&` makes a raw recorder ignore SIGINT, which is
    #    why the inline version corrupted bags). record.sh also captures /gmpc/*.
    echo "[$(date +%T)] [3/5] start recording -> ${bag_dir}"
    "${HERE}/record.sh" "$METHOD" "$run_tag" "$DURATION" \
        >> "$log_file" 2>&1 < /dev/null &
    REC_PID=$!
    PIDS+=( $REC_PID )
    sleep 3

    # 4. publish goal once >=2 subscribers (DDS late-subscriber race fix)
    echo "[$(date +%T)] [4/5] wait for /goal_pose subscribers, then publish ..."
    for i in $(seq 1 30); do
        count=$(ros2 topic info /goal_pose 2>/dev/null | awk '/[Ss]ubscri.*[Cc]ount/ {print $NF; exit}')
        count=${count:-0}
        [ "$count" -ge 2 ] && break
        sleep 1
    done
    ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped \
        "{ header: { frame_id: 'map' }, pose: { position: { x: ${GOAL_X}, y: ${GOAL_Y}, z: 0.0 }, orientation: { w: 1.0 } } }" \
        >> "$log_file" 2>&1 || true

    # 5. goal_watcher races the recorder. Whichever ends first wins:
    #    goal reached -> stop recorder early; else recorder hits DURATION cap.
    echo "[$(date +%T)] [5/5] waiting for goal (tol=0.25 m, cap ${DURATION}s) ..."
    python3 "${HERE}/goal_watcher.py" \
        --goal-x "$GOAL_X" --goal-y "$GOAL_Y" --tol 0.25 --timeout "$DURATION" \
        >> "$log_file" 2>&1 < /dev/null &
    WATCH_PID=$!
    PIDS+=( $WATCH_PID )
    wait -n "$REC_PID" "$WATCH_PID" 2>/dev/null || true

    if kill -0 "$WATCH_PID" 2>/dev/null; then
        # watcher still alive -> recorder hit the DURATION cap first
        echo "[$(date +%T)]     ${DURATION}s cap hit without reaching goal"
        pkill -INT -P "$WATCH_PID" 2>/dev/null || true
        kill  -INT    "$WATCH_PID" 2>/dev/null || true
    else
        # watcher exited 0 -> goal reached; flush the recorder. SIGINT must reach
        # record.sh's `timeout`/`ros2 bag` grandchild, so pkill the descendants.
        echo "[$(date +%T)]     goal reached -> flushing recorder"
        sleep 2   # tail so the bag captures the stop / zero-twist
        pkill -INT -P "$REC_PID" 2>/dev/null || true
        kill  -INT    "$REC_PID" 2>/dev/null || true
    fi
    # let record.sh finish finalizing the bag before the violent cleanup
    for _ in $(seq 1 15); do kill -0 "$REC_PID" 2>/dev/null || break; sleep 1; done

    cleanup

    # analyze this trial
    echo "[$(date +%T)] analyze ${run_tag} ..."
    python3 "${HERE}/analyze.py" "$bag_dir" --method "$METHOD" --run "$run_tag" --out "$OUT_CSV" \
        || echo "    analyze.py failed for ${run_tag}"
}

echo "[$(date +%T)] === omni_bot static batch: METHOD=${METHOD} (cbf=${CBF_FLAG}) N=${N_TRIALS}, dur=${DURATION}s, goal=(${GOAL_X},${GOAL_Y}) ==="
for s in $(seq 1 "$N_TRIALS"); do
    run_trial "$s"
done

echo
echo "[$(date +%T)] === DONE. results: ${OUT_CSV} ==="
column -s, -t "$OUT_CSV" 2>/dev/null | cut -c1-160 || cat "$OUT_CSV"
echo "next: paste ${OUT_CSV} here, or run  python3 ${HERE}/plot.py ${OUT_CSV}"
