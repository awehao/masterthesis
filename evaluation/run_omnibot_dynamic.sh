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

# POSES_CSV=<file> draws this trial's start AND goal from that file's row for
# the seed, so a long batch covers many different traverses instead of
# repeating one. Without it the fixed GOAL_X/GOAL_Y and a (0,0) spawn are used,
# which is what every earlier result did.
pose_for_seed() {
    local seed="$1"
    [ -z "${POSES_CSV:-}" ] && return 1
    [ -f "$POSES_CSV" ]     || { echo "    POSES_CSV not found: $POSES_CSV"; return 1; }
    local row
    row="$(awk -F, -v s="$seed" 'NR>1 && $1==s {print; exit}' "$POSES_CSV")"
    [ -n "$row" ] || return 1
    SPAWN_X="$(echo "$row" | cut -d, -f2)"; SPAWN_Y="$(echo "$row" | cut -d, -f3)"
    GOAL_X="$(echo  "$row" | cut -d, -f4)"; GOAL_Y="$(echo  "$row" | cut -d, -f5)"
    export SPAWN_X SPAWN_Y
    return 0
}

run_trial() {
    local seed="$1"
    pose_for_seed "$seed" || true
    local run_tag="${TAG}_seed${seed}"
    local log_file="${HERE}/logs/${AMETHOD}__${run_tag}.log"
    local bag_dir="${HERE}/bags/${AMETHOD}__${run_tag}"
    # Move aside rather than delete: an anomaly worth analysing was lost once
    # because the next batch reused this directory. Old copies accumulate under
    # __prev_<time> and can be cleared by hand when no longer wanted.
    [ -d "$bag_dir" ] && mv "$bag_dir" "${bag_dir}__prev_$(date +%H%M%S)" 2>/dev/null
    true
    rm -f  "$log_file"

    echo "=========================================================="
    echo "[$(date +%T)] TRIAL ${seed}/${N_TRIALS}  method=${METHOD}  start=(${SPAWN_X:-0.0},${SPAWN_Y:-0.0})  goal=(${GOAL_X},${GOAL_Y})  dur=${DURATION}s"
    echo "=========================================================="

    # 1. headless dynamic world + omni_bot + (GMPC-CBF | MPPI | RPP) + perception
    echo "[$(date +%T)] [1/5] launch ($([ "$GUI" = "1" ] && echo "GUI" || echo "headless")\
$([ "$ARM" = "1" ] && echo ", arm")$([ "${DETOUR:-0}" = "1" ] && echo ", detour"), method=${METHOD}) ..."
    # Record the configuration INTO the log. Without this a bag cannot be
    # attributed to a configuration after the fact, which already led to a
    # figure being labelled with settings that were never verified.
    {
      echo "### CONFIG $(date +%T)"
      echo "###   launch: $LAUNCH_ARGS"
      echo "###   traverse: (${SPAWN_X:-0.0},${SPAWN_Y:-0.0}) -> (${GOAL_X},${GOAL_Y})"
      for v in BIGARENA ARENA TRAJ GUI ARM DETOUR DETOUR_OFFSET DETOUR_VX_FLOOR DETOUR_CLEAR_REF \
               DETOUR_CLEAR_PAD DETOUR_SIDE_PROJ PLAN_BLEND \
               HORIZON SPAWN_X SPAWN_Y POSES_CSV GOAL_DELAY_MIN GOAL_DELAY_MAX INFLATION CBF_VEL_MARGIN CBF_PRUNE_RANGE CBF_FAR_STRIDE ST_WEIGHT PROG_WEIGHT CBF_ALPHA CBF_SAFE_MARGIN PLANNER_SCAN CBF_SLACK_W STUCK_WINDOW STUCK_PROGRESS STATIC_MARGIN \
               CBF_MARGIN_GROWTH AX_MAX AY_MAX AZ_MAX \
               STATIC_WINDOW MIN_NET_SPEED STATIC_KEEP_VEL MIN_CLUSTER_PTS EKF_REJECT REPLAN BASE_ACCEL; do
        echo "###   $v=${!v-<unset>}"
      done
    } >> "$log_file"
    ros2 launch my_omnibot_description $LAUNCH_ARGS \
        >> "$log_file" 2>&1 < /dev/null &
    PIDS+=( $! )
    # Wait on real readiness signals rather than a flat sleep sized for the
    # worst case; see trial_start.py. Costs ~87 s of the old shell version.
    echo "[$(date +%T)] [2/5] wait for stack + reset AMCL ..."
    # PIPESTATUS, not the pipeline's status: that would be tee's, which always
    # succeeds, so a failed bring-up would sail through and record a dead trial.
    python3 "${HERE}/trial_start.py" --phase prepare \
        --goal-x "$GOAL_X" --goal-y "$GOAL_Y" \
        --start-x "${SPAWN_X:-0.0}" --start-y "${SPAWN_Y:-0.0}" 2>&1 | tee -a "$log_file"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "[$(date +%T)]     bring-up failed -- skipping trial ${seed}"
        cleanup
        return 1
    fi

    # 3. record (record.sh already captures /gmpc/obstacles + diagnostics)
    echo "[$(date +%T)] [3/5] start recording -> ${bag_dir}"
    "${HERE}/record.sh" "$AMETHOD" "$run_tag" "$DURATION" \
        >> "$log_file" 2>&1 < /dev/null &
    REC_PID=$!
    PIDS+=( $REC_PID )

    # 4. publish the goal, retrying until /plan comes back
    echo "[$(date +%T)] [4/5] publish goal, wait for /plan ..."
    # --seed makes the release-to-goal delay reproducible per trial: the
    # obstacles wait on their start points, so this delay is the only thing
    # setting their phase when the robot sets off.
    python3 "${HERE}/trial_start.py" --phase goal \
        --goal-x "$GOAL_X" --goal-y "$GOAL_Y" --seed "$seed" \
        --goal-delay-min "${GOAL_DELAY_MIN:-1.0}" \
        --goal-delay-max "${GOAL_DELAY_MAX:-5.0}" 2>&1 | tee -a "$log_file"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "[$(date +%T)]     no plan -- skipping trial ${seed}"
        cleanup
        return 1
    fi

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
    run_trial "$s" || echo "[$(date +%T)] trial ${s} skipped"
done

echo
echo "[$(date +%T)] === DONE. results: ${OUT_CSV} ==="
column -s, -t "$OUT_CSV" 2>/dev/null | cut -c1-170 || cat "$OUT_CSV"
echo "next: paste ${OUT_CSV} here"
