#!/usr/bin/env bash
# One bigarena trial in Isaac Sim, phase-4 chain, arm carried tucked.
#
# Only the simulator differs from the Gazebo runs: omni_bot_dynamic.launch.py
# is started with NO_GZ=1, which omits exactly three actions (gz sim, the
# ros_gz bridge, the model spawn) and keeps every other node, parameter and
# ordering. Isaac supplies /clock, /odom_raw, /scan_raw, consumes /cmd_vel and
# handles /model/<dyn>/{cmd_vel,pose}.
#
# Usage: ./run_bigarena_isaac.sh [METHOD] [SEED] [DURATION_S]
WS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/jazzy/setup.bash
source "${WS_ROOT}/install/setup.bash"
set -u

METHOD="${1:-gmpc_scan}"; SEED="${2:-1}"; DURATION="${3:-180}"
POSES_CSV="${POSES_CSV:-${HERE}/results/bigarena_poses.csv}"
ISAAC_PY="${ISAAC_PY:-$HOME/venvs/isaacsim-6.0.1/bin/python}"
# Every run gets its own directory. The previous scheme reused one name per
# (method, seed), so preparing a re-run meant deleting the last one -- which is
# how the earlier seed-1 bag was lost and its task time became a reconstruction
# rather than a measurement. RUN_ID can be set to label a run; it never
# overwrites, because the timestamp is always part of the path.
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
TAG="isaac_bigarena_${METHOD}__seed${SEED}__${RUN_ID}"
RUN_DIR="${HERE}/runs/${TAG}"
if [ -e "$RUN_DIR" ]; then
    echo "ERROR: $RUN_DIR 已存在，拒絕覆蓋（改 RUN_ID）"; exit 1
fi
mkdir -p "$RUN_DIR"
LOG="${RUN_DIR}/run.log"
read -r SX SY GX GY < <(awk -F, -v s="$SEED" 'NR>1 && $1==s {print $2,$3,$4,$5; exit}' "$POSES_CSV")
if [ -z "${SX:-}" ]; then echo "POSES_CSV 沒有 seed=$SEED"; exit 1; fi
echo "[$(date +%T)] 案例 method=$METHOD seed=$SEED start=($SX,$SY) goal=($GX,$GY) dur=${DURATION}s"
echo "[$(date +%T)] POSES_CSV=$POSES_CSV  TRAJ=bigarena_traffic  BIGARENA=1"

PIDS=()
NODE_PAT='isaac_bigarena_sim|ros2 launch|ros2 bag|ros2 topic pub|nav2_|map_server|amcl'
NODE_PAT+='|planner_server|lifecycle_manager|velocity_smoother|gmpc_node|scan_relay'
NODE_PAT+='|scan_obstacle_tracker|obstacle_aggregator|scan_safety_shield|ekf_node'
NODE_PAT+='|odom_tf_broadcaster|dynamic_obstacle_driver|robot_state_publisher'
NODE_PAT+='|goal_to_plan_relay|foxglove_bridge|goal_watcher'
cleanup() {
  echo "[$(date +%T)] cleanup ..."
  for p in "${PIDS[@]:-}"; do pkill -INT -P "$p" 2>/dev/null; kill -INT "$p" 2>/dev/null; done
  sleep 3; pkill -INT -f "$NODE_PAT" 2>/dev/null; sleep 3
  pkill -KILL -f "$NODE_PAT" 2>/dev/null; sleep 1
  ros2 daemon stop >/dev/null 2>&1; sleep 1; ros2 daemon start >/dev/null 2>&1
  echo "[$(date +%T)] cleanup done"
}
trap cleanup EXIT INT TERM

# 0. 溫度取樣：在 Isaac 之前就開始，用牆鐘每秒取樣
THERM_CSV="${RUN_DIR}/thermal.csv"
ISAAC_PIDFILE="${RUN_DIR}/isaac.pid"
rm -f "$ISAAC_PIDFILE"
"${HERE}/thermal_sampler.sh" "$THERM_CSV" "${CPU_LIMIT:-88}" "$ISAAC_PIDFILE" \
    >> "$LOG" 2>&1 < /dev/null &
THERM_PID=$!; PIDS+=( $THERM_PID )
echo "[$(date +%T)] [0/6] 溫度取樣已啟動 -> $THERM_CSV（上限 ${CPU_LIMIT:-88} °C）"
sleep 3

# 1. Isaac
HEADLESS="${HEADLESS:-true}"
echo "[$(date +%T)] [1/6] 啟動 Isaac（headless=$HEADLESS, cpu_threads=${CPU_THREADS:-8}）..."
"$ISAAC_PY" "${HERE}/isaac_bigarena_sim.py" --seed "$SEED" --method "$METHOD" \
    --traj bigarena_traffic --headless "$HEADLESS" \
    --duration "${SIM_BUDGET:-900}" --task-limit "$DURATION" \
    --wall-limit "${WALL_LIMIT:-1200}" \
    --render-hz "${RENDER_HZ:-12}" --cpu-limit "${CPU_LIMIT:-88}" \
    --cpu-threads "${CPU_THREADS:-8}" \
    --out "${RUN_DIR}/isaac_run.json" >> "$LOG" 2>&1 < /dev/null &
ISAAC_PID=$!; echo "$ISAAC_PID" > "$ISAAC_PIDFILE"; PIDS+=( $ISAAC_PID )
# `ros2 topic echo --once` exits 0 even when it printed nothing, so the first
# version of this wait passed after 5 s while Isaac was still loading -- the
# navigation chain and the readiness gate then both ran before the simulator
# published anything. Require two clock samples that actually advance.
echo "[$(date +%T)] [1/6] 等 /clock 真的在前進（最多 300 s）..."
clock_ok=0
for i in $(seq 1 100); do
  if ! kill -0 "$ISAAC_PID" 2>/dev/null; then
    echo "[$(date +%T)] [1/6] **Isaac 已退出，停止等待**"
    echo "[$(date +%T)] [1/6]   最後幾行："; tail -6 "$LOG" | sed 's/^/      /'
    grep -a "溫度中止\|thermal_abort\|Traceback" "$LOG" | tail -2 | sed 's/^/      /'
    exit 3
  fi
  s1=$(timeout 3 ros2 topic echo --once --field clock.sec /clock 2>/dev/null | head -1)
  if [ -n "${s1:-}" ]; then
    sleep 2
    s2=$(timeout 3 ros2 topic echo --once --field clock.sec /clock 2>/dev/null | head -1)
    if [ -n "${s2:-}" ] && [ "${s2}" -gt "${s1}" ] 2>/dev/null; then
      echo "[$(date +%T)] [1/6]   /clock 前進中 ($s1 -> $s2)，用時約 $((i*3))s"
      clock_ok=1; break
    fi
  fi
  sleep 1
done
if [ "$clock_ok" -ne 1 ]; then
  echo "[$(date +%T)] [1/6] **/clock 未前進，中止**"; exit 1
fi

# 2. 導航鏈：同一支 launch，只跳過 gz 三個節點
echo "[$(date +%T)] [2/6] 啟動導航鏈（NO_GZ=1, BIGARENA=1, TRAJ=bigarena_traffic）..."
NO_GZ=1 BIGARENA=1 TRAJ=bigarena_traffic SPAWN_X="$SX" SPAWN_Y="$SY" \
  ros2 launch my_omnibot_description omni_bot_dynamic.launch.py \
  gui:=false use_arm:=true >> "$LOG" 2>&1 < /dev/null &
PIDS+=( $! )
sleep 25

# 3. AMCL 初始位姿：起點，不是原點
echo "[$(date +%T)] [3/6] 設定 AMCL 初始位姿 ($SX, $SY) ..."
timeout 10 ros2 topic pub -t 3 -r 1 /initialpose \
  geometry_msgs/msg/PoseWithCovarianceStamped \
  "{header: {frame_id: 'map'}, pose: {pose: {position: {x: $SX, y: $SY, z: 0.0}, orientation: {w: 1.0}}}}" \
  >> "$LOG" 2>&1 || true
sleep 5

# 4. 錄製：先開始，涵蓋啟動與就緒檢查；上限只是防呆，正常由 Isaac 結束後才停
# The previous run ended because THIS timer expired 180 s after recording
# began, which included start-up and the readiness gate: the task itself had
# only had 140.7 s of simulated time and was still closing on the goal. The
# task clock now lives in the simulator and starts at the goal; this one is
# only a backstop.
REC_CAP="${REC_CAP:-1500}"
echo "[$(date +%T)] [4/6] 開始錄製（防呆上限 ${REC_CAP}s，任務時限由模擬時間另計）..."
timeout --foreground --signal=INT --kill-after=5 "${REC_CAP}s" \
  ros2 bag record -o "${RUN_DIR}/bag" \
  /clock /odom /odom_raw /odometry/filtered /amcl_pose /model/omni_bot/pose \
  /cmd_vel /cmd_vel_nav /cmd_vel_pre_shield /scan /scan_raw /plan /goal_pose \
  /tf /tf_static /gmpc/solve_time_ms /gmpc/min_h /gmpc/diag /joint_states \
  >> "$LOG" 2>&1 < /dev/null &
REC=$!; PIDS+=( $REC )
sleep 3

# 5. 就緒檢查：全部通過才發目標
# The previous attempt published the goal on a fixed sleep and produced a bag
# with two topics. Two causes were tangled there -- the run lasted 2 s, and
# nothing had confirmed the interfaces were live -- so each is now checked
# explicitly and its time recorded.
READY_LOG="${RUN_DIR}/readiness.log"
: > "$READY_LOG"
mark() { echo "[$(date +%T)] [5/6]   $1"; echo "$(date +%s) $1" >> "$READY_LOG"; }
gate_fail=0

# Each condition is given time to come up rather than judged once: a node that
# is 10 s from being ready is not the same as one that is broken.
wait_for() {  # wait_for <秒數> <說明> <指令...>
    local lim="$1" desc="$2"; shift 2
    local t0=$SECONDS
    while [ $((SECONDS - t0)) -lt "$lim" ]; do
        if "$@" >/dev/null 2>&1; then
            mark "$desc（$((SECONDS - t0))s）"; return 0
        fi
        sleep 2
    done
    mark "!! $desc 逾時 ${lim}s"; gate_fail=1; return 1
}

clock_moved() {
    local x y
    x=$(timeout 3 ros2 topic echo --once --field clock.sec /clock 2>/dev/null | head -1)
    [ -n "${x:-}" ] || return 1
    sleep 1
    y=$(timeout 3 ros2 topic echo --once --field clock.sec /clock 2>/dev/null | head -1)
    [ -n "${y:-}" ] && [ "$y" -gt "$x" ] 2>/dev/null
}
has_data() { timeout 6 ros2 topic echo --once "$1" 2>/dev/null | grep -q . ; }
lifecycle_active() { timeout 6 ros2 lifecycle get "$1" 2>/dev/null | grep -q active ; }

isaac_alive() { kill -0 "$ISAAC_PID" 2>/dev/null; }
if ! isaac_alive; then
    mark "!! Isaac 在就緒檢查前已退出"; gate_fail=1
fi
wait_for 60 "clock 在前進" clock_moved
for t in /scan /scan_raw /odom /odom_raw; do
    wait_for 40 "$t 有資料" has_data "$t"
done
# tf2_echo runs on wall time by default and cannot see a simulation-time TF
# buffer at all; tf_ready_check.py runs with use_sim_time, looks up the latest
# available transform, and separates a missing frame from an extrapolation.
for pair in "map odom" "odom base_footprint" "map base_footprint"; do
    set -- $pair
    wait_for 60 "TF $1 -> $2 可查且新鮮" \
        python3 "${HERE}/tf_ready_check.py" "$1" "$2" --timeout 12 --max-age 2.0
done
# Evidence for WHY tf2_echo failed last time, rather than the inference that it
# must have been the wall-clock/sim-time gap: run it once and keep its output.
echo "--- tf2_echo(預設 wall time) 的原始輸出，供根因佐證 ---" >> "$READY_LOG"
timeout 8 ros2 run tf2_ros tf2_echo odom base_footprint \
    >> "$READY_LOG" 2>&1 || echo "  (tf2_echo 退出碼 $?)" >> "$READY_LOG"
echo "--- tf2_echo(use_sim_time:=true) ---" >> "$READY_LOG"
timeout 8 ros2 run tf2_ros tf2_echo odom base_footprint --ros-args \
    -p use_sim_time:=true >> "$READY_LOG" 2>&1 || \
    echo "  (退出碼 $?)" >> "$READY_LOG"
for n in /map_server /amcl /planner_server; do
    wait_for 60 "lifecycle $n active" lifecycle_active "$n"
done

# (e) 錄製端確實訂閱了關鍵 topic（從 recorder 自己的日誌確認）
for t in /scan /odom /cmd_vel /amcl_pose /model/omni_bot/pose /tf; do
    if grep -aq "Subscribed to topic '${t}'" "$LOG" 2>/dev/null; then
        mark "bag 已訂閱 $t"
    else
        mark "!! bag 未訂閱 $t"; gate_fail=1
    fi
done

if [ "$gate_fail" -ne 0 ]; then
    echo "[$(date +%T)] [5/6] **就緒檢查未通過，不發目標**（見 $READY_LOG）"
    sleep 5
    exit 2
fi
echo "[$(date +%T)] [5/6] 就緒檢查全部通過"

echo "[$(date +%T)] [5/6] 等 /goal_pose 訂閱者後發布目標 ($GX, $GY) ..."
for i in $(seq 1 30); do
  c=$(ros2 topic info /goal_pose 2>/dev/null | awk '/[Ss]ubscri.*[Cc]ount/ {print $NF; exit}')
  [ "${c:-0}" -ge 2 ] && { echo "[$(date +%T)] [5/6]   訂閱者=$c（${i}s）"; break; }
  sleep 1
done
echo "[$(date +%T)] [5/6] **案例開始時間 $(date +%T)**：發布目標"
echo "$(date +%s) case_start_goal_published" >> "$READY_LOG"
timeout 15 ros2 topic pub -t 5 -r 1 /goal_pose geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: 'map'}, pose: {position: {x: $GX, y: $GY, z: 0.0}, orientation: {w: 1.0}}}" \
  >> "$LOG" 2>&1 || true

# 6. 等 Isaac 自己結束：它會先把底盤歸零、步進生效，再寫入停止原因與最後樣本。
# 只有在那之後才停止錄製並清理，否則保存會被 cleanup 截斷（上一趟就是如此，
# JSON 的 stop_reason 是 None）。
echo "[$(date +%T)] [6/6] 等 Isaac 完成任務並保存（任務時限 ${DURATION}s 模擬時間）..."
while kill -0 "$ISAAC_PID" 2>/dev/null; do
    if ! kill -0 "$REC" 2>/dev/null; then
        echo "[$(date +%T)] [6/6] 錄製已先結束（防呆上限），繼續等 Isaac 保存"
    fi
    sleep 2
done
echo "[$(date +%T)] [6/6] Isaac 已結束並保存，停止錄製"
grep -a "停止原因" "$LOG" | tail -1
pkill -INT -P "$REC" 2>/dev/null; kill -INT "$REC" 2>/dev/null
wait $REC 2>/dev/null || true
echo "[$(date +%T)] === 結束：$TAG ==="
echo "[$(date +%T)]     資料目錄: ${RUN_DIR}"
echo "[$(date +%T)]     log: $LOG"
