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
cleanup() {
  trap - EXIT INT TERM
  echo "[$(date +%T)] cleanup ..."
  # Each registered child is launched with setsid below; its PID is its PGID.
  # Signal only this run's groups, never unrelated ROS/Kit sessions by name.
  for p in "${PIDS[@]}"; do kill -INT -- "-$p" 2>/dev/null || true; done
  sleep 10
  for p in "${PIDS[@]}"; do kill -TERM -- "-$p" 2>/dev/null || true; done
  sleep 3
  for p in "${PIDS[@]}"; do kill -KILL -- "-$p" 2>/dev/null || true; done
  echo "[$(date +%T)] cleanup done"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# 0. 溫度取樣：在 Isaac 之前就開始，用牆鐘每秒取樣
THERM_CSV="${RUN_DIR}/thermal.csv"
ISAAC_PIDFILE="${RUN_DIR}/isaac.pid"
rm -f "$ISAAC_PIDFILE"
setsid "${HERE}/thermal_sampler.sh" "$THERM_CSV" "${CPU_LIMIT:-88}" "$ISAAC_PIDFILE" \
    >> "$LOG" 2>&1 < /dev/null &
THERM_PID=$!; PIDS+=( $THERM_PID )
echo "[$(date +%T)] [0/6] 溫度取樣已啟動 -> $THERM_CSV（上限 ${CPU_LIMIT:-88} °C）"
sleep 3

# 1. Isaac
HEADLESS="${HEADLESS:-true}"
echo "[$(date +%T)] [1/6] 啟動 Isaac（headless=$HEADLESS, cpu_threads=${CPU_THREADS:-8}）..."
setsid "$ISAAC_PY" "${HERE}/isaac_bigarena_sim.py" --seed "$SEED" --method "$METHOD" \
    --traj bigarena_traffic --headless "$HEADLESS" \
    --experience "${EXPERIENCE:-}" --width "${VIEW_WIDTH:-1600}" --height "${VIEW_HEIGHT:-900}" \
    --poses-csv "$POSES_CSV" \
    --duration "${SIM_BUDGET:-900}" --task-limit "$DURATION" \
    --wall-limit "${WALL_LIMIT:-1200}" \
    --render-hz "${RENDER_HZ:-12}" --cpu-limit "${CPU_LIMIT:-88}" \
    --arrive-tol "${ARRIVE_TOL:-0.30}" \
    --mover-phase-yaml "${AMMR_PHASE_YAML:-}" \
    --camera "${CAMERA:-false}" --cam-width "${CAM_W:-640}" \
    --cam-height "${CAM_H:-480}" --cam-hz "${CAM_HZ:-10}" \
    --cam-save "${CAM_SAVE:-0}" --cam-raw "${CAM_RAW:-true}" \
    --cam-async "${CAM_ASYNC:-true}" --cam-jpeg-q "${CAM_Q:-80}" \
    --cpu-threads "${CPU_THREADS:-8}" \
    --out "${RUN_DIR}/isaac_run.json" >> "$LOG" 2>&1 < /dev/null &
ISAAC_PID=$!; echo "$ISAAC_PID" > "$ISAAC_PIDFILE"; PIDS+=( $ISAAC_PID )
echo "[$(date +%T)] Isaac PID=$ISAAC_PID（獨立 session），experience=${EXPERIENCE:-default}"
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
# The launch's gui:= argument is its Gazebo-GUI switch, but it ALSO gates
# foxglove_bridge. Passing false to avoid a gz window silently disabled the
# bridge as well, which is why no Foxglove was ever reachable. NO_GZ=1
# already removes gz entirely, so gui:=true here only enables the bridge.
echo "[$(date +%T)] [2/6] 啟動導航鏈（NO_GZ=1, BIGARENA=1, TRAJ=bigarena_traffic）..."
NO_GZ=1 BIGARENA=1 TRAJ=bigarena_traffic SPAWN_X="$SX" SPAWN_Y="$SY" \
  HEADING="${HEADING:-0}" NO_TRAFFIC="${NO_TRAFFIC:-0}" \
  setsid ros2 launch my_omnibot_description omni_bot_dynamic.launch.py \
  gui:="${LAUNCH_GUI:-true}" use_arm:=true >> "$LOG" 2>&1 < /dev/null &
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
setsid timeout --foreground --signal=INT --kill-after=5 "${REC_CAP}s" \
  ros2 bag record -o "${RUN_DIR}/bag" \
  /clock /odom /odom_raw /odometry/filtered /amcl_pose /model/omni_bot/pose \
  /cmd_vel /cmd_vel_nav /cmd_vel_pre_shield /scan /scan_raw /plan /goal_pose \
  /tf /tf_static /gmpc/solve_time_ms /gmpc/min_h /gmpc/diag /joint_states \
  /gmpc/heading /case_start \
  /dynamic_obstacles/target /dynamic_obstacles/phase_epoch \
  /dynamic_obstacles/ground_truth \
  /model/dyn_obs_0/pose /model/dyn_obs_1/pose /model/dyn_obs_2/pose \
  /model/dyn_obs_3/pose /model/dyn_obs_4/pose /model/dyn_obs_5/pose \
  /model/dyn_obs_6/pose /model/dyn_obs_7/pose /model/dyn_obs_8/pose \
  /model/dyn_obs_9/pose \
  /base_camera/color/image_raw /base_camera/color/camera_info \
  /base_camera/color/image_raw/compressed \
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
        if ! kill -0 "$ISAAC_PID" 2>/dev/null; then
            mark "!! Isaac 已退出，立即停止就緒檢查"; exit 3
        fi
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
    mark "!! Isaac 在就緒檢查前已退出"; exit 3
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
# A one-shot grep raced the recorder: /cmd_vel does not exist until gmpc_node
# publishes its first message, so the subscription can legitimately appear a
# few seconds after the other topics. That is "not ready yet", not "broken" --
# the same distinction the other checks already make -- so it is given time.
bag_subscribed() { grep -aq "Subscribed to topic '${1}'" "$LOG" 2>/dev/null; }
for t in /scan /odom /cmd_vel /amcl_pose /model/omni_bot/pose /tf; do
    wait_for 30 "bag 已訂閱 $t" bag_subscribed "$t"
done

# (f) 靜止檢查：NO_TRAFFIC=1 只保證沒有下命令，不保證物體不動。
# 記錄輸出，不論通過與否，讓「殘留速度」在事後可查而不是靠推論。
STILL_LOG="${RUN_DIR}/stillness.log"
# NO_TRAFFIC=1 -> 全部應靜止；traffic 開啟 -> 只要求機器人靜止，
# 並反過來要求移動體確實在動。
# scheduled 情境在 /case_start 之前障礙物本來就靜止（驅動送零速度），因此不能
# 用 --traffic（那會要求移動體正在動）。移動確認改在 /case_start 之後單獨做。
STILL_ARGS=""
if [ "${NO_TRAFFIC:-0}" != "1" ] && \
   [ "${AMMR_OBSTACLE_MODE:-legacy}" != "scheduled" ]; then
    STILL_ARGS="--traffic"
fi
if timeout 40 python3 "${HERE}/stillness_check.py" --window "${STILL_WINDOW:-3.0}" \
      $STILL_ARGS > "$STILL_LOG" 2>&1; then
    mark "靜止檢查通過（見 stillness.log）"
else
    mark "!! 靜止檢查未通過（見 stillness.log）"
    [ "${STILL_STRICT:-1}" = "1" ] && gate_fail=1
fi
cat "$STILL_LOG" >> "$READY_LOG"

if [ "$gate_fail" -ne 0 ]; then
    echo "[$(date +%T)] [5/6] **就緒檢查未通過，不發目標**（見 $READY_LOG）"
    sleep 5
    exit 2
fi
echo "[$(date +%T)] [5/6] 就緒檢查全部通過"

# ---- scheduled 情境：定位 -> /case_start -> 移動確認 --------------------
CS_EPOCH=""
if [ "${AMMR_OBSTACLE_MODE:-legacy}" = "scheduled" ]; then
    echo "[$(date +%T)] [5/6] 相位 0 定位檢查（任務開始前就放好並確認穩定）..."
    if ! timeout 40 python3 "${HERE}/case_start_check.py" preposition \
          --traj "$AMMR_TRAJ_FILE" --out "${RUN_DIR}/preposition.json" \
          2>&1 | tee -a "$LOG" | sed "s/^/[$(date +%T)] [5\/6]   /"; then
        echo "[$(date +%T)] [5/6] **相位 0 定位未通過，不發 /case_start，中止**"
        exit 4
    fi
    # Publish ONCE. The driver resets its phase epoch on EVERY /case_start, so a
    # burst would set the zero point to the LAST message; the gz runner already
    # hit that. Publish one, then verify the driver adopted it.
    CS_OK=0
    for attempt in 1 2 3; do
        echo "[$(date +%T)] [5/6] 等 /case_start 訂閱者（第 ${attempt} 次）..."
        for i in $(seq 1 30); do
            cs=$(ros2 topic info /case_start 2>/dev/null \
                 | awk '/[Ss]ubscri.*[Cc]ount/ {print $NF; exit}')
            [ "${cs:-0}" -ge 1 ] && break
            sleep 1
        done
        echo "[$(date +%T)] [5/6] 發布 /case_start（僅一次）..."
        timeout 8 ros2 topic pub -t 1 /case_start std_msgs/msg/Empty "{}" \
            >> "$LOG" 2>&1 || true
        if timeout 30 python3 "${HERE}/case_start_check.py" epoch --window 5 \
              --out "${RUN_DIR}/phase_epoch.json" \
              2>&1 | tee -a "$LOG" | sed "s/^/[$(date +%T)] [5\/6]   /"; then
            CS_OK=1; break
        fi
    done
    if [ "$CS_OK" -ne 1 ]; then
        echo "[$(date +%T)] [5/6] **驅動未採用 /case_start，障礙物不會依排程移動，中止**"
        exit 5
    fi
    CS_EPOCH=$(python3 -c "import json;print(json.load(open('${RUN_DIR}/phase_epoch.json'))['phase_epoch'])")
    echo "[$(date +%T)] [5/6] phase_epoch = ${CS_EPOCH} s（模擬時間）"
    echo "[$(date +%T)] [5/6] /case_start 之後的移動確認..."
    if ! timeout 40 python3 "${HERE}/case_start_check.py" moving \
          --traj "$AMMR_TRAJ_FILE" --out "${RUN_DIR}/movers_moving.json" \
          2>&1 | tee -a "$LOG" | sed "s/^/[$(date +%T)] [5\/6]   /"; then
        echo "[$(date +%T)] [5/6] **移動體未如預期啟動，中止**"
        exit 6
    fi
fi

echo "[$(date +%T)] [5/6] 等 /goal_pose 訂閱者後發布目標 ($GX, $GY) ...
for i in $(seq 1 30); do
  c=$(ros2 topic info /goal_pose 2>/dev/null | awk '/[Ss]ubscri.*[Cc]ount/ {print $NF; exit}')
  [ "${c:-0}" -ge 2 ] && { echo "[$(date +%T)] [5/6]   訂閱者=$c（${i}s）"; break; }
  sleep 1
done
# `ros2 topic pub` leaves header.stamp at zero, so the publish instant could not
# be checked afterwards and the reports had to fall back on "the time the
# simulator's callback ran" -- which was measured running LATE, after the first
# /plan and after motion had started. publish_goal.py stamps each message with
# simulation time and writes the publish events to JSON, so the task clock can
# be tied to a verifiable publish event.
echo "[$(date +%T)] [5/6] **案例開始時間 $(date +%T)**：發布目標"
echo "$(date +%s) case_start_goal_published" >> "$READY_LOG"
timeout 60 python3 "${HERE}/publish_goal.py" --x "$GX" --y "$GY" \
  --count 5 --period 1.0 --out "${RUN_DIR}/goal_publish.json" \
  2>&1 | tee -a "$LOG" | sed "s/^/[$(date +%T)] [5\/6]   /"
# epoch 與目標發布之間流逝的相位必須被記錄，不能只說「設計上同時」：
# 就緒等待若在兩趟長短不同，會吃掉不同長度的障礙物相位。
if [ -n "$CS_EPOCH" ]; then
    python3 - "$RUN_DIR" "$CS_EPOCH" <<'PYEOF' | tee -a "$LOG"
import json, sys
d, ep = sys.argv[1], float(sys.argv[2])
g = json.load(open(f'{d}/goal_publish.json'))
gp = g['first_publish_sim_t']
rec = dict(phase_epoch_sim_t=ep, first_goal_publish_sim_t=gp,
           phase_consumed_before_goal_s=gp - ep)
json.dump(rec, open(f'{d}/phase_alignment.json', 'w'), indent=1)
print(f'  相位零點 {ep:.3f} s，目標發布 {gp:.3f} s，'
      f'發目標前已流逝相位 {gp-ep:+.3f} s')
PYEOF
fi

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
kill -INT -- "-$REC" 2>/dev/null || true
wait $REC 2>/dev/null || true
echo "[$(date +%T)] === 結束：$TAG ==="
echo "[$(date +%T)]     資料目錄: ${RUN_DIR}"
echo "[$(date +%T)]     log: $LOG"
