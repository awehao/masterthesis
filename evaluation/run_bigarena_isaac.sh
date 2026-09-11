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
source "${HERE}/lib/run_guards.sh"

METHOD="${1:-gmpc_scan}"; SEED="${2:-1}"; DURATION="${3:-180}"
# METHOD used to name the output directory and nothing else, while the heading
# objective came from a separate HEADING variable -- so a directory called
# ...gmpc_scan_heading... could hold a run with the heading objective off, and
# an ON/OFF pair could silently be two OFFs. The method now DECIDES the
# configuration, an unknown name is refused rather than falling through to the
# default chain, and a conflicting HEADING in the environment is an error, not a
# silent override.
# method_config 的說明走 stderr（已直接顯示），所以這裡只補上收尾訊息。
if ! _MCFG=$(method_config "$METHOD"); then
    echo "ERROR: 方法 '$METHOD' 未定義於 method_config()，拒絕執行"; exit 1
fi
_HEADING_WANT="${_MCFG#HEADING=}"
if [ -n "${HEADING:-}" ] && [ "${HEADING}" != "$_HEADING_WANT" ]; then
    echo "ERROR: HEADING=${HEADING} 與方法 $METHOD 要求的 HEADING=${_HEADING_WANT} 衝突"
    echo "       方法名稱決定設定；要跑別的組合請用對應的方法名稱。"
    exit 1
fi
HEADING="$_HEADING_WANT"
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
    --odom-vel-source "${ODOM_VEL_SOURCE:-cmd}" \
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
  HEADING="${HEADING}" NO_TRAFFIC="${NO_TRAFFIC:-0}" \
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
  /gmpc/diag_v2 /cmd_vel_smoothed /wheel_guard/status \
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
# lifecycle_active() now comes from lib/run_guards.sh: `grep -q active` also
# matched "inactive", so an unconfigured node passed this gate.

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

# 命令鏈的中間輸出與 guard 內部狀態。v2_off_093455 那趟最終命令的輪速檢查
# 通過，但因為沒錄這幾項，無法判斷每一筆命令是原樣通過還是被改過，也拿不到
# guard 的逐筆 dt 與縮放量。訂閱只代表 recorder 接上了，收到幾筆要在事後從
# bag 的 message_count 確認，所以兩件事分開做。
for t in /cmd_vel_smoothed /wheel_guard/status /gmpc/diag_v2; do
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

# 方法名稱說的是「應該」，這裡讀回節點「實際」的設定。兩者不符就中止：
# 一趟目錄名寫 _heading 而 heading_enable=False 的執行，比沒有這趟更糟，
# 因為它會被當成 ON 進入配對統計。
echo "[$(date +%T)] [5/6] 讀回控制器實際參數並與方法核對 ..."
_HEADING_EXPECT=$([ "$HEADING" = "1" ] && echo True || echo False)
# 這裡刻意不用 `if ! cmd | tee`：那樣判的是 tee 的退出碼，正是 run_step
# 存在的原因。先跑、留下退出碼，再把輸出送進紀錄。
assert_param /gmpc_controller heading_enable "$_HEADING_EXPECT" \
    > "${RUN_DIR}/param_readback.log" 2>&1
_rc=$?
tee -a "$LOG" "$READY_LOG" < "${RUN_DIR}/param_readback.log" >/dev/null
sed 's/^/[5\/6]   /' "${RUN_DIR}/param_readback.log"
if [ "$_rc" -ne 0 ]; then
    echo "[$(date +%T)] [5/6] **實際設定與方法 $METHOD 不符**"; gate_fail=1
fi
# 完整參數快照：之後要重現或申訴某一趟的設定，只能靠這份，不能靠目錄名稱。
timeout 20 ros2 param dump /gmpc_controller --output-dir "$RUN_DIR" \
    >> "$LOG" 2>&1 || echo "[$(date +%T)] [5/6]   （參數快照失敗，不中止）"
python3 - "$RUN_DIR" "$METHOD" "$SEED" "$HEADING" <<'PYEOF2' | tee -a "$LOG"
import json, sys
d, method, seed, heading = sys.argv[1:5]
json.dump(dict(method=method, seed=int(seed), heading=int(heading),
               resolved_from='method_config() in evaluation/lib/run_guards.sh'),
          open(f'{d}/method_manifest.json', 'w'), indent=1)
print(f'  method={method} seed={seed} HEADING={heading} -> method_manifest.json')
PYEOF2

if [ "$gate_fail" -ne 0 ]; then
    echo "[$(date +%T)] [5/6] **就緒檢查未通過，不發目標**（見 $READY_LOG）"
    sleep 5
    exit 2
fi
echo "[$(date +%T)] [5/6] 就緒檢查全部通過"

# ---- 端點身分檢查：guard 必須是 /cmd_vel 的唯一發布者 -------------------
# 只看訂閱者「計數」無法辨識來源；先前一趟 ON 正是因同 domain 的殘留導航鏈而作廢。
if [ "${GUARD:-0}" = "1" ]; then
    timeout 60 python3 "${HERE}/endpoint_check.py" \
        --out "${RUN_DIR}/endpoints.json" \
        --sole-publisher "/cmd_vel=wheel_limit_guard" \
        > "${RUN_DIR}/endpoints.log" 2>&1
    _rc=$?
    sed "s/^/[$(date +%T)] [5\/6]   /" "${RUN_DIR}/endpoints.log" | tee -a "$LOG"
    if [ "$_rc" -ne 0 ]; then
        echo "[$(date +%T)] [5/6] **端點身分檢查未通過，中止**"; exit 8
    fi
fi

# ---- scheduled 情境：定位 -> /case_start -> 移動確認 --------------------
CS_EPOCH=""
if [ "${AMMR_OBSTACLE_MODE:-legacy}" = "scheduled" ]; then
    echo "[$(date +%T)] [5/6] 相位 0 定位檢查（任務開始前就放好並確認穩定）..."
    # `cmd | tee | sed` 會讓 if 判定 sed 的退出碼，python 的失敗被吞掉。
    # 實測有一趟移動確認明確印出「未通過」卻仍繼續執行。先存檔再判定。
    run_step "$LOG" "[$(date +%T)] [5/6]   " "${RUN_DIR}/preposition.log" \
        timeout 40 python3 "${HERE}/case_start_check.py" preposition \
              --traj "$AMMR_TRAJ_FILE" --out "${RUN_DIR}/preposition.json"
    if [ $? -ne 0 ]; then
        echo "[$(date +%T)] [5/6] **相位 0 定位未通過，不發 /case_start，中止**"
        exit 4
    fi
    # Publish ONCE. The driver resets its phase epoch on EVERY /case_start, so a
    # burst would set the zero point to the LAST message; the gz runner already
    # hit that. Publish one, then verify the driver adopted it.
    # 發送與確認必須分開。舊的迴圈每次重試都會再發一次 /case_start，而驅動
    # 收到每一則都會重設相位零點——所以「第一次其實已被採用、只是確認讀取
    # 逾時」的情況下，第二次重試會把相位整個推掉，兩趟的遭遇時序就不同了。
    # 現在只發一次；重試的只有讀取。
    # 端點匹配後單次發布。只看「訂閱者數 >= 1」不足以確認對方是本趟的
    # dynamic_obstacle_driver，也不確認型別與 QoS 相容；publish_case_start.py
    # 逐項核對後才發，且發布器保持存活、只重讀不重發（driver 每收到一則都會
    # 重設相位零點，重送會改掉零點）。
    run_step "$LOG" "[$(date +%T)] [5/6]   " "${RUN_DIR}/case_start.log" \
        timeout 90 python3 "${HERE}/publish_case_start.py" \
              --out "${RUN_DIR}/case_start.json"
    if [ $? -ne 0 ]; then
        echo "[$(date +%T)] [5/6] **/case_start 握手未完成，障礙物不會依排程移動，中止**"
        exit 5
    fi
    CS_EPOCH=$(python3 -c "import json;print(json.load(open('${RUN_DIR}/case_start.json'))['phase_epoch'])")
    echo "[$(date +%T)] [5/6] phase_epoch = ${CS_EPOCH} s（模擬時間）"
    # 有限值只證明「有某個 epoch」，不證明「是這一趟的 epoch」。
    python3 - "$CS_EPOCH" "${CS_T_PRE:-nan}" <<'PYEOF3' || exit 5
import math, sys
ep, pre = float(sys.argv[1]), float(sys.argv[2])
if not math.isfinite(ep):
    sys.exit('!! phase_epoch 非有限值')
if math.isfinite(pre) and ep < pre - 1.0:
    sys.exit(f'!! phase_epoch {ep:.3f} 早於本次發布時刻 {pre:.1f}：'
             '驅動採用的是先前的 /case_start，相位不屬於這一趟')
print(f'  phase_epoch {ep:.3f} s 屬於本次發布事件（發布前 clock={pre:.1f} s）')
PYEOF3
    echo "[$(date +%T)] [5/6] /case_start 之後的移動確認..."
    # `cmd | tee | sed` 會讓 if 判定 sed 的退出碼，python 的失敗被吞掉。
    # 實測有一趟移動確認明確印出「未通過」卻仍繼續執行。先存檔再判定。
    run_step "$LOG" "[$(date +%T)] [5/6]   " "${RUN_DIR}/moving.log" \
        timeout 40 python3 "${HERE}/case_start_check.py" moving \
              --traj "$AMMR_TRAJ_FILE" --out "${RUN_DIR}/movers_moving.json"
    if [ $? -ne 0 ]; then
        echo "[$(date +%T)] [5/6] **移動體未如預期啟動，中止**"
        exit 6
    fi
fi

echo "[$(date +%T)] [5/6] 等 /goal_pose 訂閱者後發布目標 ($GX, $GY) ..."
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
# 固定的 epoch -> 目標發布延遲。事前設定，兩趟共用；前置檢查必須在它之前跑完，
# 錯過就中止不順延，否則檢查耗時會決定遭遇相位。
GOAL_AT_ARGS=""
if [ -n "$CS_EPOCH" ]; then
    PHASE_DELTA="${PHASE_DELTA:-30.0}"
    GOAL_AT=$(python3 -c "
import math, sys
e = float('${CS_EPOCH}')
if not math.isfinite(e):
    sys.exit('phase_epoch 非有限值')
print(f'{e + ${PHASE_DELTA}:.6f}')") || {
        echo "[$(date +%T)] [5/6] **phase_epoch 無效，中止**"; exit 7; }
    echo "[$(date +%T)] [5/6] 排定目標發布時刻 = epoch ${CS_EPOCH} + Δ ${PHASE_DELTA} = ${GOAL_AT} s"
    GOAL_AT_ARGS="--at-sim-time $GOAL_AT"
fi
# `python | tee | sed` 讓 shell 判 sed 的退出碼：模擬發布器以退出碼 3 結束，
# 整條管線仍回報 0，整趟會在沒有目標的情況下繼續跑到逾時。
run_step "$LOG" "[$(date +%T)] [5/6]   " "${RUN_DIR}/goal_publish.log" \
  timeout 900 python3 "${HERE}/publish_goal.py" --x "$GX" --y "$GY" $GOAL_AT_ARGS \
    --count 5 --period 1.0 --out "${RUN_DIR}/goal_publish.json"
if [ $? -ne 0 ]; then
    echo "[$(date +%T)] [5/6] **目標發布失敗，中止（沒有目標的執行不是一趟資料）**"
    exit 8
fi
# epoch 與目標發布之間流逝的相位必須被記錄，不能只說「設計上同時」：
# 就緒等待若在兩趟長短不同，會吃掉不同長度的障礙物相位。
if [ -n "$CS_EPOCH" ]; then
    python3 - "$RUN_DIR" "$CS_EPOCH" <<'PYEOF' | tee -a "$LOG"
import json, sys
d, ep = sys.argv[1], float(sys.argv[2])
g = json.load(open(f'{d}/goal_publish.json'))
gp = g['first_publish_sim_t']
rec = dict(phase_epoch_sim_t=ep, first_goal_publish_sim_t=gp,
           phase_consumed_before_goal_s=gp - ep,
           scheduled_sim_t=g.get('scheduled_sim_t'),
           schedule_error_s=g.get('schedule_error_s'))
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
# 「程序消失」不等於「完成並保存」。Isaac 是 setsid 起的背景子程序，wait 能
# 取得它的退出狀態；取不到就退回檢查結果檔。之前這裡無論它是正常收尾還是
# 中途崩潰，都一律印「已結束並保存」。
wait "$ISAAC_PID" 2>/dev/null; ISAAC_RC=$?
echo "[$(date +%T)] [6/6] Isaac 程序結束，退出碼 ${ISAAC_RC}"
if [ "$ISAAC_RC" -ge 128 ]; then
    echo "[$(date +%T)] [6/6] **Isaac 被訊號 $((ISAAC_RC - 128)) 中止，不是正常收尾**"
fi
RUN_OK=0
verify_result_json "${RUN_DIR}/isaac_run.json" > "${RUN_DIR}/result_check.log" 2>&1 \
    && RUN_OK=1
sed "s|^|[$(date +%T)] [6/6] |" "${RUN_DIR}/result_check.log" | tee -a "$LOG"
if [ "$RUN_OK" -ne 1 ]; then
    echo "[$(date +%T)] [6/6] **結果檔不完整：這一趟不可評分**" | tee -a "$LOG"
    echo "結果檔不完整（見 result_check.log）" > "${RUN_DIR}/INCOMPLETE.txt"
fi
echo "[$(date +%T)] [6/6] 停止錄製"
grep -a "停止原因" "$LOG" | tail -1
kill -INT -- "-$REC" 2>/dev/null || true
wait $REC 2>/dev/null || true
echo "[$(date +%T)] === 結束：$TAG ==="
echo "[$(date +%T)]     資料目錄: ${RUN_DIR}"
echo "[$(date +%T)]     log: $LOG"
# 收尾的退出碼要說出這一趟能不能用，否則批次腳本無從分辨「跑完」與「跑壞」。
if [ "$RUN_OK" -ne 1 ]; then exit 9; fi
