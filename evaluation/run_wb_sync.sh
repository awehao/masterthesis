#!/usr/bin/env bash
# Isaac 全身執行端：**只跑 sync 一趟**。
#
# 準備階段與任務時間**分開管理**：Isaac 的 --sim-limit 從它啟動就開始算，
# 而後面還有節點啟動與 preflight。放行前必須確認**剩餘模擬時間**足夠跑完
# 命令剖面與停止觀察窗，否則就是先前踩過的「命令還沒發、模擬已結束」。
#
# 清理只針對本趟記錄的 PID，每個子程序自成 process group；
# 不用 pkill、不做字串比對、不影響其他工作階段。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; WS="$(dirname "$HERE")"
cd "$WS"

# ROS 與本 workspace：`ros2 run ammr_wholebody_mpc ...` 需要 install/ 在環境裡，
# 否則會是 "Package 'ammr_wholebody_mpc' not found"。
# ROS 的 setup.bash 會引用未綁定變數，與 `set -u` 衝突；只在 source 期間關閉。
set +u
# shellcheck disable=SC1091
. /opt/ros/jazzy/setup.bash
if [ -f "$WS/install/setup.bash" ]; then
  # shellcheck disable=SC1091
  . "$WS/install/setup.bash"
else
  set -u; echo "**找不到 $WS/install/setup.bash —— 請先 colcon build**"; exit 65
fi
set -u

# ---- 1 獨立 domain：**未設即拒絕啟動** ----
if [ -z "${ROS_DOMAIN_ID:-}" ]; then
  echo "**ROS_DOMAIN_ID 未設定，拒絕啟動**。請明確指定本趟的獨立 domain，例如："
  echo "  ROS_DOMAIN_ID=95 bash evaluation/run_wb_sync.sh"
  exit 64
fi
export ROS_DOMAIN_ID            # 所有子程序沿用同一值

RUN_ID="${RUN_ID:-wb_sync_$(date +%H%M%S)}"
DIR="$WS/evaluation/runs/$RUN_ID"; mkdir -p "$DIR"
LOG="$DIR/run.log"; : > "$LOG"
ISAAC_PY="${ISAAC_PY:-$HOME/venvs/isaacsim-6.0.1/bin/python}"
# **兩份 URDF 用途不同，不可互換：**
#   manip 版     根 base_footprint → base_link → 手臂；與 Isaac 載入的一致，
#                給 robot_state_publisher 發 TF 用。
#   wholebody 版 根 world → virtual_base → base_x/y/theta → base_link，
#                底盤是**真實關節**；只當距離／安全節點的 FK **參數**，
#                拿去發 TF 會要求 base_x/y/theta 的 joint_states，且根本不同。
URDF_TF="$WS/evaluation/models/omni_bot_manip.urdf"
URDF_WB="$WS/evaluation/models/omni_bot_wholebody_expanded.urdf"

# ---- 2 準備階段與任務時間分開 ----
PROFILE_S=9.0                   # 零2 + 斜升1 + 保持3 + 斜降1 + 零2
# **判準事前固定**，見 evaluation/results/specs/wb_sync_criteria_v1.yaml，
# 由 evaluation/wb_sync_check.py 離線判定。速率不得為了通過而調整。
ARM_RATE="${ARM_RATE:-0.05}"    # joint2，rad/s；與 arm 趟次同幅度
BASE_VX="${BASE_VX:-0.03}"      # report frame +x，m/s；與 base 趟次同幅度
CRITERIA="$WS/evaluation/results/specs/wb_sync_criteria_v1.yaml"
OBSERVE_S="${OBSERVE_S:-8.0}"   # 停止觀察窗
NEED_S=$(python3 -c "print($PROFILE_S + $OBSERVE_S)")
SIM_LIMIT="${SIM_LIMIT:-70}"    # 足以涵蓋啟動 + NEED_S
PREP_TIMEOUT_S="${PREP_TIMEOUT_S:-240}"   # 準備階段的**牆鐘**逾時

PIDS=(); NAMES=(); ABORT=""
say(){ echo "[$(date +%T)] $*" | tee -a "$LOG"; }
spawn(){ local n="$1"; shift; setsid "$@" >>"$LOG" 2>&1 < /dev/null &
         PIDS+=($!); NAMES+=("$n"); say "  起 $n PID=$!"; }
fail(){ say "**失敗：$***"; echo "$*" > "$DIR/FAILED.md"; exit 1; }
cleanup(){
  say "cleanup（只針對本趟 PID）..."
  for p in "${PIDS[@]:-}"; do kill -TERM -"$p" 2>/dev/null; kill -TERM "$p" 2>/dev/null; done
  sleep 3
  for p in "${PIDS[@]:-}"; do kill -KILL -"$p" 2>/dev/null; kill -KILL "$p" 2>/dev/null; done
  say "cleanup done"
}
trap cleanup EXIT

say "=== sync 同動測試 RUN_ID=$RUN_ID domain=$ROS_DOMAIN_ID ==="
say "準備逾時 ${PREP_TIMEOUT_S}s（牆鐘）；任務需要模擬時間 ${NEED_S}s；SIM_LIMIT=${SIM_LIMIT}"
say "判準 $CRITERIA（事前定版）；底盤 ${BASE_VX} m/s、joint2 ${ARM_RATE} rad/s"
[ -f "$CRITERIA" ] || fail "找不到事前判準檔 —— 不在判準未定版時開跑"
say "判準 sha256 $(sha256sum "$CRITERIA" | cut -c1-16)（由判定器實際載入）"
say "起跑前 CPU $(python3 evaluation/cpu_temp.py)"
W0=$(date +%s)
prep_left(){ echo $(( PREP_TIMEOUT_S - ( $(date +%s) - W0 ) )); }

say "[1/8] 純邏輯測試（不開模擬器）"
python3 -u evaluation/test_wb_cmd_chain.py >>"$LOG" 2>&1 \
  || fail "命令鏈純邏輯測試未通過"
python3 -u evaluation/test_wb_sync_check.py >>"$LOG" 2>&1 \
  || fail "判定器離線測試未通過 —— 不在判定器未驗證時開跑"
say "  純邏輯測試通過（命令鏈 + 判定器）"

say "[2/8] 啟動 Isaac 執行端（--mode sync）"
spawn isaac "$ISAAC_PY" -u evaluation/isaac_wholebody_sim.py \
    --out "$DIR/sim" --mode sync --sim-limit "$SIM_LIMIT" --solver-label none

say "[3/8] 等 /clock 前進（剩餘準備時間 $(prep_left)s）"
timeout "$(prep_left)" python3 evaluation/clock_advancing.py --discover 180 \
    >>"$LOG" 2>&1 || fail "/clock 未前進或準備逾時"

say "[4/8] 啟動 robot_state_publisher、距離節點與安全層"
spawn rsp ros2 run robot_state_publisher robot_state_publisher "$URDF_TF" \
    --ros-args -p use_sim_time:=true
spawn dist ros2 run ammr_wholebody_mpc arm_link_distance --ros-args \
    -p use_sim_time:=true -p report_frame:=odom -p geometry:=links \
    -p wholebody_urdf:="$URDF_WB"
spawn safety ros2 run ammr_wholebody_mpc wholebody_safety --ros-args \
    -p use_sim_time:=true -p report_frame:=odom -p base_frame:=base_link \
    -p wholebody_urdf:="$URDF_WB"
# **Isaac 鏈的消費端是執行端本身**，不是 ros2_control 控制器。
# 關節順序核對的對象因此指向實際會執行這些數字的那一端。
spawn adapter python3 -u evaluation/arm_vel_adapter.py \
    --consumer-node /isaac_wholebody_sim
sleep 5

say "[5/8] 起動前檢查（內容 / 新鮮度 / 端點身分）"
timeout "$(prep_left)" python3 -u evaluation/wb_preflight.py --out "$DIR" \
    2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" = "0" ] || fail "起動前檢查未通過（原因見 preflight.json）"

say "[6/8] 放行前確認剩餘模擬時間"
NOW=$(timeout 15 python3 - <<'PY'
import rclpy, time
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
rclpy.init(); n = Node('wb_now'); v = []
n.create_subscription(Clock, '/clock', lambda m: v.append(m.clock.sec + m.clock.nanosec*1e-9), 10)
t0 = time.monotonic()
while time.monotonic() - t0 < 5 and len(v) < 3:
    rclpy.spin_once(n, timeout_sec=0.05)
print(f'{v[-1]:.3f}' if v else 'nan')
rclpy.try_shutdown()
PY
)
LEFT=$(python3 -c "
try:
    print(f'{$SIM_LIMIT - $NOW:.3f}')
except Exception:
    print('nan')")
say "  目前模擬時間 ${NOW}s；剩餘 ${LEFT}s；需要 ${NEED_S}s"
python3 -c "
import sys
try:
    sys.exit(0 if float('$LEFT') >= float('$NEED_S') else 1)
except Exception:
    sys.exit(1)" \
  || fail "剩餘模擬時間 ${LEFT}s 不足 ${NEED_S}s —— 不在時間不夠時放行命令源"

YAW=$(python3 -c "import json;print(json.load(open('$DIR/preflight.json')).get('base_yaw_deg') or 0.0)")
say "[7/8] 有界命令源（**sync 模式**：底盤 ${BASE_VX} m/s ＋ joint2 ${ARM_RATE} rad/s）"
say "  **同一則 9 維訊息同時帶兩個分量**，由同一個 scale 縮放"
say "  起始底盤 yaw ${YAW}°（底盤命令在 report frame +x，非本體 +x）"
timeout 120 python3 -u evaluation/wb_bounded_cmd.py --mode sync --out "$DIR" \
    --base-vx "$BASE_VX" --arm-rate "$ARM_RATE" \
    --expect-base-yaw-deg "$YAW" 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" = "0" ] || fail "命令源非零退出"

say "等執行端收尾（觀察窗 ${OBSERVE_S}s 之後）"
DONE=0
for i in $(seq 120); do
  kill -0 "${PIDS[0]}" 2>/dev/null || { DONE=1; break; }
  sleep 1
done
if [ "$DONE" != "1" ]; then
  ABORT="等待收尾逾時，強制清理"
  say "**$ABORT —— 標記為中止，不算停止行為驗收**"
  echo "$ABORT" > "$DIR/ABORTED.md"
fi

say "收尾檢查"
[ -f "$DIR/sim/wb_run.json" ] || fail "缺少 sim/wb_run.json"
[ -f "$DIR/cmd_source.json" ] || fail "缺少 cmd_source.json"
[ -f "$DIR/preflight.json" ] || fail "缺少 preflight.json"
say "收尾後 CPU $(python3 evaluation/cpu_temp.py)"
if [ -n "$ABORT" ]; then
  say "輸出 $DIR（**本趟標記為中止**，不做判定）"; exit 5
fi

say "[8/8] 離線判定（事前定版判準，不下修門檻）"
python3 -u evaluation/wb_sync_check.py --run "$DIR" 2>&1 | tee -a "$LOG"
CHECK=${PIPESTATUS[0]}
say "輸出 $DIR"
if [ "$CHECK" != "0" ]; then
  say "**判定未通過**（明細見 sync_check.json）"
  exit 6
fi
say "判定通過 —— **停下回報，不自動進 pregrasp**"
