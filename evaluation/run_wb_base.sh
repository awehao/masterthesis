#!/usr/bin/env bash
# Isaac 全身執行端：**只跑 base 一趟**。
#
# 起動前檢查未通過就收尾，不跳過。通過才啟動有界命令源。
# 每個子程序自成 process group；清理只針對本趟記錄到的 PID，
# 不用 pkill、不做字串比對、不影響其他工作階段。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; WS="$(dirname "$HERE")"
cd "$WS"
RUN_ID="${RUN_ID:-wb_base_$(date +%H%M%S)}"
DIR="$WS/evaluation/runs/$RUN_ID"; mkdir -p "$DIR"
LOG="$DIR/run.log"; : > "$LOG"
ISAAC_PY="${ISAAC_PY:-$HOME/venvs/isaacsim-6.0.1/bin/python}"
URDF="$WS/evaluation/models/omni_bot_wholebody_expanded.urdf"
SIM_LIMIT="${SIM_LIMIT:-30}"
PIDS=()
say(){ echo "[$(date +%T)] $*" | tee -a "$LOG"; }
spawn(){ local n="$1"; shift; setsid "$@" >>"$LOG" 2>&1 < /dev/null &
         PIDS+=($!); say "  起 $n PID=$!"; }
cleanup(){
  say "cleanup（只針對本趟 PID）..."
  for p in "${PIDS[@]:-}"; do kill -TERM -"$p" 2>/dev/null; kill -TERM "$p" 2>/dev/null; done
  sleep 3
  for p in "${PIDS[@]:-}"; do kill -KILL -"$p" 2>/dev/null; kill -KILL "$p" 2>/dev/null; done
  say "cleanup done"
}
trap cleanup EXIT

say "=== base 介面測試 RUN_ID=$RUN_ID domain=${ROS_DOMAIN_ID:-未設} ==="
say "起跑前 CPU $(python3 evaluation/cpu_temp.py)"

say "[1/6] 純邏輯測試（不開模擬器）"
python3 -u evaluation/test_wb_cmd_chain.py 2>&1 | tail -2 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" = "0" ] || { say "**純邏輯測試未通過，中止**"; exit 2; }

say "[2/6] 啟動 Isaac 執行端（--mode base）"
spawn isaac "$ISAAC_PY" -u evaluation/isaac_wholebody_sim.py \
    --out "$DIR/sim" --mode base --sim-limit "$SIM_LIMIT" --solver-label none

say "[3/6] 等 /clock 前進"
python3 evaluation/clock_advancing.py --discover 180 2>&1 | tee -a "$LOG" || {
    say "**/clock 未前進，中止**"; exit 3; }

say "[4/6] 啟動 robot_state_publisher、距離節點與安全層"
spawn rsp ros2 run robot_state_publisher robot_state_publisher "$URDF" \
    --ros-args -p use_sim_time:=true
spawn dist ros2 run ammr_wholebody_mpc arm_link_distance --ros-args \
    -p use_sim_time:=true -p report_frame:=odom -p geometry:=links \
    -p wholebody_urdf:="$URDF"
spawn safety ros2 run ammr_wholebody_mpc wholebody_safety --ros-args \
    -p use_sim_time:=true -p report_frame:=odom -p base_frame:=base_link \
    -p wholebody_urdf:="$URDF"
spawn adapter python3 -u evaluation/arm_vel_adapter.py
sleep 5

say "[5/6] 起動前檢查（內容 / 新鮮度 / 端點身分）"
if ! python3 -u evaluation/wb_preflight.py --out "$DIR" 2>&1 | tee -a "$LOG"; then
    say "**起動前檢查未通過 —— 不啟動命令源，保存原因後收尾**"
    exit 4
fi
YAW=$(python3 -c "import json;print(json.load(open('$DIR/preflight.json')).get('base_yaw_deg') or 0.0)")

say "[6/6] 有界命令源（report frame +x，起始 yaw ${YAW}°）"
python3 -u evaluation/wb_bounded_cmd.py --mode base --out "$DIR" \
    --expect-base-yaw-deg "$YAW" 2>&1 | tee -a "$LOG"

say "等執行端收尾"
for i in $(seq 60); do kill -0 "${PIDS[0]}" 2>/dev/null || break; sleep 1; done
say "收尾後 CPU $(python3 evaluation/cpu_temp.py)"
say "輸出 $DIR"
