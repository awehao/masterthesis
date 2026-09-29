#!/usr/bin/env bash
# 協同版的**握手接線短測**：建立連接 → 凍結手臂（不拉動）→ 釋放請求 →
# 執行 → **讀回確認**。固定底盤、不錄影、不拉動。
#
# 與 run_drawer_case.sh 的關係：那支跑 isaac_drawer_sim.py（固定底座 200 mm
# 成果的程式版本），**維持不動**。本支只跑協同版 isaac_coman_drawer_sim.py。
#
# 本流程一律為**診斷**：--handshake-test 會把趟次標記 is_diagnostic。
# RECORD=<目錄以外的任意非空值> 開啟模擬器內相機錄影（沿用既有預設鏡位）。
# 錄影會吃掉即時餘裕，可能改變 RTF；畫面內不加任何文字。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; WS="$(dirname "$HERE")"
cd "$WS"
if [ -z "${ROS_DOMAIN_ID:-}" ]; then
  echo "**ROS_DOMAIN_ID 未設定，拒絕啟動**"; exit 64
fi
export ROS_DOMAIN_ID
CASE="${CASE:-drawer_open_a_fixed}"
RUN_ID="${RUN_ID:-coman_handshake_$(date +%H%M%S)}"
DIR="$WS/evaluation/runs/$RUN_ID"; mkdir -p "$DIR"
LOG="$DIR/run.log"; : > "$LOG"
ISAAC_PY="${ISAAC_PY:-$HOME/venvs/isaacsim-6.0.1/bin/python}"
SIM_LIMIT="${SIM_LIMIT:-30}"
REL_AFTER="${REL_AFTER:-20}"          # 連接後第幾步送出釋放請求
STOP_AFTER="${STOP_AFTER:-20}"        # 確認後再跑幾步才收尾
POST_STOP_STEPS="${POST_STOP_STEPS:-20}"   # **事前指定**的停止後觀察步數
PIDS=()
say(){ echo "[$(date +%T)] $*" | tee -a "$LOG"; }
cleanup(){
  say "cleanup（只針對本趟 PID）..."
  for p in "${PIDS[@]:-}"; do kill -TERM -"$p" 2>/dev/null; kill -TERM "$p" 2>/dev/null; done
  sleep 3
  for p in "${PIDS[@]:-}"; do kill -KILL "$p" 2>/dev/null; done
  say "cleanup done"
}
trap cleanup EXIT

say "=== 協同握手短測 RUN_ID=$RUN_ID CASE=$CASE domain=$ROS_DOMAIN_ID ==="
say "起跑前 CPU $(python3 evaluation/cpu_temp.py)"

say "[1/5] 產生並驗證軌跡（沿用既有產生器；連接後的事件會被執行端攔下）"
if ! python3 -u evaluation/gen_drawer_traj.py --case "$CASE" --out "$DIR/traj" \
     --pull-time-scale "${PULL_SCALE:-2.0}" --handover "${HANDOVER:-offset}" \
     ${PULL_TARGET:+--pull-target-m "$PULL_TARGET"} \
     2>&1 | tee "$DIR/traj_gen.log" | tee -a "$LOG" >/dev/null; then
    say "**軌跡驗證未通過，中止（不啟動模擬器）**"; exit 2
fi
say "  $(grep -E '^  [0-9]+ 個設定點' "$DIR/traj_gen.log")"

say "[2/5] 啟動 Isaac（協同版）"
setsid "$ISAAC_PY" -u evaluation/isaac_coman_drawer_sim.py --case "$CASE" \
    --out "$DIR/sim" --sim-limit "$SIM_LIMIT" \
    --handshake-test --handshake-release-after-steps "$REL_AFTER" \
    --handshake-stop-after-confirm-steps "$STOP_AFTER" \
    --post-stop-steps "$POST_STOP_STEPS" \
    ${INJECT_FAULT:+--inject-fault "$INJECT_FAULT"} \
    ${INJECT_FAULT_STEP:+--inject-fault-step "$INJECT_FAULT_STEP"} \
    ${RECORD:+--record-frames "$DIR/frames"} \
    ${RECORD_FPS:+--record-fps "$RECORD_FPS"} \
    ${RECORD_RES:+--record-res "$RECORD_RES"} \
    ${RECORD_FOCAL:+--record-focal "$RECORD_FOCAL"} \
    ${RECORD_EYE:+--record-eye "$RECORD_EYE"} \
    ${RECORD_AT:+--record-at "$RECORD_AT"} \
    >> "$LOG" 2>&1 < /dev/null &
ISAAC=$!; PIDS+=($ISAAC); say "  Isaac PID=$ISAAC"

say "[3/5] 等 /clock 前進與訂閱者"
if ! python3 evaluation/clock_advancing.py --discover 180 2>&1 | tee -a "$LOG"; then
    say "**/clock 未前進，中止**"; exit 3
fi

say "[4/5] 播放（走到 engage；其後事件由執行端攔下）"
python3 -u evaluation/play_drawer_traj.py --traj "$DIR/traj/traj.csv" \
    --out "$DIR/sent.json" 2>&1 | tee "$DIR/play.log" | tee -a "$LOG" >/dev/null
say "  $(grep -E '^發布完成' "$DIR/play.log" || echo '（播放端已結束）')"

say "[5/5] 等模擬器收尾"
for i in $(seq 180); do kill -0 $ISAAC 2>/dev/null || break; sleep 1; done
say "收尾後 CPU $(python3 evaluation/cpu_temp.py)"
say "輸出目錄 $DIR"
