#!/usr/bin/env bash
# 抽屜接觸案例的一趟完整流程。
#
# 導航與第一階段維持凍結：本腳本不碰 isaac_bigarena_sim.py，也不碰
# isaac_manip_sim.py。抽屜資產只有本流程載入，bigarena.sdf 未被修改。
#
# 順序（離線檢查未過就不啟動模擬器）：
#   1 產生並逐點驗證軌跡
#   2 啟動 Isaac
#   3 等 /clock 與訂閱者
#   4 開錄
#   5 播放（含 engage / release 事件）
#   6 收工
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; WS="$(dirname "$HERE")"
cd "$WS"
CASE="${CASE:-drawer_open_a_fixed}"
RUN_ID="${RUN_ID:-drawer_$(date +%H%M%S)}"
DIR="$WS/evaluation/runs/$RUN_ID"; mkdir -p "$DIR"
LOG="$DIR/run.log"; : > "$LOG"
ISAAC_PY="${ISAAC_PY:-$HOME/venvs/isaacsim-6.0.1/bin/python}"
SIM_LIMIT="${SIM_LIMIT:-45}"
RTF="${RTF:-1.0}"
PIDS=()
say(){ echo "[$(date +%T)] $*" | tee -a "$LOG"; }
cleanup(){
  say "cleanup ..."
  for p in "${PIDS[@]:-}"; do kill -TERM -"$p" 2>/dev/null; kill -TERM "$p" 2>/dev/null; done
  sleep 3
  for p in "${PIDS[@]:-}"; do kill -KILL "$p" 2>/dev/null; done
  say "cleanup done"
}
trap cleanup EXIT

say "=== 抽屜案例 RUN_ID=$RUN_ID CASE=$CASE domain=${ROS_DOMAIN_ID:-未設} ==="
say "起跑前 CPU $(python3 evaluation/cpu_temp.py)"

say "[1/6] 產生並驗證軌跡"
if ! python3 -u evaluation/gen_drawer_traj.py --case "$CASE" --out "$DIR/traj" \
     2>&1 | tee "$DIR/traj_gen.log" | tee -a "$LOG" >/dev/null; then
    say "**軌跡驗證未通過，中止（不啟動模擬器）**"; exit 2
fi
say "  $(grep -E '^  [0-9]+ 個設定點' "$DIR/traj_gen.log")"
say "  $(grep -E '^判定：' "$DIR/traj_gen.log")"

say "[2/6] 啟動 Isaac"
setsid "$ISAAC_PY" -u evaluation/isaac_drawer_sim.py --case "$CASE" \
    --out "$DIR/sim" --sim-limit "$SIM_LIMIT" --rtf "$RTF" \
    >> "$LOG" 2>&1 < /dev/null &
ISAAC=$!; PIDS+=($ISAAC); say "  Isaac PID=$ISAAC"

say "[3/6] 等 /clock 前進與訂閱者"
if ! python3 evaluation/clock_advancing.py --discover 180 2>&1 | tee -a "$LOG"; then
    say "**/clock 未前進，中止**"; exit 3
fi
# 訂閱者的閘由播放端自己把關（play_drawer_traj.py 沒有訂閱者就 exit 3，
# 不在沒人收的情況下空跑），這裡不重複數一次。

say "[4/6] 開錄"
setsid ros2 bag record -o "$DIR/bag" \
    /arm/joint_position_cmd /manip/gripper_cmd /manip/phase_cmd \
    /joint_states /manip/tcp_pose /manip/drawer_state /manip/grasp_state \
    /manip/contact /manip/status /clock >> "$LOG" 2>&1 < /dev/null &
BAG=$!; PIDS+=($BAG); say "  bag PID=$BAG"; sleep 3

say "[5/6] 播放"
python3 -u evaluation/play_drawer_traj.py --traj "$DIR/traj/traj.csv" \
    --out "$DIR/sent.json" 2>&1 | tee "$DIR/play.log" | tee -a "$LOG" >/dev/null
say "  $(grep -E '^發布完成' "$DIR/play.log")"

say "[6/6] 等模擬器收尾"
for i in $(seq 120); do kill -0 $ISAAC 2>/dev/null || break; sleep 1; done
say "收尾後 CPU $(python3 evaluation/cpu_temp.py)"
say "輸出目錄 $DIR"
