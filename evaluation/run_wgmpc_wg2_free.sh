#!/usr/bin/env bash
# WG2 首趟自由空間整合測試。判準見
# evaluation/results/specs/wgmpc_wg2_run1_preregistration.yaml（啟動前已定版）。
#
# 清理只針對本趟記錄的 PID，每個子程序自成 process group；
# 不用 pkill、不做字串比對、不影響其他工作階段。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; WS="$(dirname "$HERE")"
cd "$WS"
set +u
. /opt/ros/jazzy/setup.bash
if [ -f "$WS/install/setup.bash" ]; then . "$WS/install/setup.bash"; else
  set -u; echo "**找不到 install/setup.bash —— 請先 colcon build**"; exit 65; fi
set -u
[ -z "${ROS_DOMAIN_ID:-}" ] && { echo "**ROS_DOMAIN_ID 未設定，拒絕啟動**"; exit 64; }
export ROS_DOMAIN_ID
RUN_ID="${RUN_ID:-wgmpc_wg2_free_$(date +%H%M%S)}"
DIR="$WS/evaluation/runs/$RUN_ID"; mkdir -p "$DIR"
LOG="$DIR/run.log"; : > "$LOG"
ISAAC_PY="${ISAAC_PY:-$HOME/venvs/isaacsim-6.0.1/bin/python}"
URDF_WB="$WS/evaluation/models/omni_bot_wholebody_expanded.urdf"
# ---- 事前定版的配置（見 preregistration）----
N=5; RATE=20; SIM_LIMIT=120; TASK_S=60
OFFSET="0.25 0.15 0.05"
REACH_P=0.005; REACH_R=0.02; HOLD_S=2.0
VMAX_BASE_LIN=0.035255; VMAX_BASE_ANG=0.199900; VMAX_ARM=0.999900

PIDS=(); NAMES=()
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
cleanup(){ say "cleanup（只針對本趟 PID）..."
  for i in "${!PIDS[@]}"; do kill -TERM -"${PIDS[$i]}" 2>/dev/null || true
                             kill -TERM "${PIDS[$i]}" 2>/dev/null || true; done
  sleep 3
  for i in "${!PIDS[@]}"; do kill -KILL "${PIDS[$i]}" 2>/dev/null || true; done
  say "cleanup done"; }
trap cleanup EXIT
spawn(){ local n="$1"; shift; setsid "$@" >>"$LOG" 2>&1 </dev/null &
         PIDS+=($!); NAMES+=("$n"); say "  起 $n PID=$!"; }

say "=== WG2 首趟自由空間整合測試 RUN_ID=$RUN_ID domain=$ROS_DOMAIN_ID ==="
say "起跑前 CPU $(python3 evaluation/cpu_temp.py)"
say "判準：N=$N dt=$(python3 -c "print(1/$RATE)") 偏移=($OFFSET) 到達≤${REACH_P}m/${REACH_R}rad 保持${HOLD_S}s"

say "[1/5] 起 Isaac 自由空間執行端（--mode sync，**無抽屜**）"
spawn isaac "$ISAAC_PY" -u evaluation/isaac_wholebody_sim_e2.py \
  --out "$DIR/sim" --mode sync --sim-limit "$SIM_LIMIT" --solver-label wgmpc_wg2
say "  等 Isaac 起 scene（最多 180 s）"
for i in $(seq 180); do
  grep -q '/joint_states\|進入物理\|physics' "$LOG" 2>/dev/null && break; sleep 1
done
sleep 25

say "[2/5] 起安全層（低速框 L1；D1 選 A ⇒ 世界逐軸 0.035255）"
spawn safety ros2 run ammr_wholebody_mpc wholebody_safety --ros-args \
  -p use_sim_time:=true -p report_frame:=odom -p base_frame:=base_link \
  -p wholebody_urdf:="$URDF_WB" -p freespace_confirmed:=false \
  -p vmax_base_lin:="$VMAX_BASE_LIN" -p vmax_base_ang:="$VMAX_BASE_ANG" \
  -p vmax_arm:="$VMAX_ARM"
sleep 6

say "[3/5] 低速框讀回比對（安全層 vs 本趟設定）"
python3 evaluation/coman_lowspeed_readback.py "$LOG" \
  "$VMAX_BASE_LIN" "$VMAX_BASE_ANG" "$VMAX_ARM" 0.05 0.2 1.0 \
  2>&1 | tee -a "$LOG" || exit 67

say "[4/5] 起 adapter（消費端為 Isaac 執行端）"
spawn adapter python3 -u evaluation/arm_vel_adapter.py \
  --consumer-node /isaac_wholebody_sim
sleep 5

say "[5/5] 起 W-GMPC 節點（N=$N、u_prev=strict ＋ 已確認初始靜止）"
python3 -u evaluation/wgmpc_wg2_node.py \
  --N "$N" --rate "$RATE" --target-offset $OFFSET \
  --reach-pos-m "$REACH_P" --reach-rot-rad "$REACH_R" --hold-s "$HOLD_S" \
  --duration-s "$TASK_S" --u-prev-policy strict --assume-initial-rest \
  --out "$DIR/wg2_out.json" 2>&1 | tee -a "$LOG"
say "收尾後 CPU $(python3 evaluation/cpu_temp.py)"
say "輸出目錄 $DIR"
