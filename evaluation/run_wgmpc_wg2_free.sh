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
# rsp 用 manip 版（底盤不是關節 ⇒ 不需要 base_x/y/theta 的 joint_states）。
URDF_TF="$WS/evaluation/models/omni_bot_manip.urdf"
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

say "[1/6] 起 Isaac 執行端（--mode solver_freespace，**無抽屜、無額外碰撞體**）"
# **--mode solver_freespace**：執行端會遍歷 UsdPhysics.CollisionAPI，
# 機器人與地面以外若有任何碰撞體就 return 8 中止
# （語意檢查，不是靠 runner 名稱放行）。
spawn isaac "$ISAAC_PY" -u evaluation/isaac_wholebody_sim_e2.py \
  --out "$DIR/sim" --mode solver_freespace --sim-limit "$SIM_LIMIT" \
  --solver-label wgmpc_wg2 \
  --run-label "WG2 自由空間閉迴路：W-GMPC N=5 在已確認空場景的到達與保持"
say "  等 Isaac 起 scene（最多 180 s）"
for i in $(seq 180); do
  grep -q '/joint_states\|進入物理\|physics' "$LOG" 2>/dev/null && break; sleep 1
done
sleep 25

say "[2/6] 等 /clock 前進"
timeout 200 python3 evaluation/clock_advancing.py --discover 180 \
  >>"$LOG" 2>&1 || { echo "**/clock 未前進**" | tee -a "$LOG"; exit 68; }

say "[3/6] 起 robot_state_publisher 與距離節點（**外部障礙物集合為空**）"
# **兩份 URDF 用途不同**：manip 版給 rsp 發 TF；wholebody 版只當距離／安全節點
# 的 FK 參數（底盤是真實關節，拿去發 TF 會要求 base_x/y/theta 的 joint_states）。
spawn rsp ros2 run robot_state_publisher robot_state_publisher "$URDF_TF" \
  --ros-args -p use_sim_time:=true
# `geometry:=links`：**不給 obstacles 參數** ⇒ live 為空 ⇒ 每個連桿仍發一列
# STATUS_NODATA（arm_link_distance 的 `if T is None or not live:` 路徑）。
# 這與「根本沒收到距離資料（pts is None ⇒ reason 5 停止）」是**不同**情形。
spawn dist ros2 run ammr_wholebody_mpc arm_link_distance --ros-args \
  -p use_sim_time:=true -p report_frame:=odom -p geometry:=links \
  -p wholebody_urdf:="$URDF_WB"
sleep 8

say "[4/6] 起安全層（低速框 L1；D1 選 A ⇒ 世界逐軸 0.035255）"
spawn safety ros2 run ammr_wholebody_mpc wholebody_safety --ros-args \
  -p use_sim_time:=true -p report_frame:=odom -p base_frame:=base_link \
  -p wholebody_urdf:="$URDF_WB" \
  -p vmax_base_lin:="$VMAX_BASE_LIN" -p vmax_base_ang:="$VMAX_BASE_ANG" \
  -p vmax_arm:="$VMAX_ARM" \
  -p freespace_confirmed:=true
sleep 6

say "[5/6] 低速框讀回比對 ＋ TF／NODATA 通路核對"
python3 evaluation/coman_lowspeed_readback.py "$LOG" \
  "$VMAX_BASE_LIN" "$VMAX_BASE_ANG" "$VMAX_ARM" 0.05 0.2 1.0 \
  2>&1 | tee -a "$LOG" || exit 67

# **TF 缺失也會產生 NODATA** ⇒ 不能只看 NODATA 就宣稱空場景。
# 這裡另核必要 TF 是否有效，以及 NODATA 列是否真的發布並被安全層解析。
python3 evaluation/wgmpc_wg2_freespace_check.py --out "$DIR/freespace_check.json" \
  2>&1 | tee -a "$LOG" || exit 69

say "[6/6] 起 adapter 與 W-GMPC 節點"
spawn adapter python3 -u evaluation/arm_vel_adapter.py \
  --consumer-node /isaac_wholebody_sim
sleep 5

say "  起 W-GMPC 節點（N=$N、u_prev=strict ＋ 已確認初始靜止）"
python3 -u evaluation/wgmpc_wg2_node.py \
  --N "$N" --rate "$RATE" --target-offset $OFFSET \
  --reach-pos-m "$REACH_P" --reach-rot-rad "$REACH_R" --hold-s "$HOLD_S" \
  --duration-s "$TASK_S" --u-prev-policy strict --assume-initial-rest \
  --out "$DIR/wg2_out.json" 2>&1 | tee -a "$LOG"

# ---- 受控停止與落盤**先於** cleanup ----
# 先前的缺陷：節點先結束 ⇒ trap cleanup 立刻 SIGTERM Isaac，
# 而執行端只在自己的迴圈結束後才落盤 ⇒ **能說明為何停止套用的紀錄被弄丟**。
# 現在：請執行端走受控停止（送停止請求），等它自己完成封存，
# **等待有上限**；超時才升級終止並標記「封存不完整」。
ARCHIVE_WAIT_S="${ARCHIVE_WAIT_S:-180}"
say "[收尾 1/3] 請執行端受控停止（/wb_sim/stop_request）"
timeout 10 ros2 topic pub --once /wb_sim/stop_request std_msgs/msg/String \
  "{data: 'wgmpc_wg2 節點已結束，請走受控停止與停止觀察後封存'}" \
  >>"$LOG" 2>&1 || say "  （停止請求發布失敗，改等自然收尾）"

say "[收尾 2/3] 等執行端完成封存（上限 ${ARCHIVE_WAIT_S} s）"
ARCHIVE_OK=0
for i in $(seq "$ARCHIVE_WAIT_S"); do
  # 執行端的封存檔出現且非空 ⇒ 完成
  if ls "$DIR/sim/"*.json >/dev/null 2>&1; then ARCHIVE_OK=1; break; fi
  sleep 1
done
if [ "$ARCHIVE_OK" = "1" ]; then
  say "  **封存完成**（$(ls "$DIR/sim/"*.json | tr '\n' ' ')）"
  echo '{"archive_complete": true}' > "$DIR/archive_status.json"
else
  say "  **等待逾時 ⇒ 升級終止，標記「封存不完整」**"
  echo '{"archive_complete": false, "reason": "執行端未在 '"$ARCHIVE_WAIT_S"' s 內完成封存，由 cleanup 升級終止"}' \
    > "$DIR/archive_status.json"
fi

say "[收尾 3/3] cleanup 由 trap 執行（只針對本趟 PID）"
say "收尾後 CPU $(python3 evaluation/cpu_temp.py)"
say "輸出目錄 $DIR"
