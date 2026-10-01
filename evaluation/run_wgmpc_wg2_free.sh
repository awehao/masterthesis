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
N=5; RATE=20; SIM_LIMIT=120
OFFSET="0.25 0.15 0.05"
REACH_P=0.005; REACH_R=0.02; HOLD_S=2.0
VMAX_BASE_LIN=0.035255; VMAX_BASE_ANG=0.199900; VMAX_ARM=0.999900
# **模型選擇要顯式傳入**：節點預設是 ideal，不傳就會跑成原核心。
ARM_MODEL="${ARM_MODEL:-setpoint}"
ARM_IDENT="$WS/evaluation/results/wgmpc_arm_sp_ident_free4.json"
# **迴路延遲補償**：rec7 實錄量到端到端延遲 ≈ 1.40 個控制週期
#（發布延遲 0.60 ＋ cmd_age 0.60 ＋ 一個物理步）。
# 離線閉迴路在延遲 1.4 週期、補償 0 時重現 rec7 的不收斂；
# 補償 1.4 時 1.35 s 到達並保持。**不改權重、視界或任何限制。**
DELAY_COMP="${DELAY_COMP:-1.4}"
# **任務時間預算**：牆鐘上限留寬，由**模擬時間**預算與 free4 對齊。
# 錄影會拖慢 sim:wall（free4 無錄影時為 0.997），只靠牆鐘會讓任務
# 拿到的模擬時間比 free4 少。free4 任務覆蓋模擬 59.61 s。
TASK_SIM_S=60; TASK_WALL_S=400
# ---- 錄影（模擬器內相機 /World/rec_cam，**不是桌面錄製**）----
# 解析度與 fps 明確指定，不用較高負載的預設（1920x1080／30 fps）。
REC_RES="1280x720"; REC_FPS=10
# 取景由 URDF 連桿原點 ∪ free4 實際軌跡 ∪ 目標算出，含 0.12 m 幾何加厚。
# 距離 **3.80 m**：rec5 趟次的實拍影格反推實際 vFOV ≈ 26–27.6°
#（與「水平光圈 20.955 mm、焦距 24 mm」的假設相符），
# 故不再用悲觀 20° 的 4.45 m —— 那讓畫面過空、機器人只佔 313/720 像素。
# 3.80 m 對 vFOV 24° 仍涵蓋八個角點，機器人高約 367/720 像素。
REC_AT="0.1322,0.0000,0.4468"
REC_EYE="2.1183,-2.9471,1.7923"
# 目標標記：名目 FK（q=0）＋偏移。與 free4 實錄差 0.0016 m，遠小於標記半徑
# 0.020 m。**權威目標是節點在趟中算出並寫進 wg2_out.json 的那一個。**
REC_TARGET="0.44700,0.15000,0.44999"

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
say "**手臂執行模型：$ARM_MODEL**（辨識檔 $(basename "$ARM_IDENT")）"
say "**迴路延遲補償：$DELAY_COMP 個控制週期**（實測端到端 ≈ 1.40）"
say "錄影：$REC_RES @ ${REC_FPS}fps、模擬器內相機、at=$REC_AT eye=$REC_EYE target=$REC_TARGET"
say "時間預算：模擬 ${TASK_SIM_S}s（與 free4 的 59.61s 對齊）、牆鐘上限 ${TASK_WALL_S}s"
python3 - <<EOF | tee -a "$LOG"
import json, os
json.dump({'arm_model': '$ARM_MODEL', 'arm_ident': '$ARM_IDENT',
           'N': $N, 'rate_hz': $RATE, 'offset_m': [$(echo $OFFSET | tr ' ' ',')],
           'reach_pos_m': $REACH_P, 'reach_rot_rad': $REACH_R,
           'hold_s': $HOLD_S, 'sim_limit_s': $SIM_LIMIT,
           'task_sim_s': $TASK_SIM_S, 'task_wall_s': $TASK_WALL_S,
           'recording': {'res': '$REC_RES', 'fps': $REC_FPS,
                         'at': '$REC_AT', 'eye': '$REC_EYE',
                         'target_marker': '$REC_TARGET',
                         'source': '模擬器內相機 /World/rec_cam，非桌面錄製'},
           'low_speed_box_l1': {'base_lin': $VMAX_BASE_LIN,
                                'base_ang': $VMAX_BASE_ANG,
                                'arm': $VMAX_ARM}},
          open('$DIR/run_config.json', 'w'), ensure_ascii=False, indent=1)
print('啟動配置已落盤 $DIR/run_config.json')
EOF

say "[1/6] 起 Isaac 執行端（--mode solver_freespace，**無抽屜、無額外碰撞體**）"
# **--mode solver_freespace**：執行端會遍歷 UsdPhysics.CollisionAPI，
# 機器人與地面以外若有任何碰撞體就 return 8 中止
# （語意檢查，不是靠 runner 名稱放行）。
spawn isaac "$ISAAC_PY" -u evaluation/isaac_wholebody_sim_e2.py \
  --out "$DIR/sim" --mode solver_freespace --sim-limit "$SIM_LIMIT" \
  --solver-label wgmpc_wg2 \
  --record-frames "$DIR/frames" --record-res "$REC_RES" \
  --record-fps "$REC_FPS" --record-at "$REC_AT" --record-eye "$REC_EYE" \
  --record-target "$REC_TARGET" \
  --run-label "WG2 自由空間閉迴路：W-GMPC N=5（$ARM_MODEL 模型）到達與保持"
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

say "  起 W-GMPC 節點（**--arm-model $ARM_MODEL**、N=$N、u_prev=strict ＋ 已確認初始靜止）"
[ "$ARM_MODEL" = "setpoint" ] && { [ -f "$ARM_IDENT" ] || {
  echo "**找不到辨識檔 $ARM_IDENT**" | tee -a "$LOG"; exit 66; }; }
python3 -u evaluation/wgmpc_wg2_node.py \
  --arm-model "$ARM_MODEL" --arm-ident "$ARM_IDENT" \
  --delay-comp-cycles "$DELAY_COMP" \
  --N "$N" --rate "$RATE" --target-offset $OFFSET \
  --reach-pos-m "$REACH_P" --reach-rot-rad "$REACH_R" --hold-s "$HOLD_S" \
  --duration-s "$TASK_WALL_S" --duration-sim-s "$TASK_SIM_S" \
  --u-prev-policy strict --assume-initial-rest \
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
# **不以「目錄內有任意 JSON」判定**：加入錄影後目錄會有其他產物，
# 存在不等於執行結果已落盤。改為核對 sim/wb_run.json 能解析、
# 必要紀錄齊備、且錄影索引與磁碟影格數一致。
ARCHIVE_OK=0
for i in $(seq "$ARCHIVE_WAIT_S"); do
  if [ -f "$DIR/sim/wb_run.json" ]; then
    if python3 evaluation/wgmpc_wg2_archive_check.py "$DIR" \
         --expect-recording --expect-arm-model "$ARM_MODEL" \
         --out "$DIR/archive_check.json" >/dev/null 2>&1; then
      ARCHIVE_OK=1; break
    fi
  fi
  sleep 1
done
say "  封存內容核對結果："
python3 evaluation/wgmpc_wg2_archive_check.py "$DIR" \
  --expect-recording --expect-arm-model "$ARM_MODEL" \
  --out "$DIR/archive_check.json" 2>&1 | tee -a "$LOG"
if [ "$ARCHIVE_OK" = "1" ]; then
  say "  **封存完整**（內容已核對，不只是檔案存在）"
  echo '{"archive_complete": true, "basis": "wgmpc_wg2_archive_check.py 內容核對"}' \
    > "$DIR/archive_status.json"
else
  say "  **封存不完整 ⇒ 升級終止；結果保留，不自動重跑**"
  echo '{"archive_complete": false, "basis": "wgmpc_wg2_archive_check.py 內容核對未通過", "reason": "執行端未在 '"$ARCHIVE_WAIT_S"' s 內完成可核對的封存"}' \
    > "$DIR/archive_status.json"
fi

if [ "$ARCHIVE_OK" = "1" ]; then
  say "[收尾 2b/3] 轉成可播放 MP4（**影格完整才轉**）"
  python3 evaluation/wgmpc_wg2_make_video.py "$DIR" --fps "$REC_FPS" \
    --skip-archive-check 2>&1 | tee -a "$LOG"
else
  say "[收尾 2b/3] 封存不完整 ⇒ **不轉檔**；影格原樣保留"
fi

say "[收尾 3/3] cleanup 由 trap 執行（只針對本趟 PID）"
say "收尾後 CPU $(python3 evaluation/cpu_temp.py)"
say "輸出目錄 $DIR"
