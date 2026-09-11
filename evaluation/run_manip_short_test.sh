#!/usr/bin/env bash
# 固定底盤的手臂位置命令短測試。
#
# 這**不是**導航停放驗收：底盤由初始化程序直接放在案例的精確停放位姿，
# 之後每一步歸零保持不動。guard 以零命令持續發布 /cmd_vel，維持它是唯一
# 發布者 —— 先把交接後的拓樸跑通。
#
# 導航實驗維持凍結：本腳本不碰 isaac_bigarena_sim.py 也不啟動導航鏈。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; WS="$(dirname "$HERE")"
cd "$WS"
CASE="${CASE:-}"
RUN_ID="${RUN_ID:-manip_$(date +%H%M%S)}"
DIR="$WS/evaluation/runs/$RUN_ID"; mkdir -p "$DIR"
LOG="$DIR/run.log"; : > "$LOG"
URDF="$WS/evaluation/models/omni_bot_manip.urdf"
ISAAC_PY="${ISAAC_PY:-$HOME/venvs/isaacsim-6.0.1/bin/python}"
PIDS=()
say(){ echo "[$(date +%T)] $*" | tee -a "$LOG"; }
cleanup(){ say "cleanup ..."; for p in "${PIDS[@]:-}"; do kill -TERM -"$p" 2>/dev/null; kill -TERM "$p" 2>/dev/null; done; sleep 3; for p in "${PIDS[@]:-}"; do kill -KILL "$p" 2>/dev/null; done; say "cleanup done"; }
trap cleanup EXIT

say "=== 手臂短測試 RUN_ID=$RUN_ID  domain=${ROS_DOMAIN_ID:-未設} ==="

say "[1/7] 重新展開共用模型"
bash evaluation/expand_wholebody_urdf.sh "$URDF" 2>&1 | tee -a "$LOG"

say "[2/7] 離線沿途碰撞檢查（未過就不執行）"
if ! python3 evaluation/check_arm_path.py ${CASE:+--case "$CASE"} 2>&1 | tee "$DIR/path_check.log" | tee -a "$LOG"; then
    say "**沿途檢查未通過，中止**"; exit 2
fi

say "[3/7] 啟動 Isaac（固定底盤）"
setsid "$ISAAC_PY" evaluation/isaac_manip_sim.py ${CASE:+--case "$CASE"} \
    --urdf "$URDF" --sim-limit "${SIM_LIMIT:-60}" \
    --out "$DIR/manip_run.json" >> "$LOG" 2>&1 < /dev/null &
ISAAC=$!; PIDS+=($ISAAC)
say "  Isaac PID=$ISAAC"

say "[4/7] 啟動 robot_state_publisher（**同一份檔案**）與 guard"
# URDF 內容不能走命令列：ros2 的參數解析會對 XML 內的字元失敗
# （實測 "Failed to parse global arguments"，與長度無關 —— 90 KB 遠低於 ARG_MAX
# 的 2 MB）。改用 params 檔，內容仍然是同一個 $URDF。
PARAMS="$DIR/rsp_params.yaml"
python3 - "$URDF" "$PARAMS" <<'PYP'
import sys, yaml
xml = open(sys.argv[1]).read()
yaml.safe_dump({'robot_state_publisher': {'ros__parameters': {
    'use_sim_time': True, 'robot_description': xml}}},
    open(sys.argv[2], 'w'), allow_unicode=True, default_flow_style=False)
PYP
setsid ros2 run robot_state_publisher robot_state_publisher \
    --ros-args --params-file "$PARAMS" >> "$LOG" 2>&1 < /dev/null &
PIDS+=($!)
setsid ros2 run ammr_wholebody_mpc wheel_limit_guard --ros-args \
    -r __node:=wheel_limit_guard \
    -p use_sim_time:=true -p cmd_in_topic:=/cmd_vel_smoothed \
    -p cmd_out_topic:=/cmd_vel -p odom_topic:=/odom \
    -p wheel_radius:=0.05 -p wheel_base_L:=0.245 \
    -p wheel_w_max:=5.55 -p wheel_a_max:=125.0 \
    -p publish_rate:=20.0 -p input_timeout:=0.25 -p dt_max:=0.50 \
    -p assume_stopped_at_start:=true \
    >> "$LOG" 2>&1 < /dev/null &
PIDS+=($!)

say "[5/7] 等 /clock 前進，再做執行期核對"
timeout 120 python3 evaluation/clock_advancing.py --discover 90 --window 10 2>&1 | tee -a "$LOG" || { say "**/clock 沒有前進，中止**"; exit 3; }

fail=0
say "  -- 執行期模型一致性 --"
python3 evaluation/check_model_runtime.py --file "$URDF" 2>&1 | tee "$DIR/model_runtime.log" | tee -a "$LOG" || fail=1
say "  -- /cmd_vel 唯一發布者 --"
python3 evaluation/endpoint_check.py \
    --topics /clock,/cmd_vel,/joint_states,/manip/tcp_pose,/arm/joint_position_cmd \
    --resolve-wait 40 \
    --require-endpoint /arm/joint_position_cmd=sub \
    --sole-publisher /cmd_vel=wheel_limit_guard --out "$DIR/endpoints.json" \
    2>&1 | tee -a "$LOG" || fail=1
say "  -- TF 是否到得了 link_tcp --"
if timeout 30 ros2 run tf2_ros tf2_echo base_footprint link_tcp --ros-args -p use_sim_time:=true >> "$LOG" 2>&1; then :; fi
grep -q "At time" "$LOG" && say "    base_footprint -> link_tcp 可查" || { say "    !! base_footprint -> link_tcp 查不到"; fail=1; }
[ "$fail" -ne 0 ] && { say "**執行期核對未通過，不播放軌跡**"; exit 4; }

say "  -- 第三方計數：guard 是否真的在發（含零命令）--"
python3 evaluation/count_topics.py --seconds 20 --out "$DIR/third_party_counts.json" \
    --topic /cmd_vel=geometry_msgs/msg/Twist \
    --topic /wheel_guard/status=std_msgs/msg/String \
    --topic /joint_states=sensor_msgs/msg/JointState \
    2>&1 | tee -a "$LOG"

say "  -- 診斷：手臂命令 topic 在圖上的狀態 --"
{ echo "### ros2 topic list"; ros2 topic list 2>&1 | sort
  echo "### ros2 topic info -v /arm/joint_position_cmd"; ros2 topic info -v /arm/joint_position_cmd 2>&1
  echo "### ros2 node list"; ros2 node list 2>&1
} >> "$LOG" 2>&1

say "[6/7] 播放已檢查的軌跡"
python3 evaluation/play_arm_traj.py ${CASE:+--case "$CASE"} --out "$DIR/traj_sent.json" 2>&1 | tee -a "$LOG"

say "[7/7] 等 Isaac 存檔"
wait $ISAAC; rc=$?
say "  Isaac 退出碼 $rc"
[ -f "$DIR/manip_run.json" ] && python3 evaluation/report_manip_run.py \
    "$DIR/manip_run.json" --sent "$DIR/traj_sent.json" 2>&1 | tee -a "$LOG"
say "資料目錄 $DIR"
