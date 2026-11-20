#!/usr/bin/env bash
# D2 同鏈乾跑（Codex 20261117_113330）：world_d2_node（受控對象＋D2 執行器：抓持→綁定→開門）＋原導航鏈（GMPC_TASK=1、GMPC_CMD_TOPIC=cmd_vel_nav_raw）＋監督 b_d2_node。不起 Isaac。
# 用法：bash evaluation/motm_dry/run_d2.sh <輸出前綴> <模擬上限 s>
set -u
WS="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$WS"
OUT="$1"; TEND="${2:-80}"
SSX="$(echo "${D2_START:-1.5,-2.8}" | cut -d, -f1)"; SSY="$(echo "${D2_START:-1.5,-2.8}" | cut -d, -f2)"   # 起點（同 world／替身的 D2_START）
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-77}" ROS_LOCALHOST_ONLY=1
set +u; source /opt/ros/jazzy/setup.bash; source install/setup.bash; set -u
PAT="[s]tub_isaac_d2|[s]can_safety_shield|[w]heel_limit_guard|[v]elocity_smoother|[g]mpc_task_node|[g]mpc_node|[p]lanner_server|[w]orld_node.py|[w]orld_d2_node|[w]orld_ahead_node|[b]_capture_node|[b]_d2_node"
for i in $(seq 15); do n=$(pgrep -fc "$PAT" || true); [ "$n" = "0" ] && break; sleep 2; done
[ "$n" = "0" ] || { echo "已有相關節點在跑（$n）⇒ 拒跑"; exit 99; }
PIDS=()
cleanup(){ for p in "${PIDS[@]}"; do kill -INT -- "-$p" 2>/dev/null; done; sleep 5; for p in "${PIDS[@]}"; do kill -KILL -- "-$p" 2>/dev/null; done; }
trap cleanup EXIT
PYS="env -u VIRTUAL_ENV PATH=/usr/local/bin:/usr/bin:/bin /usr/bin/python3"
export D2_RTF="${D2_RTF:-0.3}" B_RTF="${D2_RTF:-0.3}" B_SWEEP_S="${B_SWEEP_S:-0.030}" B_POOL_N="${B_POOL_N:-2}" B_POOL_N_PL="${B_POOL_N_PL:-2}" B_PLANNER_CPUS="${B_PLANNER_CPUS:-4,12}"
export D2_PLANT=isaac D2_LENIENT=0 B_APP_HO=1 B_DEADLINE_MS="${B_DEADLINE_MS:-150}" AHEAD_DIAG_DEADLINE_MS="${AHEAD_DIAG_DEADLINE_MS:-150}" CAND_STREAM="${CAND_STREAM:-0}"   # 介面替身核對：與 run_d2_isaac.sh 同設定（不寬鬆）
if [ "${D2_LENIENT:-0}" = "1" ]; then export B_DEADLINE_MS="${B_DEADLINE_MS:-150}" AHEAD_DIAG_DEADLINE_MS="${AHEAD_DIAG_DEADLINE_MS:-150}" CAND_STREAM="${CAND_STREAM:-0}"; fi   # 寬鬆模式：先成功後驗證（期限 150 ms 牆鐘＜RTF 0.3 一拍 167 ms，物理仍不等計算）
export DOOR_BASE_CLEAR_M=0.10 DOOR_BASE_CLEAR_PLAN_M=0.12 GRASP_TAIL_CERT=${GRASP_TAIL_CERT:-0} TAIL_CERT=${TAIL_CERT:-1} POOL_N=${POOL_N:-6} CAND_STREAM_CHUNK=${CAND_STREAM_CHUNK:-4} AHEAD_GC_FREEZE=${AHEAD_GC_FREEZE:-1} SHARED_POOL=1 DOOR_ARM_MODE="${DOOR_ARM_MODE:-qp}" DOOR_WHEEL_FRAC="${DOOR_WHEEL_FRAC:-0.95,0.75}"
setsid taskset -c "${WORLD_CPUS:-4,12,5,13,6,14}" $PYS -u evaluation/motm_dry/world_d2_node.py "$OUT" "$TEND" > "$OUT.world.log" 2>&1 < /dev/null & PIDS+=( $! ); WPID=$!
for i in $(seq 240); do grep -q "world_ready" "$OUT.world.log" 2>/dev/null && break; kill -0 "$WPID" 2>/dev/null || { echo "世界節點提前結束"; tail -20 "$OUT.world.log"; exit 3; }; sleep 1; done
grep -q "world_ready" "$OUT.world.log" || { echo "世界節點未就緒"; exit 3; }
setsid taskset -c "${STUB_CPUS:-0,8,1,9}" $PYS -u evaluation/motm_dry/stub_isaac_d2.py "$OUT" "$TEND" > "$OUT.stub.log" 2>&1 < /dev/null & PIDS+=( $! )
sleep 3
if [ "${D2_DOOR_CLOSE:-0}" = "1" ]; then                      # 回程關門：Shield 改吃 /scan_shield（綁定共轉期間遮門扇；Howard 11/20 選 b）
  export SHIELD_SCAN_TOPIC=/scan_shield
  setsid taskset -c "${SUP_CPUS:-3,11}" $PYS -u evaluation/door_leaf_scan_mask.py > "$OUT.mask.log" 2>&1 < /dev/null & PIDS+=( $! )
fi
NO_GZ=1 BIGARENA=1 SCENE="${M8_MAP:-mission_house_m8}" NO_TRAFFIC=1 HEADING=1 GUARD=1 SHIELD=1 GMPC_TASK=1 GMPC_CMD_TOPIC=cmd_vel_nav_raw SPAWN_X="$SSX" SPAWN_Y="$SSY" \
  setsid taskset -c "${NAV_CPUS:-2,10,7,15}" ros2 launch my_omnibot_description omni_bot_dynamic.launch.py gui:=false use_arm:=true > "$OUT.launch.log" 2>&1 < /dev/null & PIDS+=( $! )
sleep 15
for i in $(seq 40); do timeout 6 ros2 lifecycle get /amcl 2>/dev/null | grep -q "^active" && break; sleep 2; done
timeout 10 ros2 topic pub -t 3 -r 1 /initialpose geometry_msgs/msg/PoseWithCovarianceStamped "{header: {frame_id: 'map'}, pose: {pose: {position: {x: $SSX, y: $SSY, z: 0.0}, orientation: {z: 0.7071, w: 0.7071}}}}" > "$OUT.initpose.log" 2>&1 || true
sleep 3
setsid taskset -c "${SUP_CPUS:-3,11}" $PYS -u evaluation/motm_b/b_d2_node.py "$OUT" > "$OUT.sup.log" 2>&1 < /dev/null & PIDS+=( $! ); SPID=$!
for i in $(seq $((TEND * 4 + 120))); do kill -0 "$WPID" 2>/dev/null || break; sleep 1; done
sleep 3; kill -INT -- "-$SPID" 2>/dev/null; sleep 4
echo done
