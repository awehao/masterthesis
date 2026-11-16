#!/usr/bin/env bash
# 主線 B 取像趟監督節點的 ROS 乾跑（不起 Isaac）：遠距乾跑世界節點（運動學受控對象、obs 模式＝以歷史影像估計合成 /far/handle_obs，1.6 m 內標有效——只驗接線，不是視覺）
# ＋原導航鏈真節點（GMPC_TASK=1 任務擴充、GMPC_CMD_TOPIC=cmd_vel_nav_raw 由監督節點轉送）＋取像監督節點。
# 用法：bash evaluation/motm_b/run_bcap_dry.sh <輸出前綴> <模擬時長上限 s>
set -u
WS="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$WS"
OUT="$1"; TEND="${2:-120}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-75}" ROS_LOCALHOST_ONLY=1
set +u; source /opt/ros/jazzy/setup.bash; source install/setup.bash; set -u
PAT="[s]can_safety_shield|[w]heel_limit_guard|[v]elocity_smoother|[g]mpc_task_node|[g]mpc_node|[p]lanner_server|[w]orld_node.py|[b]_capture_node|[m]erge_node.py|[w]orld_ahead_node.py"
for i in $(seq 15); do n=$(pgrep -fc "$PAT" || true); [ "$n" = "0" ] && break; sleep 2; done
[ "$n" = "0" ] || { echo "已有相關節點在跑（$n）⇒ 拒跑"; exit 99; }
PIDS=()
cleanup(){ for p in "${PIDS[@]}"; do kill -INT -- "-$p" 2>/dev/null; done; sleep 4; for p in "${PIDS[@]}"; do kill -KILL -- "-$p" 2>/dev/null; done; }
trap cleanup EXIT
Q0="1.5,-2.8,1.5708,-0.0128,-0.0519,0.086,0.0116,-1.6249,-0.0018"
PYS="env -u VIRTUAL_ENV PATH=/usr/local/bin:/usr/bin:/bin /usr/bin/python3"
NI="nice -n ${NICE:-10}"
FAR_GRAV=1 setsid $NI $PYS -u evaluation/far_dry/world_node.py "$OUT" "$Q0" "$TEND" obs > "$OUT.world.log" 2>&1 < /dev/null & PIDS+=( $! ); WPID=$!
sleep 3
NO_GZ=1 BIGARENA=1 SCENE=mission_house_m8 NO_TRAFFIC=1 HEADING=1 GUARD=1 SHIELD=1 GMPC_TASK=1 GMPC_CMD_TOPIC=cmd_vel_nav_raw SPAWN_X=1.5 SPAWN_Y=-2.8 \
  setsid $NI ros2 launch my_omnibot_description omni_bot_dynamic.launch.py gui:=false use_arm:=true > "$OUT.launch.log" 2>&1 < /dev/null & PIDS+=( $! )
sleep 15
for i in $(seq 40); do timeout 6 ros2 lifecycle get /amcl 2>/dev/null | grep -q "^active" && break; sleep 2; done
echo "amcl $(timeout 6 ros2 lifecycle get /amcl 2>/dev/null | head -1)" > "$OUT.lifecycle.txt"
timeout 10 ros2 topic pub -t 3 -r 1 /initialpose geometry_msgs/msg/PoseWithCovarianceStamped "{header: {frame_id: 'map'}, pose: {pose: {position: {x: 1.5, y: -2.8, z: 0.0}, orientation: {z: 0.7071, w: 0.7071}}}}" > "$OUT.initpose.log" 2>&1 || true
sleep 3
setsid $NI $PYS -u evaluation/motm_b/${B_NODE:-b_capture_node2.py} "$OUT" > "$OUT.sup.log" 2>&1 < /dev/null & PIDS+=( $! ); SPID=$!
for i in $(seq $((TEND * 3 + 60))); do kill -0 "$WPID" 2>/dev/null || break; kill -0 "$SPID" 2>/dev/null || break; sleep 1; done
kill -INT -- "-$SPID" 2>/dev/null; sleep 5
echo done
