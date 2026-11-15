#!/usr/bin/env bash
# 遠距隔離真節點乾跑（進線模式；不起 Isaac）：世界節點（合成感測＋8 步延遲）＋原導航鏈真節點＋進線監督節點（命令選擇、進線 GMPC、手臂追蹤）。
#   與基準相同的開關；另設 GMPC_CMD_TOPIC=cmd_vel_nav_raw 讓命令選擇器插在平滑器之前（平滑器→Shield→GUARD 不變）。目標由監督節點依把手估計送出。
# 用法：bash evaluation/far_dry/run_far_merge.sh <輸出前綴> <起點9維,逗號> <時長上限s>
set -u
WS="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$WS"
OUT="$1"; Q0="$2"; TEND="${3:-150}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-72}" ROS_LOCALHOST_ONLY=1
set +u; source /opt/ros/jazzy/setup.bash; source install/setup.bash; set -u
n=$(pgrep -fc "[s]can_safety_shield|[w]heel_limit_guard|[v]elocity_smoother|[g]mpc_node|[p]lanner_server|[w]orld_node|[m]erge_node" || true)
[ "$n" = "0" ] || { echo "已有相關節點在跑（$n）⇒ 拒跑"; exit 99; }
PIDS=()
cleanup(){ for p in "${PIDS[@]}"; do kill -INT -- "-$p" 2>/dev/null; done; pkill -INT -f "[g]mpc_node.*gmpc_merge" 2>/dev/null; sleep 3; for p in "${PIDS[@]}"; do kill -KILL -- "-$p" 2>/dev/null; done; pkill -KILL -f "[g]mpc_node.*gmpc_merge" 2>/dev/null; }
trap cleanup EXIT
SX=$(echo "$Q0" | cut -d, -f1); SY=$(echo "$Q0" | cut -d, -f2); SYAW=$(echo "$Q0" | cut -d, -f3)
PYS="env -u VIRTUAL_ENV /usr/bin/python3"
setsid $PYS -u evaluation/far_dry/world_node.py "$OUT" "$Q0" "$TEND" > "$OUT.world.log" 2>&1 < /dev/null & PIDS+=( $! ); WPID=$!
sleep 2
NO_GZ=1 BIGARENA=1 SCENE=mission_house_m8 NO_TRAFFIC=1 HEADING=1 GUARD=1 SHIELD=1 GMPC_CMD_TOPIC=cmd_vel_nav_raw SPAWN_X="$SX" SPAWN_Y="$SY" \
  setsid ros2 launch my_omnibot_description omni_bot_dynamic.launch.py gui:=false use_arm:=true > "$OUT.launch.log" 2>&1 < /dev/null & PIDS+=( $! )
setsid $PYS -u evaluation/far_dry/logger_node.py "$OUT" > "$OUT.logger.log" 2>&1 < /dev/null & PIDS+=( $! )
sleep 22
QZ=$(python3 -c "import math;print(math.sin($SYAW/2))"); QW=$(python3 -c "import math;print(math.cos($SYAW/2))")
timeout 10 ros2 topic pub -t 3 -r 1 /initialpose geometry_msgs/msg/PoseWithCovarianceStamped "{header: {frame_id: 'map'}, pose: {pose: {position: {x: $SX, y: $SY, z: 0.0}, orientation: {z: $QZ, w: $QW}}}}" > "$OUT.initpose.log" 2>&1 || true
sleep 6
mkdir -p "$OUT.side"
for nd in /gmpc_controller /scan_safety_shield /wheel_limit_guard /scan_obstacle_tracker /velocity_smoother /planner_server /global_costmap/global_costmap /amcl /goal_to_plan_relay; do
  timeout 25 ros2 param dump "$nd" > "$OUT.side/params$(echo $nd | tr '/' '_').yaml" 2> /dev/null || true
done
for nd in planner_server map_server amcl velocity_smoother; do echo "$nd $(timeout 6 ros2 lifecycle get /$nd 2>/dev/null | head -1)" >> "$OUT.side/lifecycle.txt"; done
setsid $PYS -u evaluation/far_dry/merge_node.py "$OUT" > "$OUT.merge.log" 2>&1 < /dev/null & PIDS+=( $! ); MPID=$!
sleep 12
setsid timeout $((TEND + 20)) ros2 topic echo /merge/gmpc/diag_v2 std_msgs/msg/String --field data > "$OUT.merge_diag_v2.txt" 2>/dev/null < /dev/null & PIDS+=( $! )
for t in /cmd_vel /cmd_vel_smoothed /cmd_vel_pre_shield /cmd_vel_nav /cmd_vel_nav_raw /cmd_vel_merge /merge/plan /plan /goal_pose /far/arm_cmd /scan; do { echo "== $t"; timeout 10 ros2 topic info -v "$t" 2>/dev/null | grep -E "Node name|Publisher count|Subscription count|Endpoint type"; } >> "$OUT.side/topic_endpoints.txt"; done
timeout 25 ros2 param dump /gmpc_merge > "$OUT.side/params_gmpc_merge.yaml" 2>/dev/null || true
ros2 node list > "$OUT.side/nodes.txt" 2>&1
for i in $(seq $((TEND + 30))); do kill -0 "$MPID" 2>/dev/null || break; kill -0 "$WPID" 2>/dev/null || break; sleep 1; done
sleep 1
kill -INT -- "-$WPID" 2>/dev/null; sleep 2
echo done
