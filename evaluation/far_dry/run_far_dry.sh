#!/usr/bin/env bash
# 遠距隔離真節點乾跑（Codex 20261116 核可；不起 Isaac）：運動學世界節點（合成 /scan_raw、/odom_raw、/clock、/joint_states）＋原導航鏈真節點
#   （omni_bot_dynamic.launch.py：map_server、amcl、ekf、planner_server、goal_to_plan_relay、gmpc_controller、scan_obstacle_tracker、velocity_smoother、scan_safety_shield、wheel_limit_guard…）。
#   與基準趟（fullon_s8_1）相同的開關：NO_GZ=1 BIGARENA=1 HEADING=1 GUARD=1 SHIELD=1 use_arm:=true；**只換**：SCENE=mission_house_m8（地圖）、起點、目標、NO_TRAFFIC=1（無移動體）。其餘參數不帶任何覆寫。
# 用法：bash evaluation/far_dry/run_far_dry.sh <輸出前綴> <起點9維,逗號> <目標x> <目標y> <時長s>
set -u
WS="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$WS"
OUT="$1"; Q0="$2"; GX="$3"; GY="$4"; TEND="${5:-90}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-72}" ROS_LOCALHOST_ONLY=1
set +u; source /opt/ros/jazzy/setup.bash; source install/setup.bash; set -u
n=$(pgrep -fc "[s]can_safety_shield|[w]heel_limit_guard|[v]elocity_smoother|[g]mpc_node|[p]lanner_server" || true)
[ "$n" = "0" ] || { echo "已有導航節點在跑（$n）⇒ 拒跑"; exit 99; }
PIDS=()
cleanup(){ for p in "${PIDS[@]}"; do kill -INT -- "-$p" 2>/dev/null; done; sleep 3; for p in "${PIDS[@]}"; do kill -KILL -- "-$p" 2>/dev/null; done; }
trap cleanup EXIT
SX=$(echo "$Q0" | cut -d, -f1); SY=$(echo "$Q0" | cut -d, -f2); SYAW=$(echo "$Q0" | cut -d, -f3)
PYS="env -u VIRTUAL_ENV /usr/bin/python3"
setsid $PYS -u evaluation/far_dry/world_node.py "$OUT" "$Q0" "$TEND" > "$OUT.world.log" 2>&1 < /dev/null & PIDS+=( $! )
sleep 2
NO_GZ=1 BIGARENA=1 SCENE=mission_house_m8 NO_TRAFFIC=1 HEADING=1 GUARD=1 SHIELD=1 SPAWN_X="$SX" SPAWN_Y="$SY" \
  setsid ros2 launch my_omnibot_description omni_bot_dynamic.launch.py gui:=false use_arm:=true > "$OUT.launch.log" 2>&1 < /dev/null & PIDS+=( $! )
setsid $PYS -u evaluation/far_dry/logger_node.py "$OUT" > "$OUT.logger.log" 2>&1 < /dev/null & PIDS+=( $! )
sleep 22
QZ=$(python3 -c "import math;print(math.sin($SYAW/2))"); QW=$(python3 -c "import math;print(math.cos($SYAW/2))")
timeout 10 ros2 topic pub -t 3 -r 1 /initialpose geometry_msgs/msg/PoseWithCovarianceStamped "{header: {frame_id: 'map'}, pose: {pose: {position: {x: $SX, y: $SY, z: 0.0}, orientation: {z: $QZ, w: $QW}}}}" > "$OUT.initpose.log" 2>&1 || true
sleep 6
# ---- 現場快照：參數（非空才算）、端點、lifecycle
mkdir -p "$OUT.side"
for nd in /gmpc_controller /scan_safety_shield /wheel_limit_guard /scan_obstacle_tracker /velocity_smoother /planner_server /global_costmap/global_costmap /amcl /goal_to_plan_relay; do
  timeout 25 ros2 param dump "$nd" > "$OUT.side/params$(echo $nd | tr '/' '_').yaml" 2> "$OUT.side/params$(echo $nd | tr '/' '_').err" || true
done
for t in /cmd_vel /cmd_vel_smoothed /cmd_vel_pre_shield /cmd_vel_nav /scan /scan_raw /odom /odom_raw /plan /goal_pose /far/arm_cmd; do { echo "== $t"; timeout 10 ros2 topic info -v "$t" 2>&1 | grep -E "Node name|Publisher count|Subscription count|Endpoint type"; } >> "$OUT.side/topic_endpoints.txt"; done
for nd in planner_server map_server amcl velocity_smoother; do echo "$nd $(timeout 6 ros2 lifecycle get /$nd 2>/dev/null | head -1)" >> "$OUT.side/lifecycle.txt"; done
ros2 node list > "$OUT.side/nodes.txt" 2>&1
# ---- 目標
timeout 15 ros2 topic pub -t 3 -r 1 /goal_pose geometry_msgs/msg/PoseStamped "{header: {frame_id: 'map'}, pose: {position: {x: $GX, y: $GY, z: 0.0}, orientation: {w: 1.0}}}" > "$OUT.goal.log" 2>&1 || true
# 等世界節點結束
W=${PIDS[0]}
for i in $(seq $((TEND + 60))); do kill -0 "$W" 2>/dev/null || break; sleep 1; done
echo done
