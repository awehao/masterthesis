#!/usr/bin/env bash
# D1 ROS 乾跑（不起 Isaac）：世界節點（受控對象＋非阻塞執行器，控制器在提前一拍工作程序）＋原導航鏈真節點（平滑器→Shield→GUARD）＋最小選擇器（擁有者固定 GRASP）。
# 開關同遠距乾跑（run_far_merge.sh）：NO_GZ BIGARENA SCENE=mission_house_m8 NO_TRAFFIC HEADING GUARD SHIELD GMPC_CMD_TOPIC=cmd_vel_nav_raw（原導航 GMPC 的輸出由選擇器丟棄）。
# 用法：bash evaluation/motm_dry/run_d1.sh <輸出前綴> <模擬時長上限 s>；環境變數照 door_ahead_case（FAKE_START_Q9、FAKE_LOCK、POOL_N、CAND_STREAM_CHUNK、TAIL_CERT、AHEAD_LATE…）
set -u
WS="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$WS"
OUT="$1"; TEND="${2:-120}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-73}" ROS_LOCALHOST_ONLY=1
set +u; source /opt/ros/jazzy/setup.bash; source install/setup.bash; set -u
PAT="[s]can_safety_shield|[w]heel_limit_guard|[v]elocity_smoother|[g]mpc_node|[p]lanner_server|[w]orld_ahead_node.py|[g]rasp_selector.py|[w]orld_node.py|[m]erge_node.py"
for i in $(seq 15); do n=$(pgrep -fc "$PAT" || true); [ "$n" = "0" ] && break; sleep 2; done
[ "$n" = "0" ] || { echo "已有相關節點在跑（$n）⇒ 拒跑"; exit 99; }
PIDS=()
cleanup(){ for p in "${PIDS[@]}"; do kill -INT -- "-$p" 2>/dev/null; done; sleep 3; for p in "${PIDS[@]}"; do kill -KILL -- "-$p" 2>/dev/null; done; }
trap cleanup EXIT
Q0="${FAKE_START_Q9:?需要 FAKE_START_Q9}"
SX=$(echo "$Q0" | cut -d, -f1); SY=$(echo "$Q0" | cut -d, -f2); SYAW=$(echo "$Q0" | cut -d, -f3)
PYS="env -u VIRTUAL_ENV PATH=/usr/local/bin:/usr/bin:/bin /usr/bin/python3"
setsid $PYS -u evaluation/motm_dry/world_ahead_node.py "$OUT" "$TEND" > "$OUT.world.log" 2>&1 < /dev/null & PIDS+=( $! ); WPID=$!
for i in $(seq 180); do grep -q "world_ready" "$OUT.world.log" 2>/dev/null && break; kill -0 "$WPID" 2>/dev/null || { echo "世界節點提前結束"; tail -5 "$OUT.world.log"; exit 3; }; sleep 1; done
grep -q "world_ready" "$OUT.world.log" || { echo "世界節點未就緒"; exit 3; }
NO_GZ=1 BIGARENA=1 SCENE=mission_house_m8 NO_TRAFFIC=1 HEADING=1 GUARD=1 SHIELD=1 GMPC_CMD_TOPIC=cmd_vel_nav_raw SPAWN_X="$SX" SPAWN_Y="$SY" \
  setsid ros2 launch my_omnibot_description omni_bot_dynamic.launch.py gui:=false use_arm:=true > "$OUT.launch.log" 2>&1 < /dev/null & PIDS+=( $! )
setsid $PYS -u evaluation/motm_dry/grasp_selector.py "$OUT" > "$OUT.selector.log" 2>&1 < /dev/null & PIDS+=( $! )
sleep 22
QZ=$(python3 -c "import math;print(math.sin($SYAW/2))"); QW=$(python3 -c "import math;print(math.cos($SYAW/2))")
timeout 10 ros2 topic pub -t 3 -r 1 /initialpose geometry_msgs/msg/PoseWithCovarianceStamped "{header: {frame_id: 'map'}, pose: {pose: {position: {x: $SX, y: $SY, z: 0.0}, orientation: {z: $QZ, w: $QW}}}}" > "$OUT.initpose.log" 2>&1 || true
mkdir -p "$OUT.side"
for nd in /scan_safety_shield /wheel_limit_guard /velocity_smoother; do timeout 25 ros2 param dump "$nd" > "$OUT.side/params$(echo $nd | tr '/' '_').yaml" 2> /dev/null || true; done
for t in /cmd_vel /cmd_vel_smoothed /cmd_vel_pre_shield /cmd_vel_nav /cmd_vel_nav_raw /cmd_vel_grasp /scan; do { echo "== $t"; timeout 10 ros2 topic info -v "$t" 2>/dev/null | grep -E "Node name|Publisher count|Subscription count|Endpoint type"; } >> "$OUT.side/topic_endpoints.txt"; done
ros2 node list > "$OUT.side/nodes.txt" 2>&1
for i in $(seq $((TEND * 2 + 120))); do kill -0 "$WPID" 2>/dev/null || break; sleep 1; done
echo done
