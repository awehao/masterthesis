#!/usr/bin/env bash
# 起遠距乾跑世界節點（靜止，不送命令以外不動）＋原導航鏈，送候選目標錄 /plan。用法：bash run_probe.sh OUT_PREFIX "goals"
set -u
WS="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$WS"; OUT="$1"; GOALS="$2"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-74}" ROS_LOCALHOST_ONLY=1
set +u; source /opt/ros/jazzy/setup.bash; source install/setup.bash; set -u
PIDS=(); cleanup(){ for p in "${PIDS[@]}"; do kill -INT -- "-$p" 2>/dev/null; done; sleep 3; for p in "${PIDS[@]}"; do kill -KILL -- "-$p" 2>/dev/null; done; }; trap cleanup EXIT
Q0="1.5,-2.8,1.5708,-0.0128,-0.0519,0.086,0.0116,-1.6249,-0.0018"
PYS="env -u VIRTUAL_ENV PATH=/usr/local/bin:/usr/bin:/bin /usr/bin/python3"
FAR_GRAV=1 setsid $PYS -u evaluation/far_dry/world_node.py "$OUT" "$Q0" 200 > "$OUT.world.log" 2>&1 < /dev/null & PIDS+=( $! )
sleep 2
NO_GZ=1 BIGARENA=1 SCENE=mission_house_m8 NO_TRAFFIC=1 HEADING=1 GUARD=1 SHIELD=1 GMPC_TASK=1 SPAWN_X=1.5 SPAWN_Y=-2.8 setsid ros2 launch my_omnibot_description omni_bot_dynamic.launch.py gui:=false use_arm:=true > "$OUT.launch.log" 2>&1 < /dev/null & PIDS+=( $! )
sleep 22
timeout 10 ros2 topic pub -t 3 -r 1 /initialpose geometry_msgs/msg/PoseWithCovarianceStamped "{header: {frame_id: 'map'}, pose: {pose: {position: {x: 1.5, y: -2.8, z: 0.0}, orientation: {z: 0.7071, w: 0.7071}}}}" > /dev/null 2>&1 || true
sleep 4
timeout 120 $PYS -u evaluation/motm_b/${PROBE:-probe_smac_paths.py} "$OUT.json" "$GOALS" 2>&1 | grep -v WARN | tail -12
