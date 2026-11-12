#!/usr/bin/env bash
# 側錄：等主腳本建立趟次目錄並開始錄製後，補錄話題；就緒後存各節點參數與 /cmd_vel 發布者。只讀，不發任何命令。
set +u; source /opt/ros/jazzy/setup.bash; source /home/howardchen/masterthesis/install/setup.bash
cd /home/howardchen/masterthesis
for i in $(seq 600); do [ -f "$RUN_DIR/run.log" ] && grep -q '\[4/6\] 開始錄製' "$RUN_DIR/run.log" && break; sleep 1; done
[ -d "$RUN_DIR" ] || exit 0
mkdir -p "$RUN_DIR/side"
setsid ros2 bag record -o "$RUN_DIR/bag_extra" /gmpc/obstacles /gmpc/static_obstacles /gmpc/tracks_debug /shield/diag /scan_filtered > "$RUN_DIR/side/bag_extra.log" 2>&1 < /dev/null &
BAG=$!
trap 'kill -INT -- -$BAG 2>/dev/null; sleep 3; exit 0' INT TERM
for n in /gmpc_controller /scan_safety_shield /wheel_limit_guard /scan_obstacle_tracker /velocity_smoother /goal_to_plan_relay /planner_server /global_costmap/global_costmap /amcl /ekf_global; do
  timeout 25 ros2 param dump "$n" > "$RUN_DIR/side/params$(echo $n | tr '/' '_').yaml" 2> "$RUN_DIR/side/params$(echo $n | tr '/' '_').err" || echo "param dump 失敗 $n" >> "$RUN_DIR/side/side.log"
done
for t in /cmd_vel /cmd_vel_smoothed /cmd_vel_pre_shield /cmd_vel_nav /scan /gmpc/obstacles; do { echo "== $t"; timeout 10 ros2 topic info -v "$t" | grep -E "Node name|Publisher count|Subscription count|Endpoint type"; } >> "$RUN_DIR/side/topic_endpoints.txt" 2>&1; done
echo "側錄就緒 $(date +%T)" >> "$RUN_DIR/side/side.log"
wait $BAG
