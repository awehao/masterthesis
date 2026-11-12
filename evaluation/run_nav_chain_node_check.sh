#!/usr/bin/env bash
# GUARD＋Shield 串接：真節點隔離鏈（不起 Isaac）。用法：bash evaluation/run_nav_chain_node_check.sh <full|direct> <輸出前綴>
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"; cd "$WS"
MODE="$1"; OUT="$2"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-71}" ROS_LOCALHOST_ONLY=1
set +u; source /opt/ros/jazzy/setup.bash; source install/setup.bash; set -u
n=$(pgrep -fc "[s]can_safety_shield|[w]heel_limit_guard|[v]elocity_smoother" || true)
[ "$n" = "0" ] || { echo "已有相關節點在跑（$n）⇒ 拒跑"; exit 99; }
PIDS=()
cleanup(){ for p in "${PIDS[@]}"; do kill -INT -- "-$p" 2>/dev/null; done; sleep 2; for p in "${PIDS[@]}"; do kill -KILL -- "-$p" 2>/dev/null; done; }
trap cleanup EXIT
sp(){ setsid "$@" > "$OUT.$1.log" 2>&1 < /dev/null & PIDS+=( $! ); }
spn(){ local nm="$1"; shift; setsid "$@" > "$OUT.$nm.log" 2>&1 < /dev/null & PIDS+=( $! ); }
spn shield ros2 run ammr_wholebody_mpc scan_safety_shield --ros-args -p enable:=true -p scan_topic:=/scan -p cmd_in_topic:=/cmd_vel_pre_shield -p cmd_out_topic:=/cmd_vel_smoothed \
  -p robot_radius:=0.30 -p alpha:=2.0 -p a_brake:=6.25 -p tau:=0.15 -p vx_max:=0.2775 -p vy_max:=0.2775 -p wz_max:=1.1327 -p wheel_rows:=true -p final_cmd_topic:=/cmd_vel -p wheel_dt:=0.045
spn guard ros2 run ammr_wholebody_mpc wheel_limit_guard --ros-args -p cmd_in_topic:=/cmd_vel_smoothed -p cmd_out_topic:=/cmd_vel -p odom_topic:=/odom -p wheel_radius:=0.05 -p wheel_base_L:=0.245 \
  -p wheel_w_max:=5.55 -p wheel_a_max:=125.0 -p publish_rate:=20.0 -p input_timeout:=0.25 -p dt_max:=0.50 -p assume_stopped_at_start:=true
if [ "$MODE" = full ]; then
  spn smoother ros2 run nav2_velocity_smoother velocity_smoother --ros-args -p smoothing_frequency:=20.0 -p scale_velocities:=false -p feedback:=OPEN_LOOP -p max_velocity:="[0.2775, 0.2775, 1.1327]" -p min_velocity:="[-0.2775, -0.2775, -1.1327]" \
    -p max_accel:="[6.25, 6.25, 25.51]" -p max_decel:="[-6.25, -6.25, -25.51]" -p odom_topic:=/odom -p odom_duration:=0.1 -p deadband_velocity:="[0.0, 0.0, 0.0]" -p velocity_timeout:=1.0 -r cmd_vel:=cmd_vel_nav -r cmd_vel_smoothed:=cmd_vel_pre_shield
  spn smlife ros2 run nav2_lifecycle_manager lifecycle_manager --ros-args -r __node:=lifecycle_manager_smoother -p autostart:=true -p node_names:="['velocity_smoother']"
  ok=0; for i in $(seq 30); do grep -q 'Managed nodes are active' "$OUT.smlife.log" 2>/dev/null && { ok=1; break; }; sleep 1; done
  [ "$ok" = 1 ] || { echo "平滑器未 active ⇒ 中止"; exit 86; }
fi
sleep 3
python3 -u evaluation/nav_chain_node_driver.py "$MODE" "$OUT"
