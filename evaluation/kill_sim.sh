#!/usr/bin/env bash
# Clean shutdown of a gz/ROS simulation session. Patterns are kept in variables
# so this script's own command line does not match them (pkill -f would
# otherwise kill the caller).
P1='gz'; P1="${P1} sim"
pkill -f "$P1"                       2>/dev/null
pkill -f 'ros_gz_sim'                2>/dev/null
pkill -f 'ros2 launch'               2>/dev/null
pkill -f 'scan_obstacle_tracker'     2>/dev/null
pkill -f 'gmpc_node'                 2>/dev/null
pkill -f 'dynamic_obstacle_driver'   2>/dev/null
pkill -f 'foxglove_bridge'           2>/dev/null
pkill -f 'parameter_bridge'          2>/dev/null
pkill -f 'nav2\|amcl\|map_server\|planner_server\|velocity_smoother\|lifecycle_manager\|ekf_node' 2>/dev/null
sleep 2
echo "cleaned."
