#!/usr/bin/env bash
# 起跑前：收尾並等導航／乾跑節點歸零（最多 90 s），歸零後清 FastDDS 殭屍共用記憶體連接埠（11/23：收節點後殘留 ⇒ 下一趟交接卡死）
bash $(dirname "$0")/cleanup.sh >/dev/null 2>&1
for i in $(seq 1 45); do
  n=$(pgrep -fc "[s]can_safety_shield|[a]mcl --ros-args|[m]ap_server --ros-args|[l]ifecycle_manager --ros-args|[r]obot_state_publisher --ros-args|[o]dom_tf_broadcaster|[s]can_relay --ros-args|[e]kf_node --ros-args|[g]oal_to_plan_relay|[s]can_obstacle_tracker|[w]heel_limit_guard|[v]elocity_smoother|[g]mpc_node|[g]mpc_task_node|[p]lanner_server|[w]orld_node.py|[w]orld_d2_node|[m]erge_node.py|[b]_capture_node|[b]_d2_node|[s]tub_isaac_d2" || true)
  if [ "$n" = "0" ]; then
    sleep 3; source /opt/ros/jazzy/setup.bash >/dev/null 2>&1; timeout 60 fastdds shm clean 2>&1 | tail -1
    bash $(dirname "$0")/clean_carb_shm.sh
    echo clean; exit 0
  fi
  sleep 2
done
echo "still $n"; exit 1
