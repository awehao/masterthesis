#!/usr/bin/env bash
# 收掉所有替身／導航程序（以行程群組 TERM）；不比對自己
me=$$
for g in $(for p in $(pgrep -f "[s]tub_isaac_d2|[b]_d2_node|[w]orld_d2_node|[r]un_d2_stub|[g]mpc_task_node|[p]lanner_server|[v]elocity_smoother|[w]heel_limit_guard|[s]can_safety_shield|[a]mcl --ros-args|[m]ap_server --ros-args|[l]ifecycle_manager --ros-args|[r]obot_state_publisher --ros-args|[o]dom_tf_broadcaster|[s]can_relay --ros-args|[e]kf_node --ros-args|[g]oal_to_plan_relay|[s]can_obstacle_tracker|[a]mcl --ros-args|[m]ap_server --ros-args|[l]ifecycle_manager --ros-args|[r]obot_state_publisher --ros-args|[o]dom_tf_broadcaster|[s]can_relay --ros-args|[e]kf_node --ros-args|[g]oal_to_plan_relay|[s]can_obstacle_tracker"); do ps -o pgid= -p $p; done | sort -u); do
  [ "$g" = "$(ps -o pgid= -p $me | tr -d ' ')" ] && continue
  kill -TERM -- -$g 2>/dev/null
done
sleep 5                                                              # 收不到 TERM 的（例：卡在生命週期設定的 lifecycle_manager）⇒ KILL
for p in $(pgrep -f "[s]tub_isaac_d2|[b]_d2_node|[w]orld_d2_node|[r]un_d2_stub|[g]mpc_task_node|[p]lanner_server|[v]elocity_smoother|[w]heel_limit_guard|[s]can_safety_shield|[a]mcl --ros-args|[m]ap_server --ros-args|[l]ifecycle_manager --ros-args|[r]obot_state_publisher --ros-args|[o]dom_tf_broadcaster|[s]can_relay --ros-args|[e]kf_node --ros-args|[g]oal_to_plan_relay|[s]can_obstacle_tracker"); do kill -KILL $p 2>/dev/null; done
