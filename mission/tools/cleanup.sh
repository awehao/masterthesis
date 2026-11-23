#!/usr/bin/env bash
PAT="[w]orld_d2_node|[g]mpc_task_node|[p]lanner_server|[b]_d2_node|[a]mcl|[s]tub_isaac_d2|[c]omponent_container|[m]ap_server|[s]can_safety_shield|[w]heel_limit_guard|[v]elocity_smoother|[s]can_obstacle_tracker|[g]oal_to_plan_relay|[e]kf_node|[i]saac_m8_house"
echo "running: $(pgrep -fc "$PAT")"
