#!/usr/bin/env bash
# 收掉所有替身／導航／監督程序（整個行程群組，避免留孤兒 amcl 等），並確認歸零。
source "$(dirname "$0")/_env.sh"
bash "$MISSION/tools/kill_all.sh"
for i in 1 2 3; do bash "$MISSION/tools/wait_clean.sh" && exit 0; done
echo "仍有殘留，列出："; ps -eo pid,etime,cmd | grep -E "[a]mcl --ros|[m]ap_server --ros|[l]ifecycle_manager --ros|[g]mpc_task_node|[w]orld_d2_node|[b]_d2_node|[s]tub_isaac_d2" | cut -c1-120
