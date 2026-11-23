#!/usr/bin/env bash
# 替身看門狗：Nav2 生命週期在 LIMIT 秒內沒出現「Managed nodes are active」（偶發卡在設定 amcl／map_server）⇒ 收掉、以 <TAG>r 重跑一次（同參數）。
# 用法（run_stub.sh 自動呼叫）：nav_watchdog.sh <TAG> <LIMIT 秒> <原參數...>
TAG=$1; LIMIT=$2; shift 2
M=$(cd "$(dirname "$0")/.." && pwd); LOG=$(dirname "$M")/evaluation/results/motm_direct/d2/$TAG.launch.log
for i in $(seq 1 $((LIMIT / 5))); do grep -q "Managed nodes are active" "$LOG" 2>/dev/null && { echo "nav ok $TAG"; exit 0; }; sleep 5; done
echo "$(date +%H:%M) $TAG：Nav2 $LIMIT s 未啟動完成 ⇒ 收掉、以 ${TAG}r 重跑"
bash "$M/tools/kill_all.sh"; sleep 3
case "$TAG" in *r) echo "已重跑過一次，不再重跑"; exit 1;; esac
bash "$M/launch/run_stub.sh" "${TAG}r" "$@"
