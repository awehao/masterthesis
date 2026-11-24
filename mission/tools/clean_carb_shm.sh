#!/usr/bin/env bash
# 清掉已結束程序留下的 Isaac carb 共用記憶體段（/dev/shm/carb-RStringInternals-<pid>）。
# 11/24 第 99 趟：PID 回捲後新 Isaac 拿到 11/20 舊程序的同一 PID，接上舊段裡沒釋放的鎖 ⇒ SimulationApp 啟動卡在 futex（CPU 0%、日誌停在 #EARLY r4）。
# 只刪「PID 已不存在」的段；活著的程序的段不動。
n=0
# 11/25 第 135 趟同樣卡在 #EARLY r4：段已清，但同名號誌 /dev/shm/sem.carb-RStringInternals-<pid> 累積 98 個沒清 ⇒ 一併清（同樣只刪 PID 已不存在者）
for f in /dev/shm/carb-RStringInternals-* /dev/shm/sem.carb-RStringInternals-*; do
  [ -e "$f" ] || continue
  p=${f##*-}
  case "$p" in ''|*[!0-9]*) continue;; esac
  if ! kill -0 "$p" 2>/dev/null && [ ! -d "/proc/$p" ]; then rm -f "$f" && n=$((n+1)); fi
done
echo "carb stale shm cleaned: $n"
