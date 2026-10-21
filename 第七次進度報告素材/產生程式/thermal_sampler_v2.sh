#!/usr/bin/env bash
# Independent temperature sampler on WALL time.
#
# The in-simulator guard samples every N physics steps, which is not a fixed
# amount of real time: during scene load and any stall it records nothing at
# all, which is exactly the window where the temperature rose. This runs beside
# the simulator, starts BEFORE it, samples every second, and enforces the same
# limit.
#
# Usage: thermal_sampler_v2.sh <csv> <limit_c> [pidfile]
#
# v2（2026-10-21）：thermal_sampler.sh（凍結，不改）的複本。差異只有 CPU 讀值：
#   原版只找 k10temp（AMD），在 Intel 機器上回傳 0 ⇒ 守護正確拒跑。v2 與 cpu_temp.py 同順序
#   k10temp → coretemp，取該晶片所有 temp*_input 的最大值（k10temp 原版只取 temp1＝Tctl；最大值不低於它）。
#   讀值必須是 1–150000 毫度的整數；空值、非數字、讀檔失敗都不算有效 ⇒ 該晶片視為無讀值。
#   都沒有 ⇒ cpu_c 仍寫 0（守護視為無效：起跑前拒跑 92、運行中中止 94）。
#   cpu_c 留在原第 3 欄；實際來源另加在最後一欄 cpu_src。中止線沿用（守護只接受 ≤ 92）。
#   HWMON_ROOT 只供離線測試覆寫（預設 /sys/class/hwmon）。
OUT="$1"; LIMIT="${2:-88}"; PIDFILE="${3:-}"
HWMON_ROOT="${HWMON_ROOT:-/sys/class/hwmon}"
cpu_read() {  # 印「<毫度> <來源>」；無有效讀值印「0 none」
  local name h f v best
  for name in k10temp coretemp; do
    best=''
    for h in "$HWMON_ROOT"/*/; do
      [ "$(cat "$h/name" 2>/dev/null)" = "$name" ] || continue
      for f in "$h"temp*_input; do
        v=$(cat "$f" 2>/dev/null) || continue
        [[ "$v" =~ ^[0-9]+$ ]] && [ "$v" -ge 1 ] && [ "$v" -le 150000 ] || continue
        { [ -z "$best" ] || [ "$v" -gt "$best" ]; } && best=$v
      done
    done
    [ -n "$best" ] && { echo "$best hwmon:$name"; return; }
  done
  echo "0 none"
}
echo "wall_s,iso,cpu_c,load1,gpu_util,gpu_c,cpu_src" > "$OUT"
t0=$(date +%s)
watched_pid=''
watched_start=''
while true; do
    # Do not keep collecting idle temperatures after Isaac has exited, and
    # never signal an unrelated process if the PID has been reused.
    if [ -z "$watched_pid" ] && [ -n "$PIDFILE" ] && [ -s "$PIDFILE" ]; then
        read -r watched_pid < "$PIDFILE"
        watched_start=$(awk '{print $22}' "/proc/$watched_pid/stat" 2>/dev/null)
    fi
    if [ -n "$watched_pid" ]; then
        current_start=$(awk '{print $22}' "/proc/$watched_pid/stat" 2>/dev/null)
        if [ -z "$current_start" ] || [ "$current_start" != "$watched_start" ]; then
            echo "$(date +%H:%M:%S) Isaac exited; temperature sampling stopped"
            exit 0
        fi
    fi
    read -r mc src < <(cpu_read)
    c=$(( mc / 1000 ))
    l=$(awk '{print $1}' /proc/loadavg)
    g=$(nvidia-smi --query-gpu=utilization.gpu,temperature.gpu --format=csv,noheader 2>/dev/null | tr -d ' %' | tr ',' ',')
    echo "$(( $(date +%s) - t0 )),$(date +%H:%M:%S),$c,$l,${g:-,},$src" >> "$OUT"
    if [ "$c" -ge "$LIMIT" ]; then
        echo "$(( $(date +%s) - t0 )),$(date +%H:%M:%S),LIMIT_HIT_${c}C" >> "$OUT"
        [ -n "$watched_pid" ] && kill "$watched_pid" 2>/dev/null
        exit 3
    fi
    sleep 1
done
