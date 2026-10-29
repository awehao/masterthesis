#!/usr/bin/env bash
# Independent temperature sampler on WALL time.
#
# The in-simulator guard samples every N physics steps, which is not a fixed
# amount of real time: during scene load and any stall it records nothing at
# all, which is exactly the window where the temperature rose. This runs beside
# the simulator, starts BEFORE it, samples every second, and enforces the same
# limit.
#
# Usage: thermal_sampler.sh <csv> <limit_c> [pidfile]
OUT="$1"; LIMIT="${2:-88}"; PIDFILE="${3:-}"
k10() { for f in /sys/class/hwmon/*/; do
    [ "$(cat "$f/name" 2>/dev/null)" = k10temp ] && { cat "$f/temp1_input"; return; }; done; echo 0; }
echo "wall_s,iso,cpu_c,load1,gpu_util,gpu_c" > "$OUT"
t0=$(date +%s)
watched_pid=''
watched_start=''
# v3（2026-10-29；door_m4r_5 與 m4r_2 重播：Isaac 初始化 GPU 時 nvidia-smi 阻塞 8–9 s，使主迴圈 CPU 溫度讀值逾 5 s 未更新 ⇒ 守護以監看失效中止）：
#   GPU 查詢移到背景迴圈（每 2 s、timeout -k 1 2），寫快取檔；主迴圈只讀快取。CPU 溫度來源、每秒取樣、上限比較、離開碼完全不變。
GPUF="${OUT}.gpu"; : > "$GPUF"
( while kill -0 $$ 2>/dev/null; do v=$(timeout -k 1 2 nvidia-smi --query-gpu=utilization.gpu,temperature.gpu --format=csv,noheader 2>/dev/null | tr -d ' %'); [ -n "$v" ] && echo "$v" > "$GPUF.tmp" && mv -f "$GPUF.tmp" "$GPUF"; sleep 2; done ) &
GPU_BG=$!
trap "kill $GPU_BG 2>/dev/null" EXIT; trap "kill $GPU_BG 2>/dev/null; exit 143" TERM INT
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
    c=$(( $(k10) / 1000 ))
    l=$(awk '{print $1}' /proc/loadavg)
    g=$(cat "$GPUF" 2>/dev/null)                                                          # v3：讀背景 GPU 快取，不在主迴圈呼叫 nvidia-smi
    echo "$(( $(date +%s) - t0 )),$(date +%H:%M:%S),$c,$l,${g:-,}" >> "$OUT"
    if [ "$c" -ge "$LIMIT" ]; then
        echo "$(( $(date +%s) - t0 )),$(date +%H:%M:%S),LIMIT_HIT_${c}C" >> "$OUT"
        [ -n "$watched_pid" ] && kill "$watched_pid" 2>/dev/null
        exit 3
    fi
    sleep 1
done
