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
while true; do
    c=$(( $(k10) / 1000 ))
    l=$(awk '{print $1}' /proc/loadavg)
    g=$(nvidia-smi --query-gpu=utilization.gpu,temperature.gpu --format=csv,noheader 2>/dev/null | tr -d ' %' | tr ',' ',')
    echo "$(( $(date +%s) - t0 )),$(date +%H:%M:%S),$c,$l,${g:-,}" >> "$OUT"
    if [ "$c" -ge "$LIMIT" ]; then
        echo "$(( $(date +%s) - t0 )),$(date +%H:%M:%S),LIMIT_HIT_${c}C" >> "$OUT"
        [ -n "$PIDFILE" ] && [ -f "$PIDFILE" ] && kill "$(cat "$PIDFILE")" 2>/dev/null
        exit 3
    fi
    sleep 1
done
