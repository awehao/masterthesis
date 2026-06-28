#!/usr/bin/env bash
# 每 20s 記錄 CPU(k10temp)/GPU/nvme 溫度,給過夜 benchmark 控溫用。
# 用法:  ./evaluation/temp_logger.sh  &   (背景跑,Ctrl-C 或 kill 停)
OUT="${1:-evaluation/results/temps.csv}"
echo "time,cpu_k10temp_C,amdgpu_C,nvme_C,nvidia_C" > "$OUT"
cpu(){ for h in /sys/class/hwmon/hwmon*; do [ "$(cat $h/name 2>/dev/null)" = "k10temp" ] && echo $(( $(cat $h/temp1_input)/1000 )); done; }
amd(){ for h in /sys/class/hwmon/hwmon*; do [ "$(cat $h/name 2>/dev/null)" = "amdgpu" ] && echo $(( $(cat $h/temp1_input)/1000 )); done; }
nvm(){ for h in /sys/class/hwmon/hwmon*; do [ "$(cat $h/name 2>/dev/null)" = "nvme" ] && echo $(( $(cat $h/temp1_input)/1000 )); done; }
nvd(){ nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1; }
while true; do
  echo "$(date +%H:%M:%S),$(cpu),$(amd),$(nvm),$(nvd)" >> "$OUT"
  sleep 20
done
