#!/usr/bin/env bash
# Isaac 實跑完整任務（一次只能一個 Isaac；CPU ≤55 °C 才起跑；92 °C 熱中止）。約 20–25 分鐘。
# 先登錄：mission/launch/register_isaac.sh <N> ...
# 用法：mission/launch/run_isaac.sh <N> [KEY=VAL ...]
# 結果：evaluation/runs/m8house_d2_unified_<N>/（result.json、run.log、frames/）；終端輸出：mission/logs/isaac_<N>.out
source "$(dirname "$0")/_env.sh"
N=${1:?要給趟號 N}; shift
FRZ=evaluation/experiments/20261119_m8_house_d2_unified${N}_freeze.sha256
[ -f "$WS/$FRZ" ] || { echo "沒有登錄／凍結檔 $FRZ：先跑 register_isaac.sh"; exit 1; }
pgrep -f "[i]saac_m8_house" >/dev/null && { echo "已有 Isaac 在跑，一次只能一個"; exit 1; }
bash "$MISSION/tools/wait_clean.sh" || exit 1
load_env task.env isaac.env
apply_overrides "$@"
[ -n "$B_PICK_RETURN" ] || export B_PICK_RETURN="$START"         # 回程終點預設＝起點（只改 START 即可換起點）
export RUN_ID=m8house_d2_unified_$N FREEZE=$FRZ
K=$(grep -l k10temp /sys/class/hwmon/*/name | head -1 | xargs dirname)
cd "$WS"
(until [ $(( $(cat $K/temp1_input) / 1000 )) -le 55 ]; do sleep 15; done; bash "$MISSION/tools/clean_carb_shm.sh" > /dev/null 2>&1; setsid bash evaluation/run_guarded_v3.sh > "$MISSION/logs/isaac_$N.out" 2>&1 < /dev/null; echo rc=$? >> "$MISSION/logs/isaac_$N.out") > /dev/null 2>&1 &
echo "$(date +%H:%M) Isaac 第 $N 趟已排入（等 CPU ≤55 °C）；看進度：tail -f mission/logs/isaac_$N.out"
