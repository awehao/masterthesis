#!/usr/bin/env bash
# N1 正式第二版（11/28 Howard）：第一輪原版 GMPC 未到達的 15 條路線降難度（見 experiments/n1_scenarios/README_r2.md），
# 只重跑這 15 條 × 4 方法＝60 趟（Howard 00:45：其餘 45 條情境與條件不變，沿用 n1main 結果；最後把這 15 條換進 N1）。條件與 run_n1_batch.sh 完全相同。
# 用法：run_n1_r2.sh [批次名=n1r2]
cd "$(dirname "$0")/.."
WS=$PWD
B=${1:-n1r2}
D=$WS/evaluation/experiments/n1_scenarios; R=$WS/evaluation/results/$B; L=$WS/evaluation/logs/$B
REG=$WS/evaluation/results/plan1127/registry.csv
mkdir -p "$R" "$L"
LOG=$R/$B.log
source /opt/ros/jazzy/setup.bash; source install/setup.bash
set -u
GMPC_HOUSE="HEADING=1 GMPC_TASK=1 NAV_YIELD=1 D2_YIELD_SIDE=1 YS_PRED=1"
CHANGED="n1_bigarena:4 n1_bigarena:5 n1_bigarena:9 n1_bigarena:10 n1_bigarena:12 n1_bigarena:19 n1_office:7 n1_office:11 n1_office:12 n1_office:17 n1_office:20 n1_warehouse:6 n1_warehouse:11 n1_warehouse:15 n1_warehouse:17"
CASES="$CHANGED"
echo "$(date +%F_%T) batch $B cases=$(echo $CASES | wc -w)（15 條降難度路線 × 4 方法） NAV_VMAX=0.26 git=$(git rev-parse --short HEAD)" >> "$LOG"
for C in $CASES; do
  LY=${C%:*}; K=${C#*:}; CD=crossing
  F=$D/${LY}_${CD}_k${K}.yaml
  [ -f "$F" ] || { echo "$(date +%T) NOSCEN $LY k$K" >> "$LOG"; continue; }
  ORDER=$(python3 -c "import random,zlib;m=['gmpc_scan','mppi','dwb','rpp'];random.Random(zlib.crc32(b'$LY|$CD|$K')).shuffle(m);print(' '.join(m))")
  for M in $ORDER; do
    [ -f "$R/stop" ] && { echo "$(date +%T) stop file" >> "$LOG"; exit 0; }
    EXTRA=""; [ "$M" = "gmpc_scan" ] && EXTRA="$GMPC_HOUSE M8_MAP=$LY"
    ST=infra_fail
    for TRY in 1 2 3; do
      TAG=_${B}_${LY}_${CD}_k${K}_t${TRY}
      OUT=$L/${LY}_${CD}_k${K}_${M}_t${TRY}.out
      bash mission/tools/wait_clean.sh > /dev/null 2>&1
      env POSES_CSV=$WS/evaluation/results/n1_poses_${LY}.csv BIGARENA=1 SCENE=$LY TRAJ=bigarena_traffic GUI=0 SEED_FIRST=$K \
        SHIELD=1 GUARD=1 NAV_VMAX=0.26 AMMR_OBSTACLE_MODE=scheduled AMMR_TRAJ_FILE=$F \
        TAG_SUFFIX=$TAG OUT_CSV=$R/${LY}_${CD}_k${K}_${M}.csv $EXTRA \
        timeout 700 ./evaluation/run_omnibot_dynamic.sh 1 200 0 0 "$M" > "$OUT" 2>&1 < /dev/null
      RES=$(grep -E 'goal reached|cap hit|skipping' "$OUT" | tail -1 | sed 's/^\[[0-9:]*\] *//')
      echo "$(date +%T) $LY $CD k$K $M try$TRY : ${RES:-no_result}" >> "$LOG"
      case "$RES" in *skipping*|"") sleep 10; continue;; esac
      ST=done
      break
    done
    echo "N1/N/$LY/$CD/seed$(printf %02d "$K")/$M,$ST,$B,$R/${LY}_${CD}_k${K}_${M}.csv,$(date +%F_%T)" >> "$REG"
    sleep 5
  done
done
echo "$(date +%F_%T) BATCH DONE $B" >> "$LOG"
