#!/usr/bin/env bash
# N1 補跑（Howard「系統失敗要補齊趟次」）：run_n1_fill.sh <批次名>
# 從 results/plan1127/registry.csv 找本批次 status=infra_fail 的趟次（3 次都在起跑前／bring-up 失敗），用同一份情境、同樣條件再跑；
# 趟次編號接在既有最大 t 之後（舊 out／bag 不覆蓋）；成功後補一列 done 登記。紀錄寫在 results/<批次>/<批次>.log（行首 FILL）。
cd "$(dirname "$0")/.."
WS=$PWD
B=$1
D=$WS/evaluation/experiments/n1_scenarios; R=$WS/evaluation/results/$B; L=$WS/evaluation/logs/$B
REG=$WS/evaluation/results/plan1127/registry.csv
LOG=$R/$B.log
source /opt/ros/jazzy/setup.bash; source install/setup.bash
set -u
GMPC_HOUSE="HEADING=1 GMPC_TASK=1 NAV_YIELD=1 D2_YIELD_SIDE=1 YS_PRED=1"
LIST=$(awk -F, -v b="$B" '$3==b && $2=="infra_fail"{print $1}' "$REG" | sort -u)
DONE=$(awk -F, -v b="$B" '$3==b && $2=="done"{print $1}' "$REG" | sort -u)
echo "$(date +%F_%T) FILL start $B" >> "$LOG"
for KEY in $LIST; do
  echo "$DONE" | grep -qx "$KEY" && continue                           # 之後已補成功
  IFS=/ read -r _ _ LY CD SEEDS M <<< "$KEY"; K=$((10#${SEEDS#seed}))
  F=$D/${LY}_${CD}_k${K}.yaml
  EXTRA=""; [ "$M" = "gmpc_scan" ] && EXTRA="$GMPC_HOUSE M8_MAP=$LY"
  T0=$(ls "$L"/${LY}_${CD}_k${K}_${M}_t*.out 2>/dev/null | sed -E 's/.*_t([0-9]+)\.out$/\1/' | sort -n | tail -1); T0=${T0:-0}
  ST=infra_fail
  for TRY in $((T0+1)) $((T0+2)) $((T0+3)); do
    TAG=_${B}_${LY}_${CD}_k${K}_t${TRY}
    OUT=$L/${LY}_${CD}_k${K}_${M}_t${TRY}.out
    bash mission/tools/wait_clean.sh > /dev/null 2>&1
    env POSES_CSV=$WS/evaluation/results/n1_poses_${LY}.csv BIGARENA=1 SCENE=$LY TRAJ=bigarena_traffic GUI=0 SEED_FIRST=$K \
      SHIELD=1 GUARD=1 NAV_VMAX=0.26 AMMR_OBSTACLE_MODE=scheduled AMMR_TRAJ_FILE=$F \
      TAG_SUFFIX=$TAG OUT_CSV=$R/${LY}_${CD}_k${K}_${M}.csv $EXTRA \
      timeout 700 ./evaluation/run_omnibot_dynamic.sh 1 200 0 0 "$M" > "$OUT" 2>&1 < /dev/null
    RES=$(grep -E 'goal reached|cap hit|skipping' "$OUT" | tail -1 | sed 's/^\[[0-9:]*\] *//')
    echo "$(date +%T) FILL $LY $CD k$K $M try$TRY : ${RES:-no_result}" >> "$LOG"
    case "$RES" in *skipping*|"") sleep 10; continue;; esac
    ST=done; break
  done
  echo "$KEY,$ST,$B,$R/${LY}_${CD}_k${K}_${M}.csv,$(date +%F_%T)" >> "$REG"
done
echo "$(date +%F_%T) FILL DONE $B" >> "$LOG"
