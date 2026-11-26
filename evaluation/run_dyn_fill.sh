#!/usr/bin/env bash
# 導航動態批次補跑（11/26 Howard「系統失敗要補齊趟次」）：run_dyn_fill.sh <批次名> <清單檔>
# 清單每行：<速度> <型態> <路線> <方法>（# 開頭為註解）。來源＝dyn_health.py 判 INFRA 的趟次＋產生器失敗而整案例缺漏的案例。
# 條件與 run_dyn_batch.sh 完全相同（同一情境檔、SCENE=bigarena_ghost、scheduled、SHIELD=1 GUARD=1、時限 200 s）。
# 趟次編號接在既有最大 t 之後（舊 bag／log 不覆蓋）；若已有結果 CSV，先改名成 <csv>.prev_t<k> 保留再跑。
# 3b 未過／bring-up 失敗／無路徑＝再重來，至多 3 次。全部記在 results/<批次名>/<批次名>.log（行首標 FILL）。
cd "$(dirname "$0")/.."
WS=$PWD
B=$1; LIST=$2; K=${3:-1}
D=$WS/evaluation/experiments/dyn_scenarios_$B; R=$WS/evaluation/results/$B; L=$WS/evaluation/logs/$B
mkdir -p "$D" "$R" "$L"
LOG=$R/$B.log
source /opt/ros/jazzy/setup.bash; source install/setup.bash
set -u
echo "$(date +%F_%T) FILL start list=$LIST git=$(git rev-parse --short HEAD)" >> "$LOG"
grep -vE '^\s*(#|$)' "$LIST" | while read -r SP PT RT M; do
  [ -f "$R/stop" ] && { echo "$(date +%T) FILL stop file" >> "$LOG"; exit 0; }
  F=$D/bigarena_route_${SP}_${PT}_r${RT}_k${K}.yaml
  [ -f "$F" ] || python3 src/ammr_bringup/scripts/make_bigarena_route_traffic.py --speed "$SP" --pattern "$PT" --route "$RT" --seed "$K" --out-dir "$D" > /dev/null \
    || { echo "$(date +%T) FILL GENFAIL $SP $PT r$RT" >> "$LOG"; continue; }
  CSV=$R/${SP}_${PT}_r${RT}_${M}.csv
  T0=$(ls -d "$L"/${SP}_${PT}_r${RT}_${M}_t*.out 2>/dev/null | sed -E 's/.*_t([0-9]+)\.out$/\1/' | sort -n | tail -1); T0=${T0:-0}
  [ -f "$CSV" ] && mv "$CSV" "$CSV.prev_t$T0"
  for TRY in $((T0+1)) $((T0+2)) $((T0+3)); do
    TAG=_${B}_${SP}_${PT}_r${RT}_t${TRY}
    OUT=$L/${SP}_${PT}_r${RT}_${M}_t${TRY}.out
    bash mission/tools/wait_clean.sh > /dev/null 2>&1
    POSES_CSV=$WS/evaluation/results/bigarena_poses.csv BIGARENA=1 TRAJ=bigarena_traffic GUI=0 SEED_FIRST=$RT \
      SHIELD=1 GUARD=1 SCENE=bigarena_ghost AMMR_OBSTACLE_MODE=scheduled AMMR_TRAJ_FILE=$F \
      TAG_SUFFIX=$TAG OUT_CSV=$CSV \
      timeout 700 ./evaluation/run_omnibot_dynamic.sh 1 200 0 0 "$M" > "$OUT" 2>&1 < /dev/null
    RES=$(grep -E 'goal reached|cap hit|skipping' "$OUT" | tail -1 | sed 's/^\[[0-9:]*\] *//')
    echo "$(date +%T) FILL $SP $PT r$RT $M try$TRY : ${RES:-no_result}" >> "$LOG"
    case "$RES" in *skipping*) sleep 10; continue;; esac
    break
  done
  sleep 5
done
echo "$(date +%F_%T) FILL DONE $LIST" >> "$LOG"
