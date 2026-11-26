#!/usr/bin/env bash
# 導航動態能力批次（11/26，Howard「導航要比較動態的能力」「終點附近不要放，路上多點」）
# 用法：run_dyn_batch.sh <批次名> <速度,...> <型態,...> <路線起> <路線迄> [種子=1]
#   例：run_dyn_batch.sh dyn1126 walk,fast const,stopgo 1 10
# 每個案例（速度×型態×路線）四個方法連跑，方法順序依案例雜湊打亂（區塊內隨機化）。
# 共同條件：SCENE=bigarena_ghost（行人無碰撞體）、AMMR_OBSTACLE_MODE=scheduled、SHIELD=1 GUARD=1（共用防護＋輪級限制）、時限 200 s。
# 3b 步驟（相位 0 定位／移動確認）未過＝基礎設施失敗：同案例同方法自動重來至多 2 次，全部記在 <批次名>.log（不刪紀錄）。
# 情境檔產生在 experiments/dyn_scenarios_<批次名>/（新檔名，跑批次中不改名）。結果 CSV：results/<批次名>/<速度>_<型態>_r<路線>_<方法>.csv
set -u
cd "$(dirname "$0")/.."
WS=$PWD
B=$1; IFS=, read -ra SPEEDS <<< "$2"; IFS=, read -ra PATS <<< "$3"; R0=$4; R1=$5; K=${6:-1}
D=$WS/evaluation/experiments/dyn_scenarios_$B; R=$WS/evaluation/results/$B; L=$WS/evaluation/logs/$B
mkdir -p "$D" "$R" "$L"
LOG=$R/$B.log
source /opt/ros/jazzy/setup.bash; source install/setup.bash
echo "$(date +%F_%T) batch $B speeds=${SPEEDS[*]} patterns=${PATS[*]} routes=$R0-$R1 seed=$K git=$(git rev-parse --short HEAD)" >> "$LOG"
for RT in $(seq "$R0" "$R1"); do
  for SP in "${SPEEDS[@]}"; do
    for PT in "${PATS[@]}"; do
      F=$D/bigarena_route_${SP}_${PT}_r${RT}_k${K}.yaml
      [ -f "$F" ] || python3 src/ammr_bringup/scripts/make_bigarena_route_traffic.py --speed "$SP" --pattern "$PT" --route "$RT" --seed "$K" --out-dir "$D" > /dev/null \
        || { echo "$(date +%T) GENFAIL $SP $PT r$RT" >> "$LOG"; continue; }
      ORDER=$(python3 -c "import random,zlib;m=['gmpc_scan','mppi','dwb','rpp'];random.Random(zlib.crc32(b'$SP|$PT|$RT|$K')).shuffle(m);print(' '.join(m))")
      for M in $ORDER; do
        [ -f "$R/stop" ] && { echo "$(date +%T) stop file" >> "$LOG"; exit 0; }
        for TRY in 1 2 3; do
          TAG=_${B}_${SP}_${PT}_r${RT}_t${TRY}
          OUT=$L/${SP}_${PT}_r${RT}_${M}_t${TRY}.out
          bash mission/tools/wait_clean.sh > /dev/null 2>&1
          POSES_CSV=$WS/evaluation/results/bigarena_poses.csv BIGARENA=1 TRAJ=bigarena_traffic GUI=0 SEED_FIRST=$RT \
            SHIELD=1 GUARD=1 SCENE=bigarena_ghost AMMR_OBSTACLE_MODE=scheduled AMMR_TRAJ_FILE=$F \
            TAG_SUFFIX=$TAG OUT_CSV=$R/${SP}_${PT}_r${RT}_${M}.csv \
            timeout 700 ./evaluation/run_omnibot_dynamic.sh 1 200 0 0 "$M" > "$OUT" 2>&1
          RES=$(grep -E 'goal reached|cap hit|skipping' "$OUT" | tail -1 | sed 's/^\[[0-9:]*\] *//')
          echo "$(date +%T) $SP $PT r$RT $M try$TRY : ${RES:-no_result}" >> "$LOG"
          case "$RES" in *skipping*) sleep 10; continue;; esac   # 任何跳過（scheduled 未過、bring-up 失敗、無路徑）＝基礎設施失敗，重來
          break
        done
        sleep 5
      done
    done
  done
done
echo "$(date +%F_%T) BATCH DONE $B" >> "$LOG"
