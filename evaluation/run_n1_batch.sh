#!/usr/bin/env bash
# N1 導航外部基線批次（11/27 新規劃；Howard：三種複雜布局、行人 0.10–0.25 m/s、未知靜態障礙、隨機起終點、障礙不刻意）
# 用法：run_n1_batch.sh <批次名> <布局,...> <條件,...> <種子起> <種子迄>
#   例：run_n1_batch.sh n1pilot n1_bigarena,n1_office,n1_warehouse static,crossing 101 101
# 種子 1–20＝正式測試（登記 status=done；11/27 Howard：每布局 20 種子、全部交會）；101 以後＝先導／開發（登記 status=pilot）。
# 共同條件：SCENE=<布局>（世界＋地圖）、scheduled 時序、SHIELD=1 GUARD=1、NAV_VMAX=0.26（四方法平移上限一致，Howard 11/27 選「全部 0.26」）、時限 200 s。
# GMPC（gmpc_scan）＝住宅任務版（Howard 11/27 選）：與 run_d2_isaac.sh 相同的導航開關 HEADING=1 GMPC_TASK=1（6 個 CBF／規劃參數用程式預設，同住宅任務），
#   加會車層 NAV_YIELD=1 D2_YIELD_SIDE=1 YS_PRED=1（住宅任務 task.env 同值；地圖淨空 M8_MAP=<布局>），接在任務擴充節點的參考視窗（gmpc_task_node）。
#   9 月鎖定版（chain_final120.sh）不用；dyn1126 兩者都沒帶（memory feedback_gmpc_locked_config）。
# 每個案例（布局×條件×種子）四方法連跑、順序依案例雜湊打亂；任何「skipping」＝系統失敗，同趟重來至多 3 次。
# 情境檔：experiments/n1_scenarios/<布局>_<條件>_k<種子>.yaml（同一布局＋種子永遠同一份；已存在就不重產）。
# 結果：results/<批次名>/<布局>_<條件>_k<種子>_<方法>.csv；紀錄 results/<批次名>/<批次名>.log；停止檔 results/<批次名>/stop。
cd "$(dirname "$0")/.."
WS=$PWD
B=$1; IFS=, read -ra LAYS <<< "$2"; IFS=, read -ra CONDS <<< "$3"; K0=$4; K1=$5
D=$WS/evaluation/experiments/n1_scenarios; R=$WS/evaluation/results/$B; L=$WS/evaluation/logs/$B
REG=$WS/evaluation/results/plan1127/registry.csv
mkdir -p "$D" "$R" "$L" "$(dirname "$REG")"
[ -f "$REG" ] || echo "run_key,status,batch,result_csv,time" > "$REG"
LOG=$R/$B.log
source /opt/ros/jazzy/setup.bash; source install/setup.bash
set -u
GMPC_HOUSE="HEADING=1 GMPC_TASK=1 NAV_YIELD=1 D2_YIELD_SIDE=1 YS_PRED=1"
echo "$(date +%F_%T) batch $B layouts=${LAYS[*]} conds=${CONDS[*]} seeds=$K0-$K1 NAV_VMAX=0.26 git=$(git rev-parse --short HEAD)" >> "$LOG"
for K in $(seq "$K0" "$K1"); do
  for LY in "${LAYS[@]}"; do
    if [ ! -f "$D/${LY}_static_k${K}.yaml" ] || [ ! -f "$D/${LY}_crossing_k${K}.yaml" ]; then
      python3 src/ammr_bringup/scripts/make_n1_scenario.py --layout "$LY" --seed "$K" --out-dir "$D" >> "$LOG" 2>&1 \
        || { echo "$(date +%T) GENFAIL $LY k$K" >> "$LOG"; continue; }
    fi
    for CD in "${CONDS[@]}"; do
      F=$D/${LY}_${CD}_k${K}.yaml
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
          if [ "$K" -le 20 ]; then ST=done; else ST=pilot; fi
          break
        done
        echo "N1/N/$LY/$CD/seed$(printf %02d "$K")/$M,$ST,$B,$R/${LY}_${CD}_k${K}_${M}.csv,$(date +%F_%T)" >> "$REG"
        sleep 5
      done
    done
  done
done
echo "$(date +%F_%T) BATCH DONE $B" >> "$LOG"
