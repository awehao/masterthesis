#!/usr/bin/env bash
# N1 平行車道批次（11/27；Howard 要起床看到實驗完成）：同時跑 NLANE 個模擬，每條車道各自 ROS_DOMAIN_ID／GZ_PARTITION／鎖檔，
# 收尾只砍自己 session 的程序（run_omnibot_dynamic.sh 的 PAR_SAFE=1）。跑法、條件、GMPC 設定與 run_n1_batch.sh 相同。
# 用法：run_n1_par.sh <批次名> <布局,...> <條件,...> <種子起> <種子迄> [車道數=3]
#   例：run_n1_par.sh n1hardpilot n1_bigarena,n1_office,n1_warehouse crossing 101 102 3
# 種子 1–20＝正式（登記 done）；101 以後＝先導／開發（登記 pilot）。
# 共同條件：SCENE=<布局>、scheduled、SHIELD=1 GUARD=1、NAV_VMAX=0.26；GMPC＝住宅任務版（HEADING=1 GMPC_TASK=1 NAV_YIELD=1 D2_YIELD_SIDE=1 YS_PRED=1 M8_MAP=<布局>）。
# 情境：experiments/n1_scenarios/（先全部產生好再開車道，避免並行寫起終點檔）。系統失敗（skipping／無結果）同趟重來至多 3 次。
cd "$(dirname "$0")/.."
WS=$PWD
B=$1; IFS=, read -ra LAYS <<< "$2"; IFS=, read -ra CONDS <<< "$3"; K0=$4; K1=$5; NL=${6:-3}
D=$WS/evaluation/experiments/n1_scenarios; R=$WS/evaluation/results/$B; L=$WS/evaluation/logs/$B
REG=$WS/evaluation/results/plan1127/registry.csv
mkdir -p "$D" "$R" "$L" "$(dirname "$REG")"
[ -f "$REG" ] || echo "run_key,status,batch,result_csv,time" > "$REG"
LOG=$R/$B.log
source /opt/ros/jazzy/setup.bash; source install/setup.bash
set -u
GMPC_HOUSE="HEADING=1 GMPC_TASK=1 NAV_YIELD=1 D2_YIELD_SIDE=1 YS_PRED=1"
echo "$(date +%F_%T) batch $B layouts=${LAYS[*]} conds=${CONDS[*]} seeds=$K0-$K1 lanes=$NL NAV_VMAX=0.26 git=$(git rev-parse --short HEAD)" >> "$LOG"
# 1) 先產生全部情境（單一程序）
CASES=()
for K in $(seq "$K0" "$K1"); do
  for LY in "${LAYS[@]}"; do
    if [ ! -f "$D/${LY}_crossing_k${K}.yaml" ] || [ ! -f "$D/${LY}_static_k${K}.yaml" ]; then
      python3 src/ammr_bringup/scripts/make_n1_scenario.py --layout "$LY" --seed "$K" --out-dir "$D" >> "$LOG" 2>&1 \
        || { echo "$(date +%T) GENFAIL $LY k$K" >> "$LOG"; continue; }
    fi
    for CD in "${CONDS[@]}"; do CASES+=("$LY $CD $K"); done
  done
done
echo "$(date +%T) cases=${#CASES[@]}" >> "$LOG"

run_case() {   # $1 lane  $2 layout  $3 cond  $4 seed
  local LN=$1 LY=$2 CD=$3 K=$4
  local F=$D/${LY}_${CD}_k${K}.yaml
  local ORDER M TRY TAG OUT RES ST EXTRA pid
  ORDER=$(python3 -c "import random,zlib;m=['gmpc_scan','mppi','dwb','rpp'];random.Random(zlib.crc32(b'$LY|$CD|$K')).shuffle(m);print(' '.join(m))")
  for M in $ORDER; do
    [ -f "$R/stop" ] && return 0
    EXTRA=""; [ "$M" = "gmpc_scan" ] && EXTRA="$GMPC_HOUSE M8_MAP=$LY"
    ST=infra_fail
    for TRY in 1 2 3; do
      TAG=_${B}_${LY}_${CD}_k${K}_t${TRY}
      OUT=$L/${LY}_${CD}_k${K}_${M}_t${TRY}.out
      setsid env ROS_DOMAIN_ID=$((60 + LN)) GZ_PARTITION=n1lane${LN} PAR_SAFE=1 PAR_ID=$LN \
        POSES_CSV=$WS/evaluation/results/n1_poses_${LY}.csv BIGARENA=1 SCENE=$LY TRAJ=bigarena_traffic GUI=0 SEED_FIRST=$K \
        SHIELD=1 GUARD=1 NAV_VMAX=0.26 AMMR_OBSTACLE_MODE=scheduled AMMR_TRAJ_FILE=$F \
        TAG_SUFFIX=$TAG OUT_CSV=$R/${LY}_${CD}_k${K}_${M}.csv $EXTRA \
        timeout 700 ./evaluation/run_omnibot_dynamic.sh 1 200 0 0 "$M" > "$OUT" 2>&1 < /dev/null &
      pid=$!
      wait $pid
      pkill -INT -s $pid 2>/dev/null; sleep 2; pkill -KILL -s $pid 2>/dev/null    # 車道保險：這趟 session 殘留的程序全收
      RES=$(grep -E 'goal reached|cap hit|skipping' "$OUT" | tail -1 | sed 's/^\[[0-9:]*\] *//')
      echo "$(date +%T) L$LN $LY $CD k$K $M try$TRY : ${RES:-no_result}" >> "$LOG"
      case "$RES" in *skipping*|"") sleep 8; continue;; esac
      if [ "$K" -le 20 ]; then ST=done; else ST=pilot; fi
      break
    done
    echo "N1/N/$LY/$CD/seed$(printf %02d "$K")/$M,$ST,$B,$R/${LY}_${CD}_k${K}_${M}.csv,$(date +%F_%T)" >> "$REG"
    sleep 3
  done
}

lane() {   # $1 lane id：拿第 i 個案例（i ≡ lane mod NL）
  local LN=$1 i c
  sleep $((LN * 25))                                                  # 錯開啟動，避免同時開 gz
  for i in "${!CASES[@]}"; do
    [ $((i % NL)) -eq "$LN" ] || continue
    [ -f "$R/stop" ] && break
    c=(${CASES[$i]})
    run_case "$LN" "${c[0]}" "${c[1]}" "${c[2]}"
  done
  echo "$(date +%T) lane $LN done" >> "$LOG"
}

for LN in $(seq 0 $((NL - 1))); do lane "$LN" & done
wait
echo "$(date +%F_%T) BATCH DONE $B" >> "$LOG"
