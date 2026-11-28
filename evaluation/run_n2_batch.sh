#!/usr/bin/env bash
# N2 導航消融批次（11/27 新規劃；N1 跑完才跑）：GMPC 住宅任務版的四個消融配置，情境沿用 N1 正式的同一份（不重產）。
# 用法：run_n2_batch.sh <批次名> <布局,...> <種子起> <種子迄> [配置,...=B,C,D,E]
#   例：run_n2_batch.sh n2main n1_bigarena,n1_office 1 10
# 配置（A＝完整方法，直接沿用 N1 n1main 同布局同種子的 GMPC 結果，不重跑）：
#   B 不用障礙速度  NAV_ZERO_OBS_VEL=1（只在會車層候選外推與 CBF 把行人速度設 0；追蹤器、關聯、目前幾何不變）
#   C 移除 CBF      CBF_ENABLE=0
#   D 移除參考調整  D2_YIELD_SIDE=0（不做會車層的減速與側移）
#   E 僅作速度調整  YS_PRED_SPEED_ONLY=1（會車層只選減速、不側移；yield_side.py 的這個開關要在 N1 跑完後加）
# 共同：同 N1（SCENE、scheduled、SHIELD=1 GUARD=1、NAV_VMAX=0.26、時限 200 s、HEADING=1 GMPC_TASK=1 NAV_YIELD=1 YS_PRED=1 M8_MAP=<布局>）。
# 結果：results/<批次>/<布局>_crossing_k<種子>_<配置>.csv；登記 ${MODULE:-N2}/N/<布局>/crossing/seedNN/<配置>（種子 ≤20＝done）。
# 開發比較（GMPC 改進版）：MODULE=DEV、配置 H／U／UF、開發種子 101 起（登記 pilot，不計入任何正式模組）。
cd "$(dirname "$0")/.."
WS=$PWD
B=$1; IFS=, read -ra LAYS <<< "$2"; K0=$3; K1=$4; IFS=, read -ra CFGS <<< "${5:-B,C,D,E}"
D=${SCEN_DIR:-$WS/evaluation/experiments/n1_scenarios}; R=$WS/evaluation/results/$B; L=$WS/evaluation/logs/$B   # SCEN_DIR／POSES_PREFIX：11/28 快速行人開發比較（n1_scenarios_fast）
REG=$WS/evaluation/results/plan1127/registry.csv
mkdir -p "$R" "$L"
LOG=$R/$B.log
source /opt/ros/jazzy/setup.bash; source install/setup.bash
set -u
cfg_env() {
  case "$1" in
    H) echo "";;                                                            # 開發比較：住宅任務原版
    U) echo "NAV_UNSTUCK=1";;                                               # 開發比較：＋卡住解鎖
    UF) echo "NAV_UNSTUCK=1 PLANNER_SCAN=/scan_filtered";;                  # 開發比較：＋卡住解鎖＋規劃器遮掉行人（9 月鎖定設定的同一開關）
    UE) echo "NAV_UNSTUCK=1 YS_ESCAPE=1";;                                  # 開發比較：＋卡住解鎖＋緊急避讓（會車層加後退候選）
    UFE) echo "NAV_UNSTUCK=1 PLANNER_SCAN=/scan_filtered YS_ESCAPE=1";;     # 開發比較：三項全開
    B) echo "NAV_ZERO_OBS_VEL=1";;
    C) echo "CBF_ENABLE=0";;
    D) echo "D2_YIELD_SIDE=0";;
    E) echo "YS_PRED_SPEED_ONLY=1";;
  esac
}
echo "$(date +%F_%T) batch $B layouts=${LAYS[*]} conds=crossing seeds=$K0-$K1 cfgs=${CFGS[*]} NAV_VMAX=0.26 scen=$D git=$(git rev-parse --short HEAD)" >> "$LOG"
for K in $(seq "$K0" "$K1"); do
  for LY in "${LAYS[@]}"; do
    F=$D/${LY}_crossing_k${K}.yaml
    [ -f "$F" ] || { echo "$(date +%T) NOSCEN $LY k$K（N1 尚未產生）" >> "$LOG"; continue; }
    ORDER=$(python3 -c "import random,zlib;m='${CFGS[*]}'.split();random.Random(zlib.crc32(b'n2|$LY|$K')).shuffle(m);print(' '.join(m))")
    for CF in $ORDER; do
      [ -f "$R/stop" ] && { echo "$(date +%T) stop file" >> "$LOG"; exit 0; }
      ST=infra_fail
      for TRY in 1 2 3; do
        TAG=_${B}_${CF}_${LY}_crossing_k${K}_t${TRY}
        OUT=$L/${LY}_crossing_k${K}_${CF}_t${TRY}.out
        bash mission/tools/wait_clean.sh > /dev/null 2>&1
        env POSES_CSV=${POSES_PREFIX:-$WS/evaluation/results/n1_poses_}${LY}.csv BIGARENA=1 SCENE=$LY TRAJ=bigarena_traffic GUI=0 SEED_FIRST=$K \
          SHIELD=1 GUARD=1 NAV_VMAX=0.26 AMMR_OBSTACLE_MODE=scheduled AMMR_TRAJ_FILE=$F \
          HEADING=1 GMPC_TASK=1 NAV_YIELD=1 D2_YIELD_SIDE=1 YS_PRED=1 M8_MAP=$LY $(cfg_env "$CF") \
          TAG_SUFFIX=$TAG OUT_CSV=$R/${LY}_crossing_k${K}_${CF}.csv \
          timeout 700 ./evaluation/run_omnibot_dynamic.sh 1 200 0 0 gmpc_scan > "$OUT" 2>&1 < /dev/null
        RES=$(grep -E 'goal reached|cap hit|skipping' "$OUT" | tail -1 | sed 's/^\[[0-9:]*\] *//')
        echo "$(date +%T) $LY crossing k$K $CF try$TRY : ${RES:-no_result}" >> "$LOG"
        case "$RES" in *skipping*|"") sleep 10; continue;; esac
        if [ "$K" -le 20 ]; then ST=done; else ST=pilot; fi
        break
      done
      echo "${MODULE:-N2}/N/$LY/crossing/seed$(printf %02d "$K")/$CF,$ST,$B,$R/${LY}_crossing_k${K}_${CF}.csv,$(date +%F_%T)" >> "$REG"
      sleep 5
    done
  done
done
echo "$(date +%F_%T) BATCH DONE $B" >> "$LOG"
