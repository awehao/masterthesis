#!/usr/bin/env bash
# heading OFF／ON 確認性批次：10 seed × 2 重複 × 2 條件 = 40 趟。
#
# 分析計畫在批次開始前已寫定，見
#   evaluation/results/PREREG_heading_10seed_20260912.md
# 設定沿用 evaluation/results/FROZEN_v2_config.md。
# **批次期間不得重建、不得調參、不得改情境或軌跡。**
#
# seed 抽樣由本腳本自行重算，不依賴任何外部清單：
#   母體 = 30 個 seed 扣掉探索批已用的 {1,7,8,13,24}
#   random.Random(20260912).sample(母體, 12) 取前 10 個
# 因此抽樣可由本檔本身重現，且未經人工挑選。
#
# 用法：
#   cd ~/masterthesis && source /opt/ros/jazzy/setup.bash && source install/setup.bash
#   bash evaluation/run_heading_10seed_batch.sh 2>&1 | tee /tmp/heading40.log
set +u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; WS="$(dirname "$HERE")"
cd "$WS"

V="$WS/src/ammr_bringup/config/dynamic_trajectories_bigarena_traffic_v3.yaml"
export ROS_LOCALHOST_ONLY=1
export AMMR_OBSTACLE_MODE=scheduled AMMR_TRAJ_FILE="$V" AMMR_PHASE_YAML="$V"
export PHASE_DELTA=30.0 GUARD=1
export CPU_LIMIT=92 CPU_THREADS=8 HEADLESS=true
export CAMERA=false RENDER_HZ=12 VIEW_WIDTH=1600 VIEW_HEIGHT=900

OUT="$WS/evaluation/results/batch40_state"
mkdir -p "$OUT"
FAILS="$OUT/fails.txt"; DONE="$OUT/done.txt"
touch "$FAILS" "$DONE"

# ---- 起跑前核對凍結版本；不符就不跑 ----------------------------------------
declare -A WANT=(
  [src/ammr_wholebody_mpc/ammr_wholebody_mpc/gmpc.py]=e2ded59dbe2a8b78
  [src/ammr_wholebody_mpc/ammr_wholebody_mpc/gmpc_node.py]=b83d4d2072c7b240
  [src/ammr_wholebody_mpc/ammr_wholebody_mpc/wheel_limit_guard.py]=e319aa88e7485642
  [src/ammr_bringup/config/dynamic_trajectories_bigarena_traffic_v3.yaml]=2e32069f3786b10d
)
bad=0
for f in "${!WANT[@]}"; do
  got=$(sha256sum "$WS/$f" | cut -c1-16)
  if [ "$got" != "${WANT[$f]}" ]; then
    echo "!! 版本不符 $f  實際 $got  應為 ${WANT[$f]}"; bad=1
  fi
done
[ "$bad" -ne 0 ] && { echo "**凍結版本核對失敗，不執行**"; exit 1; }
echo "凍結版本核對通過"

# ---- 抽樣與排程（腳本自行重算）---------------------------------------------
PLAN=$(python3 - <<'PY'
import random
pool = [s for s in range(1, 31) if s not in {1, 7, 8, 13, 24}]
seeds = random.Random(20260912).sample(pool, 12)[:10]
dom = 20; out = []
for s in seeds:
    for rep in (1, 2):
        # 每組配對內交替先後，避免順序效應固定偏向同一邊
        for w in (('off', 'on') if rep == 1 else ('on', 'off')):
            out.append(f'{s}:r{rep}:{w}:{dom}'); dom += 1
print(' '.join(out))
PY
)
N=$(echo $PLAN | wc -w)
echo "排程 $N 趟；seed $(echo $PLAN | tr ' ' '\n' | cut -d: -f1 | sort -un | tr '\n' ' ')"

T0=$(date +%s); i=0
for item in $PLAN; do
  i=$((i+1))
  SEED=${item%%:*}; r=${item#*:}; REP=${r%%:*}; r=${r#*:}
  WHICH=${r%%:*}; DOM=${r##*:}
  KEY="s${SEED}_${REP}_${WHICH}"
  # 已完成的跳過 —— 中斷後可直接重跑本腳本接續
  grep -qx "$KEY" "$DONE" && { echo "#### [$i/$N] $KEY 已完成，跳過"; continue; }

  if [ "$WHICH" = "off" ]; then export HEADING=0; M=gmpc_scan
  else export HEADING=1; M=gmpc_scan_heading; fi
  export ROS_DOMAIN_ID="$DOM"
  export RUN_ID="c10_${KEY}_$(date +%H%M%S)"
  EL=$(( ($(date +%s)-T0)/60 ))
  echo "######## [$i/$N] seed=$SEED $REP ${WHICH^^} domain=$DOM  已耗時 ${EL} 分  $(date +%T) ########"
  bash evaluation/run_bigarena_isaac.sh "$M" "$SEED" 180
  rc=$?
  echo "#### [$i/$N] $KEY exit=$rc $(date +%T)"
  if [ "$rc" -eq 0 ]; then echo "$KEY" >> "$DONE"
  else echo "$RUN_ID $KEY exit=$rc" >> "$FAILS"; fi
  sleep 20
done
echo "總耗時 $(( ($(date +%s)-T0)/60 )) 分；失敗 $(wc -l < "$FAILS") 趟（見 $FAILS）"
echo "BATCH40_DONE"
