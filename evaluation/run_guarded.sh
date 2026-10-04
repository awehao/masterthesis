#!/usr/bin/env bash
# 以 92 °C 牆鐘熱中止包住 run_nav_handover.sh（不改運行器本身）。
#
# 取樣器先起（場景載入期間也在量），模擬器 PID 一出現就交給它監看；
# 超溫時取樣器砍模擬器，本腳本再 TERM 運行器，讓它的 cleanup 收掉同趟其他 PID。
# 中止線只能調低、不能調高；趟次目錄已存在就拒跑（不覆寫既有實錄）。
#
#   ROS_DOMAIN_ID=94 RUN_ID=h5_log_check1 <其他環境變數> bash evaluation/run_guarded.sh
#
# 產物：runs/<RUN_ID>/thermal.csv（每秒一筆）、thermal.log、guard.log
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"
: "${ROS_DOMAIN_ID:?ROS_DOMAIN_ID 未設定}"
: "${RUN_ID:?RUN_ID 要明設（守護需要先知道趟次目錄）}"
LIMIT="${THERMAL_LIMIT:-92}"
if [ "$LIMIT" -gt 92 ]; then echo "THERMAL_LIMIT=$LIMIT 高於 92 °C，拒跑"; exit 90; fi
RUNNER="${RUNNER:-$WS/evaluation/run_nav_handover.sh}"
DIR="$WS/evaluation/runs/$RUN_ID"
if [ -e "$DIR" ]; then echo "$DIR 已存在，拒跑（不覆寫既有實錄）"; exit 91; fi
mkdir -p "$DIR"
G="$DIR/guard.log"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$G"; }

PIDF="$DIR/sim.pid"; : > "$PIDF"
bash "$WS/evaluation/thermal_sampler.sh" "$DIR/thermal.csv" "$LIMIT" "$PIDF" \
  > "$DIR/thermal.log" 2>&1 &
TS=$!
sleep 2
c=$(awk -F, 'NR==2{print $3}' "$DIR/thermal.csv" 2>/dev/null)
if grep -q LIMIT_HIT "$DIR/thermal.csv" 2>/dev/null; then
  say "起跑前已達中止線（$(grep LIMIT_HIT "$DIR/thermal.csv")），不起跑"; exit 93
fi
if ! kill -0 "$TS" 2>/dev/null || [ -z "$c" ] || [ "$c" -le 0 ]; then
  say "熱取樣器沒有有效讀值（cpu_c='${c:-}'），不起跑"
  kill "$TS" 2>/dev/null; exit 92
fi
say "熱中止啟用：上限 ${LIMIT} °C，目前 ${c} °C，取樣器 PID=$TS"

bash "$RUNNER" & RN=$!
say "運行器 PID=$RN（$RUNNER）"
sim=''
while kill -0 "$RN" 2>/dev/null; do
  if [ -z "$sim" ]; then
    sim=$(grep -oP '起 sim PID=\K[0-9]+' "$DIR/run.log" 2>/dev/null | head -1)
    if [ -n "$sim" ]; then echo "$sim" > "$PIDF"; say "模擬器 PID=$sim 已交給取樣器"; fi
  fi
  if grep -q LIMIT_HIT "$DIR/thermal.csv" 2>/dev/null; then
    say "**熱中止**：$(grep LIMIT_HIT "$DIR/thermal.csv" | tail -1)；TERM 運行器"
    kill -TERM "$RN" 2>/dev/null
    break
  fi
  if [ -z "$sim" ] && ! kill -0 "$TS" 2>/dev/null; then
    say "取樣器在模擬器起來前就結束；TERM 運行器"
    kill -TERM "$RN" 2>/dev/null
    break
  fi
  sleep 1
done
wait "$RN"; rc=$?
kill "$TS" 2>/dev/null
peak=$(awk -F, 'NR>1 && $3+0>0{if($3>m)m=$3}END{print m+0}' "$DIR/thermal.csv")
hit=$(grep -c LIMIT_HIT "$DIR/thermal.csv")
say "運行器離開碼 $rc；CPU 峰值 ${peak} °C；熱中止 ${hit} 次"
[ "$hit" -gt 0 ] && exit 93
exit "$rc"
