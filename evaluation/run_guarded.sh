#!/usr/bin/env bash
# 以 92 °C 牆鐘熱中止包住 run_nav_handover.sh（不改運行器本身）。
#
# 取樣器先起（場景載入期間也在量），模擬器 PID 一出現就交給它監看；
# 超溫時取樣器砍模擬器，本腳本再 TERM 運行器，讓它的 cleanup 收掉同趟其他 PID。
# **全程**核對取樣器：活著、讀值新鮮（≤ STALE_S 秒）、讀值有效（> 0 °C）；
# 模擬器還在跑而監看失效 ⇒ 本腳本自己砍模擬器並中止，記錄原因。
# 中止線只接受 1–92 的整數（取樣器以整數比較）；趟次目錄已存在就拒跑。
#
#   ROS_DOMAIN_ID=94 RUN_ID=h5_log_check1 <其他環境變數> bash evaluation/run_guarded.sh
#
# 產物：runs/<RUN_ID>/thermal.csv（每秒一筆）、thermal.log、guard.log
# 離開碼：運行器原碼；90 上限不合法；91 目錄已存在；92 取樣器起不來；
#        93 熱中止；94 監看失效而中止
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"
: "${ROS_DOMAIN_ID:?ROS_DOMAIN_ID 未設定}"
: "${RUN_ID:?RUN_ID 要明設（守護需要先知道趟次目錄）}"
LIMIT="${THERMAL_LIMIT:-92}"
if ! [[ "$LIMIT" =~ ^[1-9][0-9]*$ ]] || [ "$LIMIT" -gt 92 ]; then
  echo "THERMAL_LIMIT='$LIMIT' 不合法（只接受 1–92 的整數），拒跑"; exit 90
fi
STALE_S=5
RUNNER="${RUNNER:-$WS/evaluation/run_nav_handover.sh}"
DIR="$WS/evaluation/runs/$RUN_ID"
if [ -e "$DIR" ]; then echo "$DIR 已存在，拒跑（不覆寫既有實錄）"; exit 91; fi
mkdir -p "$DIR"
G="$DIR/guard.log"
CSV="$DIR/thermal.csv"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$G"; }
last_c(){ tail -1 "$CSV" 2>/dev/null | awk -F, '{print $3}'; }
csv_age(){ echo $(( $(date +%s) - $(stat -c %Y "$CSV" 2>/dev/null || echo 0) )); }

PIDF="$DIR/sim.pid"; : > "$PIDF"
bash "$WS/evaluation/thermal_sampler.sh" "$CSV" "$LIMIT" "$PIDF" \
  > "$DIR/thermal.log" 2>&1 &
TS=$!
sleep 2
if grep -q LIMIT_HIT "$CSV" 2>/dev/null; then
  say "起跑前已達中止線（$(grep LIMIT_HIT "$CSV")），不起跑"; exit 93
fi
c=$(last_c)
if ! kill -0 "$TS" 2>/dev/null || ! [[ "$c" =~ ^[0-9]+$ ]] || [ "$c" -le 0 ]; then
  say "熱取樣器沒有有效讀值（cpu_c='${c:-}'），不起跑"
  kill "$TS" 2>/dev/null; exit 92
fi
say "熱中止啟用：上限 ${LIMIT} °C，目前 ${c} °C，取樣器 PID=$TS，讀值逾 ${STALE_S} s 未更新即中止"

bash "$RUNNER" & RN=$!
say "運行器 PID=$RN（$RUNNER）"
sim=''
guard_rc=0
abort(){  # $1 = 原因, $2 = 離開碼
  say "$1"
  if [ -n "$sim" ] && kill -0 "$sim" 2>/dev/null; then
    kill -TERM "$sim" 2>/dev/null; say "已 TERM 模擬器 PID=$sim"
  fi
  kill -TERM "$RN" 2>/dev/null; say "已 TERM 運行器 PID=$RN"
  guard_rc=$2
}
while kill -0 "$RN" 2>/dev/null; do
  if [ -z "$sim" ]; then
    sim=$(grep -oP '起 sim PID=\K[0-9]+' "$DIR/run.log" 2>/dev/null | head -1)
    if [ -n "$sim" ]; then echo "$sim" > "$PIDF"; say "模擬器 PID=$sim 已交給取樣器"; fi
  fi
  if grep -q LIMIT_HIT "$CSV" 2>/dev/null; then
    abort "**熱中止**：$(grep LIMIT_HIT "$CSV" | tail -1)" 93; break
  fi
  sim_alive=0
  [ -n "$sim" ] && kill -0 "$sim" 2>/dev/null && sim_alive=1
  if ! kill -0 "$TS" 2>/dev/null; then
    # 取樣器在模擬器結束後自行停止是正常的；其餘都是監看失效
    if [ -z "$sim" ] || [ "$sim_alive" = 1 ]; then
      abort "**監看失效**：取樣器已結束而模擬器$([ -z "$sim" ] && echo '尚未起來' || echo '仍在跑')（thermal.log：$(tail -1 "$DIR/thermal.log" 2>/dev/null)）" 94
      break
    fi
  else
    a=$(csv_age); c=$(last_c)
    if [ "$a" -gt "$STALE_S" ]; then
      abort "**監看失效**：溫度讀值 ${a} s 未更新（上限 ${STALE_S} s）" 94; break
    fi
    if ! [[ "$c" =~ ^[0-9]+$ ]] || [ "$c" -le 0 ]; then
      abort "**監看失效**：最新溫度讀值無效（cpu_c='${c}'）" 94; break
    fi
  fi
  sleep 1
done
wait "$RN"; rc=$?
kill "$TS" 2>/dev/null
peak=$(awk -F, 'NR>1 && $3 ~ /^[0-9]+$/ {if($3+0>m)m=$3+0}END{print m+0}' "$CSV")
hit=$(grep -c LIMIT_HIT "$CSV")
say "運行器離開碼 $rc；CPU 峰值 ${peak} °C；熱中止 ${hit} 次；守護離開碼 ${guard_rc}"
[ "$guard_rc" -ne 0 ] && exit "$guard_rc"
exit "$rc"
