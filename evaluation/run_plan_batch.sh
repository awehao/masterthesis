#!/usr/bin/env bash
# 依「逐趟排程檔」執行一批（C1 起用；C0 的 run_c0_batch.sh 保留不動）。
#
# 排程檔（TSV，含表頭）：idx rid case open_m method N pair order
# 每趟：凍結核對 → 清本 domain 殘留（依 environ 的 ROS_DOMAIN_ID、依 PID）→ 記運算環境
#       → run_guarded.sh（92 °C 熱中止）＋每秒 CPU 頻率取樣 → 清殘留 → 再核凍結。
# 只有 OPEN_M 與 WGMPC_N 隨排程變；其餘環境變數與 C0 相同。
#
# 停止條件：凍結不符（95）、守護層中止（90–94）。任務失敗照樣往下跑（失敗留在分母）。
# 批次中途改程式：舊資料標為舊版本批次、保留、停止合併（另建批次），不刪。
#
#   PLAN=evaluation/results/horizon_ablation/c1a_schedule.tsv \
#   FREEZE=evaluation/results/horizon_ablation/freeze_c1a.sha256 \
#   BATCH=c1a_b1 DOMAIN=94 bash evaluation/run_plan_batch.sh
#   續跑：RESUME_FROM=<idx>（前面各趟必須守護離開碼 0；中斷事件寫進同一份批次紀錄）
#   只跑某幾趟（功能確認）：ONLY_IDX="01"
# 本腳本自己**不設** ROS_DOMAIN_ID，清殘留時才不會誤判自己。
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WS"
PLAN="${PLAN:?PLAN 要明設}"
FREEZE="${FREEZE:?FREEZE 要明設}"
BATCH="${BATCH:?BATCH 要明設}"
DOMAIN="${DOMAIN:-94}"
RESUME_FROM="${RESUME_FROM:-}"
ONLY_IDX="${ONLY_IDX:-}"
LOG="$WS/evaluation/runs/${BATCH}_batch.log"
mkdir -p "$WS/evaluation/runs"
if [ -z "$RESUME_FROM" ] && [ -z "$ONLY_IDX" ]; then
  [ -e "$LOG" ] && { echo "$LOG 已存在，拒跑"; exit 91; }
fi
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

clean_domain(){
  local n=0 p
  for p in $(pgrep -u "$USER"); do
    [ "$p" = "$$" ] && continue
    if tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -qx "ROS_DOMAIN_ID=$DOMAIN"; then
      say "  清殘留 PID=$p $(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-80)"
      kill -TERM "$p" 2>/dev/null; n=$((n+1))
    fi
  done
  [ "$n" -gt 0 ] && sleep 3
  for p in $(pgrep -u "$USER"); do
    tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -qx "ROS_DOMAIN_ID=$DOMAIN" \
      && kill -KILL "$p" 2>/dev/null
  done
  return 0
}
freeze_ok(){ sha256sum -c --quiet "$FREEZE" >> "$LOG" 2>&1; }

# 運算環境快照（R4）
env_snapshot(){  # $1 = 輸出檔
  python3 - "$1" <<'PY'
import json, os, sys, glob, platform
def rd(p):
    try: return open(p).read().strip()
    except OSError: return None
fr = [int(rd(p)) for p in glob.glob('/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq') if rd(p)]
temp = None
for h in glob.glob('/sys/class/hwmon/*/'):
    if rd(h + 'name') == 'k10temp':
        temp = int(rd(h + 'temp1_input')) / 1000
out = {'kernel': platform.release(),
       'platform_profile': rd('/sys/firmware/acpi/platform_profile'),
       'epp': rd('/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference'),
       'governor': rd('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'),
       'driver': rd('/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver'),
       'boost': rd('/sys/devices/system/cpu/cpufreq/boost'),
       'ac_online': rd('/sys/class/power_supply/ADP1/online'),
       'cpu_mhz_avg_at_start': round(sum(fr) / len(fr) / 1000) if fr else None,
       'cpu_c_at_start': temp}
json.dump(out, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False))
PY
}
freq_sampler(){  # $1 = 輸出 csv；每秒平均／最高頻率（MHz）
  echo "wall_s,iso,cpu_mhz_avg,cpu_mhz_max" > "$1"
  local t0; t0=$(date +%s)
  while true; do
    awk '{s+=$1; if($1>m)m=$1; n++} END{printf "%d,%d\n", s/n/1000, m/1000}' \
      /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>/dev/null \
      | sed "s/^/$(( $(date +%s) - t0 )),$(date +%H:%M:%S),/" >> "$1"
    sleep 1
  done
}

say "=== 批次 $BATCH 排程 $PLAN（sha256 $(sha256sum "$PLAN" | cut -c1-16)）domain=$DOMAIN ==="
say "批次腳本 sha256 $(sha256sum "$0" | cut -c1-16)；凍結清單 $FREEZE（$(sha256sum "$FREEZE" | cut -c1-16)）；kernel $(uname -r)"
[ -n "$RESUME_FROM" ] && say "  續跑：自 idx $RESUME_FROM（前一個批次程序中斷；非程式變更）"
[ -n "$ONLY_IDX" ] && say "  只跑 idx：$ONLY_IDX"

while IFS=$'\t' read -r IDX RID CASE OPEN_M METHOD N PAIR ORDERTAG <&3; do
  if [ -n "$ONLY_IDX" ]; then
    case " $ONLY_IDX " in *" $IDX "*) ;; *) continue ;; esac
  elif [ -n "$RESUME_FROM" ] && [ "$((10#$IDX))" -lt "$((10#$RESUME_FROM))" ]; then
    tail -1 "$WS/evaluation/runs/$RID/guard.log" 2>/dev/null | grep -q '守護離開碼 0' \
      || { say "**idx $IDX $RID 沒有完整收尾紀錄**，拒絕續跑"; exit 96; }
    continue
  fi
  [ -e "$WS/evaluation/runs/$RID" ] && { say "**$RID 已存在**，拒跑（不覆寫）"; exit 91; }
  freeze_ok || { say "**凍結不符**（$RID 起跑前），整批停止"; exit 95; }
  clean_domain
  say "--- idx $IDX $RID（$CASE open_m=$OPEN_M $METHOD N=$N，$PAIR $ORDERTAG）起跑"
  say "    環境 $(env_snapshot "$WS/evaluation/runs/${RID}.env.json")"
  freq_sampler "$WS/evaluation/runs/${RID}.freq.csv" < /dev/null & FS=$!
  ROS_DOMAIN_ID="$DOMAIN" RUN_ID="$RID" WGMPC_N="$N" HEADING=1 MOTM=1 CAM=false OPEN_M="$OPEN_M" \
  FINGER_COL=split \
  DRAWER_ASSET="$WS/src/my_omnibot_description/config/drawer_unit_bar26.yaml" \
  GRASP_DEPTH_M=0.0068 \
  MOTM_TASK_ARGS="--standoff-m 0.03 --pre-ramp-mps 0.03 --pre-settle-s 0.5 --restow-mode sync --approach-rate 0.02 --motm-w-qn 0.8 --motm-w-vref-pre 0.3" \
  SOLVER_EXTRA="--offset-init-static" \
  bash "$WS/evaluation/run_guarded.sh" > "$WS/evaluation/runs/${RID}.guard.out" 2>&1 < /dev/null
  rc=$?
  kill "$FS" 2>/dev/null
  say "    $RID 結束 rc=$rc；$(tail -1 "$WS/evaluation/runs/$RID/guard.log" 2>/dev/null)"
  clean_domain
  freeze_ok || { say "**凍結不符**（$RID 跑完後），整批停止"; exit 95; }
  case "$rc" in
    90|91|92|93|94) say "**守護層中止 rc=$rc**，整批停止"; exit "$rc" ;;
  esac
done 3< <(tail -n +2 "$PLAN")
say "=== 批次結束 ==="
