#!/usr/bin/env bash
# 依「逐趟排程檔」執行一批（C1 起用；C0 的 run_c0_batch.sh 保留不動）。
#
# 排程檔（TSV，含表頭）：idx rid case open_m method N pair order
# 每趟：凍結核對 → 清本批次殘留 → 記運算環境 → run_guarded.sh（92 °C 熱中止）＋每秒頻率取樣
#       → 寫逐趟狀態檔 <rid>.status.json → 清殘留 → 再核凍結。
# 只有 OPEN_M 與 WGMPC_N 隨排程變；其餘環境變數與 C0 相同。
#
# **逐趟分類（status.json 的 class）**
#   executed          運行器 rc 0、守護 rc 0、必要封存檔齊全 ⇒ 繼續下一趟。
#                     任務判準成敗由事後判定（任務失敗但紀錄有效 ⇒ 保留，不重跑）
#   guard_abort       守護層 90–94（熱中止／監看失效等）⇒ 停批
#   startup_failure   運行器 83 場景未建起／84 模擬器已結束／85 缺辨識檔／86 平滑器未啟用 ⇒ 停批
#   handover_failure  運行器 1／2（導航交棒節點回報失敗）⇒ 停批，交人工依登錄規則分類
#   missing_artifacts rc 0 但封存檔不齊 ⇒ 停批（證據／協定問題）
#   unclassified      其他非零 ⇒ 停批
#   interrupted       批次程序被中斷（INT／TERM）⇒ 收掉本趟帶標記的程序、離開碼 99；續跑不接受
#   任何非 executed 都停批並保存原因；不自動略過、不自動重跑。
#
# **清程序範圍**：每趟的子程序帶 C1_RUN_TAG=<BATCH>:<idx>；只清帶本批次標記的程序，
#   兩輪都排除本程序與所有祖先。domain 上若有**不帶本批次標記**的程序 ⇒ 停批（不殺別人的程序）。
#   父環境若帶 ROS_DOMAIN_ID，本腳本先以 env -u 去掉後重新 exec 自己，fork 出的子程序才不會帶著它。
#
#   PLAN=evaluation/results/horizon_ablation/c1a_schedule.tsv \
#   FREEZE=evaluation/results/horizon_ablation/freeze_c1a.sha256 \
#   BATCH=c1a_b1 DOMAIN=94 bash evaluation/run_plan_batch.sh
#   續跑：RESUME_FROM=<idx>（之前各趟的 status.json 必須是 executed；中斷事件寫進同一份批次紀錄）
#   只跑某幾趟（功能確認）：ONLY_IDX="01"
set -u
# 父環境帶 ROS_DOMAIN_ID 時，/proc/<pid>/environ 保留的是 exec 當下的環境，unset 改不到；
# 本程序 fork 出的子 shell 也會帶著它 ⇒ 會被誤認為佔用 domain。故先去掉它重新 exec 自己。
if [ -n "${ROS_DOMAIN_ID+x}" ]; then
  exec env -u ROS_DOMAIN_ID bash "$0" "$@"
fi
WS="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WS"
PLAN="${PLAN:?PLAN 要明設}"
FREEZE="${FREEZE:?FREEZE 要明設}"
BATCH="${BATCH:?BATCH 要明設}"
DOMAIN="${DOMAIN:-94}"
RESUME_FROM="${RESUME_FROM:-}"
ONLY_IDX="${ONLY_IDX:-}"
RUNS="$WS/evaluation/runs"
LOG="$RUNS/${BATCH}_batch.log"
mkdir -p "$RUNS"
if [ -z "$RESUME_FROM" ] && [ -z "$ONLY_IDX" ]; then
  [ -e "$LOG" ] && { echo "$LOG 已存在，拒跑"; exit 91; }
fi
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
FS=''
CUR_IDX=''
CUR_RID=''

# 本程序與所有祖先（不可被清理）
ANCESTORS=" $$ "
p=$$
while [ "$p" -gt 1 ] 2>/dev/null; do
  p=$(awk '{print $4}' "/proc/$p/stat" 2>/dev/null) || break
  [ -n "$p" ] || break
  ANCESTORS="$ANCESTORS$p "
done

env_has(){ tr '\0' '\n' < "/proc/$1/environ" 2>/dev/null | grep -qx "$2"; }
domain_procs(){  # 印出 "pid tagged|foreign"
  local q
  for q in $(pgrep -u "$USER"); do
    case "$ANCESTORS" in *" $q "*) continue ;; esac
    env_has "$q" "ROS_DOMAIN_ID=$DOMAIN" || continue
    if tr '\0' '\n' < "/proc/$q/environ" 2>/dev/null | grep -q "^C1_RUN_TAG=${BATCH}:"; then
      echo "$q tagged"
    else
      echo "$q foreign"
    fi
  done
}
clean_domain(){  # 只清本批次標記的程序；有外來程序 ⇒ 回傳 1
  local n=0 q kind foreign=0
  while read -r q kind; do
    [ -z "$q" ] && continue
    if [ "$kind" = tagged ]; then
      say "  清殘留 PID=$q $(tr '\0' ' ' < /proc/$q/cmdline 2>/dev/null | cut -c1-80)"
      kill -TERM "$q" 2>/dev/null; n=$((n+1))
    else
      say "  **domain $DOMAIN 有不屬於本批次的程序** PID=$q $(tr '\0' ' ' < /proc/$q/cmdline 2>/dev/null | cut -c1-80)"
      foreign=1
    fi
  done < <(domain_procs)
  [ "$n" -gt 0 ] && sleep 3
  while read -r q kind; do
    [ "$kind" = tagged ] && kill -KILL "$q" 2>/dev/null
  done < <(domain_procs)
  return $foreign
}
freeze_ok(){ sha256sum -c --quiet "$FREEZE" >> "$LOG" 2>&1; }
kill_tag(){  # $1 = 完整標記值；TERM 後 KILL
  local q
  for q in $(pgrep -u "$USER"); do
    case "$ANCESTORS" in *" $q "*) continue ;; esac
    env_has "$q" "C1_RUN_TAG=$1" && kill -TERM "$q" 2>/dev/null
  done
  sleep 3
  for q in $(pgrep -u "$USER"); do
    case "$ANCESTORS" in *" $q "*) continue ;; esac
    env_has "$q" "C1_RUN_TAG=$1" && kill -KILL "$q" 2>/dev/null
  done
  return 0
}

env_snapshot(){  # $1 = 輸出檔
  python3 - "$1" <<'PY'
import json, sys, glob, platform
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
classify(){  # $1 = 守護（run_guarded）rc；$2 = 趟次目錄 ⇒ 印出 "class runner_rc"
  local grc="$1" d="$2" rrc
  rrc=$(grep -oP '運行器離開碼 \K[0-9]+' "$d/guard.log" 2>/dev/null | tail -1)
  rrc="${rrc:-NA}"
  case "$grc" in 90|91|92|93|94) echo "guard_abort $rrc"; return ;; esac
  case "$rrc" in
    0) ;;
    83|84|85|86) echo "startup_failure $rrc"; return ;;
    1|2) echo "handover_failure $rrc"; return ;;
    *) echo "unclassified $rrc"; return ;;
  esac
  [ "$grc" = 0 ] || { echo "unclassified $rrc"; return; }
  local f
  for f in align_solver.json task.json room_run.json mission.json wholebody.json guard.log thermal.csv; do
    [ -s "$d/$f" ] || { echo "missing_artifacts $rrc"; return; }
  done
  grep -q '=== 結束 rc=0 ===' "$d/run.log" 2>/dev/null || { echo "missing_artifacts $rrc"; return; }
  echo "executed $rrc"
}
write_status(){  # rid idx class runner_rc guard_rc
  python3 - "$RUNS/$1.status.json" "$1" "$2" "$3" "$4" "$5" <<'PY'
import json, sys, time
f, rid, idx, cls, rrc, grc = sys.argv[1:]
json.dump({'rid': rid, 'idx': idx, 'class': cls, 'runner_rc': rrc, 'guard_rc': grc,
           'written_at': time.strftime('%Y-%m-%dT%H:%M:%S')}, open(f, 'w'),
          ensure_ascii=False, indent=1)
PY
}

on_interrupt(){
  [ -n "$FS" ] && kill "$FS" 2>/dev/null
  if [ -n "$CUR_IDX" ]; then
    kill_tag "${BATCH}:${CUR_IDX}"
    write_status "$CUR_RID" "$CUR_IDX" interrupted NA NA
    say "**批次被中斷**：idx $CUR_IDX $CUR_RID 記為 interrupted，已收掉本趟程序"
  else
    say "**批次被中斷**（趟次之間）"
  fi
  exit 99
}
trap on_interrupt INT TERM
trap '[ -n "$FS" ] && kill "$FS" 2>/dev/null' EXIT

say "=== 批次 $BATCH 排程 $PLAN（sha256 $(sha256sum "$PLAN" | cut -c1-16)）domain=$DOMAIN ==="
say "批次腳本 sha256 $(sha256sum "$0" | cut -c1-16)；凍結清單 $FREEZE（$(sha256sum "$FREEZE" | cut -c1-16)）；kernel $(uname -r)"
[ -n "$RESUME_FROM" ] && say "  續跑：自 idx $RESUME_FROM（前一個批次程序中斷；非程式變更）"
[ -n "$ONLY_IDX" ] && say "  只跑 idx：$ONLY_IDX"

while IFS=$'\t' read -r IDX RID CASE OPEN_M METHOD N PAIR ORDERTAG <&3; do
  if [ -n "$ONLY_IDX" ]; then
    case " $ONLY_IDX " in *" $IDX "*) ;; *) continue ;; esac
  elif [ -n "$RESUME_FROM" ] && [ "$((10#$IDX))" -lt "$((10#$RESUME_FROM))" ]; then
    cls=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['class'])" \
          "$RUNS/$RID.status.json" 2>/dev/null)
    [ "$cls" = executed ] || { say "**idx $IDX $RID 狀態不是 executed（${cls:-無狀態檔}）**，拒絕續跑"; exit 96; }
    say "  已執行：idx $IDX $RID"
    continue
  fi
  [ -e "$RUNS/$RID" ] && { say "**$RID 已存在**，拒跑（不覆寫）"; exit 91; }
  freeze_ok || { say "**凍結不符**（$RID 起跑前），整批停止"; exit 95; }
  clean_domain || { say "**domain 被其他程序佔用**，整批停止（不殺不屬於本批次的程序）"; exit 98; }
  say "--- idx $IDX $RID（$CASE open_m=$OPEN_M $METHOD N=$N，$PAIR $ORDERTAG）起跑"
  say "    環境 $(env_snapshot "$RUNS/${RID}.env.json")"
  CUR_IDX="$IDX"; CUR_RID="$RID"
  freq_sampler "$RUNS/${RID}.freq.csv" < /dev/null & FS=$!
  # method：H5／H1（時域消融，行為與既有完全相同）；MOTM／PARK（MotM 時間比較）；
  #         B1（WG4-B：任務參數與 MOTM 完全相同，只把求解節點換成單步 QP，kp 取 B1_KP_SET，預設 1.0）
  BASE_ARGS="--standoff-m 0.03 --pre-ramp-mps 0.03 --pre-settle-s 0.5 --restow-mode sync --approach-rate 0.02"
  SK=""; BKP=""
  case "$METHOD" in
    PARK) MOTM_V=0; TASK_ARGS="$BASE_ARGS"; AREF="" ;;
    B1)   MOTM_V=1; TASK_ARGS="$BASE_ARGS --motm-w-qn 0.8 --motm-w-vref-pre 0.3 ${MOTM_EXTRA_ARGS:-}"; AREF="${MOTM_A_REF_SET:-}"
          SK=b1; BKP="${B1_KP_SET:-1.0}" ;;
    MOTM) MOTM_V=1; TASK_ARGS="$BASE_ARGS --motm-w-qn 0.8 --motm-w-vref-pre 0.3 ${MOTM_EXTRA_ARGS:-}"; AREF="${MOTM_A_REF_SET:-}" ;;
    *)    MOTM_V=1; TASK_ARGS="$BASE_ARGS --motm-w-qn 0.8 --motm-w-vref-pre 0.3"; AREF="" ;;
  esac
  say "    方法 $METHOD：MOTM=$MOTM_V MOTM_A_REF=${AREF:-（預設）} 任務參數 $TASK_ARGS"
  if [ -n "$AREF" ]; then export MOTM_A_REF="$AREF"; else unset MOTM_A_REF; fi
  if [ -n "$SK" ]; then export SOLVER_KIND="$SK" B1_KP="$BKP"; say "    求解節點：$SK kp=$BKP"
  else unset SOLVER_KIND B1_KP; fi
  C1_RUN_TAG="${BATCH}:${IDX}" ROS_DOMAIN_ID="$DOMAIN" RUN_ID="$RID" WGMPC_N="$N" HEADING=1 MOTM="$MOTM_V" CAM=false OPEN_M="$OPEN_M" \
  FINGER_COL=split \
  DRAWER_ASSET="$WS/src/my_omnibot_description/config/drawer_unit_bar26.yaml" \
  GRASP_DEPTH_M=0.0068 \
  MOTM_TASK_ARGS="$TASK_ARGS" \
  SOLVER_EXTRA="--offset-init-static" \
  bash "$WS/evaluation/run_guarded.sh" > "$RUNS/${RID}.guard.out" 2>&1 < /dev/null &
  GP=$!
  wait "$GP"; grc=$?          # wait 可被訊號打斷 ⇒ 中斷時 trap 立刻處理
  CUR_IDX=''; CUR_RID=''
  kill "$FS" 2>/dev/null; FS=''
  read -r CLS RRC < <(classify "$grc" "$RUNS/$RID")
  write_status "$RID" "$IDX" "$CLS" "$RRC" "$grc"
  say "    $RID 結束：class=$CLS 運行器 rc=$RRC 守護 rc=$grc；$(tail -1 "$RUNS/$RID/guard.log" 2>/dev/null)"
  clean_domain || say "  **收尾時 domain 有不屬於本批次的程序**（未處理）"
  freeze_ok || { say "**凍結不符**（$RID 跑完後），整批停止"; exit 95; }
  if [ "$CLS" != executed ]; then
    say "**$RID 分類 $CLS**，整批停止；依登錄規則人工處置（不自動略過、不自動重跑）"
    case "$CLS" in guard_abort) exit "$grc" ;; *) exit 97 ;; esac
  fi
done 3< <(tail -n +2 "$PLAN")
say "=== 批次結束 ==="
