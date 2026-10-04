#!/usr/bin/env bash
# C0 時域消融正式批次：3 對交錯 H5, H1, H5, H1, H5, H1（experiment_spec_C0.yaml schedule）。
#
# 每趟：凍結核對 → 清掉本 domain 的殘留程序（只看 environ 的 ROS_DOMAIN_ID，依 PID）
#       → run_guarded.sh（92 °C 熱中止）→ 再清殘留 → 再核凍結。
# 凍結不符 ⇒ 整批停止（C0：批次中途改程式 ⇒ 作廢）。熱中止／監看失效 ⇒ 整批停止。
# 任務失敗（運行器非 0）照樣繼續 —— 失敗是結果，留在分母，不補跑。
#
#   BATCH=c0b1 DOMAIN=94 bash evaluation/run_c0_batch.sh
# 本腳本自己**不設** ROS_DOMAIN_ID，清殘留時才不會誤判自己。
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WS"
BATCH="${BATCH:?BATCH 要明設}"
DOMAIN="${DOMAIN:-94}"
FREEZE="$WS/evaluation/results/horizon_ablation/freeze_c0.sha256"
LOG="$WS/evaluation/runs/${BATCH}_batch.log"
[ -e "$LOG" ] && { echo "$LOG 已存在，拒跑"; exit 91; }
mkdir -p "$WS/evaluation/runs"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
ORDER=(H5 H1 H5 H1 H5 H1)

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

say "=== C0 批次 $BATCH domain=$DOMAIN 順序 ${ORDER[*]} ==="
say "批次腳本 sha256 $(sha256sum "$0" | cut -c1-16)；凍結清單 $(sha256sum "$FREEZE" | cut -c1-16)"
k=0
for M in "${ORDER[@]}"; do
  k=$((k+1)); pair=$(( (k+1)/2 ))
  RID="${BATCH}_p${pair}_${M}"
  N=$([ "$M" = H1 ] && echo 1 || echo 5)
  freeze_ok || { say "**凍結不符**（$RID 起跑前），整批停止"; exit 95; }
  clean_domain
  say "--- 第 $k 趟 $RID（N=$N）起跑"
  ROS_DOMAIN_ID="$DOMAIN" RUN_ID="$RID" WGMPC_N="$N" HEADING=1 MOTM=1 CAM=false OPEN_M=0.200 \
  FINGER_COL=split \
  DRAWER_ASSET="$WS/src/my_omnibot_description/config/drawer_unit_bar26.yaml" \
  GRASP_DEPTH_M=0.0068 \
  MOTM_TASK_ARGS="--standoff-m 0.03 --pre-ramp-mps 0.03 --pre-settle-s 0.5 --restow-mode sync --approach-rate 0.02 --motm-w-qn 0.8 --motm-w-vref-pre 0.3" \
  SOLVER_EXTRA="--offset-init-static" \
  bash "$WS/evaluation/run_guarded.sh" > "$WS/evaluation/runs/${RID}.guard.out" 2>&1
  rc=$?
  say "    $RID 結束 rc=$rc；$(tail -1 "$WS/evaluation/runs/$RID/guard.log" 2>/dev/null)"
  clean_domain
  freeze_ok || { say "**凍結不符**（$RID 跑完後），整批停止"; exit 95; }
  case "$rc" in
    90|91|92|93|94) say "**守護層中止 rc=$rc**，整批停止"; exit "$rc" ;;
  esac
done
say "=== 批次完成 ==="
