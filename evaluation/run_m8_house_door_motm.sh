#!/usr/bin/env bash
# run_m8_house_door_motm（2026-11-13）：住宅門 MotM 名目展示（前導接近→移動中插入／閉爪→READY 交接綁定→P2 開門到 85° 停穩）；SMOKE=1 為接線冒煙（前導 2 s、只送首段、不閉爪不綁）。
#   理想感知＋理想抓持、非即時。由 run_guarded_v3.sh 包住（Isaac 互斥鎖＋92 °C 熱中止）。起跑前逐檔核對凍結清單（FREEZE），不符拒跑。異常不自動重跑。
# 用法：RUN_ID=m8house_door_motm_smoke_1 MODE=smoke SMOKE=1 FREEZE=evaluation/experiments/<...>.sha256 ROS_DOMAIN_ID=95 RUNNER=evaluation/run_m8_house_door_motm.sh bash evaluation/run_guarded_v3.sh
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WS"
: "${RUN_ID:?RUN_ID 未設定}"
: "${MODE:?MODE 未設定（本腳本只用作標記）}"
: "${FREEZE:?FREEZE（凍結清單）未設定}"
ISAAC_PY="${ISAAC_PY:-$HOME/venvs/isaacsim-6.0.1/bin/python}"
DIR="$WS/evaluation/runs/$RUN_ID"; mkdir -p "$DIR"
MEM_MAX="${MEM_MAX:-24G}"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$DIR/run.log"; }
if ! sha256sum -c --quiet "$FREEZE" > "$DIR/freeze_check.log" 2>&1; then
  say "凍結清單核對不符（$FREEZE）⇒ 拒跑：$(head -3 "$DIR/freeze_check.log")"; exit 96
fi
cp "$FREEZE" "$DIR/versions.sha256"
say "=== 門 MotM 名目展示：RUN_ID=$RUN_ID MODE=$MODE SMOKE=${SMOKE:-0}；凍結 $FREEZE 全部相符；機器 $(hostname)／$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)／$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)／kernel $(uname -r)；GUI 開；非即時；理想感知＋理想抓持"
setsid systemd-run --user --scope -q -p MemoryMax="$MEM_MAX" -p MemorySwapMax=0 env -u VIRTUAL_ENV "$ISAAC_PY" -u evaluation/isaac_m8_house_door_motm.py --out "$DIR" $( [ "${SMOKE:-0}" = "1" ] && echo --smoke ) >> "$DIR/sim.log" 2>&1 &
SIM=$!
sleep 1
P=$(pgrep -f "[i]saac_m8_house_door_motm.py.*runs/$RUN_ID" | head -1)
say "  起 sim PID=${P:-$SIM}"
trap 'say "收到 TERM"; [ -n "${P:-}" ] && kill -TERM "$P" 2>/dev/null; sleep 3; [ -n "${P:-}" ] && kill -KILL "$P" 2>/dev/null; pkill -TERM -f "[i]saac_m8_house_door_motm.py.*runs/$RUN_ID" 2>/dev/null; exit 143' TERM
trap 'say "收到 INT"; [ -n "${P:-}" ] && kill -TERM "$P" 2>/dev/null; exit 130' INT
wait "$SIM"; rc=$?
say "模擬器結束 rc=$rc"
grep -E "^\[motm\] |拒跑|Traceback" "$DIR/sim.log" | tail -30 | cut -c1-1200 | tee -a "$DIR/run.log"
exit "$rc"
