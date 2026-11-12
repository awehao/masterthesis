#!/usr/bin/env bash
# run_m8_house_door_approach（2026-11-12；Codex 20261112_120637）：住宅場景門把手非接觸接近（Isaac；非即時閉迴路）。不起 ROS、不插入、不閉爪、不開門、無行人。
#   由 run_guarded_v3.sh 包住（Isaac 互斥鎖＋92 °C 熱中止）。起跑前逐檔核對凍結清單（FREEZE），不符拒跑。異常不自動重跑。
# 用法：RUN_ID=m8house_da_smoke_1 MODE=smoke FREEZE=evaluation/experiments/<...>.sha256 ROS_DOMAIN_ID=95 RUNNER=evaluation/run_m8_house_door_approach.sh bash evaluation/run_guarded_v3.sh
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WS"
: "${RUN_ID:?RUN_ID 未設定}"
: "${MODE:?MODE（smoke｜approach）未設定}"
: "${FREEZE:?FREEZE（凍結清單）未設定}"
ISAAC_PY="${ISAAC_PY:-$HOME/venvs/isaacsim-6.0.1/bin/python}"
DIR="$WS/evaluation/runs/$RUN_ID"; mkdir -p "$DIR"
MEM_MAX="${MEM_MAX:-20G}"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$DIR/run.log"; }
if ! sha256sum -c --quiet "$FREEZE" > "$DIR/freeze_check.log" 2>&1; then
  say "凍結清單核對不符（$FREEZE）⇒ 拒跑：$(head -3 "$DIR/freeze_check.log")"; exit 96
fi
cp "$FREEZE" "$DIR/versions.sha256"
say "=== 門把手非接觸接近：RUN_ID=$RUN_ID MODE=$MODE；凍結 $FREEZE 全部相符；機器 $(hostname)／$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)／$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)／kernel $(uname -r)；GUI 開；非即時計算"
setsid systemd-run --user --scope -q -p MemoryMax="$MEM_MAX" -p MemorySwapMax=0 env -u VIRTUAL_ENV "$ISAAC_PY" -u evaluation/isaac_m8_house_door_approach.py --mode "$MODE" --out "$DIR" >> "$DIR/sim.log" 2>&1 &
SIM=$!
sleep 1
P=$(pgrep -f "[i]saac_m8_house_door_approach.py.*runs/$RUN_ID" | head -1)
say "  起 sim PID=${P:-$SIM}"
trap 'say "收到 TERM"; [ -n "${P:-}" ] && kill -TERM "$P" 2>/dev/null; sleep 3; [ -n "${P:-}" ] && kill -KILL "$P" 2>/dev/null; exit 143' TERM
trap 'say "收到 INT"; [ -n "${P:-}" ] && kill -TERM "$P" 2>/dev/null; exit 130' INT
wait "$SIM"; rc=$?
say "模擬器結束 rc=$rc"
grep -E "^\[da\] (完成|評估|\*\*鎖存|例外)|拒跑|Traceback" "$DIR/sim.log" | tail -8 | cut -c1-1500 | tee -a "$DIR/run.log"
exit "$rc"
