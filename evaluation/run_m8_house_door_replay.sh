#!/usr/bin/env bash
# 住宅場景門把手接近的 Isaac 重播（只擺位＋算圖）。由 run_guarded_v3.sh 包住（Isaac 互斥鎖＋熱中止）。
# 用法：RUN_ID=m8house_da_approach_1_replay SRC_RUN=m8house_da_approach_1 ROS_DOMAIN_ID=95 RUNNER=evaluation/run_m8_house_door_replay.sh bash evaluation/run_guarded_v3.sh
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WS"
: "${RUN_ID:?}"; : "${SRC_RUN:?}"
ISAAC_PY="${ISAAC_PY:-$HOME/venvs/isaacsim-6.0.1/bin/python}"
DIR="$WS/evaluation/runs/$RUN_ID"; mkdir -p "$DIR"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$DIR/run.log"; }
say "=== 重播（擺位＋算圖，非物理）：來源 $SRC_RUN → $RUN_ID"
setsid systemd-run --user --scope -q -p MemoryMax=20G -p MemorySwapMax=0 env -u VIRTUAL_ENV "$ISAAC_PY" -u evaluation/isaac_m8_house_door_replay.py --run "$SRC_RUN" --out "$DIR/frames" >> "$DIR/sim.log" 2>&1 &
SIM=$!
sleep 1
P=$(pgrep -f "[i]saac_m8_house_door_replay.py.*runs/$RUN_ID" | head -1)
say "  起 sim PID=${P:-$SIM}"
trap '[ -n "${P:-}" ] && kill -TERM "$P" 2>/dev/null; exit 143' TERM INT
wait "$SIM"; rc=$?
say "模擬器結束 rc=$rc"
grep -E "^\[replay\] (完成|例外)|Traceback" "$DIR/sim.log" | tail -3 | tee -a "$DIR/run.log"
exit "$rc"
