#!/usr/bin/env bash
# run_bottle_smoke（2026-10-31；Codex 20261031 核可一次）：只放桌子與瓶子的 Isaac 冒煙（不載入機器人、不起 ROS）。
# 由 run_guarded_v3.sh 包 92 °C 熱中止：RUN_ID=bottle_smoke_1 RUNNER=evaluation/run_bottle_smoke.sh bash evaluation/run_guarded_v3.sh
# 開 Isaac 視窗（Howard 10/31：同一程序持續更新 viewport；不錄影、不逐步寫圖）。
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WS"
: "${RUN_ID:?RUN_ID 未設定}"
DIR="$WS/evaluation/runs/$RUN_ID"; mkdir -p "$DIR"
ISAAC_PY="${ISAAC_PY:-$HOME/venvs/isaacsim-6.0.1/bin/python}"
ASSET=src/my_omnibot_description/config/bottle_s0.yaml
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$DIR/run.log"; }
say "=== 瓶子冒煙（只放桌子＋瓶子）RUN_ID=$RUN_ID 資產 $ASSET sha256 $(sha256sum $ASSET | cut -c1-16) ==="
sha256sum evaluation/isaac_bottle_smoke.py evaluation/run_bottle_smoke.sh "$ASSET" evaluation/isaac_common.py > "$DIR/versions.sha256"
"$ISAAC_PY" -u evaluation/isaac_bottle_smoke.py --asset "$ASSET" --out "$DIR" --sim-s "${SIM_S:-2.0}" --hold-s "${HOLD_S:-3.0}" --headless false > "$DIR/sim.log" 2>&1 &
SIM=$!
say "起 sim PID=$SIM"
wait $SIM; RC=$?
say "模擬器離開碼 $RC"
grep -aE "\[bottle\]" "$DIR/sim.log" | tee -a "$DIR/run.log"
exit $RC
