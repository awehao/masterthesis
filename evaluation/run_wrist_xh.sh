#!/usr/bin/env bash
# 【XH2 版】run_wrist_v0.sh（sha256 e7fe869d71e61402c6c509068338af9d773d6a9749ea07afcd2e89d0a03df0d4）的複本：差異只有模擬器改 isaac_drawer_room_sim_xh.py、資產改由 DRAWER_ASSET 必設（不預設 bar26）。
# V0 腕部 RGB-D 擷取的運行器（由 run_guarded.sh 包 92 °C 熱中止）：
#   ROS_DOMAIN_ID=94 RUN_ID=v0_wrist_1 RUNNER=evaluation/run_wrist_v0.sh bash evaluation/run_guarded.sh
# 只起模擬器（--wrist-v0：擺位＋算圖＋發布腕部話題），不起任何控制節點。
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WS"
: "${RUN_ID:?RUN_ID 未設定}"
: "${DRAWER_ASSET:?DRAWER_ASSET 未設定（XH 資產檔絕對路徑）}"
DIR="$WS/evaluation/runs/$RUN_ID"; mkdir -p "$DIR"
ISAAC_PY="${ISAAC_PY:-$HOME/venvs/isaacsim-6.0.1/bin/python}"
"$ISAAC_PY" -u evaluation/isaac_drawer_room_sim_xh.py --wrist-v0 --cam false \
  --finger-collision split \
  --drawer-asset "$DRAWER_ASSET" \
  --out "$DIR" ${WRIST_EXTRA:-} > "$DIR/sim.log" 2>&1 &
SP=$!
echo "[$(date +%H:%M:%S)]   起 sim PID=$SP" >> "$DIR/run.log"
wait "$SP"; rc=$?
echo "=== 結束 rc=$rc ===" >> "$DIR/run.log"
exit "$rc"
