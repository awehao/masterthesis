#!/usr/bin/env bash
# K1 被動門短測的運行器（由 run_guarded_v2.sh 包 92 °C 熱中止；只起 Isaac，不起任何 ROS 節點、不放機器人）：
#   RUN_ID=door_k1_1 ROS_DOMAIN_ID=96 RUNNER=evaluation/run_door_k1.sh bash evaluation/run_guarded_v2.sh
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WS"
: "${RUN_ID:?RUN_ID 未設定}"
DIR="$WS/evaluation/runs/$RUN_ID"; mkdir -p "$DIR"
sha256sum -c --quiet evaluation/results/door/freeze_door_k1_dev.sha256 >> "$DIR/run.log" 2>&1 || { echo "K1 開發凍結不符 ⇒ 拒跑" >> "$DIR/run.log"; exit 95; }
ISAAC_PY="${ISAAC_PY:-$HOME/venvs/isaacsim-6.0.1/bin/python}"
"$ISAAC_PY" -u evaluation/door_k1_sim.py --out "$DIR" > "$DIR/sim.log" 2>&1 &
SP=$!
echo "[$(date +%H:%M:%S)]   起 sim PID=$SP" >> "$DIR/run.log"
wait "$SP"; rc=$?
echo "=== 結束 rc=$rc ===" >> "$DIR/run.log"
exit "$rc"
