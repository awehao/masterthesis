#!/usr/bin/env bash
# 手臂候選姿態靜態圖（isaac_arm_pose_stills.py），由 run_guarded_v3.sh 包 92 °C 熱中止。
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WS"
: "${RUN_ID:?RUN_ID 未設定}"
: "${POSES:?POSES 未設定}"
DIR="$WS/evaluation/runs/$RUN_ID"; mkdir -p "$DIR"
"${ISAAC_PY:-$HOME/venvs/isaacsim-6.0.1/bin/python}" -u evaluation/isaac_arm_pose_stills.py --out "$DIR" --poses "$POSES" > "$DIR/sim.log" 2>&1 &
SIM=$!
wait $SIM
