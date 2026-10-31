#!/usr/bin/env bash
# run_bottle_s0a（2026-10-31；**須 Codex 核可才執行**，核可一趟跑一趟；中止不自動補跑、不延伸 S0b）。
# 由 run_guarded_v3.sh 包 92 °C 熱中止：RUN_ID=bottle_s0a_1 ROS_DOMAIN_ID=95 RUNNER=evaluation/run_bottle_s0a.sh bash evaluation/run_guarded_v3.sh
# 同一 Isaac 程序開視窗持續更新畫面（Howard 10/31）；不錄影、不逐步寫圖。
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WS"
: "${RUN_ID:?RUN_ID 未設定}"
DIR="$WS/evaluation/runs/$RUN_ID"; mkdir -p "$DIR"
ISAAC_PY="${ISAAC_PY:-$HOME/venvs/isaacsim-6.0.1/bin/python}"
ASSET=src/my_omnibot_description/config/bottle_s0.yaml
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$DIR/run.log"; }
say "=== 瓶子 S0a 停車抓取 RUN_ID=$RUN_ID 資產 $ASSET sha256 $(sha256sum $ASSET | cut -c1-16)；git $(git rev-parse --short HEAD) ==="
sha256sum evaluation/isaac_bottle_s0a.py evaluation/bottle_s0_ctrl.py evaluation/bottle_s0_logic.py evaluation/bottle_ref.py src/ammr_wholebody_mpc/ammr_wholebody_mpc/wgmpc_core_sp_fr.py evaluation/run_bottle_s0a.sh "$ASSET" > "$DIR/versions.sha256"
"$ISAAC_PY" -u evaluation/isaac_bottle_s0a.py --asset "$ASSET" --out "$DIR" --headless false > "$DIR/sim.log" 2>&1 &
SIM=$!
say "起 sim PID=$SIM"
wait $SIM; RC=$?
say "模擬器離開碼 $RC"
grep -aE "\[s0a\]" "$DIR/sim.log" | tee -a "$DIR/run.log"
exit $RC
