#!/usr/bin/env bash
# run_bottle_probe（2026-10-31；S0c 受控碰觸；**須 Codex 核可才執行**，核可一趟跑一趟；不自動補跑）。
# 由 run_guarded_v3.sh 包 92 °C 熱中止：RUN_ID=bottle_probe_1 ROS_DOMAIN_ID=95 RUNNER=evaluation/run_bottle_probe.sh bash evaluation/run_guarded_v3.sh
# 同一 Isaac 程序開視窗持續更新畫面（Howard 10/31）；不錄影、不逐步寫圖。
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WS"
: "${RUN_ID:?RUN_ID 未設定}"
DIR="$WS/evaluation/runs/$RUN_ID"; mkdir -p "$DIR"
ISAAC_PY="${ISAAC_PY:-$HOME/venvs/isaacsim-6.0.1/bin/python}"
ASSET=src/my_omnibot_description/config/bottle_s0.yaml
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$DIR/run.log"; }
say "=== 瓶子 S0c 受控碰觸 RUN_ID=$RUN_ID 資產 $ASSET sha256 $(sha256sum $ASSET | cut -c1-16)；git $(git rev-parse --short HEAD) ==="
sha256sum evaluation/isaac_bottle_probe.py evaluation/bottle_s0_probe.py evaluation/bottle_s0_ctrl.py evaluation/bottle_s0_logic.py evaluation/bottle_ref.py src/ammr_wholebody_mpc/ammr_wholebody_mpc/wgmpc_core_sp_fr.py evaluation/run_bottle_probe.sh "$ASSET" > "$DIR/versions.sha256"
"$ISAAC_PY" -u evaluation/isaac_bottle_probe.py --asset "$ASSET" --out "$DIR" --headless false > "$DIR/sim.log" 2>&1 &
SIM=$!
say "起 sim PID=$SIM"
wait $SIM; RC=$?
RC2=$(cat "$DIR/exit_rc.txt" 2>/dev/null || echo 99)   # sim_app.close() 會吞掉離開碼；檔案缺失算失敗
say "程序離開碼 $RC；腳本結果碼 $RC2（0＝預期通道觸發且停止有效、5＝其他判定先觸發、6＝未接觸、7＝停止未驗／無效、68＝接觸視圖拒跑、99＝無結果檔）"
grep -aE "\[s0a\]" "$DIR/sim.log" | tee -a "$DIR/run.log"
[ "$RC" -ne 0 ] && exit "$RC"
exit "$RC2"
