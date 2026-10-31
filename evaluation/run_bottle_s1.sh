#!/usr/bin/env bash
# run_bottle_s1（2026-10-31；**須 Codex 核可才執行**，核可一趟跑一趟；中止不自動補跑、不延伸 S2）。核心 gmpc_wb；障礙真值注入 CBF，無 LiDAR、無 Raw-scan Shield。
# 由 run_guarded_v3.sh 包 92 °C 熱中止：RUN_ID=bottle_s1_1 ROS_DOMAIN_ID=95 RUNNER=evaluation/run_bottle_s1.sh bash evaluation/run_guarded_v3.sh
# 同一 Isaac 程序開視窗持續更新畫面（Howard 10/31）；不錄影、不逐步寫圖。
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WS"
: "${RUN_ID:?RUN_ID 未設定}"
DIR="$WS/evaluation/runs/$RUN_ID"; mkdir -p "$DIR"
ISAAC_PY="${ISAAC_PY:-$HOME/venvs/isaacsim-6.0.1/bin/python}"
ASSET=src/my_omnibot_description/config/bottle_s0.yaml
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$DIR/run.log"; }
say "=== 瓶子 S1 候選完整任務（gmpc_wb，真值障礙） RUN_ID=$RUN_ID 資產 $ASSET sha256 $(sha256sum $ASSET | cut -c1-16)；git $(git rev-parse --short HEAD) ==="
sha256sum evaluation/isaac_bottle_s1.py evaluation/bottle_s1_ctrl.py evaluation/bottle_s0b_ctrl.py evaluation/bottle_branch.py evaluation/bottle_s0_ctrl.py evaluation/bottle_s0_logic.py evaluation/bottle_ref.py src/ammr_wholebody_mpc/ammr_wholebody_mpc/gmpc_wb.py src/ammr_wholebody_mpc/ammr_wholebody_mpc/gmpc.py evaluation/run_bottle_s1.sh "$ASSET" > "$DIR/versions.sha256"
"$ISAAC_PY" -u evaluation/isaac_bottle_s1.py --asset "$ASSET" --out "$DIR" --headless false > "$DIR/sim.log" 2>&1 &
SIM=$!
say "起 sim PID=$SIM"
wait $SIM; RC=$?
RC2=$(cat "$DIR/exit_rc.txt" 2>/dev/null || echo 99)   # sim_app.close() 會吞掉離開碼；檔案缺失算失敗
say "程序離開碼 $RC；腳本結果碼 $RC2（0＝完整成功、3＝已停止且有效、4＝停止未驗／無效、68＝接觸視圖拒跑、99＝無結果檔）"
grep -aE "\[s1\]" "$DIR/sim.log" | tee -a "$DIR/run.log"
[ "$RC" -ne 0 ] && exit "$RC"
exit "$RC2"
