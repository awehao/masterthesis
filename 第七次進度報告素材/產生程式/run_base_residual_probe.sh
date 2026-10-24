#!/usr/bin/env bash
# B0 無門底盤表觀速度殘差短測：單一條件運行器（規格 results/specs/B0_base_residual_probe_spec.md；Codex 20261025_020934）。
# 由 run_guarded_v2.sh 包 92 °C 熱中止；一個 RUN_ID＝一個條件＝一個 Isaac 程序（獨立初始化）。
# r2（Codex 20261025_021820 必修②）：已有 probe.json 或啟動標記 ⇒ 拒跑；每條件牆鐘上限 TIMEOUT_S（預設 180 s），逾時以 k2_proc.sh
#   有界 TERM→KILL→wait 收尾；中斷時 Isaac 端保存已得資料並標 incomplete；彙整不完整 ⇒ 非 0 離開（批次停批、不補跑）。
#   env -u VIRTUAL_ENV PATH=<去掉 .venv-dl0> COND=k2_ready__zero RUN_ID=b0_k2_ready__zero_1 ROS_DOMAIN_ID=97 \
#     RUNNER=evaluation/run_base_residual_probe.sh bash evaluation/run_guarded_v2.sh
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WS"
: "${RUN_ID:?RUN_ID 未設定}"
: "${COND:?COND 未設定}"
ISAAC_PY="${ISAAC_PY:-$HOME/venvs/isaacsim-6.0.1/bin/python}"
DIR="${B0_RUNS_DIR:-$WS/evaluation/runs}/$RUN_ID"; mkdir -p "$DIR"   # B0_RUNS_DIR 只供離線測試覆寫（同 GUARD_RUNS_DIR）
FRZ=evaluation/results/door/freeze_base_residual_probe_dev_r2.sha256
TIMEOUT_S="${TIMEOUT_S:-180}"
source "$WS/evaluation/k2_proc.sh"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$DIR/run.log"; }
SIM=''
cleanup(){ if [ -n "$SIM" ]; then stop_bounded "$SIM" 10; say "cleanup：sim $STOP_HOW rc=$STOP_RC"; fi; true; }
trap cleanup EXIT
trap 'say "收到 TERM"; exit 143' TERM
trap 'say "收到 INT"; exit 130' INT
for f in probe.json started.marker; do
  [ -e "$DIR/$f" ] && { echo "**$DIR/$f 已存在** ⇒ 拒跑（不覆寫既有實錄）"; exit 91; }
done
sha256sum -c --quiet "$FRZ" >> "$DIR/run.log" 2>&1 || { say "**B0 凍結不符** ⇒ 拒跑"; exit 95; }
python3 -c "import sys; sys.path.insert(0,'evaluation'); import base_residual_probe_core as C; C.parse_condition('$COND')" \
  || { say "**條件名不合法：$COND** ⇒ 拒跑"; exit 97; }
say "=== B0 RUN_ID=$RUN_ID 條件 $COND（無門；觀察 10 s＋暖機 1 s）==="
date +%s > "$DIR/started.marker"
"$ISAAC_PY" -u evaluation/isaac_base_residual_probe.py --condition "$COND" --out "$DIR" >>"$DIR/sim.log" 2>&1 &
SIM=$!; say "  起 sim PID=$SIM"
wait_or_stop "$SIM" "$TIMEOUT_S" 15; RC=$STOP_RC; HOW=$STOP_HOW; SIM=''
grep -E '^\[probe\]' "$DIR/sim.log" | tee -a "$DIR/run.log"
say "  sim 結束方式 $HOW rc=$RC（上限 ${TIMEOUT_S} s）"
[ -s "$DIR/probe.json" ] || { say "**沒有 probe.json**（$HOW rc=$RC）⇒ incomplete"; exit 1; }
COMPLETE=$(python3 -c "import json;print(bool(json.load(open('$DIR/probe.json'))['summary'].get('complete')))" 2>/dev/null)
[ "$HOW" = exited ] && [ "$RC" = 0 ] && [ "$COMPLETE" = True ] || { say "**incomplete**（$HOW rc=$RC complete=$COMPLETE）⇒ 非 0 離開"; exit 3; }
say "=== 結束 rc=0（完整；只描述，判讀依規格 §4 預先登錄類別）==="
exit 0
