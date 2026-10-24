#!/usr/bin/env bash
# B0 無門底盤表觀速度殘差短測：單一條件運行器（規格 results/specs/B0_base_residual_probe_spec.md；Codex 20261025_020934）。
# 由 run_guarded_v2.sh 包 92 °C 熱中止；一個 RUN_ID＝一個條件＝一個 Isaac 程序（獨立初始化）。未核可前不得執行。
#   env -u VIRTUAL_ENV PATH=<去掉 .venv-dl0> COND=k2_ready__zero RUN_ID=b0_k2_ready__zero_1 ROS_DOMAIN_ID=97 \
#     RUNNER=evaluation/run_base_residual_probe.sh bash evaluation/run_guarded_v2.sh
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WS"
: "${RUN_ID:?RUN_ID 未設定}"
: "${COND:?COND 未設定}"
ISAAC_PY="${ISAAC_PY:-$HOME/venvs/isaacsim-6.0.1/bin/python}"
DIR="$WS/evaluation/runs/$RUN_ID"; mkdir -p "$DIR"
FRZ=evaluation/results/door/freeze_base_residual_probe_dev_r1.sha256
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$DIR/run.log"; }
SIM=''
cleanup(){ [ -n "$SIM" ] && kill -0 "$SIM" 2>/dev/null && { kill -TERM "$SIM" 2>/dev/null; sleep 3; kill -KILL "$SIM" 2>/dev/null; }; true; }
trap cleanup EXIT
trap 'say "收到 TERM"; exit 143' TERM
trap 'say "收到 INT"; exit 130' INT
sha256sum -c --quiet "$FRZ" >> "$DIR/run.log" 2>&1 || { say "**B0 凍結不符** ⇒ 拒跑"; exit 95; }
python3 -c "import sys; sys.path.insert(0,'evaluation'); import base_residual_probe_core as C; C.parse_condition('$COND')" \
  || { say "**條件名不合法：$COND** ⇒ 拒跑"; exit 97; }
say "=== B0 RUN_ID=$RUN_ID 條件 $COND（無門；觀察 10 s＋暖機 1 s）==="
"$ISAAC_PY" -u evaluation/isaac_base_residual_probe.py --condition "$COND" --out "$DIR" >>"$DIR/sim.log" 2>&1 &
SIM=$!; say "  起 sim PID=$SIM"
wait "$SIM"; RC=$?; SIM=''
grep -E '^\[probe\]' "$DIR/sim.log" | tee -a "$DIR/run.log"
[ -s "$DIR/probe.json" ] || { say "**沒有 probe.json** rc=$RC"; exit 1; }
say "=== 結束 rc=$RC（只描述；判讀依規格 §4 預先登錄類別）==="
exit "$RC"
