#!/usr/bin/env bash
# run_bottle_replay：瓶子任務該趟實錄重播，只渲染、不跑控制（isaac_bottle_replay.py）。開窗跑，不改任何既有實錄。
# 由 run_guarded_v3.sh 包 92 °C 熱中止：RUN_ID=bottle_s3_1_replay SRC_RUN=bottle_s3_1 ROS_DOMAIN_ID=96 RUNNER=evaluation/run_bottle_replay.sh bash evaluation/run_guarded_v3.sh
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WS"
: "${RUN_ID:?RUN_ID 未設定}"
: "${SRC_RUN:?SRC_RUN 未設定}"
DIR="$WS/evaluation/runs/$RUN_ID"; mkdir -p "$DIR"
ISAAC_PY="${ISAAC_PY:-$HOME/venvs/isaacsim-6.0.1/bin/python}"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$DIR/run.log"; }
[ -s "evaluation/runs/$SRC_RUN/s1_steps.npz" ] || { say "**找不到來源實錄** $SRC_RUN"; exit 81; }
say "=== 瓶子實錄重播 RUN_ID=$RUN_ID 來源 $SRC_RUN（steps sha256 $(sha256sum evaluation/runs/$SRC_RUN/s1_steps.npz | cut -c1-16)）參數［${RP_ARGS:-無}］；git $(git rev-parse --short HEAD)"
"$ISAAC_PY" -u "${RP_SCRIPT:-evaluation/isaac_bottle_replay.py}" --src "$SRC_RUN" --out "$DIR" --headless false ${RP_ARGS:-} > "$DIR/sim.log" 2>&1 &
SIM=$!
say "起 sim PID=$SIM"
(
  for i in $(seq 1 90); do
    W=$(xdotool search --onlyvisible --name "Isaac Sim" 2>/dev/null | tail -1)
    if [ -n "$W" ]; then
      xdotool windowmap "$W" windowraise "$W" windowactivate "$W" 2>/dev/null
      say "視窗 $W 已置前（第 $i 秒）"
      break
    fi
    kill -0 "$SIM" 2>/dev/null || break
    read -t 1 < /dev/zero
  done
) &
wait $SIM; RC=$?
RC2=$(cat "$DIR/exit_rc.txt" 2>/dev/null || echo 99)
say "程序離開碼 $RC；重播結果碼 $RC2（0＝影格齊全且姿態在容差內、5＝有缺圖或姿態超容差、99＝無結果檔）；影格 $(ls "$DIR/frames" 2>/dev/null | wc -l)"
grep -aE "\[rp\]" "$DIR/sim.log" | grep -av '/' | tee -a "$DIR/run.log"
[ "$RC" -ne 0 ] && exit "$RC"
exit "$RC2"
