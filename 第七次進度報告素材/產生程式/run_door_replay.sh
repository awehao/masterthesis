#!/usr/bin/env bash
# 門場景「該趟實錄重播」：只渲染、不跑控制、不起 ROS（isaac_door_k2_sim_ev2r.py --replay；規格 K3_root_evidence_spec_draft2.md §5）。
# 逐幀設底盤位姿、手臂與手指關節、門扇繞鉸鏈轉到紀錄門角；每幀速度歸零、推一個物理步讓變換生效後擷取。
# 輸出：evaluation/runs/$RUN_ID/frames/f_NNNNN.png 與 frames.csv。由 run_guarded_v2.sh 包 92 °C 熱中止：
#   RUN_ID=door_k3ab_lw_1_replay_side SRC_RUN=door_k3ab_lw_1 VIEW=side ROS_DOMAIN_ID=98 RUNNER=evaluation/run_door_replay.sh bash evaluation/run_guarded_v2.sh
# VIEW：side（側後方全景）｜close（夾爪與把手近景）。T0／T1：重播的模擬時間範圍（預設整趟）。HZ：每秒幀數（模擬時間；預設 10）。
# 收尾只停本趟 PID。重播不改任何既有實錄；來源 room_run.json 以 sha256 記入 run.log。
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WS"
: "${RUN_ID:?RUN_ID 未設定}"
: "${SRC_RUN:?SRC_RUN 未設定}"
VIEW="${VIEW:-side}"
T0="${T0:-0}"; T1="${T1:-1000000000}"; HZ="${HZ:-10}"
ISAAC_PY="${ISAAC_PY:-$HOME/venvs/isaacsim-6.0.1/bin/python}"
DIR="$WS/evaluation/runs/$RUN_ID"; mkdir -p "$DIR"
SRC="$WS/evaluation/runs/$SRC_RUN/room_run.json"
STAND='0.6,-0.5332,1.570796'
K2_READY='0,0,0.035,0,-1.5707963,0'
case "$VIEW" in
  side)  EYE='2.4,-2.3,1.7';  AT='0.55,-0.25,0.75'; FOC=18 ;;
  close) EYE='1.55,-0.95,1.3'; AT='0.8,-0.1,0.85';  FOC=24 ;;
  *) echo "VIEW 只接受 side｜close"; exit 2 ;;
esac
PIDS=()
source "$WS/evaluation/k2_proc.sh"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$DIR/run.log"; }
cleanup(){ for p in "${PIDS[@]}"; do kill -TERM "$p" 2>/dev/null || true; done; sleep 3; for p in "${PIDS[@]}"; do kill -KILL "$p" 2>/dev/null || true; done; }
trap cleanup EXIT
[ -s "$SRC" ] || { say "**找不到來源實錄** $SRC"; exit 81; }
say "=== 門場景實錄重播 RUN_ID=$RUN_ID 來源 $SRC_RUN（sha256 $(sha256sum "$SRC" | cut -c1-16)）VIEW=$VIEW eye=$EYE at=$AT foc=$FOC Hz=$HZ 範圍 [$T0, $T1] ==="
"$ISAAC_PY" -u evaluation/isaac_door_k2_sim_ev2r.py --replay "$SRC" --cam true --cam-hz "$HZ" --cam-res 1280x720 \
  --cam-eye "$EYE" --cam-at "$AT" --cam-foc "$FOC" --replay-t0 "$T0" --replay-t1 "$T1" \
  --finger-collision split --park-hold --start-pose "$STAND" --park-pose "$STAND" --stow-q "$K2_READY" --out "$DIR" \
  >> "$DIR/sim.log" 2>&1 &
PIDS+=("$!"); say "  起 sim PID=${PIDS[0]}"
wait_or_stop "${PIDS[0]}" 1800 15; say "  重播結束：$STOP_HOW rc=$STOP_RC；幀數 $(ls "$DIR/frames" 2>/dev/null | wc -l)"
grep -E '^\[door\] (重播|\*\*重播)' "$DIR/sim.log" | tee -a "$DIR/run.log"
[ "$STOP_HOW" = exited ] && [ "$STOP_RC" = 0 ] && exit 0
exit 1
