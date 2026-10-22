#!/usr/bin/env bash
# K3-A：局部通行門 v2 的夾持 → 小角度開門 8° → SETTLE → 保留抓持受控停止（不張爪、不退離）。
# 規格 evaluation/results/specs/K3_door_small_open_spec.md（draft-1～3；範圍依 Codex 20261021 draft-3 審查縮為 K3-A）。
# run_door_k2.sh 的版本化複本：只換任務節點（door_k3a_task_node.py）、凍結清單與紀錄字樣；執行端／展開／轉接／求解器與參數同 K2。
# **INT 版**（run_door_k3a.sh 的複本）：唯一差異＝求解節點改用 dl3_wgmpc_wg2_adm2_node.py 並加 --offset-observer interval
# （有限區間等效偏移觀測；Codex 20261022 ADM2 接入審查通過）。其餘參數、K3-A 範圍、判定器全同；不帶 --offset-admission。
# 由 run_guarded_v2.sh 包 92 °C 熱中止：
#   RUN_ID=door_k3a_int_1 ROS_DOMAIN_ID=97 RUNNER=evaluation/run_door_k3a_int.sh bash evaluation/run_guarded_v2.sh
# 節點：門場景執行端（PARK_HOLD）→ 展開節點（K2_READY → Q_PRE_DOOR）→ K2 任務節點 → 轉接器 → W-GMPC 求解器。
# **不起導航**（直接生成在停位）。求解器參數＝runs/dl3v_M/align_solver.json args，**只差**：--park-hold（K2 底盤停車伺服）、
# 不接 --coord-topic（無 MotM）。收尾只停本趟 PID。
OFFSET_MOVING=1
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WS"
: "${ROS_DOMAIN_ID:?ROS_DOMAIN_ID 未設定}"
: "${RUN_ID:?RUN_ID 未設定}"
ISAAC_PY="${ISAAC_PY:-$HOME/venvs/isaacsim-6.0.1/bin/python}"
DIR="$WS/evaluation/runs/$RUN_ID"; mkdir -p "$DIR"
FRZ=evaluation/results/door/freeze_door_k3a_int_dev.sha256
STAND='0.6,-0.5332,1.570796'
K2_READY='0,0,0.035,0,-1.5707963,0'
Q_PRE='-0.9471,0.0758,1.6024,-1.539,-0.9479,-0.0544'
PIDS=(); NAMES=()
source "$WS/evaluation/k2_proc.sh"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$DIR/run.log"; }
spawn(){ local n="$1"; shift; "$@" >>"$DIR/$n.log" 2>&1 & PIDS+=("$!"); NAMES+=("$n"); say "  起 $n PID=${PIDS[-1]}"; }
cleanup(){ say "cleanup（只針對本趟 PID；TERM → 3 s → KILL）"
  for i in "${!PIDS[@]}"; do kill -TERM "${PIDS[$i]}" 2>/dev/null || true; done
  sleep 3
  for i in "${!PIDS[@]}"; do kill -KILL "${PIDS[$i]}" 2>/dev/null || true; done; }
trap cleanup EXIT
trap 'say "收到 TERM"; exit 143' TERM
trap 'say "收到 INT"; exit 130' INT

sha256sum -c --quiet "$FRZ" >> "$DIR/run.log" 2>&1 || { say "**K3-A INT 開發凍結不符** ⇒ 拒跑"; exit 95; }
say "=== K3-A（INT：--offset-observer interval）RUN_ID=$RUN_ID domain=$ROS_DOMAIN_ID；停位 $STAND；起始 K2_READY；夾持→開到 8°→保持（不張爪、不退離）==="

spawn sim "$ISAAC_PY" -u evaluation/isaac_door_k2_sim.py --sim-limit 240 --contact-min-n 0.5 \
  --finger-collision split --park-hold --start-pose "$STAND" --park-pose "$STAND" --stow-q "$K2_READY" --out "$DIR"
say "  等場景建起（最多 180 s）"
for i in $(seq 180); do grep -q '進入主迴圈' "$DIR/sim.log" 2>/dev/null && break; kill -0 "${PIDS[0]}" 2>/dev/null || break; sleep 1; done
grep -q '進入主迴圈' "$DIR/sim.log" 2>/dev/null || { say "**場景未建起** ⇒ 中止"; tail -8 "$DIR/sim.log" | tee -a "$DIR/run.log"; exit 83; }
grep -E '^\[(room|door)\] (零摩擦|夾持面|摩擦核對|門建於|初始門角|門扇接觸|牆接觸)' "$DIR/sim.log" | tee -a "$DIR/run.log"

spawn wholebody python3 -u evaluation/door_k2_wholebody_node.py --solver-handshake --park-hold --pos-tol 0.005 \
  --stow-q "$K2_READY" --park "$STAND" --q-unfold="$Q_PRE" --out "$DIR/wholebody.json"
spawn task python3 -u evaluation/door_k3a_task_node.py --out "$DIR/task.json"
ARM_IDENT="$WS/evaluation/results/wgmpc_arm_sp_ident_free4.json"
[ -f "$ARM_IDENT" ] || { say "**找不到手臂辨識檔** ⇒ 中止"; exit 85; }
spawn adapter python3 -u evaluation/arm_vel_adapter.py --in-topic /wgmpc/cmd_world --consumer-node /door_k2_sim
spawn solver python3 -u evaluation/dl3_wgmpc_wg2_adm2_node.py --require-topic-target --park-hold --offset-observer interval \
  --offset-moving --offset-free \
  --continuous --wait-for-drawer-handover --arm-model setpoint --arm-ident "$ARM_IDENT" \
  --cmd-topic /wgmpc/cmd_world --target-topic /drawer/tcp_target --target-offset 0 0 0 \
  --N 5 --rate 20 --delay-comp-state-cycles 1.6 --delay-comp-cmd-cycles 1.0 \
  --near-target-gamma 1.0 --w-s 0.001 --w-a 0.05 --w-s-arm 0.05 --margin-guard \
  --reach-pos-m 0.005 --reach-rot-rad 0.02 --hold-s 2.0 --duration-s 600 --duration-sim-s 400 \
  --u-prev-policy strict --assume-initial-rest --out "$DIR/align_solver.json"

say "[任務] 等 K3-A 任務節點結束（牆鐘最多 660 s）"
SIM_PID="${PIDS[0]}"; WB_PID="${PIDS[1]}"; TASK_PID="${PIDS[2]}"; SOLVER_PID="${PIDS[4]}"
wait_or_stop "$TASK_PID" 660 10; RC_TASK=$STOP_RC; HOW_TASK=$STOP_HOW
grep -E '^\[k3a\]' "$DIR/task.log" | tee -a "$DIR/run.log"
say "[收尾] 等求解節點停止並封存（最多 30 s，之後 TERM→10 s→KILL）"
wait_or_stop "$SOLVER_PID" 30 10; RC_SOLVER=$STOP_RC; HOW_SOLVER=$STOP_HOW
wait_or_stop "$WB_PID" 10 10; RC_WB=$STOP_RC; HOW_WB=$STOP_HOW
say "[收尾] 等執行端封存 room_run.json（最多 60 s；任務節點已送 /room/stop）"
wait_or_stop "$SIM_PID" 60 15; RC_SIM=$STOP_RC; HOW_SIM=$STOP_HOW
say "  結束方式：task $HOW_TASK、solver $HOW_SOLVER、wholebody $HOW_WB、sim $HOW_SIM（非 exited 即非正常結束）"
OK_ROOM=0; [ -s "$DIR/room_run.json" ] && OK_ROOM=1
OK_SOLVER=0; [ -s "$DIR/align_solver.json" ] && OK_SOLVER=1
OK_DONE=0; grep -q '^\[k3a\] 結束：DONE' "$DIR/task.log" && OK_DONE=1
say "  rc：task $RC_TASK、solver $RC_SOLVER、wholebody $RC_WB、sim $RC_SIM；room_run $OK_ROOM、align_solver $OK_SOLVER、DONE $OK_DONE"
RC=1
[ "$HOW_TASK" = exited ] && [ "$HOW_SOLVER" = exited ] && [ "$HOW_WB" = exited ] && [ "$HOW_SIM" = exited ] && [ "$RC_TASK" = 0 ] && [ "$RC_SOLVER" = 0 ] && [ "$RC_WB" = 0 ] && [ "$RC_SIM" = 0 ] && [ "$OK_ROOM" = 1 ] && [ "$OK_SOLVER" = 1 ] && [ "$OK_DONE" = 1 ] && RC=0
say "=== 結束 rc=$RC（0 只表示所有節點正常結束且到 DONE；通過與否由 door_k3a_check.py 判定）==="
exit "$RC"
