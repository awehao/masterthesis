#!/usr/bin/env bash
# 連續協同通行：核心段接觸驗證跑法（run_door_wide_w1f2.sh 的版本化複本；規格 evaluation/results/specs/COOPEN_core_spec_draft2.md；
#   Codex 20261026_235425 核可準備，**未核可 Isaac 前不得執行**）。
# 只驗：繞緣持握（φ 60°、s 0.85、roll 0）成立後協同開門到 48°（門速 0.6 °/s、底盤引導 w_vref 1、引導限幅 0.035 m/s／0.10 rad/s），
#   觀察開門末段的出口繞行，再沿既有受控結束路徑（SETTLE → 受控暫停 → 主動保持停止 → DONE）收尾；不放手、不收臂、不穿越。
# 實驗初始化（揭露）：底盤以 --start-pose 直接放在停位（候選 0° 底盤位姿），不經導航接近；手臂由 K2_READY 經既有展開節點到繞緣預抓取。
# 相對 W1f2 的差異：登錄檔固定 coopen_core_params_48_r5.json（θ、門速、走廊／引導表、包絡、觀察器上限、停位、Q_PRE 皆取自登錄）；
#   任務節點 door_coopen_core_task_node.py＋--grasp-basis coopen_core_grasp_basis_r4.json；凍結 freeze_door_coopen_core_dev_r2（r1 漏列 4 檔，作廢保留）；
#   判定器 door_coopen_check_core.py。執行端 ev5、轉接、求解器 ADM9、觀察器、增益、權重、限制、門檻、手指上限與暫停定義全同 W1f2。
OFFSET_MOVING=1
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WS"
: "${ROS_DOMAIN_ID:?ROS_DOMAIN_ID 未設定}"
: "${RUN_ID:?RUN_ID 未設定}"
ISAAC_PY="${ISAAC_PY:-$HOME/venvs/isaacsim-6.0.1/bin/python}"
DIR="$WS/evaluation/runs/$RUN_ID"; mkdir -p "$DIR"
FRZ=evaluation/results/door/freeze_door_coopen_core_dev_r2.sha256   # r1 漏列執行時讀取的 4 檔（稽核掛勾查得），作廢保留
PRM=evaluation/results/door/coopen_core_params_48_r5.json   # 核心段：固定登錄（不由環境變數改）
GB=evaluation/results/door/coopen_core_grasp_basis_r4.json
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

sha256sum -c --quiet "$FRZ" >> "$DIR/run.log" 2>&1 || { say "**核心段開發凍結不符** ⇒ 拒跑"; exit 95; }
say "python3＝$(command -v python3)；VIRTUAL_ENV=${VIRTUAL_ENV:-（無）}"
python3 -c "import osqp" 2>/dev/null || { say "**python3 環境錯誤（無法匯入 osqp；可能誤啟用虛擬環境）** ⇒ 拒跑"; exit 96; }
[ -s "$PRM" ] || { say "**核心段登錄檔不存在：$PRM** ⇒ 拒跑"; exit 97; }
[ -s "$GB" ] || { say "**抓取基準不存在：$GB** ⇒ 拒跑"; exit 97; }
GB_SHA=$(python3 -c "import json;print(json.load(open('$PRM'))['grasp']['basis_sha256'])")
[ "$(sha256sum "$GB" | cut -d' ' -f1)" = "$GB_SHA" ] || { say "**抓取基準 sha256 與登錄不符** ⇒ 拒跑"; exit 97; }
W1B_THETA=$(python3 -c "import json;print(json.load(open('$PRM'))['theta_goal_deg'])")
STAND=$(python3 -c "import json;print(json.load(open('$PRM'))['start']['stand_cli'])")
K2_READY=$(python3 -c "import json;print(json.load(open('$PRM'))['start']['k2_ready_cli'])")
Q_PRE=$(python3 -c "import json;print(json.load(open('$PRM'))['start']['q_pre_cli'])")
W1B_TABLE=$(python3 -c "import json;print(json.load(open('$PRM'))['protection_scope_change']['corridor_table'])")
W1B_ENV=$(python3 -c "import json;print(json.load(open('$PRM'))['protection_scope_change']['envelope']['cli'])")
W1B_OBS=$(python3 -c "import json;print(json.load(open('$PRM'))['protection_scope_change']['observer_door_max_deg'])")
W1B_COORD=$(python3 -c "import json;print(json.load(open('$PRM'))['control_cost_change']['coord_topic'])")
W1E_FEFF=$(python3 -c "import json;print(json.load(open('$PRM'))['executor_change']['pause_finger_max_effort'])")
W1E_FEFF0=$(python3 -c "import json;print(json.load(open('$PRM'))['executor_change']['original_effort'])")
say "W1e 執行端變更：暫停期手指驅動輸出上限 ${W1E_FEFF0} → ${W1E_FEFF} N（ev5 r2；啟動讀回不符拒跑、切換／定期讀回異常走中止閂鎖）"
say "W1b 登錄：$PRM（sha256 $(sha256sum "$PRM" | cut -c1-16)）θ=${W1B_THETA}° 走廊表 $W1B_TABLE 包絡 $W1B_ENV 觀察器上限 ${W1B_OBS}° 協同 $W1B_COORD"
say "=== 核心段（繞緣持握＋協同開門＋出口繞行觀察）RUN_ID=$RUN_ID domain=$ROS_DOMAIN_ID；停位 $STAND（實驗初始化）；起始 K2_READY；Q_PRE $Q_PRE；夾持→開到 ${W1B_THETA}°→受控暫停→停止（不張爪、不收臂、不穿越）==="

spawn sim "$ISAAC_PY" -u evaluation/isaac_door_k2_sim_ev5.py --pause-finger-max-effort "$W1E_FEFF" --pause-finger-original-effort "$W1E_FEFF0" --cmd-env --evidence-out "$DIR/evidence.jsonl" --park-gate-mobile --run-id "$RUN_ID" --corridor-table "$W1B_TABLE" --envelope="$W1B_ENV" --sim-limit 240 --contact-min-n 0.5 \
  --finger-collision split --park-hold --start-pose "$STAND" --park-pose "$STAND" --stow-q "$K2_READY" --out "$DIR"
say "  等場景建起（最多 180 s）"
for i in $(seq 180); do grep -q '進入主迴圈' "$DIR/sim.log" 2>/dev/null && break; kill -0 "${PIDS[0]}" 2>/dev/null || break; sleep 1; done
grep -q '進入主迴圈' "$DIR/sim.log" 2>/dev/null || { say "**場景未建起** ⇒ 中止"; tail -8 "$DIR/sim.log" | tee -a "$DIR/run.log"; exit 83; }
grep -E '^\[(room|door)\] (零摩擦|夾持面|摩擦核對|門建於|初始門角|門扇接觸|牆接觸)' "$DIR/sim.log" | tee -a "$DIR/run.log"

spawn wholebody python3 -u evaluation/door_k2_wholebody_node.py --solver-handshake --park-hold --pos-tol 0.005 \
  --stow-q "$K2_READY" --park "$STAND" --q-unfold="$Q_PRE" --out "$DIR/wholebody.json"
spawn task python3 -u evaluation/door_coopen_core_task_node.py --out "$DIR/task.json" --run-id "$RUN_ID" --grasp-basis "$GB" --w1b-params "$PRM" --corridor-table "$W1B_TABLE"
ARM_IDENT="$WS/evaluation/results/wgmpc_arm_sp_ident_free4.json"
[ -f "$ARM_IDENT" ] || { say "**找不到手臂辨識檔** ⇒ 中止"; exit 85; }
spawn adapter python3 -u evaluation/arm_vel_adapter_k3m.py --in-topic /wgmpc/cmd_world --consumer-node /door_k2_sim --cmd-env --env-in-topic /wgmpc/cmd_env
spawn solver python3 -u evaluation/dl3_wgmpc_wg2_adm9_node.py --require-topic-target --park-hold --offset-observer interval --mode-switch-run-id "$RUN_ID" --cmd-env --run-id "$RUN_ID" --pause --pause-arm-zero --pause-active-hold --pause-arm-win-s 0.5 --coord-topic "$W1B_COORD" --offset-hold-on-free \
  --offset-moving --offset-free \
  --continuous --wait-for-drawer-handover --arm-model setpoint --arm-ident "$ARM_IDENT" \
  --cmd-topic /wgmpc/cmd_world --target-topic /drawer/tcp_target --target-offset 0 0 0 \
  --N 5 --rate 20 --delay-comp-state-cycles 1.6 --delay-comp-cmd-cycles 1.0 \
  --near-target-gamma 1.0 --w-s 0.001 --w-a 0.05 --w-s-arm 0.05 --margin-guard \
  --reach-pos-m 0.005 --reach-rot-rad 0.02 --hold-s 2.0 --duration-s 600 --duration-sim-s 400 \
  --u-prev-policy strict --assume-initial-rest --out "$DIR/align_solver.json"
spawn lw_observer python3 -u evaluation/door_k3m_observer.py --out "$DIR/lw_observer.json" --door-max-deg "$W1B_OBS"

say "[任務] 等 K3-A 任務節點結束（牆鐘最多 660 s）"
SIM_PID="${PIDS[0]}"; WB_PID="${PIDS[1]}"; TASK_PID="${PIDS[2]}"; SOLVER_PID="${PIDS[4]}"; OBS_PID="${PIDS[5]}"
sleep 5; kill -0 "$OBS_PID" 2>/dev/null || { say "**獨立觀察器未存活** ⇒ 中止（沒有觀察期保護不得繼續）"; exit 86; }
wait_or_stop "$TASK_PID" 660 10; RC_TASK=$STOP_RC; HOW_TASK=$STOP_HOW
grep -E '^\[k3a\]' "$DIR/task.log" | tee -a "$DIR/run.log"
say "[收尾] 等求解節點停止並封存（最多 30 s，之後 TERM→10 s→KILL）"
wait_or_stop "$SOLVER_PID" 30 10; RC_SOLVER=$STOP_RC; HOW_SOLVER=$STOP_HOW
wait_or_stop "$WB_PID" 10 10; RC_WB=$STOP_RC; HOW_WB=$STOP_HOW
say "[收尾] 等執行端封存 room_run.json（最多 60 s；任務節點已送 /room/stop）"
wait_or_stop "$SIM_PID" 60 15; RC_SIM=$STOP_RC; HOW_SIM=$STOP_HOW
kill -TERM "$OBS_PID" 2>/dev/null || true
wait_or_stop "$OBS_PID" 10 5; RC_OBS=$STOP_RC; HOW_OBS=$STOP_HOW
say "LW_HEALTH sim=$HOW_SIM sim_rc=$RC_SIM obs=$HOW_OBS obs_rc=$RC_OBS task=$HOW_TASK task_rc=$RC_TASK"
grep -qE '"lw_observer_(unhealthy|not_ready)"' "$DIR/task.json" 2>/dev/null && say "**觀察器失效（任務節點記錄）** ⇒ 本趟記觀察提前終止"
say "  結束方式：task $HOW_TASK、solver $HOW_SOLVER、wholebody $HOW_WB、sim $HOW_SIM（非 exited 即非正常結束）"
OK_ROOM=0; [ -s "$DIR/room_run.json" ] && OK_ROOM=1
OK_SOLVER=0; [ -s "$DIR/align_solver.json" ] && OK_SOLVER=1
OK_DONE=0; grep -q '^\[k3a\] 結束：DONE' "$DIR/task.log" && OK_DONE=1
say "  rc：task $RC_TASK、solver $RC_SOLVER、wholebody $RC_WB、sim $RC_SIM；room_run $OK_ROOM、align_solver $OK_SOLVER、DONE $OK_DONE"
RC=1
[ "$HOW_TASK" = exited ] && [ "$HOW_SOLVER" = exited ] && [ "$HOW_WB" = exited ] && [ "$HOW_SIM" = exited ] && [ "$RC_TASK" = 0 ] && [ "$RC_SOLVER" = 0 ] && [ "$RC_WB" = 0 ] && [ "$RC_SIM" = 0 ] && [ "$OK_ROOM" = 1 ] && [ "$OK_SOLVER" = 1 ] && [ "$OK_DONE" = 1 ] && RC=0
say "=== 結束 rc=$RC（0 只表示所有節點正常結束且到 DONE；通過與否由 door_coopen_check_core.py 判定（w1f2 全部判準＋門速重建＋抓取登錄；A-2 量測只報告））==="
exit "$RC"
