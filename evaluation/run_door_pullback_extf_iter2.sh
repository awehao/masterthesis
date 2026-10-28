#!/usr/bin/env bash
# extf_iter2 版（2026-10-29；Codex 核可一次）：extf_iter 跑法＋執行端 ev9r8l --robot-self-collision-off（明寫 physxArticulation:enabledSelfCollisions＝False）；其餘同 extf_iter。
# 以下為 extf_iter 版說明。extf_iter 版（2026-10-29；Codex 核可單一候選一趟）：extf_leader 跑法＋執行端 ev9r8k --robot-pos-iters 64（機器人 articulation 位置迭代顯式指定 64；原有效值未知）；
# 參考預設改 ref pb1e。其餘同 extf_leader。
# 以下為 extf_leader 版說明。extf_leader 版（2026-10-29；Codex 核可 (i) 一趟）：extf_flc 跑法＋執行端 ev9r8j --finger-leader-only（只驅動 finger_joint1、上限 5 N；finger_joint2 kp＝kd＝0 由 NewtonMimicAPI 連動；
# 位置目標只送手臂與 leader；啟動主從讀回；閉爪無接觸段跟隨核對失敗即停）。Δ 雙指目標已封存不含。其餘同 extf_flc。
# 以下為 extf_flc 版說明。extf_flc 版（2026-10-29；Codex extf_1 回覆核可一次新 RUN_ID）：extf 跑法＋手指失接觸逐物理步連續判定（執行端 ev9r8f --finger-loss-cont、任務端 node10c --finger-loss-cont）；
# 停止判準時間語義變更（原窗口最小力跨窗口累計）。extf=true、原模型、深度、夾持上限與其餘保護不變。
# 以下為 extf 版說明。extf 版（2026-10-29；Codex 核可單一候選、新 RUN_ID 單跑一次連續展示）：diag 跑法＋執行端 ev9r8e --ext-forces-every-iter（場景 enableExternalForcesEveryIteration＝True）；其餘同 diag。
# 以下為 diag 版說明。診斷版（2026-10-28；Codex 核可一次）：＝pb1_4 的跑法（commit 285ccb94e：登錄 r3、正面深度 6.8 mm、ref pb1、node10、ADM20、暫停上限 5 N、偏移凍結、速度迭代預設 1）
# 只把執行端換成 ev9r8d --diag-6d（取證另記門扇／兩指／掌心 6 維狀態；不改控制）。保護觸發即停，不為補資料延後中止。
# 用法：RUN_ID=door_pullback_extf_iter2_1 ROS_DOMAIN_ID=97 RUNNER=evaluation/run_door_pullback_extf_iter2.sh bash evaluation/run_guarded.sh
# 往後拉開 48° ＋換抓＋通過中拉關＋放手退出 Isaac 功能展示 pb1（Howard 10/28「不要邊繞，往後拉開」；Codex 20261028_195306 條件式核可一次新 RUN_ID）。
# run_door_regrasp_bench1.sh 的版本化複本（bench1 不改），只換三個登錄：
#   PRM＝results/door/pullback_params_48_r1.json（W1 正面開門配置：走廊表 door_wide_corridor_table_90、包絡 θ 48 重算、停位 W1、抓取初態／初始偏移取 w1f_60_3 夾持靜止窗、暫停期手指上限 20 N）；
#   GB＝runs/phf_01_M/task.json（正面抓法）；REF＝results/door/door_regrasp_ref_pb1.json（mode 3／4 由 W1 47.89° 實測末態重規劃，mode 5–8 同 ref v5 演算法）。
#   節點、旗標、模擬上限、守護與收尾全同 bench1。
# 用法：RUN_ID=door_pullback_pb1_1 ROS_DOMAIN_ID=97 RUNNER=evaluation/run_door_pullback_bench1.sh bash evaluation/run_guarded.sh
OFFSET_MOVING=1
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WS"
: "${ROS_DOMAIN_ID:?ROS_DOMAIN_ID 未設定}"
: "${RUN_ID:?RUN_ID 未設定}"
ISAAC_PY="${ISAAC_PY:-$HOME/venvs/isaacsim-6.0.1/bin/python}"
DIR="$WS/evaluation/runs/$RUN_ID"; mkdir -p "$DIR"
PRM=evaluation/results/door/pullback_params_48_r3.json   # pb1 r3：r2＋暫停期手指上限 5 N（pb1_3 張爪中止後，Codex 選 (a)）
GB=evaluation/runs/phf_01_M/task.json
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

REF="${PB1_REF:-evaluation/results/door/door_regrasp_ref_pb1e.json}"   # 加速變體可用 PB1_REF 指定 *_wp.json
DOOR3=src/my_omnibot_description/config/door_local_v3.yaml
sha256sum evaluation/isaac_door_k2_sim_ev9r8l.py evaluation/door_coopen_core_task_node10c.py evaluation/dl3_wgmpc_wg2_adm20_node.py evaluation/arm_vel_adapter_k3m3.py \
  evaluation/door_regrasp_*.py evaluation/door_finger_loss_*.py evaluation/door_mimic_follow.py evaluation/door_k3m_observer.py "$PRM" "$GB" "$REF" "$DOOR3" > "$DIR/versions.sha256" || { say "**版本紀錄失敗** ⇒ 拒跑"; exit 95; }
say "版本紀錄 $DIR/versions.sha256（$(wc -l < "$DIR/versions.sha256") 檔；git $(git rev-parse --short HEAD)）"
say "python3＝$(command -v python3)；VIRTUAL_ENV=${VIRTUAL_ENV:-（無）}"
python3 -c "import osqp" 2>/dev/null || { say "**python3 環境錯誤（無法匯入 osqp；可能誤啟用虛擬環境）** ⇒ 拒跑"; exit 96; }
[ -s "$PRM" ] || { say "**核心段登錄檔不存在：$PRM** ⇒ 拒跑"; exit 97; }
[ -s "$GB" ] || { say "**抓取基準不存在：$GB** ⇒ 拒跑"; exit 97; }
[ "$(python3 -c "import json;print(json.load(open('$PRM'))['solver_change'].get('offset_update_in_open') is False)")" = True ] || { say "**登錄 r2 須明確 offset_update_in_open＝False（W1 語義）** ⇒ 拒跑"; exit 97; }
GB_SHA=$(python3 -c "import json;print(json.load(open('$PRM'))['grasp']['basis_sha256'])")
[ "$(sha256sum "$GB" | cut -d' ' -f1)" = "$GB_SHA" ] || { say "**抓取基準 sha256 與登錄不符** ⇒ 拒跑"; exit 97; }
W1B_THETA=$(python3 -c "import json;print(json.load(open('$PRM'))['theta_goal_deg'])")
STAND=$(python3 -c "import json;print(json.load(open('$PRM'))['start']['stand_cli'])")
K2_READY=$(python3 -c "import json;print(json.load(open('$PRM'))['start']['k2_ready_cli'])")
Q_PRE=$(python3 -c "import json;print(json.load(open('$PRM'))['start']['q_pre_cli'])")
Q_INIT=$(python3 -c "import json;print(json.load(open('$PRM'))['bench']['q_nominal_cli'])")   # bench：名目初態（關節生成）
S_INIT=$(python3 -c "import json;print(json.load(open('$PRM'))['bench']['s_init_cli'])")   # bench：寫入致動器初始設定點（q_nominal − d_init）
W1B_TABLE=$(python3 -c "import json;print(json.load(open('$PRM'))['protection_scope_change']['corridor_table'])")
W1B_ENV=$(python3 -c "import json;print(json.load(open('$PRM'))['protection_scope_change']['envelope']['cli'])")
W1B_OBS=$(python3 -c "import json;print(json.load(open('$PRM'))['protection_scope_change']['observer_door_max_deg'])")
W1B_COORD=$(python3 -c "import json;print(json.load(open('$PRM'))['control_cost_change']['coord_topic'])")
W1E_FEFF=$(python3 -c "import json;print(json.load(open('$PRM'))['executor_change']['pause_finger_max_effort'])")
W1E_FEFF0=$(python3 -c "import json;print(json.load(open('$PRM'))['executor_change']['original_effort'])")
say "W1e 執行端變更：暫停期手指驅動輸出上限 ${W1E_FEFF0} → ${W1E_FEFF} N（ev5 r2；啟動讀回不符拒跑、切換／定期讀回異常走中止閂鎖）"
say "W1b 登錄：$PRM（sha256 $(sha256sum "$PRM" | cut -c1-16)）θ=${W1B_THETA}° 走廊表 $W1B_TABLE 包絡 $W1B_ENV 觀察器上限 ${W1B_OBS}° 協同 $W1B_COORD"
say "=== 往後拉開＋換抓＋拉關 pb1（ev9r8＋v10＋ADM20＋k3m3；ref pb1；W1 正面開門配置＋開門段偏移凍結＋一次性 gc.freeze（失敗拒跑）＋暫停上限 ${W1E_FEFF} N；台架初態 Q_INIT $Q_INIT、寫入設定點 $S_INIT，實際閉爪取證＋正對門往後拉開）RUN_ID=$RUN_ID domain=$ROS_DOMAIN_ID；停位 $STAND（實驗初始化）；起始＝指定抓取初態（不經 K2_READY／Q_PRE 與接近段）；夾持→開到 ${W1B_THETA}°→受控暫停→HELD→換抓 1–6→通過中拉關 7→放手退出 8 ==="

spawn sim "$ISAAC_PY" -u evaluation/isaac_door_k2_sim_ev9r8l.py --robot-self-collision-off --robot-pos-iters 64 --finger-leader-only --finger-loss-cont --ext-forces-every-iter --diag-6d --regrasp --regrasp-ref "$REF" --w2-mode-run-id "$RUN_ID" --contact-v3-out "$DIR/contact_v3.jsonl" --door-asset "$DOOR3" --gc-freeze-before-release --bench-start --bench-params "$PRM" --bench-setpoint="$S_INIT" --pause-finger-max-effort "$W1E_FEFF" --pause-finger-original-effort "$W1E_FEFF0" --cmd-env --evidence-out "$DIR/evidence.jsonl" --park-gate-mobile --run-id "$RUN_ID" --corridor-table "$W1B_TABLE" --envelope="$W1B_ENV" --sim-limit 400 --contact-min-n 0.5 \
  --finger-collision split --park-hold --start-pose "$STAND" --park-pose "$STAND" --stow-q="$Q_INIT" --out "$DIR"
say "  等場景建起（最多 180 s）"
for i in $(seq 180); do grep -q '進入主迴圈' "$DIR/sim.log" 2>/dev/null && break; kill -0 "${PIDS[0]}" 2>/dev/null || break; sleep 1; done
grep -q '進入主迴圈' "$DIR/sim.log" 2>/dev/null || { say "**場景未建起** ⇒ 中止"; tail -8 "$DIR/sim.log" | tee -a "$DIR/run.log"; exit 83; }
grep -E '^\[(room|door)\] (零摩擦|夾持面|摩擦核對|門建於|場景求解|主從讀回|機器人求解迭代|初始門角|門扇接觸|牆接觸)' "$DIR/sim.log" | tee -a "$DIR/run.log"

spawn task python3 -u evaluation/door_coopen_core_task_node10c.py --out "$DIR/task.json" --run-id "$RUN_ID" --grasp-basis "$GB" --w1b-params "$PRM" --corridor-table "$W1B_TABLE" --finger-loss-cont --regrasp --regrasp-ref "$REF" --w2-mode-run-id "$RUN_ID" --wall-total-s 1500
ARM_IDENT="$WS/evaluation/results/wgmpc_arm_sp_ident_free4.json"
[ -f "$ARM_IDENT" ] || { say "**找不到手臂辨識檔** ⇒ 中止"; exit 85; }
spawn adapter python3 -u evaluation/arm_vel_adapter_k3m3.py --in-topic /wgmpc/cmd_world --consumer-node /door_k2_sim --cmd-env --cmd-env-v3 --env-in-topic /wgmpc/cmd_env --bench-ready-run-id "$RUN_ID"
spawn solver python3 -u evaluation/dl3_wgmpc_wg2_adm20_node.py --w2-retreat --w2-hold-r9 --w2-held --w2-held-max-s 60 --w2-mode-run-id "$RUN_ID" --regrasp --bench-start --bench-params "$PRM" --require-topic-target --park-hold --offset-observer interval --mode-switch-run-id "$RUN_ID" --cmd-env --run-id "$RUN_ID" --pause --pause-arm-zero --pause-active-hold --pause-arm-win-s 0.5 --coord-topic "$W1B_COORD" --offset-hold-on-free \
  --offset-moving --offset-free \
  --continuous --arm-model setpoint --arm-ident "$ARM_IDENT" \
  --cmd-topic /wgmpc/cmd_world --target-topic /drawer/tcp_target --target-offset 0 0 0 \
  --N 5 --rate 20 --delay-comp-state-cycles 1.6 --delay-comp-cmd-cycles 1.0 \
  --near-target-gamma 1.0 --w-s 0.001 --w-a 0.05 --w-s-arm 0.05 --margin-guard \
  --reach-pos-m 0.005 --reach-rot-rad 0.02 --hold-s 2.0 --duration-s 1600 --duration-sim-s 450 \
  --u-prev-policy strict --assume-initial-rest --out "$DIR/align_solver.json"
spawn lw_observer python3 -u evaluation/door_k3m_observer.py --out "$DIR/lw_observer.json" --door-max-deg "$W1B_OBS"

say "[任務] 等任務節點結束（牆鐘最多 1600 s）"
SIM_PID="${PIDS[0]}"; TASK_PID="${PIDS[1]}"; SOLVER_PID="${PIDS[3]}"; OBS_PID="${PIDS[4]}"   # bench：無展開節點
sleep 5; kill -0 "$OBS_PID" 2>/dev/null || { say "**獨立觀察器未存活** ⇒ 中止（沒有觀察期保護不得繼續）"; exit 86; }
wait_or_stop "$TASK_PID" 1600 10; RC_TASK=$STOP_RC; HOW_TASK=$STOP_HOW
grep -E '^\[k3a\]' "$DIR/task.log" | tee -a "$DIR/run.log"
say "[收尾] 等求解節點停止並封存（最多 30 s，之後 TERM→10 s→KILL）"
wait_or_stop "$SOLVER_PID" 30 10; RC_SOLVER=$STOP_RC; HOW_SOLVER=$STOP_HOW
RC_WB=0; HOW_WB=not_started                                   # bench：無展開節點
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
[ "$HOW_TASK" = exited ] && [ "$HOW_SOLVER" = exited ] && [ "$HOW_SIM" = exited ] && [ "$RC_TASK" = 0 ] && [ "$RC_SOLVER" = 0 ] && [ "$RC_WB" = 0 ] && [ "$RC_SIM" = 0 ] && [ "$OK_ROOM" = 1 ] && [ "$OK_SOLVER" = 1 ] && [ "$OK_DONE" = 1 ] && RC=0
say "=== 結束 rc=$RC（0 只表示所有節點正常結束且到 DONE；指定初態功能展示，不是正式評估；拉關到約 2° 只能稱近關閉）==="
exit "$RC"
