#!/usr/bin/env bash
# 換抓＋通過中拉關＋放手退出 Isaac 功能展示 bench1（Codex r68 條件式核可一次；bench9 跑法的版本化複本，bench9 不改）。
# 同趟：台架初態 → 開門 48° → 受控暫停／HELD → mode 1–6 換抓到內側把手 → mode 7 通過中拉關（參考至約 2°）→ mode 8 放手退出收臂。
# 相對 bench9 只換：執行端 isaac_door_k2_sim_ev9r8.py（--regrasp、--regrasp-ref ref v5、--w2-mode-run-id、--contact-v3-out、門規格 door_local_v3）、
#   任務節點 door_coopen_core_task_node10.py（--regrasp：BG(IG2(AG2(RegraspPolicy8)))）、求解端 dl3_wgmpc_wg2_adm20_node.py（＋W2：--w2-retreat --w2-hold-r9
#   --w2-held --w2-held-max-s 60 --regrasp）、轉接器 arm_vel_adapter_k3m3.py（--cmd-env-v3）；模擬上限 400 s、牆鐘上限放寬；
#   不建新凍結（Codex r68：先不要新凍結包），改為起跑時把所用檔案 sha256 寫入本趟 versions.sha256。其餘（登錄 r11、92 °C 守護、環境核對）同 bench9。
# 用法：RUN_ID=door_regrasp_bench1_1 ROS_DOMAIN_ID=97 RUNNER=evaluation/run_door_regrasp_bench1.sh bash evaluation/run_guarded_v2.sh
OFFSET_MOVING=1
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WS"
: "${ROS_DOMAIN_ID:?ROS_DOMAIN_ID 未設定}"
: "${RUN_ID:?RUN_ID 未設定}"
ISAAC_PY="${ISAAC_PY:-$HOME/venvs/isaacsim-6.0.1/bin/python}"
DIR="$WS/evaluation/runs/$RUN_ID"; mkdir -p "$DIR"
PRM=evaluation/results/door/coopen_core_params_48_r11.json   # bench9：固定登錄 r11（r10＋暫停期手指上限 5 N；不由環境變數改）
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

REF=evaluation/results/door/door_regrasp_ref_v5.json
DOOR3=src/my_omnibot_description/config/door_local_v3.yaml
sha256sum evaluation/isaac_door_k2_sim_ev9r8.py evaluation/door_coopen_core_task_node10.py evaluation/dl3_wgmpc_wg2_adm20_node.py evaluation/arm_vel_adapter_k3m3.py \
  evaluation/door_regrasp_*.py evaluation/door_k3m_observer.py "$PRM" "$GB" "$REF" "$DOOR3" > "$DIR/versions.sha256" || { say "**版本紀錄失敗** ⇒ 拒跑"; exit 95; }
say "版本紀錄 $DIR/versions.sha256（$(wc -l < "$DIR/versions.sha256") 檔；git $(git rev-parse --short HEAD)）"
say "python3＝$(command -v python3)；VIRTUAL_ENV=${VIRTUAL_ENV:-（無）}"
python3 -c "import osqp" 2>/dev/null || { say "**python3 環境錯誤（無法匯入 osqp；可能誤啟用虛擬環境）** ⇒ 拒跑"; exit 96; }
[ -s "$PRM" ] || { say "**核心段登錄檔不存在：$PRM** ⇒ 拒跑"; exit 97; }
[ -s "$GB" ] || { say "**抓取基準不存在：$GB** ⇒ 拒跑"; exit 97; }
[ "$(python3 -c "import json;print(json.load(open('$PRM'))['solver_change'].get('offset_update_in_open') is True)")" = True ] || { say "**登錄 r10 未設 offset_update_in_open** ⇒ 拒跑"; exit 97; }
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
say "=== 換抓＋拉關 bench1（ev9r8＋v10＋ADM20＋k3m3；ref v5）＋核心段 bench9 配置（聯合配置：偏移更新＋停頓診斷＋一次性 gc.freeze（失敗拒跑）＋暫停上限 5 N；台架初態 Q_INIT $Q_INIT、寫入設定點 $S_INIT，實際閉爪取證＋協同開門＋出口繞行觀察）RUN_ID=$RUN_ID domain=$ROS_DOMAIN_ID；停位 $STAND（實驗初始化）；起始＝指定抓取初態（不經 K2_READY／Q_PRE 與接近段）；夾持→開到 ${W1B_THETA}°→受控暫停→HELD→換抓 1–6→通過中拉關 7→放手退出 8 ==="

spawn sim "$ISAAC_PY" -u evaluation/isaac_door_k2_sim_ev9r8.py --regrasp --regrasp-ref "$REF" --w2-mode-run-id "$RUN_ID" --contact-v3-out "$DIR/contact_v3.jsonl" --door-asset "$DOOR3" --gc-freeze-before-release --bench-start --bench-params "$PRM" --bench-setpoint "$S_INIT" --pause-finger-max-effort "$W1E_FEFF" --pause-finger-original-effort "$W1E_FEFF0" --cmd-env --evidence-out "$DIR/evidence.jsonl" --park-gate-mobile --run-id "$RUN_ID" --corridor-table "$W1B_TABLE" --envelope="$W1B_ENV" --sim-limit 400 --contact-min-n 0.5 \
  --finger-collision split --park-hold --start-pose "$STAND" --park-pose "$STAND" --stow-q "$Q_INIT" --out "$DIR"
say "  等場景建起（最多 180 s）"
for i in $(seq 180); do grep -q '進入主迴圈' "$DIR/sim.log" 2>/dev/null && break; kill -0 "${PIDS[0]}" 2>/dev/null || break; sleep 1; done
grep -q '進入主迴圈' "$DIR/sim.log" 2>/dev/null || { say "**場景未建起** ⇒ 中止"; tail -8 "$DIR/sim.log" | tee -a "$DIR/run.log"; exit 83; }
grep -E '^\[(room|door)\] (零摩擦|夾持面|摩擦核對|門建於|初始門角|門扇接觸|牆接觸)' "$DIR/sim.log" | tee -a "$DIR/run.log"

spawn task python3 -u evaluation/door_coopen_core_task_node10.py --out "$DIR/task.json" --run-id "$RUN_ID" --grasp-basis "$GB" --w1b-params "$PRM" --corridor-table "$W1B_TABLE" --regrasp --regrasp-ref "$REF" --w2-mode-run-id "$RUN_ID" --wall-total-s 1500
ARM_IDENT="$WS/evaluation/results/wgmpc_arm_sp_ident_free4.json"
[ -f "$ARM_IDENT" ] || { say "**找不到手臂辨識檔** ⇒ 中止"; exit 85; }
spawn adapter python3 -u evaluation/arm_vel_adapter_k3m3.py --in-topic /wgmpc/cmd_world --consumer-node /door_k2_sim --cmd-env --cmd-env-v3 --env-in-topic /wgmpc/cmd_env --bench-ready-run-id "$RUN_ID"
spawn solver python3 -u evaluation/dl3_wgmpc_wg2_adm20_node.py --w2-retreat --w2-hold-r9 --w2-held --w2-held-max-s 60 --w2-mode-run-id "$RUN_ID" --regrasp --bench-start --bench-params "$PRM" --require-topic-target --park-hold --offset-observer interval --mode-switch-run-id "$RUN_ID" --cmd-env --run-id "$RUN_ID" --pause --pause-arm-zero --pause-active-hold --pause-arm-win-s 0.5 --coord-topic "$W1B_COORD" --offset-hold-on-free --offset-update-in-open \
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
