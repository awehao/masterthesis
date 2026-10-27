#!/usr/bin/env bash
# 連續協同通行：核心段台架初態啟動跑法 bench10 版（run_door_coopen_core_bench9.sh 的版本化複本；Codex r35／r36：C2 持握閉合修正引導）：
# 相對 bench9 只換：凍結 freeze_door_coopen_core_bench_dev_r10、登錄 r12（r11＋control_cost_change.closure_guidance）、任務節點 door_coopen_core_task_node9.py
# （closure_cycle 只平移引導參考位姿；前饋與限幅不變）、判定器字樣 door_coopen_check_core13.py。執行端 ev9r5、求解端 ADM17r4、轉接器、觀察器與所有門檻全同 bench9。
# 跑後以 run_door_replay2.sh（coopen_wide／coopen_close）製作實錄姿態重播影片（只渲染，不跑控制）。bench9 保留。
# bench9 版說明：# 連續協同通行：核心段台架初態啟動跑法 bench9 版（run_door_coopen_core_bench8.sh 的版本化複本；Codex r31：聯合配置驗證 door_coopen_core_bench_6）：
# 相對 bench8 只換：凍結 freeze_door_coopen_core_bench_dev_r9、登錄 r11（暫停期手指上限 20 → 5 N＝維持原值）、執行端 isaac_door_k2_sim_ev9r5.py
# （gc.freeze 啟用失敗 ⇒ 物理釋放前拒跑並封存）。相對 bench_5 同時有 GC 處置與暫停上限兩項變更 ⇒ 聯合配置驗證，不作嚴格單變因歸因。
# 求解端 ADM17r4、任務節點、轉接器、觀察器、判定器 r12 與所有門檻（0.2 s、0.001 rad/s、5 mm…）全同 bench8。bench8 保留。
# bench8 版說明：# 連續協同通行：核心段台架初態啟動跑法 bench8 版（run_door_coopen_core_bench7.sh 的版本化複本；Codex r30 §2：一次性 gc.freeze 候選）：
# 相對 bench7 只換：凍結 freeze_door_coopen_core_bench_dev_r8、執行端 isaac_door_k2_sim_ev9r4.py --gc-freeze-before-release（台架凍結前置、釋放計時之前一次性
# gc.freeze；不關自動 GC、不改門檻、不 collect）。停頓診斷保留；登錄 r10、求解端 ADM17r4、任務節點、轉接器、觀察器、判定器 r12 與所有門檻
# （含 0.2 s 監看時效、暫停 0.001 rad/s）全同 bench7。bench7 保留。
# bench7 版說明：# 連續協同通行：核心段台架初態啟動跑法 bench7 版（run_door_coopen_core_bench6.sh 的版本化複本；Codex r29：沿用 bench_4 控制配置，只加執行端停頓診斷）：
# 相對 bench6 只換：凍結 freeze_door_coopen_core_bench_dev_r7、執行端 isaac_door_k2_sim_ev9r3.py（ev9r2＋exec_timing_diag 只做插入：逐步區段牆鐘、GC 各代起訖；
# 結束時 <out>/exec_timing.npz 與 room_run exec_timing）。登錄 r10、求解端 ADM17r4 --offset-update-in-open、任務節點、轉接器、觀察器、判定器 r12、
# 所有門檻（含 0.2 s 監看時效）全同 bench6。bench6 保留。
# bench6 版說明：# 連續協同通行：核心段台架初態啟動跑法 bench6 版（run_door_coopen_core_bench5.sh 的版本化複本；Codex r27 唯一候選：開門段恢復偏移估計更新；規格 COOPEN_core_spec_draft14.md）：
# 相對 bench5 只換：凍結 freeze_door_coopen_core_bench_dev_r6、登錄 r10（r9＋solver_change.offset_update_in_open）、
# 求解端 dl3_wgmpc_wg2_adm17r4_node.py --offset-update-in-open（保持改由暫停觸發；OPEN 期間照既有 interval 規則更新）、判定器字樣 door_coopen_check_core12.py。
# 執行端、任務節點、轉接器、觀察器、權重、限制、門檻、接觸／漂移／幾何／停止判準全同 bench5。bench5 保留。
# bench5 版說明：# 連續協同通行：核心段台架初態啟動跑法 bench5 版（run_door_coopen_core_bench4.sh 的版本化複本；door_coopen_core_bench_2 啟動缺口修正；Codex r24；規格 COOPEN_core_spec_draft13.md）：
# 相對 bench4 只換：凍結 freeze_door_coopen_core_bench_dev_r5、轉接器 arm_vel_adapter_k3m2.py --bench-ready-run-id（台架就緒回報）、
# 求解端 dl3_wgmpc_wg2_adm17r3_node.py（正式首解前等真正轉接器就緒，上限 30 s）。bench4 保留。
# bench4 版說明：# 連續協同通行：核心段台架初態啟動跑法 bench4 版（run_door_coopen_core_bench3.sh 的版本化複本；door_coopen_core_bench_1 啟動缺口修正；規格 COOPEN_core_spec_draft12.md）：
# 相對 bench3 只換：凍結 freeze_door_coopen_core_bench_dev_r4、求解端 dl3_wgmpc_wg2_adm17r2_node.py（預熱前先等台架錨點，上限 10 s）。bench3 保留。
# bench3 版說明：# 連續協同通行：核心段台架初態啟動跑法 bench3 版（run_door_coopen_core_bench2.sh 的版本化複本；Codex r22：判定器 r11 原始證據核對＋完整正例；規格 COOPEN_core_spec_draft11.md）：
# 相對 bench2 只換：凍結 freeze_door_coopen_core_bench_dev_r3、判定器字樣 door_coopen_check_core11.py；節點、執行端、求解端全同 bench2。bench2 保留不跑。
# bench2 版說明：# 連續協同通行：核心段台架初態啟動跑法 bench2 版（run_door_coopen_core_bench.sh 的版本化複本；Codex r21 三項必修；規格 evaluation/results/specs/COOPEN_core_spec_draft10.md）：
# 相對 bench 只換：凍結 freeze_door_coopen_core_bench_dev_r2、執行端 isaac_door_k2_sim_ev9r2.py（首筆經真入口進鏈、第 1 步核對）、任務節點 door_coopen_core_task_node8.py
# （確認凍結才跳過監看、StateFilter）、判定器 door_coopen_check_core10.py（錨點交叉核對後的鏈閘門視圖、不豁免）；bench 保留不跑。
# bench 版說明：# 連續協同通行：核心段**台架初態啟動**跑法 bench 版（run_door_coopen_core_iso2.sh 的版本化複本；Codex r19／r20；規格 evaluation/results/specs/COOPEN_core_spec_draft9.md）：
# 相對 iso2 只換：登錄 r9（r8＋bench）、凍結 freeze_door_coopen_core_bench_dev_r1、執行端 isaac_door_k2_sim_ev9.py --bench-start（關節生成 q_nominal、寫入 s_init、
# 凍結至求解端宣告首解）、**不啟動展開節點**（控制權自始為 wholebody，不經交棒與 UNFOLD）、求解端 ADM17 --bench-start（偏移估計初值 d_init、
# 無 --wait-for-drawer-handover）、任務節點 door_coopen_core_task_node7.py、判定器 door_coopen_check_core9.py；程序索引因少一個節點而重排。iso2 保留。
# iso2 版說明：# 連續協同通行：核心段指定初態隔離驗證跑法 iso2 版（run_door_coopen_core_iso.sh 的版本化複本；Codex r16 必修兩項；規格 evaluation/results/specs/COOPEN_core_spec_draft8.md）：
# 相對 iso 只換：凍結 freeze_door_coopen_core_iso_dev_r2、任務節點 door_coopen_core_task_node6.py（INIT_CHECK 連續確認 r2＋trace init）、判定器 door_coopen_check_core7.py（J6 r2）；iso（r1）保留不跑。
# iso 版說明：連續協同通行：核心段**指定初態隔離驗證**跑法 iso 版（run_door_coopen_core_g2.sh 的版本化複本；Howard 2026-10-27 確認、Codex r15 補充裁定；規格 evaluation/results/specs/COOPEN_core_spec_draft7.md）：
# 相對 g2 只換：登錄 r8（r6＋grasp_init；**不含 r7 接近段成本**）、凍結 freeze_door_coopen_core_iso_dev_r1、執行端與展開節點 --stow-q＝qa_grasp（手指張開生成於門關閉抓取構型）、
# 展開目標 --q-unfold＝qa_grasp（零位移）、任務節點 door_coopen_core_task_node5.py（UNFOLD → INIT_CHECK → 實際閉爪 ENGAGE；不經 ALIGN／APPROACH）、判定器 door_coopen_check_core6.py（J1–J4＋J6）。
# 成果口徑：「指定初態、經實際夾持取證後的持握開門與出口繞行核心驗證」；跳過自主接近，不代表接近已解決、不稱完整自主開門通行。g2 保留。
# g2 版說明：連續協同通行：核心段接觸驗證跑法 g2 版（run_door_coopen_core_g.sh 的版本化複本；Codex 20261027_004139 閘控 r2；規格 evaluation/results/specs/COOPEN_core_spec_draft5.md；
# 相對 g 版只換：任務節點 door_coopen_core_task_node3.py（閘控 r2）、凍結 freeze_door_coopen_core_g_dev_r2、判定器 door_coopen_check_core4.py；g 版保留。
# g 版說明：連續協同通行：核心段接觸驗證跑法 g 版（run_door_coopen_core.sh 的版本化複本；Codex 20261027_003024 接近進度閘控；規格 evaluation/results/specs/COOPEN_core_spec_draft4.md；
# 相對原版只換：登錄 r6（＋approach_gate）、任務節點 door_coopen_core_task_node2.py、凍結 freeze_door_coopen_core_g_dev_r1、判定器 door_coopen_check_core3.py；原版與 door_coopen_core_1 結果保留。
# 原版說明：
#   Codex 20261026_235425 核可準備；Codex 20261027_002013 條件式核可單趟 door_coopen_core_1，只跑一次）。
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
FRZ=evaluation/results/door/freeze_door_coopen_core_bench_dev_r10.sha256   # bench10：bench r9 清單＋C2 模組／任務節點 v9／判定器 r13／登錄 r12／測試（bench r9 保留）
PRM=evaluation/results/door/coopen_core_params_48_r12.json   # bench10：固定登錄 r12（r11＋closure_guidance；不由環境變數改）
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
[ "$(python3 -c "import json;print(json.load(open('$PRM'))['solver_change'].get('offset_update_in_open') is True)")" = True ] || { say "**登錄 r10 未設 offset_update_in_open** ⇒ 拒跑"; exit 97; }
[ "$(python3 -c "import json;print(json.load(open('$PRM'))['control_cost_change'].get('closure_guidance',{}).get('enabled') is True)")" = True ] || { say "**登錄 r12 未設 closure_guidance** ⇒ 拒跑"; exit 97; }
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
say "=== 核心段 bench10（bench9 聯合配置＋C2 持握閉合修正引導；台架初態 Q_INIT $Q_INIT、寫入設定點 $S_INIT，實際閉爪取證＋協同開門＋出口繞行觀察）RUN_ID=$RUN_ID domain=$ROS_DOMAIN_ID；停位 $STAND（實驗初始化）；起始＝指定抓取初態（不經 K2_READY／Q_PRE 與接近段）；夾持→開到 ${W1B_THETA}°→受控暫停→停止（不張爪、不收臂、不穿越）==="

spawn sim "$ISAAC_PY" -u evaluation/isaac_door_k2_sim_ev9r5.py --gc-freeze-before-release --bench-start --bench-params "$PRM" --bench-setpoint "$S_INIT" --pause-finger-max-effort "$W1E_FEFF" --pause-finger-original-effort "$W1E_FEFF0" --cmd-env --evidence-out "$DIR/evidence.jsonl" --park-gate-mobile --run-id "$RUN_ID" --corridor-table "$W1B_TABLE" --envelope="$W1B_ENV" --sim-limit 240 --contact-min-n 0.5 \
  --finger-collision split --park-hold --start-pose "$STAND" --park-pose "$STAND" --stow-q "$Q_INIT" --out "$DIR"
say "  等場景建起（最多 180 s）"
for i in $(seq 180); do grep -q '進入主迴圈' "$DIR/sim.log" 2>/dev/null && break; kill -0 "${PIDS[0]}" 2>/dev/null || break; sleep 1; done
grep -q '進入主迴圈' "$DIR/sim.log" 2>/dev/null || { say "**場景未建起** ⇒ 中止"; tail -8 "$DIR/sim.log" | tee -a "$DIR/run.log"; exit 83; }
grep -E '^\[(room|door)\] (零摩擦|夾持面|摩擦核對|門建於|初始門角|門扇接觸|牆接觸)' "$DIR/sim.log" | tee -a "$DIR/run.log"

spawn task python3 -u evaluation/door_coopen_core_task_node9.py --out "$DIR/task.json" --run-id "$RUN_ID" --grasp-basis "$GB" --w1b-params "$PRM" --corridor-table "$W1B_TABLE"
ARM_IDENT="$WS/evaluation/results/wgmpc_arm_sp_ident_free4.json"
[ -f "$ARM_IDENT" ] || { say "**找不到手臂辨識檔** ⇒ 中止"; exit 85; }
spawn adapter python3 -u evaluation/arm_vel_adapter_k3m2.py --in-topic /wgmpc/cmd_world --consumer-node /door_k2_sim --cmd-env --env-in-topic /wgmpc/cmd_env --bench-ready-run-id "$RUN_ID"
spawn solver python3 -u evaluation/dl3_wgmpc_wg2_adm17r4_node.py --bench-start --bench-params "$PRM" --require-topic-target --park-hold --offset-observer interval --mode-switch-run-id "$RUN_ID" --cmd-env --run-id "$RUN_ID" --pause --pause-arm-zero --pause-active-hold --pause-arm-win-s 0.5 --coord-topic "$W1B_COORD" --offset-hold-on-free --offset-update-in-open \
  --offset-moving --offset-free \
  --continuous --arm-model setpoint --arm-ident "$ARM_IDENT" \
  --cmd-topic /wgmpc/cmd_world --target-topic /drawer/tcp_target --target-offset 0 0 0 \
  --N 5 --rate 20 --delay-comp-state-cycles 1.6 --delay-comp-cmd-cycles 1.0 \
  --near-target-gamma 1.0 --w-s 0.001 --w-a 0.05 --w-s-arm 0.05 --margin-guard \
  --reach-pos-m 0.005 --reach-rot-rad 0.02 --hold-s 2.0 --duration-s 600 --duration-sim-s 400 \
  --u-prev-policy strict --assume-initial-rest --out "$DIR/align_solver.json"
spawn lw_observer python3 -u evaluation/door_k3m_observer.py --out "$DIR/lw_observer.json" --door-max-deg "$W1B_OBS"

say "[任務] 等 K3-A 任務節點結束（牆鐘最多 660 s）"
SIM_PID="${PIDS[0]}"; TASK_PID="${PIDS[1]}"; SOLVER_PID="${PIDS[3]}"; OBS_PID="${PIDS[4]}"   # bench：無展開節點
sleep 5; kill -0 "$OBS_PID" 2>/dev/null || { say "**獨立觀察器未存活** ⇒ 中止（沒有觀察期保護不得繼續）"; exit 86; }
wait_or_stop "$TASK_PID" 660 10; RC_TASK=$STOP_RC; HOW_TASK=$STOP_HOW
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
say "=== 結束 rc=$RC（0 只表示所有節點正常結束且到 DONE；舊鏈通過與否由 door_coopen_check_core13.py 的 chain_pass 判定（r12 全部判準＋C2 接線 Q1–Q8 獨立核對）；chain_pass 不等於邊開邊繞成功；核心動作（邊開邊繞）證據另報 core，不自動通過）==="
exit "$RC"
