#!/usr/bin/env bash
# K3-A：局部通行門 v2 的夾持 → 小角度開門 8° → SETTLE → 保留抓持受控停止（不張爪、不退離）。
# 規格 evaluation/results/specs/K3_door_small_open_spec.md（draft-1～3；範圍依 Codex 20261021 draft-3 審查縮為 K3-A）。
# run_door_k2.sh 的版本化複本：只換任務節點（door_k3a_task_node.py）、凍結清單與紀錄字樣；執行端／展開／轉接／求解器與參數同 K2。
# **INT 版**（run_door_k3a.sh 的複本）：唯一差異＝求解節點改用 dl3_wgmpc_wg2_adm2_node.py 並加 --offset-observer interval
# （有限區間等效偏移觀測；Codex 20261022 ADM2 接入審查通過）。其餘參數、K3-A 範圍、判定器全同；不帶 --offset-admission。
# **EV 版**（run_door_k3a_int.sh 的複本）：唯一差異＝執行端改用 isaac_door_k2_sim_ev.py 並加 --evidence-out（接觸承載取證，只加紀錄、不改控制）。
# **K3-M 版**（run_door_k3a_int_ev.sh 的複本）：底盤可動的小角度協同開門（規格 K3_mobile_coop_small_open_spec.md）。差異只有：執行端改 isaac_door_k2_sim_ev3.py
#   ＋--park-gate-mobile --run-id --corridor-table（模式權威、逐步走廊監看）；求解器改 dl3_wgmpc_wg2_adm3_node.py＋--mode-switch-run-id（握手後切入 free）；
#   任務節點改 door_k3m_task_node.py＋--run-id。其餘參數全同（含 --park-hold：HOLD 前照 PARK_HOLD）。判定器 door_k3m_check.py。
# **K3-M 受控暫停版**（run_door_k3m_ev.sh 的複本；規格 K3M_pause_integration_spec_draft1.md）：差異只有
#   執行端 isaac_door_k2_sim_ev4.py＋--cmd-env（封裝入口、/door/applied_src、暫停握手權威）；adapter arm_vel_adapter_k3m.py＋--cmd-env --env-in-topic /wgmpc/cmd_env；
#   求解器 dl3_wgmpc_wg2_adm4_node.py＋--cmd-env --run-id --pause；任務節點 door_k3m_task_node_p.py（完整 SETTLE ⇒ PAUSE）；
#   獨立觀察器 door_k3m_observer.py --door-max-deg 12（中止後實驗保護；心跳由任務節點監看）；凍結 freeze_door_k3m_pause_dev_r1；判定器 door_k3m_check_p.py。
#   增益、權重、限制、門檻、走廊與幾何監看不改。單趟；未核可前不得執行。結束模擬不算實際停止。
# 由 run_guarded_v2.sh 包 92 °C 熱中止：
#   RUN_ID=door_k3m_pause_1 ROS_DOMAIN_ID=97 RUNNER=evaluation/run_door_k3m_pause.sh bash evaluation/run_guarded_v2.sh
# 節點：門場景執行端（PARK_HOLD）→ 展開節點（K2_READY → Q_PRE_DOOR）→ K2 任務節點 → 轉接器 → W-GMPC 求解器。
# **不起導航**（直接生成在停位）。求解器參數＝runs/dl3v_M/align_solver.json args，**只差**：--park-hold（K2 底盤停車伺服）、
# 不接 --coord-topic（無 MotM）。收尾只停本趟 PID。
# **W1f2 版**（run_door_wide_w1f.sh 的複本；Codex 20261025_034022：任務端最小紀錄修正）：唯一差異＝任務節點改 door_k3m_task_node_w1b2.py
#   （有界非阻塞逐列紀錄，取代迴圈內每 100 筆同步重寫 task.json）、判定器 door_k3m_check_w1f2.py（w1e2 全部＋紀錄完整性；trace_gap 0.25 s 不變）、
#   凍結 freeze_door_wide_w1f2_dev_r4（Codex 20261025_144446 條件式核可單趟 60°；r3／r4＝r2 後追加文件致凍結不符而重凍，控制與紀錄程式不變）。登錄、控制、模型、門檻全部同 W1f r2。
#   env -u VIRTUAL_ENV PATH=<去掉 .venv-dl0> RUN_ID=door_wide_w1f_60_2 ROS_DOMAIN_ID=97 RUNNER=evaluation/run_door_wide_w1f2.sh bash evaluation/run_guarded_v2.sh
# **W1f 版**（run_door_wide_w1e_r3.sh 的複本；Codex 20261025_030420，規格 W1f_wide_open_60_spec.md）：W1e 控制配置不變，唯一差異＝θ 60°
#   （登錄 w1f_params_60.json＝door_w1b_params.py 60 推得＋W1e 的 solver_change／executor_change）；判定器 w1e2（登錄驅動）；凍結 freeze_door_wide_w1f_dev_r2（Codex 20261025_031137 條件式核可單趟）。60° 不作通行成果。
#  env -u VIRTUAL_ENV PATH=<去掉 .venv-dl0> RUN_ID=door_wide_w1f_60_1 ROS_DOMAIN_ID=97 RUNNER=evaluation/run_door_wide_w1f.sh bash evaluation/run_guarded_v2.sh
# **W1e r3 版**（run_door_wide_w1e.sh 的複本；Codex 20261025_025410 條件式核可重複一趟 door_wide_w1e_30_2）：唯一差異＝凍結 freeze_door_wide_w1e_dev_r3
#   （w1e r2 全部＋判定器 w1e2 與其測試）、結尾判定器名稱 w1e2。控制、20 N 切換時機、模型、判準全部不變。
#   env -u VIRTUAL_ENV PATH=<去掉 .venv-dl0> RUN_ID=door_wide_w1e_30_2 ROS_DOMAIN_ID=97 RUNNER=evaluation/run_door_wide_w1e_r3.sh bash evaluation/run_guarded_v2.sh
# **W1e 版**（run_door_wide_w1c.sh 的複本；Codex 20261025_023608，規格 W1e_pause_finger_effort_spec.md）：W1c r2 全部設定不變（不疊加 W1d），
#   唯一差異＝執行端改 isaac_door_k2_sim_ev5.py＋--pause-finger-max-effort（取自登錄 executor_change，20 N；暫停切換那一步起兩指驅動輸出上限 5→20 N）；
#   登錄 w1e_params_30.json；判定器 door_k3m_check_w1e.py；凍結 freeze_door_wide_w1e_dev_r2（Codex 20261025_024339 條件式核可單趟）。單一變因候選，非已確認解法；報告標為執行器上限變更。
#   env -u VIRTUAL_ENV PATH=<去掉 .venv-dl0> RUN_ID=door_wide_w1e_30_1 ROS_DOMAIN_ID=97 RUNNER=evaluation/run_door_wide_w1e.sh bash evaluation/run_guarded_v2.sh
# **W1c 版**（run_door_wide_w1b.sh 的複本；Codex 20261025_002129，規格 W1c_offset_hold_spec.md）：w1b r2 全部設定不變，
#   唯一差異＝求解器改 dl3_wgmpc_wg2_adm9_node.py＋--offset-hold-on-free（首次切 free 起偏移估計保持）；登錄 w1c_params_30.json；
#   判定器 door_k3m_check_w1c.py；凍結 freeze_door_wide_w1c_dev_r1。單一變因候選，非已確認解法。
#   env -u VIRTUAL_ENV PATH=<去掉 .venv-dl0> RUN_ID=door_wide_w1c_30_1 ROS_DOMAIN_ID=97 RUNNER=evaluation/run_door_wide_w1c.sh bash evaluation/run_guarded_v2.sh
# **W1b 版**（run_door_k3m_pause_p4b.sh 的複本；Codex 20261024_231908 B 節，規格 W1b_wide_open_30_spec.md）：大角度協同開門 θ（預設 30°）＋p4b 受控暫停。
#   唯一參數來源＝results/door/w1b_params_<θ>.json（door_w1b_params.py 推得）：任務節點 door_k3m_task_node_w1b.py --w1b-params；
#   延伸走廊表 door_wide_corridor_table_90.json（執行端與任務端同一表）；執行端 --envelope＝登錄包絡；觀察器 --door-max-deg＝θ＋4°；
#   求解器 ADM8 另加 --coord-topic（底盤路徑引導，軟成本）；判定器 door_k3m_check_w1b.py（r2）；凍結 freeze_door_wide_w1b_dev_r2。
#   env -u VIRTUAL_ENV PATH=<去掉 .venv-dl0> RUN_ID=door_wide_w1b_30_1 ROS_DOMAIN_ID=97 RUNNER=evaluation/run_door_wide_w1b.sh bash evaluation/run_guarded_v2.sh
# **p4b 版**（run_door_k3m_pause_p4.sh 的複本；Howard 2026-10-24 選定手臂 0.5 s 窗口，規格 K3M_pause_p4b_arm_window.md）：
#   唯一差異＝求解器改 dl3_wgmpc_wg2_adm8_node.py＋--pause-arm-win-s 0.5；判定器 door_k3m_check_p4b.py；凍結 freeze_door_k3m_pause_p4b_dev_r1。
#   env -u VIRTUAL_ENV PATH=<去掉 .venv-dl0> RUN_ID=door_k3m_pause_p4b_1 ROS_DOMAIN_ID=97 RUNNER=evaluation/run_door_k3m_pause_p4b.sh bash evaluation/run_guarded_v2.sh
# **p4 版**（run_door_k3m_pause_p3b.sh 的複本；Howard 2026-10-24 選定「主動保持停止」，規格 K3M_pause_p4_active_hold_stop.md）：
#   唯一差異＝求解器改 dl3_wgmpc_wg2_adm7_node.py＋--pause-arm-zero --pause-active-hold（停止＝手臂 0＋底盤主動保持，STOPPED 後持續保持發布到觀察滿 10 s）；
#   凍結 freeze_door_k3m_pause_p4_dev_r1；判定器 door_k3m_check_p4.py。門檻、觀察期、保護不變。
#   env -u VIRTUAL_ENV PATH=<去掉 .venv-dl0> RUN_ID=door_k3m_pause_p4_1 ROS_DOMAIN_ID=97 RUNNER=evaluation/run_door_k3m_pause_p4.sh bash evaluation/run_guarded_v2.sh
# **p3b 版**（run_door_k3m_pause_p3.sh 的複本；Claude 依 Howard 授權裁定，規格 K3M_pause_p3b_arm_zero_base_hold.md）：
#   唯一差異＝求解器改 dl3_wgmpc_wg2_adm6_node.py＋--pause-arm-zero（手臂歸零、底盤維持 ADM4 主動保持）；凍結 freeze_door_k3m_pause_p3b_dev_r1。
#   env -u VIRTUAL_ENV PATH=<去掉 .venv-dl0> RUN_ID=door_k3m_pause_p3b_1 ROS_DOMAIN_ID=97 RUNNER=evaluation/run_door_k3m_pause_p3b.sh bash evaluation/run_guarded_v2.sh
# **p3 版**（run_door_k3m_pause_p2c.sh 的複本；Codex 無法使用，經 Howard 授權由 Claude 裁定，規格 K3M_pause_p3_all_zero_hold.md）：
#   唯一差異＝求解器改 dl3_wgmpc_wg2_adm5_node.py＋--pause-all-zero（暫停改為九軸減速到零並保持：底盤不做錨點伺服、手臂不追蹤凍結目標）；
#   凍結 freeze_door_k3m_pause_p3_dev_r1；判定器 door_k3m_check_p2r2.py。監督器、門檻、觀察期、保護全同。單趟。
#   必須以乾淨環境執行（系統 /usr/bin/python3；見 reference_venv_autoactivate）：
#   env -u VIRTUAL_ENV PATH=<去掉 .venv-dl0> RUN_ID=door_k3m_pause_p3_1 ROS_DOMAIN_ID=97 RUNNER=evaluation/run_door_k3m_pause_p3.sh bash evaluation/run_guarded_v2.sh
# **p2c 版**（run_door_k3m_pause_p2b.sh 的複本；Codex 無法使用，經 Howard 授權由 Claude 裁定，規格 K3M_pause_p2c_handover_precondition.md）：
#   唯一差異＝任務節點改 door_k3m_task_node_p2c.py（交棒請求前需已收到展開端 ≥ 20 筆新鮮 /wb_vel_cmd）；凍結 freeze_door_k3m_pause_p2c_dev_r1。
#   RUN_ID=door_k3m_pause_p2c_1 ROS_DOMAIN_ID=97 RUNNER=evaluation/run_door_k3m_pause_p2c.sh bash evaluation/run_guarded_v2.sh
# **p2b 版**（run_door_k3m_pause_p2.sh 的複本；2026-10-24 Codex 無法使用，經 Howard 授權由 Claude 裁定，規格 K3M_pause_p2b_delivery_fix.md）：
#   唯一差異＝任務節點改 door_k3m_task_node_p2b.py（SETTLE 窗口另開深度 64 的 /door/sync_state 訂閱，修正 p2 的送達缺陷）；凍結 freeze_door_k3m_pause_p2b_dev_r1。
#   判定器仍為 door_k3m_check_p2.py。判準、窗口參數、保護、走廊、增益全同。單趟。
#   RUN_ID=door_k3m_pause_p2b_1 ROS_DOMAIN_ID=97 RUNNER=evaluation/run_door_k3m_pause_p2b.sh bash evaluation/run_guarded_v2.sh
# **p2 版**（run_door_k3m_pause.sh 的複本；Codex 20261024_101421 (a)）：唯一控制差異＝任務節點改 door_k3m_task_node_p2.py
#   （SETTLE 完成判斷的角速子條件改用 50 Hz 同步流的 0.1 s 因果後向門角差分；tensor 門速 0.05 保護、其餘條件、連續 1 s、監看、握手全同）；
#   凍結 freeze_door_k3m_pause_p2_dev_r1；判定器 door_k3m_check_p2.py。不加偏航保持、不放寬走廊。單趟；未核可前不得執行。
#   RUN_ID=door_k3m_pause_p2_1 ROS_DOMAIN_ID=97 RUNNER=evaluation/run_door_k3m_pause_p2.sh bash evaluation/run_guarded_v2.sh
# **K3-B r2 版**（run_door_wide_k3b.sh 的複本；Codex 20261026_010158；實作 K3B_impl_r2.md）：唯一差異＝執行端 isaac_door_k2_sim_ev7r2.py、
#   任務節點 door_k3m_task_node_k3b2.py、判定器 door_k3m_check_k3b2.py、凍結 freeze_door_wide_k3b_dev_r2。**未核可 Isaac 實跑，未經 Codex 核可不得執行。**
#   env -u VIRTUAL_ENV PATH=<去掉 .venv-dl0> RUN_ID=door_wide_k3b_30_1 ROS_DOMAIN_ID=97 RUNNER=evaluation/run_door_wide_k3b_r2.sh bash evaluation/run_guarded_v2.sh
# **K3-B 版**（run_door_wide_w2c_r2.sh 的複本；Codex 20261026_000706：受控釋放實作包；規格 K3B_release_clear_spec_draft2.md、實作 K3B_impl_r1.md）。
#   **未核可 Isaac 實跑，未經 Codex 核可不得執行。** 差異只有：執行端 isaac_door_k2_sim_ev7.py＋--w2-release；求解器 dl3_wgmpc_wg2_adm12_node.py＋--w2-retreat；
#   任務節點 door_k3m_task_node_k3b.py＋--w2-release；判定器 door_k3m_check_k3b.py；凍結 freeze_door_wide_k3b_dev_r1。
#   範圍：30° 開門→受控暫停→HELD→mode 1 測試保持→mode 2 張爪→解除確認→mode 3 退離 0.10 m→恢復 5 N→退離終點保持 3 s→DONE；不收臂。
#   env -u VIRTUAL_ENV PATH=<去掉 .venv-dl0> RUN_ID=door_wide_k3b_30_1 ROS_DOMAIN_ID=97 RUNNER=evaluation/run_door_wide_k3b.sh bash evaluation/run_guarded_v2.sh
# **W2c r2 版**（run_door_wide_w2c.sh 的複本；Codex 20261025_223632 條件式核可單趟 door_wide_w2c_30_1；規格 W2c_run_pack_spec_r2.md）：唯一差異＝
#   執行端 isaac_door_k2_sim_ev6r3.py（r2＋W2 設定紀錄，控制不變）、判定器 door_k3m_check_w2c2.py（固定登錄值、趟別一致、事件時間交叉核對）、凍結 freeze_door_wide_w2c_dev_r2。
#   env -u VIRTUAL_ENV PATH=<去掉 .venv-dl0> RUN_ID=door_wide_w2c_30_1 ROS_DOMAIN_ID=97 RUNNER=evaluation/run_door_wide_w2c_r2.sh bash evaluation/run_guarded_v2.sh
# **W2c 版**（run_door_wide_w1f2.sh 的複本；Codex 20261025_220903：30° 開門＋HELD＋測試模式保持的開跑包；規格 W2c_run_pack_spec.md）。差異只有：
#   θ 回到 30°（登錄 w1e_params_30.json，W1e 兩趟通過的同一控制配置）；執行端 isaac_door_k2_sim_ev6r2.py＋--w2-mode-run-id（v3 唯一入口、模式權威 r4、
#   既有中止同步進權威）；adapter arm_vel_adapter_w2.py --cmd-env-v3；求解器 dl3_wgmpc_wg2_adm11_node.py＋--w2-held --w2-mode-run-id（HELD 持續保持）；
#   任務節點 door_k3m_task_node_w2.py＋--w2-mode-run-id（HELD_WAIT → 請求 → 提交確認 → 測試模式保持 3 s → DONE）；判定器 door_k3m_check_w2c.py；
#   凍結 freeze_door_wide_w2c_dev_r1。測試模式不張爪、不退離、不增加角度；增益、權重、限制、門檻、走廊、幾何、20 N 切換不改。單趟；未核可前不得執行。
#   env -u VIRTUAL_ENV PATH=<去掉 .venv-dl0> RUN_ID=door_wide_w2c_30_1 ROS_DOMAIN_ID=97 RUNNER=evaluation/run_door_wide_w2c.sh bash evaluation/run_guarded_v2.sh
OFFSET_MOVING=1
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WS"
: "${ROS_DOMAIN_ID:?ROS_DOMAIN_ID 未設定}"
: "${RUN_ID:?RUN_ID 未設定}"
ISAAC_PY="${ISAAC_PY:-$HOME/venvs/isaacsim-6.0.1/bin/python}"
DIR="$WS/evaluation/runs/$RUN_ID"; mkdir -p "$DIR"
FRZ=evaluation/results/door/freeze_door_wide_k3b_dev_r2.sha256
W1B_THETA=30                                                   # W2c：固定 30°（不由環境變數改角度）
PRM=evaluation/results/door/w1e_params_30.json                 # W2c：W1e 30° 登錄（w1e_30_1／_2 同一檔）
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

sha256sum -c --quiet "$FRZ" >> "$DIR/run.log" 2>&1 || { say "**K3-B 開發凍結不符** ⇒ 拒跑"; exit 95; }
say "python3＝$(command -v python3)；VIRTUAL_ENV=${VIRTUAL_ENV:-（無）}"
python3 -c "import osqp" 2>/dev/null || { say "**python3 環境錯誤（無法匯入 osqp；可能誤啟用虛擬環境）** ⇒ 拒跑"; exit 96; }
[ -s "$PRM" ] || { say "**W1b 登錄檔不存在：$PRM** ⇒ 拒跑"; exit 97; }
W1B_TABLE=$(python3 -c "import json;print(json.load(open('$PRM'))['protection_scope_change']['corridor_table'])")
W1B_ENV=$(python3 -c "import json;print(json.load(open('$PRM'))['protection_scope_change']['envelope']['cli'])")
W1B_OBS=$(python3 -c "import json;print(json.load(open('$PRM'))['protection_scope_change']['observer_door_max_deg'])")
W1B_COORD=$(python3 -c "import json;print(json.load(open('$PRM'))['control_cost_change']['coord_topic'])")
W1E_FEFF=$(python3 -c "import json;print(json.load(open('$PRM'))['executor_change']['pause_finger_max_effort'])")
W1E_FEFF0=$(python3 -c "import json;print(json.load(open('$PRM'))['executor_change']['original_effort'])")
say "W1e 執行端變更：暫停期手指驅動輸出上限 ${W1E_FEFF0} → ${W1E_FEFF} N（ev5 r2；啟動讀回不符拒跑、切換／定期讀回異常走中止閂鎖）"
say "W1b 登錄：$PRM（sha256 $(sha256sum "$PRM" | cut -c1-16)）θ=${W1B_THETA}° 走廊表 $W1B_TABLE 包絡 $W1B_ENV 觀察器上限 ${W1B_OBS}° 協同 $W1B_COORD"
say "=== K3-M（底盤可動協同＋EV3 取證＋INT）RUN_ID=$RUN_ID domain=$ROS_DOMAIN_ID；停位 $STAND；起始 K2_READY；夾持→開到 8°→保持（不張爪、不退離）==="

spawn sim "$ISAAC_PY" -u evaluation/isaac_door_k2_sim_ev7r2.py --w2-release --w2-mode-run-id "$RUN_ID" --pause-finger-max-effort "$W1E_FEFF" --pause-finger-original-effort "$W1E_FEFF0" --cmd-env --evidence-out "$DIR/evidence.jsonl" --park-gate-mobile --run-id "$RUN_ID" --corridor-table "$W1B_TABLE" --envelope="$W1B_ENV" --sim-limit 240 --contact-min-n 0.5 \
  --finger-collision split --park-hold --start-pose "$STAND" --park-pose "$STAND" --stow-q "$K2_READY" --out "$DIR"
say "  等場景建起（最多 180 s）"
for i in $(seq 180); do grep -q '進入主迴圈' "$DIR/sim.log" 2>/dev/null && break; kill -0 "${PIDS[0]}" 2>/dev/null || break; sleep 1; done
grep -q '進入主迴圈' "$DIR/sim.log" 2>/dev/null || { say "**場景未建起** ⇒ 中止"; tail -8 "$DIR/sim.log" | tee -a "$DIR/run.log"; exit 83; }
grep -E '^\[(room|door)\] (零摩擦|夾持面|摩擦核對|門建於|初始門角|門扇接觸|牆接觸)' "$DIR/sim.log" | tee -a "$DIR/run.log"

spawn wholebody python3 -u evaluation/door_k2_wholebody_node.py --solver-handshake --park-hold --pos-tol 0.005 \
  --stow-q "$K2_READY" --park "$STAND" --q-unfold="$Q_PRE" --out "$DIR/wholebody.json"
spawn task python3 -u evaluation/door_k3m_task_node_k3b2.py --w2-release --w2-mode-run-id "$RUN_ID" --out "$DIR/task.json" --run-id "$RUN_ID" --w1b-params "$PRM" --corridor-table "$W1B_TABLE"
ARM_IDENT="$WS/evaluation/results/wgmpc_arm_sp_ident_free4.json"
[ -f "$ARM_IDENT" ] || { say "**找不到手臂辨識檔** ⇒ 中止"; exit 85; }
spawn adapter python3 -u evaluation/arm_vel_adapter_w2.py --in-topic /wgmpc/cmd_world --consumer-node /door_k2_sim --cmd-env-v3
spawn solver python3 -u evaluation/dl3_wgmpc_wg2_adm12_node.py --w2-retreat --w2-held --w2-held-max-s 60 --w2-mode-run-id "$RUN_ID" --require-topic-target --park-hold --offset-observer interval --mode-switch-run-id "$RUN_ID" --cmd-env --run-id "$RUN_ID" --pause --pause-arm-zero --pause-active-hold --pause-arm-win-s 0.5 --coord-topic "$W1B_COORD" --offset-hold-on-free \
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
say "=== 結束 rc=$RC（0 只表示所有節點正常結束且到 DONE；通過與否由 door_k3m_check_k3b2.py 判定（既有開門／停止／保護／紀錄判準＋W2 obs_passed／held／commit／test_complete 分報；DONE 只表示測試模式保持完成））==="
exit "$RC"
