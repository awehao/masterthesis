#!/usr/bin/env bash
# 滾動版（2026-10-29；Codex 20261029_090630 選 B，核可一次功能趟）：run_door_nav1_door.sh r2 的複本，改為
#   朝前導航（折線 S→P0→D1，最後一段沿操作朝向）→ GLIDE 平順減速至入口速度框 → 底盤仍移動時滾動交棒 → 展開節點 --motm 移動中展臂 →
#   W-GMPC 底盤自由＋approach coord 協同接近（ALIGN／APPROACH）→ ENGAGE 前減速、seq 1 hold（錨點＝實測）→ 閉爪 → 既有後退開門（seq 2 free）→ 換抓…
#   執行端 ev9r8o --rolling-handover --gate6-log；求解端 ADM22 --rolling-approach；任務端 node10h --rolling；nav1 r4 --rolling；展開節點 --motm（取代 --park-hold）。
#   approach_vref 參數三處一致：v_cap 0.030、a_ref 0.002、k_yaw 1.0、w_cap 0.10、v_creep 0、overshoot 0（抽屜為 0.004／0.030：門端要在 hold 前停下，揭露）。
#   保護：接近段＝執行端接近走廊（線段偏離 ≤50 mm、偏航 ≤5°）＋任務端線段偏離代入 base_drift（20 mm／2°）＋幾何／接觸／資料年齡；hold 後同既有 PARK_HOLD；OPEN 起同既有。
# r2（Codex 20261029_123819，自 door_nav1_roll_3 起）：交棒確認後（nav1 記 HANDED_OVER＝其觀察到執行端 /handover/state owner＝wholebody）由跑法以**任務安排停止**結束
#   本趟登錄的 gmpc、tracker、smoother、smlife（TERM 本程序及其直接子程序；不用廣域 pkill、不用 SIGSTOP）；保留 scan relay、TF、地圖、護盾、nav1（零輸出）、觀察器。
#   導航 CBF 設定不關閉（導航有權威時才作用；全身接手期間由操作保護負責）。NAV2 前須重新啟動導航計算（本趟不含 NAV2）。
# 用法：RUN_ID=door_nav1_roll_1 ROS_DOMAIN_ID=97 RUNNER=evaluation/run_door_nav1_door_roll.sh bash evaluation/run_guarded.sh
# 以下為 run_door_nav1_door.sh 說明。
# NAV1＋門段同趟（2026-10-29；Codex 20261029_081047／081705 核可單跑一次新 RUN_ID）：起點 S → NAV1 → W1 停位 D1 精停 → PARK 閘門＋權威交棒 →
#   全身展開（K2_READY→Q_PRE）→ W1 正面實際接近、閉爪取證 → 既有門段（往後拉開 48°→換抓→穿門拉關至約 2°→放手退出）。
# 與 extf_leader_sc5 的差異（其餘旗標、SETTLE／HoldConfirm 語義、主從夾爪、ref pb1f、ADM21 全同）：
#   - 執行端 ev9r8n --nav-start --pub-tf true --start-pose S（取代 --bench-start／--bench-params／--bench-setpoint；--stow-q＝K2_READY）；gc.freeze 於主迴圈前一次。
#   - 任務端 node10g --nav-start（非台架換抓鏈，WAIT_HANDOVER 起；--wait-handover-timeout-s 240）；不帶 --bench-wait-contacts。
#   - 求解端 ADM21 去 --bench-start／--bench-params，改 --wait-for-drawer-handover（等展開節點握手，同 w1f2）；轉接器去 --bench-ready-run-id。
#   - 新增展開節點 door_k2_wholebody_node（同 w1f2）。
#   - 新增導航（照抄 run_drawer_phase1_nav.sh 定版參數）：map_server（maps/door_scene_v3.yaml，由 generate_door_scene_map.py 自門資產幾何生成）、
#     scan_relay（/scan_raw→/scan）、scan_obstacle_tracker、gmpc_node（GMPC＋horizon CBF α 0.5／動態 0.60／靜態 0.38）→ /cmd_vel_nav →
#     velocity_smoother → /cmd_vel_pre_mux → **door_nav1_mission_node 選擇器**（CRUISE 轉送｜PARK_ALIGN P 伺服 ≤0.03 m/s、≤0.1 rad/s｜ZERO）→
#     /cmd_vel_pre_shield → scan_safety_shield（掃描 barrier）→ /cmd_vel。
# 揭露：/odom 真值（無 AMCL）；直線計畫（非 nav2 planner）；精停段＝P 伺服＋掃描 barrier（不是 GMPC）；本場景無移動／未知障礙（本輪不加）；門外 A 房無外牆。
# r2（Codex 20261029_084216；自 door_nav1_door_5 起）：Howard「移動時車頭朝前、操作前才對位」⇒ gmpc heading_enable＝true（參考航向＝路徑前視弦方向，
#   不要求終點航向；定版其他參數不變），PARK_ALIGN 內才轉至操作朝向 90°；PARK_ALIGN 位置容差 5 → 1 mm（--tol-pos 0.001；限速、期限、閘門不變）。
# 用法：RUN_ID=door_nav1_door_1 ROS_DOMAIN_ID=97 RUNNER=evaluation/run_door_nav1_door.sh bash evaluation/run_guarded.sh
OFFSET_MOVING=1
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WS"
: "${ROS_DOMAIN_ID:?ROS_DOMAIN_ID 未設定}"
: "${RUN_ID:?RUN_ID 未設定}"
ISAAC_PY="${ISAAC_PY:-$HOME/venvs/isaacsim-6.0.1/bin/python}"
# shellcheck disable=SC1091
if [ -f "$WS/install/setup.bash" ]; then set +u; source "$WS/install/setup.bash"; set -u; fi
START_S="${START_S:--0.6,-2.8,1.570796}"
MAP="$WS/src/ammr_bringup/maps/door_scene_v3.yaml"
URDF="$WS/evaluation/models/omni_bot_manip.urdf"
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

REF="${PB1_REF:-evaluation/results/door/door_regrasp_ref_pb1f.json}"   # 加速變體可用 PB1_REF 指定 *_wp.json
DOOR3=src/my_omnibot_description/config/door_local_v3.yaml
sha256sum evaluation/isaac_door_k2_sim_ev9r8o.py evaluation/dl3_wgmpc_wg2_adm22_node.py evaluation/door_coopen_core_task_node10h.py evaluation/door_rolling_task.py evaluation/door_k3m_mode2.py evaluation/motm_coord.py evaluation/isaac_door_k2_sim_ev9r8n.py evaluation/door_coopen_core_task_node10g.py evaluation/door_nav1_mission_node.py src/ammr_bringup/scripts/generate_door_scene_map.py src/ammr_bringup/maps/door_scene_v3.yaml src/ammr_bringup/maps/door_scene_v3.pgm src/ammr_wholebody_mpc/ammr_wholebody_mpc/gmpc_node.py evaluation/door_k2_wholebody_node.py evaluation/isaac_door_k2_sim_ev9r8m.py evaluation/isaac_door_k2_sim_ev9r8j.py evaluation/door_coopen_core_task_node10f.py evaluation/door_coopen_core_task_node10e.py evaluation/door_coopen_bench_grasp3.py evaluation/door_coopen_bench_grasp2.py evaluation/door_coopen_core_task_node10d.py evaluation/door_settle_cont.py evaluation/door_coopen_core_task_node10c.py evaluation/dl3_wgmpc_wg2_adm21_node.py evaluation/arm_vel_adapter_k3m3.py \
  evaluation/door_regrasp_*.py evaluation/door_finger_loss_*.py evaluation/door_mimic_follow.py evaluation/door_k3m_observer.py "$PRM" "$GB" "$REF" "$DOOR3" > "$DIR/versions.sha256" || { say "**版本紀錄失敗** ⇒ 拒跑"; exit 95; }
say "版本紀錄 $DIR/versions.sha256（$(wc -l < "$DIR/versions.sha256") 檔；git $(git rev-parse --short HEAD)）"
say "python3＝$(command -v python3)；VIRTUAL_ENV=${VIRTUAL_ENV:-（無）}"
python3 -c "import osqp" 2>/dev/null || { say "**python3 環境錯誤（無法匯入 osqp；可能誤啟用虛擬環境）** ⇒ 拒跑"; exit 96; }
[ -s "$PRM" ] || { say "**核心段登錄檔不存在：$PRM** ⇒ 拒跑"; exit 97; }
[ -s "$GB" ] || { say "**抓取基準不存在：$GB** ⇒ 拒跑"; exit 97; }
[ -s "$MAP" ] || { say "**找不到門場景地圖 $MAP** ⇒ 拒跑"; exit 80; }
[ "$(python3 -c "import json;print(json.load(open('$PRM'))['solver_change'].get('offset_update_in_open') is False)")" = True ] || { say "**登錄 r2 須明確 offset_update_in_open＝False（W1 語義）** ⇒ 拒跑"; exit 97; }
GB_SHA=$(python3 -c "import json;print(json.load(open('$PRM'))['grasp']['basis_sha256'])")
[ "$(sha256sum "$GB" | cut -d' ' -f1)" = "$GB_SHA" ] || { say "**抓取基準 sha256 與登錄不符** ⇒ 拒跑"; exit 97; }
W1B_THETA=$(python3 -c "import json;print(json.load(open('$PRM'))['theta_goal_deg'])")
STAND=$(python3 -c "import json;print(json.load(open('$PRM'))['start']['stand_cli'])")
K2_READY=$(python3 -c "import json;print(json.load(open('$PRM'))['start']['k2_ready_cli'])")
Q_PRE=$(python3 -c "import json;print(json.load(open('$PRM'))['start']['q_pre_cli'])")
# door_nav1_door_3：登錄 start.q_pre_cli＝null（台架流程不經展開）⇒ 展開節點 --q-unfold 收到 'None' 崩潰。改用 W1 線實跑過的 Q_PRE
# （run_door_wide_w1f2.sh；door_wide_w1f_60_3 UNFOLD→ALIGN→APPROACH→ENGAGE 實跑；pullback 登錄衍生自 w1f_params_60_r2）；登錄有值時以登錄為準。
Q_PRE_W1='-0.9471,0.0758,1.6024,-1.539,-0.9479,-0.0544'
[ "$Q_PRE" = None ] && Q_PRE="$Q_PRE_W1"
Q_PRE_SRC=registry; [ "$Q_PRE" = "$Q_PRE_W1" ] && Q_PRE_SRC='run_door_wide_w1f2.sh（door_wide_w1f_60_3 實跑）'
python3 -c "import math;v=[float(x) for x in '$Q_PRE'.split(',')];assert len(v)==6 and all(math.isfinite(x) for x in v)" 2>/dev/null || { say "**Q_PRE 不合法（須六個有限數）：$Q_PRE** ⇒ 拒跑"; exit 97; }
python3 -c "import json;json.dump({'q_pre':[float(x) for x in '$Q_PRE'.split(',')],'source':'$Q_PRE_SRC','registry_q_pre_cli':None},open('$DIR/q_pre.json','w'),ensure_ascii=False)"
say "Q_PRE（展開目標）＝$Q_PRE；來源 $Q_PRE_SRC（記入 q_pre.json）"
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
say "=== NAV1＋門段同趟（ev9r8n＋node10g＋ADM21＋k3m3；ref pb1f；導航 GMPC＋CBF→平滑器→選擇器〔巡航／PARK_ALIGN〕→掃描 barrier 護盾）RUN_ID=$RUN_ID domain=$ROS_DOMAIN_ID；起點 $START_S → 停位 $STAND；交棒後 K2_READY→Q_PRE→W1 正面實際接近閉爪→開門 ${W1B_THETA}°→換抓→穿門拉關→放手退出 ==="
say "揭露：/odom 真值無 AMCL；直線計畫；巡航 heading 開（路徑前視弦）；精停段＝P 伺服＋掃描 barrier（非 GMPC，含轉至 90°，容差 1 mm／1°）；無移動／未知障礙；A 房無外牆"

spawn sim "$ISAAC_PY" -u evaluation/isaac_door_k2_sim_ev9r8o.py --rolling-handover --gate6-log --nav-start --pub-tf true --hold-confirm-cont --finger-leader-only --finger-loss-cont --ext-forces-every-iter --diag-6d --regrasp --regrasp-ref "$REF" --w2-mode-run-id "$RUN_ID" --contact-v3-out "$DIR/contact_v3.jsonl" --door-asset "$DOOR3" --gc-freeze-before-release --pause-finger-max-effort "$W1E_FEFF" --pause-finger-original-effort "$W1E_FEFF0" --cmd-env --evidence-out "$DIR/evidence.jsonl" --park-gate-mobile --run-id "$RUN_ID" --corridor-table "$W1B_TABLE" --envelope="$W1B_ENV" --sim-limit 560 --contact-min-n 0.5 \
  --finger-collision split --park-hold --start-pose="$START_S" --park-pose="$STAND" --stow-q="$K2_READY" --out "$DIR"
say "  等場景建起（最多 180 s）"
for i in $(seq 180); do grep -q '進入主迴圈' "$DIR/sim.log" 2>/dev/null && break; kill -0 "${PIDS[0]}" 2>/dev/null || break; sleep 1; done
grep -q '進入主迴圈' "$DIR/sim.log" 2>/dev/null || { say "**場景未建起** ⇒ 中止"; tail -8 "$DIR/sim.log" | tee -a "$DIR/run.log"; exit 83; }
grep -E '^\[(room|door)\] (零摩擦|夾持面|摩擦核對|門建於|場景求解|主從讀回|初始門角|門扇接觸|牆接觸)' "$DIR/sim.log" | tee -a "$DIR/run.log"

say "[導航] map_server／scan_relay／tracker／gmpc（定版）／平滑器→選擇器→護盾"
spawn rsp ros2 run robot_state_publisher robot_state_publisher "$URDF" --ros-args -p use_sim_time:=true
spawn tf_map ros2 run tf2_ros static_transform_publisher --x 0 --y 0 --z 0 --qx 0 --qy 0 --qz 0 --qw 1 --frame-id map --child-frame-id odom --ros-args -p use_sim_time:=true
spawn map ros2 run nav2_map_server map_server --ros-args -p use_sim_time:=true -p yaml_filename:="$MAP"
sleep 3
spawn maplife ros2 run nav2_lifecycle_manager lifecycle_manager --ros-args -r __node:=lifecycle_manager_map -p use_sim_time:=true -p autostart:=true -p node_names:="['map_server']"
spawn relay ros2 run ammr_bringup scan_relay --ros-args -p use_sim_time:=true
sleep 3
spawn tracker ros2 run ammr_wholebody_mpc scan_obstacle_tracker --ros-args -p use_sim_time:=true -p scan_topic:=/scan -p map_topic:=/map -p global_frame:=odom -p publish_frame:=odom -p use_map_subtraction:=true -p cluster_gap:=0.30 -p max_radius:=1.20 -p min_cluster_pts:=2 -p static_window:=2.0 -p min_net_speed:=0.05 -p surface_max_pts:=4 -p surface_range:=2.5 -p surface_min_sep:=0.20
sleep 3
spawn gmpc python3 -u -c "
import sys; sys.path.insert(0,'src/ammr_wholebody_mpc')
sys.argv=['gmpc_node','--ros-args','-p','use_sim_time:=true','-p','pose_source:=odom','-p','global_frame:=odom','-p','pose_odom_topic:=/odom','-p','plan_topic:=/plan',
 '-p','cmd_vel_topic:=/cmd_vel_nav','-p','obstacles_topic:=/gmpc/obstacles','-p','cbf_enable:=true','-p','cbf_alpha:=0.5','-p','cbf_safe_margin:=0.60',
 '-p','static_cbf_safe_margin:=0.38','-p','cbf_prune_range:=1.2','-p','cbf_near_steps:=6','-p','cbf_far_stride:=2','-p','horizon:=20','-p','control_frequency:=20.0',
 '-p','Q_xy:=15.0','-p','Qf_mult:=5.0','-p','R_vx:=2.0','-p','R_vy:=2.0','-p','R_w:=1.0','-p','S_vx:=15.0','-p','S_vy:=15.0','-p','S_w:=8.0',
 '-p','heading_enable:=true','-p','vx_min:=-0.35','-p','ax_max:=1.5','-p','ay_max:=1.0','-p','az_max:=2.0','-p','plan_blend_s:=0.0']
from ammr_wholebody_mpc.gmpc_node import main; main()"
sleep 4
spawn smoother ros2 run nav2_velocity_smoother velocity_smoother --ros-args -p use_sim_time:=true -p max_velocity:="[0.35, 0.25, 0.80]" -p min_velocity:="[-0.35, -0.25, -0.80]" -p max_accel:="[1.5, 1.0, 2.0]" -p max_decel:="[-1.5, -1.0, -2.0]" -p feedback:=OPEN_LOOP -p odom_topic:=/odom -r cmd_vel:=cmd_vel_nav -r cmd_vel_smoothed:=cmd_vel_pre_mux
spawn smlife ros2 run nav2_lifecycle_manager lifecycle_manager --ros-args -r __node:=lifecycle_manager_smoother -p use_sim_time:=true -p autostart:=true -p node_names:="['velocity_smoother']"
sleep 3
spawn shield ros2 run ammr_wholebody_mpc scan_safety_shield --ros-args -p use_sim_time:=true -p enable:=true -p scan_topic:=/scan -p cmd_in_topic:=/cmd_vel_pre_shield -p cmd_out_topic:=/cmd_vel -p robot_radius:=0.30 -p alpha:=2.0 -p a_brake:=6.25 -p tau:=0.15 -p vx_max:=0.2775 -p vy_max:=0.2775 -p wz_max:=1.1327
sleep 3
spawn nav1 python3 -u evaluation/door_nav1_mission_node.py --goal="$STAND" --out "$DIR/nav1.json" --rolling --nav-timeout-s 150
spawn wholebody python3 -u evaluation/door_k2_wholebody_node.py --solver-handshake --motm --motm-v-cap 0.030 --motm-a-ref 0.002 --motm-k-yaw 1.0 --motm-w-cap 0.10 --motm-v-creep 0.0 --motm-overshoot 0.0 --pos-tol 0.005 --stow-q "$K2_READY" --park="$STAND" --q-unfold="$Q_PRE" --out "$DIR/wholebody.json"
spawn task python3 -u evaluation/door_coopen_core_task_node10h.py --rolling --roll-goal="$STAND" --nav-start --wait-handover-timeout-s 240 --out "$DIR/task.json" --run-id "$RUN_ID" --grasp-basis "$GB" --w1b-params "$PRM" --corridor-table "$W1B_TABLE" --finger-loss-cont --settle-cont --regrasp --regrasp-ref "$REF" --w2-mode-run-id "$RUN_ID" --wall-total-s 1500
ARM_IDENT="$WS/evaluation/results/wgmpc_arm_sp_ident_free4.json"
[ -f "$ARM_IDENT" ] || { say "**找不到手臂辨識檔** ⇒ 中止"; exit 85; }
spawn adapter python3 -u evaluation/arm_vel_adapter_k3m3.py --in-topic /wgmpc/cmd_world --consumer-node /door_k2_sim --cmd-env --cmd-env-v3 --env-in-topic /wgmpc/cmd_env
spawn solver python3 -u evaluation/dl3_wgmpc_wg2_adm22_node.py --rolling-approach --w2-retreat --w2-hold-r9 --w2-held --w2-held-max-s 60 --w2-mode-run-id "$RUN_ID" --regrasp --require-topic-target --park-hold --offset-observer interval --mode-switch-run-id "$RUN_ID" --cmd-env --run-id "$RUN_ID" --pause --pause-arm-zero --pause-active-hold --pause-arm-win-s 0.5 --coord-topic "$W1B_COORD" --offset-hold-on-free \
  --offset-moving --offset-free \
  --continuous --wait-for-drawer-handover --arm-model setpoint --arm-ident "$ARM_IDENT" \
  --cmd-topic /wgmpc/cmd_world --target-topic /drawer/tcp_target --target-offset 0 0 0 \
  --N 5 --rate 20 --delay-comp-state-cycles 1.6 --delay-comp-cmd-cycles 1.0 \
  --near-target-gamma 1.0 --w-s 0.001 --w-a 0.05 --w-s-arm 0.05 --margin-guard \
  --reach-pos-m 0.005 --reach-rot-rad 0.02 --hold-s 2.0 --duration-s 1600 --duration-sim-s 560 \
  --u-prev-policy strict --assume-initial-rest --out "$DIR/align_solver.json"
spawn lw_observer python3 -u evaluation/door_k3m_observer.py --out "$DIR/lw_observer.json" --door-max-deg "$W1B_OBS"

say "[任務] 等任務節點結束（牆鐘最多 1600 s）"
pid_of(){ local i; for i in "${!NAMES[@]}"; do [ "${NAMES[$i]}" = "$1" ] && echo "${PIDS[$i]}"; done; }
SIM_PID=$(pid_of sim); TASK_PID=$(pid_of task); SOLVER_PID=$(pid_of solver); OBS_PID=$(pid_of lw_observer); WB_PID=$(pid_of wholebody); NAV1_PID=$(pid_of nav1)
say "  PID：sim $SIM_PID task $TASK_PID solver $SOLVER_PID observer $OBS_PID wholebody $WB_PID nav1 $NAV1_PID"
( until grep -q "分段 → HANDED_OVER" "$DIR/nav1.log" 2>/dev/null; do kill -0 "$TASK_PID" 2>/dev/null || exit 0; sleep 0.5; done
  say "**任務安排停止**：交棒已確認（nav1 HANDED_OVER＝執行端 owner wholebody）⇒ 結束本趟導航計算 gmpc／tracker／smoother／smlife"
  for _n in gmpc tracker smoother smlife; do _p=$(pid_of "$_n"); [ -n "$_p" ] || continue; pkill -TERM -P "$_p" 2>/dev/null; kill -TERM "$_p" 2>/dev/null; done
  sleep 3
  for _n in gmpc tracker smoother smlife; do _p=$(pid_of "$_n"); _st=gone; kill -0 "$_p" 2>/dev/null && _st=alive; say "  任務安排停止 $_n PID=$_p → $_st"; done ) &
( while kill -0 "$TASK_PID" 2>/dev/null; do kill -0 "$WB_PID" 2>/dev/null || { say "**展開節點已退出**（任務仍在跑；任務端 r3 依命令缺席判定）"; break; }; sleep 1; done ) &
sleep 5; kill -0 "$OBS_PID" 2>/dev/null || { say "**獨立觀察器未存活** ⇒ 中止（沒有觀察期保護不得繼續）"; exit 86; }
wait_or_stop "$TASK_PID" 1600 10; RC_TASK=$STOP_RC; HOW_TASK=$STOP_HOW
grep -E '^\[k3a\]' "$DIR/task.log" | tee -a "$DIR/run.log"
say "[收尾] 等求解節點停止並封存（最多 30 s，之後 TERM→10 s→KILL）"
wait_or_stop "$SOLVER_PID" 30 10; RC_SOLVER=$STOP_RC; HOW_SOLVER=$STOP_HOW
wait_or_stop "$WB_PID" 20 5; RC_WB=$STOP_RC; HOW_WB=$STOP_HOW
kill -TERM "$NAV1_PID" 2>/dev/null || true; sleep 1
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
[ "$HOW_TASK" = exited ] && [ "$HOW_SOLVER" = exited ] && [ "$HOW_SIM" = exited ] && [ "$RC_TASK" = 0 ] && [ "$RC_SOLVER" = 0 ] && [ "$RC_SIM" = 0 ] && [ "$OK_ROOM" = 1 ] && [ "$OK_SOLVER" = 1 ] && [ "$OK_DONE" = 1 ] && RC=0
say "=== 結束 rc=$RC（0 只表示所有節點正常結束且到 DONE；指定初態功能展示，不是正式評估；拉關到約 2° 只能稱近關閉）==="
exit "$RC"
