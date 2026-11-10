#!/usr/bin/env bash
# run_m8_nav（2026-11-10；第八次第一包「場景＋掃描驅動導航」，Codex 20261110_180004）：S → 門前觀察區，到門前結束（不閉爪、不開門）。
#   世界：mission_room.sdf＋門 v3＋抽屜櫃（同往返任務場景）＋ results/mission/mission_obstacles_seed1.json（r2；未知箱體＋行人，ev9r9b --obstacles-file）。
#   地圖：mission_room.yaml（不含未知箱體）。導航鏈沿用往返任務定版：scan_relay → scan_obstacle_tracker（地圖扣除；/gmpc/obstacles、/scan_filtered）
#     → gmpc_node（封存；GMPC＋horizon CBF，參數同 run_mission_roundtrip4）→ velocity_smoother → m8_nav_obs_node（選擇器）→ scan_safety_shield → /cmd_vel。
#   規劃：planner_server（SmacPlanner2D；m8_planner.yaml：靜態層＋/scan_filtered 障礙層＋膨脹）。目標＝門前觀察區（m8_scene_check.json，由門先驗幾何算）。
#   起點＝研究目標情境 S（mission_room_poses.json：−2.2, −4.3, 90°）；手臂 K2_READY。
#   紀錄：ros2 bag（mcap）錄掃描、追蹤、計畫、命令、GMPC 診斷、代價圖、導航狀態；執行端另寫 obstacles_truth.npz（真值，只供評估）。
# 模式：SMOKE_SCENE=1 只核場景建構；WIRING=1 不動作接線檢查（不起導航節點，保持 WIRING_S 秒後停）；預設＝完整一趟。
# 用法：RUN_ID=m8nav_1 ROS_DOMAIN_ID=93 RUNNER=evaluation/run_m8_nav.sh bash evaluation/run_guarded_v3.sh
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WS"
: "${ROS_DOMAIN_ID:?ROS_DOMAIN_ID 未設定}"
: "${RUN_ID:?RUN_ID 未設定}"
ISAAC_PY="${ISAAC_PY:-$HOME/venvs/isaacsim-6.0.1/bin/python}"
if [ -f "$WS/install/setup.bash" ]; then set +u; source "$WS/install/setup.bash"; set -u; fi
MAP="$WS/src/ammr_bringup/maps/mission_room.yaml"
WORLD="$WS/src/ammr_bringup/worlds/mission_room.sdf"
DOOR3=src/my_omnibot_description/config/door_local_v3.yaml
DRAWER=src/my_omnibot_description/config/drawer_unit_bar26.yaml
OBS=evaluation/results/mission/mission_obstacles_seed1.json
CHK=第八次進度報告素材/資料/m8_scene_check.json
URDF="$WS/evaluation/models/omni_bot_manip.urdf"
K2_READY='0,0,0.035,0,-1.5708,0'
START=$(python3 -c "import json,math;p=json.load(open('evaluation/results/mission/mission_room_poses.json'))['poses']['S'];print(f\"{p['x']},{p['y']},{math.radians(p['yaw_deg']):.6f}\")")
DIR="$WS/evaluation/runs/$RUN_ID"; mkdir -p "$DIR"
NAV_TIMEOUT_S="${NAV_TIMEOUT_S:-240}"
WIRING_S="${WIRING_S:-20}"
PIDS=(); NAMES=()
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$DIR/run.log"; }
spawn(){ local n="$1"; shift; setsid "$@" >>"$DIR/$n.log" 2>&1 & PIDS+=("$!"); NAMES+=("$n"); sleep 0.2; local g; g=$(ps -o pgid= -p "${PIDS[-1]}" 2>/dev/null | tr -d " "); printf "%s\t%s\t%s\n" "$n" "${PIDS[-1]}" "${g:-NA}" >> "$DIR/procs.tsv"; say "  起 $n PID=${PIDS[-1]} PGID=${g:-NA}"; }
cleanup(){ say "cleanup（只針對本趟 PID；TERM → 3 s → KILL）"
  for i in "${!PIDS[@]}"; do _g=$(awk -v p="${PIDS[$i]}" '$2==p{print $3}' "$DIR/procs.tsv"); if [ "$_g" = "${PIDS[$i]}" ]; then kill -TERM -- "-${PIDS[$i]}" 2>/dev/null || true; else kill -TERM "${PIDS[$i]}" 2>/dev/null || true; fi; done
  sleep 3
  for i in "${!PIDS[@]}"; do _g=$(awk -v p="${PIDS[$i]}" '$2==p{print $3}' "$DIR/procs.tsv"); if [ "$_g" = "${PIDS[$i]}" ]; then kill -KILL -- "-${PIDS[$i]}" 2>/dev/null || true; else kill -KILL "${PIDS[$i]}" 2>/dev/null || true; fi; done
  sleep 1; : > "$DIR/cleanup_residual.tsv"
  for i in "${!PIDS[@]}"; do _r=$(pgrep -g "${PIDS[$i]}" 2>/dev/null | tr '\n' ' '); printf "%s\t%s\t%s\n" "${NAMES[$i]}" "${PIDS[$i]}" "${_r:-none}" >> "$DIR/cleanup_residual.tsv"; done
  say "cleanup 後各群組殘留：$(awk '$3!="none"' "$DIR/cleanup_residual.tsv" | wc -l) 組有殘留（cleanup_residual.tsv）"; }
trap cleanup EXIT
trap 'say "收到 TERM"; exit 143' TERM
trap 'say "收到 INT"; exit 130' INT

if [ -n "${FREEZE:-}" ]; then                                                 # 正式趟：凍結清單逐檔核對，不符即拒跑（接線／冒煙不設）
  sha256sum -c --quiet "$FREEZE" > "$DIR/freeze_check.txt" 2>&1 || { say "**凍結核對不符（$FREEZE）** ⇒ 拒跑"; cat "$DIR/freeze_check.txt" | tee -a "$DIR/run.log"; exit 94; }
  say "凍結核對通過：$FREEZE（$(wc -l < "$FREEZE") 檔）"
fi
sha256sum evaluation/isaac_door_k2_sim_ev9r9b.py evaluation/m8_nav_obs_node.py evaluation/m8_planner.yaml evaluation/m8_scene_check.py evaluation/mission_walkers.py evaluation/run_m8_nav.sh \
  src/ammr_wholebody_mpc/ammr_wholebody_mpc/gmpc_node.py src/ammr_wholebody_mpc/ammr_wholebody_mpc/gmpc.py src/ammr_wholebody_mpc/ammr_wholebody_mpc/scan_obstacle_tracker.py \
  src/ammr_wholebody_mpc/ammr_wholebody_mpc/scan_safety_shield.py "$OBS" "$CHK" "$MAP" "$WORLD" "$DOOR3" "$DRAWER" > "$DIR/versions.sha256" || { say "**版本紀錄失敗** ⇒ 拒跑"; exit 95; }
say "版本紀錄 $DIR/versions.sha256（$(wc -l < "$DIR/versions.sha256") 檔；git $(git rev-parse --short HEAD)）"
say "python3＝$(command -v python3)；VIRTUAL_ENV=${VIRTUAL_ENV:-（無）}"
python3 -c "import osqp" 2>/dev/null || { say "**python3 環境錯誤（無法匯入 osqp）** ⇒ 拒跑"; exit 96; }
[ "$(python3 -c "import json;print(json.load(open('$CHK'))['ok'])")" = True ] || { say "**m8_scene_check 未通過** ⇒ 拒跑"; exit 97; }
OBS_SHA=$(sha256sum "$OBS" | cut -d' ' -f1)
[ "$(python3 -c "import json;print(json.load(open('$CHK'))['obstacles_file'])")" = "$OBS" ] || { say "**場景核對用的障礙檔與本趟不同** ⇒ 拒跑"; exit 97; }
ORPH=$(python3 - "$ROS_DOMAIN_ID" <<'PYEOF'
import os, sys
dom = sys.argv[1]; me = os.getpid(); n = []
for p in os.listdir('/proc'):
    if not p.isdigit() or int(p) == me:
        continue
    try:
        env = open(f'/proc/{p}/environ', 'rb').read().split(b'\0')
        cmd = open(f'/proc/{p}/cmdline', 'rb').read().replace(b'\0', b' ').decode(errors='ignore')
    except OSError:
        continue
    dm = [e for e in env if e.startswith(b'ROS_DOMAIN_ID=')]
    same = (not dm) or dm[0] == f'ROS_DOMAIN_ID={dom}'.encode()
    known = any(k in cmd for k in ('isaac_door_k2_sim', 'm8_nav_obs', 'gmpc_node', 'scan_obstacle_tracker', 'scan_safety_shield', 'planner_server'))
    if same and ('--ros-args' in cmd or known) and 'run_m8_' not in cmd and 'run_guarded' not in cmd:
        n.append(f'{p}:{cmd[:80]}')
print(len(n)); [print(x) for x in n[:10]]
PYEOF
)
[ "$(echo "$ORPH" | head -1)" = 0 ] || { say "**domain $ROS_DOMAIN_ID 已有 $(echo "$ORPH" | head -1) 個 ROS 程序（可能是前趟孤兒）⇒ 拒跑**"; echo "$ORPH" | tail -n +2 | tee -a "$DIR/run.log"; exit 99; }
say "起跑前核對：domain $ROS_DOMAIN_ID 未發現衝突程序；障礙檔 $OBS（sha256 ${OBS_SHA:0:12}）；起點 $START"

say "=== 第八次第一包：S → 門前觀察區（ev9r9b＋m8_nav_obs；RUN_ID=$RUN_ID domain=$ROS_DOMAIN_ID）"
spawn sim "$ISAAC_PY" -u evaluation/isaac_door_k2_sim_ev9r9b.py --headless false --world "$WORLD" --door-asset "$DOOR3" --with-drawer --drawer-asset "$WS/$DRAWER" \
  --nav-start --pub-tf true --nav-env-r 0.39 --finger-collision split --start-pose="$START" --stow-q="$K2_READY" --run-id "$RUN_ID" --obstacles-file "$WS/$OBS" --out "$DIR"
say "  等場景建起（最多 180 s）"
for i in $(seq 180); do grep -q '進入主迴圈' "$DIR/sim.log" 2>/dev/null && break; kill -0 "${PIDS[0]}" 2>/dev/null || break; sleep 1; done
grep -q '進入主迴圈' "$DIR/sim.log" 2>/dev/null || { say "**場景未建起** ⇒ 中止"; tail -12 "$DIR/sim.log" | tee -a "$DIR/run.log"; exit 83; }
grep -E '^\[room\] (房間|EV9r9b)' "$DIR/sim.log" | tee -a "$DIR/run.log"
if [ "${SMOKE_SCENE:-0}" = "1" ]; then
  timeout 5 ros2 topic pub --once /room/stop std_msgs/msg/String "{data: 'smoke scene'}" >/dev/null 2>&1
  for i in $(seq 30); do kill -0 "${PIDS[0]}" 2>/dev/null || break; sleep 1; done
  say "=== SMOKE_SCENE 完成 ==="; exit 0
fi

say "[導航] map_server／scan_relay／tracker／gmpc（封存定版參數）／平滑器／護盾／planner_server"
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
 '-p','cmd_vel_topic:=/cmd_vel_nav','-p','obstacles_topic:=/gmpc/obstacles','-p','cbf_enable:=true','-p','cbf_alpha:=0.5','-p','cbf_safe_margin:=0.69',
 '-p','static_cbf_safe_margin:=0.47','-p','cbf_prune_range:=1.2','-p','cbf_near_steps:=6','-p','cbf_far_stride:=2','-p','horizon:=20','-p','control_frequency:=20.0',
 '-p','Q_xy:=15.0','-p','Qf_mult:=5.0','-p','R_vx:=2.0','-p','R_vy:=2.0','-p','R_w:=1.0','-p','S_vx:=15.0','-p','S_vy:=15.0','-p','S_w:=8.0',
 '-p','heading_enable:=true','-p','wheel_fit_mode:=project','-p','vx_min:=-0.35','-p','ax_max:=1.5','-p','ay_max:=1.0','-p','az_max:=2.0','-p','plan_blend_s:=0.0']
from ammr_wholebody_mpc.gmpc_node import main; main()"
sleep 4
spawn smoother ros2 run nav2_velocity_smoother velocity_smoother --ros-args -p use_sim_time:=true -p max_velocity:="[0.35, 0.25, 0.80]" -p min_velocity:="[-0.35, -0.25, -0.80]" -p max_accel:="[1.5, 1.0, 2.0]" -p max_decel:="[-1.5, -1.0, -2.0]" -p feedback:=OPEN_LOOP -p odom_topic:=/odom -r cmd_vel:=cmd_vel_nav -r cmd_vel_smoothed:=cmd_vel_pre_mux
spawn smlife ros2 run nav2_lifecycle_manager lifecycle_manager --ros-args -r __node:=lifecycle_manager_smoother -p use_sim_time:=true -p autostart:=true -p node_names:="['velocity_smoother']"
spawn planner ros2 run nav2_planner planner_server --ros-args --params-file evaluation/m8_planner.yaml -p use_sim_time:=true
spawn planlife ros2 run nav2_lifecycle_manager lifecycle_manager --ros-args -r __node:=lifecycle_manager_planner -p use_sim_time:=true -p autostart:=true -p node_names:="['planner_server']"
for n_ in smlife planlife; do
  ok_=0; for i in $(seq 30); do grep -q 'Managed nodes are active' "$DIR/$n_.log" 2>/dev/null && { ok_=1; break; }; sleep 1; done
  [ "$ok_" = 1 ] || { say "**$n_ 未回報 active** ⇒ 中止，不空跑"; exit 86; }
done
spawn shield ros2 run ammr_wholebody_mpc scan_safety_shield --ros-args -p use_sim_time:=true -p enable:=true -p scan_topic:=/scan -p cmd_in_topic:=/cmd_vel_pre_shield -p cmd_out_topic:=/cmd_vel -p robot_radius:=0.39 -p self_return_radius:=0.30 -p alpha:=2.0 -p a_brake:=6.25 -p tau:=0.15 -p vx_max:=0.2775 -p vy_max:=0.2775 -p wz_max:=1.1327
spawn bag ros2 bag record -s mcap -o "$DIR/bag" /scan /scan_filtered /gmpc/obstacles /gmpc/static_obstacles /gmpc/tracks_debug /gmpc/diag_v2 /gmpc/plan_ack /plan /odom /cmd_vel_nav /cmd_vel_pre_mux /cmd_vel_pre_shield /cmd_vel /m8nav/state /global_costmap/costmap /handover/state
sleep 3
if [ "${WIRING:-0}" = "1" ]; then
  say "[接線檢查] 不起導航節點、底盤不動；保持 $WIRING_S 秒（牆鐘）後停"
  sleep "$WIRING_S"
else
  spawn nav python3 -u evaluation/m8_nav_obs_node.py --goal-file "$CHK" --out "$DIR/m8nav.json" --nav-timeout-s "$NAV_TIMEOUT_S"
  say "  等導航結果（模擬上限 ${NAV_TIMEOUT_S} s＋對位；牆鐘最多 1800 s）"
  for i in $(seq 1800); do
    r=$(python3 -c "import json;print(json.load(open('$DIR/m8nav.json')).get('result') or '')" 2>/dev/null)
    [ -n "$r" ] && break
    kill -0 "${PIDS[0]}" 2>/dev/null || { say "**模擬器提前結束** ⇒ 收尾"; break; }
    sleep 1
  done
  say "導航結果：${r:-（無）}"
  sleep 3
fi
say "停止世界並封存"
timeout 5 ros2 topic pub --once /room/stop std_msgs/msg/String "{data: 'm8 nav end'}" >/dev/null 2>&1
for i in $(seq 60); do kill -0 "${PIDS[0]}" 2>/dev/null || break; sleep 1; done
for n_ in nav bag; do for i in "${!NAMES[@]}"; do [ "${NAMES[$i]}" = "$n_" ] && kill -INT "${PIDS[$i]}" 2>/dev/null; done; done
sleep 3
say "=== 結束（結果 ${r:-wiring}）==="
[ "${WIRING:-0}" = "1" ] && exit 0
[ "${r:-}" = done ] && exit 0 || exit 1
