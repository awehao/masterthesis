#!/usr/bin/env bash
# run_m8_fake（2026-11-10；Codex 20261110_183705 第 3 項：短程閉迴路）：run_m8_nav.sh 的導航鏈原樣照跑（scan_relay、追蹤器、封存 gmpc_node、平滑器、
#   planner_server、Shield、導航節點），只把 Isaac 換成二維假世界 m8_fakeworld_node（不模擬接觸力；行人不推人）。不需要 Isaac 互斥鎖。
#   NAV_NODE＝m8_nav_obs_node.py（r1 基準）或 m8_nav_obs_node_r2.py（候選）；START＝x,y,yaw；T0＝世界時鐘起點（行人軌跡同一時鐘）。
# 用法：RUN_ID=m8fake_base_1 ROS_DOMAIN_ID=92 NAV_NODE=m8_nav_obs_node.py START=-0.9012,-3.0010,0.8068 T0=26.0 bash evaluation/run_m8_fake.sh
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WS"
: "${ROS_DOMAIN_ID:?ROS_DOMAIN_ID 未設定}"
: "${RUN_ID:?RUN_ID 未設定}"
: "${NAV_NODE:?NAV_NODE 未設定}"
: "${START:?START 未設定}"
T0="${T0:-0.0}"; T_END="${T_END:-130.0}"; RTF="${RTF:-1.0}"
if [ -f "$WS/install/setup.bash" ]; then set +u; source "$WS/install/setup.bash"; set -u; fi
MAP="$WS/src/ammr_bringup/maps/mission_room.yaml"
OBS=evaluation/results/mission/mission_obstacles_seed1.json
CHK=第八次進度報告素材/資料/m8_scene_check.json
URDF="$WS/evaluation/models/omni_bot_manip.urdf"
DIR="$WS/evaluation/runs/$RUN_ID"; [ -e "$DIR" ] && { echo "趟次目錄已存在 $DIR ⇒ 拒跑"; exit 91; }; mkdir -p "$DIR"
PIDS=(); NAMES=()
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$DIR/run.log"; }
spawn(){ local n="$1"; shift; setsid "$@" >>"$DIR/$n.log" 2>&1 & PIDS+=("$!"); NAMES+=("$n"); sleep 0.2; local g; g=$(ps -o pgid= -p "${PIDS[-1]}" 2>/dev/null | tr -d " "); printf "%s\t%s\t%s\n" "$n" "${PIDS[-1]}" "${g:-NA}" >> "$DIR/procs.tsv"; say "  起 $n PID=${PIDS[-1]} PGID=${g:-NA}"; }
cleanup(){ say "cleanup"; for i in "${!PIDS[@]}"; do kill -TERM -- "-${PIDS[$i]}" 2>/dev/null || kill -TERM "${PIDS[$i]}" 2>/dev/null || true; done; sleep 3
  for i in "${!PIDS[@]}"; do kill -KILL -- "-${PIDS[$i]}" 2>/dev/null || true; done; sleep 1
  : > "$DIR/cleanup_residual.tsv"; for i in "${!PIDS[@]}"; do _r=$(pgrep -g "${PIDS[$i]}" 2>/dev/null | tr '\n' ' '); printf "%s\t%s\t%s\n" "${NAMES[$i]}" "${PIDS[$i]}" "${_r:-none}" >> "$DIR/cleanup_residual.tsv"; done
  say "cleanup 後殘留 $(awk '$3!="none"' "$DIR/cleanup_residual.tsv" | wc -l) 組"; }
trap cleanup EXIT
trap 'say "收到 TERM"; exit 143' TERM
trap 'say "收到 INT"; exit 130' INT
N_OTHER=$(pgrep -f "ROS_DOMAIN_ID=$ROS_DOMAIN_ID" 2>/dev/null | wc -l)
sha256sum src/ammr_wholebody_mpc/ammr_wholebody_mpc/gmpc_node_m8.py evaluation/m8_fakeworld_node.py "evaluation/$NAV_NODE" evaluation/m8_conflict.py evaluation/m8_planner.yaml evaluation/run_m8_fake.sh src/ammr_wholebody_mpc/ammr_wholebody_mpc/gmpc_node.py \
  src/ammr_wholebody_mpc/ammr_wholebody_mpc/gmpc.py src/ammr_wholebody_mpc/ammr_wholebody_mpc/scan_obstacle_tracker.py src/ammr_wholebody_mpc/ammr_wholebody_mpc/scan_safety_shield.py "$OBS" "$CHK" > "$DIR/versions.sha256"
LEAD="${LEAD:-2.0}"
T_START=$(python3 -c "print($T0-$LEAD)")
GO="$DIR/go.flag"
say "=== 假世界短程閉迴路：$NAV_NODE；起點 $START；世界時鐘 T0 $T0；RTF $RTF；RUN_ID=$RUN_ID domain=$ROS_DOMAIN_ID"
spawn world python3 -u evaluation/m8_fakeworld_node.py --start="$START" --t0 "$T_START" --rtf "$RTF" --t-end "$T_END" --obstacles "$OBS" --out "$DIR" --go-file "$GO" ${NO_WALKERS:+--no-walkers}
sleep 2
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
 '-p','heading_enable:=true','-p','wheel_fit_mode:=project','-p','vx_min:=-0.35','-p','ax_max:=1.5','-p','ay_max:=1.0','-p','az_max:=2.0','-p','plan_blend_s:=0.0']+([ '-p','heading_source:=${HEADING_SOURCE:-chord}'] if '${GMPC_MOD:-gmpc_node}'=='gmpc_node_m8' else [])
from ammr_wholebody_mpc.${GMPC_MOD:-gmpc_node} import main; main()"
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
spawn bag ros2 bag record -s mcap -o "$DIR/bag" /shield/diag /m8/collision /gmpc/heading /scan /scan_filtered /gmpc/obstacles /gmpc/static_obstacles /gmpc/tracks_debug /gmpc/diag_v2 /gmpc/plan_ack /plan /odom /cmd_vel_nav /cmd_vel_pre_mux /cmd_vel_pre_shield /cmd_vel /m8nav/state /global_costmap/costmap /handover/state
say "  導航鏈已起；建開始旗標：行人世界時間自 $T_START（＝T0−$LEAD）起走，隨即起導航節點"
touch "$GO"
spawn nav python3 -u "evaluation/$NAV_NODE" --goal-file "$CHK" --out "$DIR/m8nav.json" --nav-timeout-s 120
say "  等導航結果（牆鐘最多 600 s）"
r=""
for i in $(seq 600); do
  r=$(python3 -c "import json;print(json.load(open('$DIR/m8nav.json')).get('result') or '')" 2>/dev/null)
  [ -n "$r" ] && break
  for j in "${!NAMES[@]}"; do [ "${NAMES[$j]}" = world ] && W_PID="${PIDS[$j]}"; done; kill -0 "$W_PID" 2>/dev/null || { say "假世界結束"; break; }
  sleep 1
done
say "導航結果：${r:-（無）}"
sleep 3
for n_ in nav bag world; do for i in "${!NAMES[@]}"; do [ "${NAMES[$i]}" = "$n_" ] && kill -INT "${PIDS[$i]}" 2>/dev/null; done; sleep 1; done
for i in $(seq 15); do kill -0 "$W_PID" 2>/dev/null || break; sleep 1; done
say "=== 結束（結果 ${r:-無}）==="
