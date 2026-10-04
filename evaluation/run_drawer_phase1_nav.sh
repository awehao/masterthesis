#!/usr/bin/env bash
# **Phase 1 定版的 GMPC + CBF（含原始掃描護盾），單純跑一次導航**，在抽屜房場景。
#
# 參數**照抄定版**（evaluation/CHANGELOG_spacetime.md 的定案表 ＋ 2026-08-06
# 的三項公平性修正），不是 launch 檔裡那組較舊的預設：
#   CBF      α 0.5、動態 margin 0.60、靜態 margin 0.38、剪枝 1.2 m / near 6 / stride 2
#   控制     horizon 20 × dt 0.05、Q_xy 15、Qf_mult 5、R 2,2,1、S 15,15,8、無航向參考
#   公平性   vx_min −0.35、加速度 1.5/1.0/2.0
#   護盾     SHIELD=1（2026-08-09 的架構性修正，讀 RAW /scan，接在平滑器之後）
#
# **本場景與原基準的差異，必須隨結果一起講：**
#   1. 定位：原基準用 AMCL＋EKF 對地圖定位；這裡 /odom 是**真值**，沒有 AMCL，
#      map→odom 是單位變換。所以本趟**不含定位誤差**這個因素。
#   2. 規劃：原基準用 nav2 SmacPlanner2D（inflation 0.70、robot_radius 0.33、
#      來源 /scan_filtered、重規劃 1.0 s）；這裡用任務節點產生的**直線計畫**。
#      房間是靜態已知的，直線淨空已用幾何核過，但這仍是**簡化**。
#   3. **這個房間沒有移動障礙物** —— CBF 的動態項沒有對象，實際被行使的只有
#      靜態 margin 與護盾。不得據此談動態避障。
#   4. 模擬器是 Isaac，不是 Gazebo。
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"; cd "$WS"
: "${ROS_DOMAIN_ID:?ROS_DOMAIN_ID 未設定}"
ISAAC_PY="${ISAAC_PY:-$HOME/venvs/isaacsim-6.0.1/bin/python}"
# **要 source 工作區** —— `ros2 run ammr_wholebody_mpc ...` 才找得到 tracker
# 與護盾。前一支 runner 是用 python 直接插路徑跑 gmpc，所以沒踩到這個。
# `set -u` 要暫時關掉：colcon 的 setup.bash 會讀未綁定的 COLCON_TRACE。
# shellcheck disable=SC1091
if [ -f "$WS/install/setup.bash" ]; then set +u; source "$WS/install/setup.bash"; set -u; fi
RUN_ID="${RUN_ID:-phase1_nav_$(date +%H%M%S)}"
DIR="$WS/evaluation/runs/$RUN_ID"; mkdir -p "$DIR"
SIM_LIMIT="${SIM_LIMIT:-180}"
MAP="$WS/src/ammr_bringup/maps/drawer_room.yaml"
URDF="$WS/evaluation/models/omni_bot_manip.urdf"
PIDS=(); NAMES=()
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$DIR/run.log"; }
spawn(){ local n="$1"; shift; "$@" >>"$DIR/$n.log" 2>&1 & PIDS+=("$!"); NAMES+=("$n");
         say "  起 $n PID=${PIDS[-1]}"; }
cleanup(){ say "cleanup（只針對本趟 PID）"
  for i in "${!PIDS[@]}"; do kill -TERM "${PIDS[$i]}" 2>/dev/null || true; done
  sleep 3
  for i in "${!PIDS[@]}"; do kill -KILL "${PIDS[$i]}" 2>/dev/null || true; done; }
trap cleanup EXIT

say "=== Phase 1 GMPC+CBF＋護盾 RUN_ID=$RUN_ID domain=$ROS_DOMAIN_ID ==="
say "差異：/odom 真值無 AMCL；直線計畫非 nav2 planner；**無移動障礙物**；Isaac"
[ -f "$MAP" ] || { say "**找不到地圖 $MAP**"; exit 80; }

spawn sim "$ISAAC_PY" -u evaluation/isaac_drawer_room_sim.py \
  --sim-limit "$SIM_LIMIT" --pub-tf true --cam "${CAM:-false}" --cam-hz 10 \
  --out "$DIR"
say "  等場景建起（最多 180 s）"
for i in $(seq 180); do grep -q '進入主迴圈' "$DIR/sim.log" 2>/dev/null && break; sleep 1; done
grep -q '進入主迴圈' "$DIR/sim.log" 2>/dev/null || { say "**場景未建起**"; exit 83; }
say "  場景已建起"

spawn rsp ros2 run robot_state_publisher robot_state_publisher "$URDF" \
  --ros-args -p use_sim_time:=true
# map→odom：真值定位 ⇒ 單位變換。**這取代 AMCL**，是本趟與原基準的差異之一。
spawn tf_map ros2 run tf2_ros static_transform_publisher \
  --x 0 --y 0 --z 0 --qx 0 --qy 0 --qz 0 --qw 1 \
  --frame-id map --child-frame-id odom --ros-args -p use_sim_time:=true
spawn map ros2 run nav2_map_server map_server --ros-args \
  -p use_sim_time:=true -p yaml_filename:="$MAP"
sleep 3
spawn maplife ros2 run nav2_lifecycle_manager lifecycle_manager --ros-args \
  -r __node:=lifecycle_manager_map \
  -p use_sim_time:=true -p autostart:=true -p node_names:="['map_server']"
spawn relay ros2 run ammr_bringup scan_relay --ros-args -p use_sim_time:=true
sleep 5

say "[感知] scan_obstacle_tracker（定版參數；在 odom 追蹤、以 odom 發布）"
spawn tracker ros2 run ammr_wholebody_mpc scan_obstacle_tracker --ros-args \
  -p use_sim_time:=true \
  -p scan_topic:=/scan -p map_topic:=/map \
  -p global_frame:=odom -p publish_frame:=odom \
  -p use_map_subtraction:=true \
  -p cluster_gap:=0.30 -p max_radius:=1.20 -p min_cluster_pts:=2 \
  -p static_window:=2.0 -p min_net_speed:=0.05 \
  -p surface_max_pts:=4 -p surface_range:=2.5 -p surface_min_sep:=0.20
sleep 4

say "[控制] gmpc_node（**定版數值**：CBF α 0.5／動態 0.60／靜態 0.38）"
spawn gmpc python3 -u -c "
import sys; sys.path.insert(0,'src/ammr_wholebody_mpc')
sys.argv=['gmpc_node','--ros-args','-p','use_sim_time:=true',
 '-p','pose_source:=odom','-p','global_frame:=odom',
 '-p','pose_odom_topic:=/odom','-p','plan_topic:=/plan',
 '-p','cmd_vel_topic:=/cmd_vel_nav',
 '-p','obstacles_topic:=/gmpc/obstacles',
 '-p','cbf_enable:=true',
 '-p','cbf_alpha:=0.5',
 '-p','cbf_safe_margin:=0.60',
 '-p','static_cbf_safe_margin:=0.38',
 '-p','cbf_prune_range:=1.2','-p','cbf_near_steps:=6','-p','cbf_far_stride:=2',
 '-p','horizon:=20','-p','control_frequency:=20.0',
 '-p','Q_xy:=15.0','-p','Qf_mult:=5.0',
 '-p','R_vx:=2.0','-p','R_vy:=2.0','-p','R_w:=1.0',
 '-p','S_vx:=15.0','-p','S_vy:=15.0','-p','S_w:=8.0',
 '-p','heading_enable:=false',
 '-p','vx_min:=-0.35',
 '-p','ax_max:=1.5','-p','ay_max:=1.0','-p','az_max:=2.0',
 '-p','plan_blend_s:=0.0']
from ammr_wholebody_mpc.gmpc_node import main; main()"
sleep 6

say "[平滑器＋護盾] gmpc → cmd_vel_nav → 平滑器 → cmd_vel_pre_shield → 護盾 → cmd_vel"
spawn smoother ros2 run nav2_velocity_smoother velocity_smoother --ros-args \
  -p use_sim_time:=true \
  -p max_velocity:="[0.35, 0.25, 0.80]" -p min_velocity:="[-0.35, -0.25, -0.80]" \
  -p max_accel:="[1.5, 1.0, 2.0]" -p max_decel:="[-1.5, -1.0, -2.0]" \
  -p feedback:=OPEN_LOOP -p odom_topic:=/odom \
  -r cmd_vel:=cmd_vel_nav -r cmd_vel_smoothed:=cmd_vel_pre_shield
# **velocity_smoother 是 lifecycle 節點**，沒有管理器會停在 unconfigured，
# 命令根本不會通過。前一趟就是停在 "Waiting on external lifecycle transitions"。
spawn smlife ros2 run nav2_lifecycle_manager lifecycle_manager --ros-args \
  -r __node:=lifecycle_manager_smoother \
  -p use_sim_time:=true -p autostart:=true \
  -p node_names:="['velocity_smoother']"
sleep 4
spawn shield ros2 run ammr_wholebody_mpc scan_safety_shield --ros-args \
  -p use_sim_time:=true -p enable:=true \
  -p scan_topic:=/scan \
  -p cmd_in_topic:=/cmd_vel_pre_shield -p cmd_out_topic:=/cmd_vel \
  -p robot_radius:=0.30 -p alpha:=2.0 -p a_brake:=6.25 -p tau:=0.15 \
  -p vx_max:=0.2775 -p vy_max:=0.2775 -p wz_max:=1.1327
sleep 5

say "[任務] 發直線計畫（**非 nav2 planner**）並監看"
python3 -u evaluation/drawer_mission_node.py --out "$DIR/mission.json" \
  --glide-entry -1.0 --slow-zone -1.0 --timeout-s 150 \
  2>&1 | tee -a "$DIR/run.log"

say "[收尾] 請模擬器受控停止並封存"
timeout 10 ros2 topic pub --once /room/stop std_msgs/msg/String \
  "{data: 'Phase 1 導航趟次結束，請封存'}" >>"$DIR/run.log" 2>&1 || true
for i in $(seq 90); do [ -f "$DIR/room_run.json" ] && break; sleep 1; done
[ -f "$DIR/room_run.json" ] && say "  封存完成 $(stat -c%s "$DIR/room_run.json") bytes" \
  || say "  **未封存**"
say "=== 結束 ==="
