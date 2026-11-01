#!/usr/bin/env bash
# run_bottle_ros（N1，2026-11-01；Codex 條件核可一趟）：瓶子任務走真 ROS 話題。獨立節點：RSP、map→odom、map_server、scan_relay、追蹤器（bottle_tracker_stamped）、Shield；模擬端同程序發 /clock /odom /tf /scan_raw。
# Isaac：RUN_ID=bottle_n1_1 ROS_DOMAIN_ID=95 RUNNER=evaluation/run_bottle_ros.sh bash evaluation/run_guarded_v3.sh
# 假世界驗收（真節點、無 Isaac）：RUN_ID=bottle_n1_fake1 ROS_DOMAIN_ID=93 SIM_MODE=fake FAKE_MODE=full bash evaluation/run_bottle_ros.sh
# 同一 Isaac 程序開視窗持續更新；不錄影、不逐步寫圖。收尾只動本趟程序群組。
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WS"
: "${RUN_ID:?RUN_ID 未設定}"
: "${ROS_DOMAIN_ID:?ROS_DOMAIN_ID 未設定}"
SIM_MODE="${SIM_MODE:-isaac}"
DIR="$WS/evaluation/runs/$RUN_ID"; mkdir -p "$DIR"
unset VIRTUAL_ENV
export PATH=/opt/ros/jazzy/bin:/usr/local/bin:/usr/bin:/bin
set +u; source /opt/ros/jazzy/setup.bash; [ -f "$WS/install/setup.bash" ] && source "$WS/install/setup.bash"; set -u
ISAAC_PY="${ISAAC_PY:-$HOME/venvs/isaacsim-6.0.1/bin/python}"
ASSET=src/my_omnibot_description/config/bottle_s0.yaml
URDF="$WS/evaluation/models/omni_bot_manip.urdf"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$DIR/run.log"; }
PIDS=(); NAMES=()
: > "$DIR/procs.tsv"
spawn(){ local n="$1"; shift; setsid "$@" >>"$DIR/$n.log" 2>&1 & PIDS+=("$!"); NAMES+=("$n"); sleep 0.2; local g; g=$(ps -o pgid= -p "${PIDS[-1]}" 2>/dev/null | tr -d ' '); printf "%s\t%s\t%s\n" "$n" "${PIDS[-1]}" "${g:-?}" >> "$DIR/procs.tsv"; }
cleanup(){ say "cleanup（只針對本趟 PID；TERM → 3 s → KILL）"
  for i in "${!PIDS[@]}"; do kill -TERM -- "-${PIDS[$i]}" 2>/dev/null; done
  read -t 3 < /dev/zero
  for i in "${!PIDS[@]}"; do kill -KILL -- "-${PIDS[$i]}" 2>/dev/null; done
  read -t 1 < /dev/zero; : > "$DIR/cleanup_residual.tsv"
  for i in "${!PIDS[@]}"; do _r=$(pgrep -g "${PIDS[$i]}" 2>/dev/null | tr '\n' ' '); printf "%s\t%s\t%s\n" "${NAMES[$i]}" "${PIDS[$i]}" "${_r:-none}" >> "$DIR/cleanup_residual.tsv"; done
  say "cleanup 後殘留 $(awk '$3!="none"' "$DIR/cleanup_residual.tsv" | wc -l) 組"; }
trap cleanup EXIT
trap 'say "收到 TERM"; exit 143' TERM
trap 'say "收到 INT"; exit 130' INT
say "=== 瓶子 N1 真話題鏈路 模式 $SIM_MODE 參數［${S1_ARGS:-無}］ RUN_ID=$RUN_ID 網域 $ROS_DOMAIN_ID；git $(git rev-parse --short HEAD)"
sha256sum evaluation/isaac_bottle_ros.py evaluation/bottle_ros_chain.py evaluation/bottle_tracker_stamped.py evaluation/bottle_map_write.py evaluation/bottle_s1_ctrl.py evaluation/bottle_s1_groups.py evaluation/motm_b1_port.py evaluation/wg4b_b1_core.py evaluation/bottle_s0b_ctrl.py evaluation/bottle_branch.py evaluation/bottle_s0_ctrl.py evaluation/bottle_s0_logic.py evaluation/bottle_ref.py src/ammr_wholebody_mpc/ammr_wholebody_mpc/scan_obstacle_tracker.py src/ammr_wholebody_mpc/ammr_wholebody_mpc/scan_safety_shield.py src/ammr_wholebody_mpc/ammr_wholebody_mpc/gmpc_wb.py src/ammr_wholebody_mpc/ammr_wholebody_mpc/gmpc.py evaluation/run_bottle_ros.sh evaluation/bottle_n2_path.py evaluation/bottle_n2_ctrl.py evaluation/bottle_n2_ros.py evaluation/bottle_n2_planner.yaml evaluation/bottle_ros_fake.py evaluation/isaac_bottle_n2.py evaluation/bottle_place_ctrl.py evaluation/bottle_place_ros.py evaluation/bottle_place_fake.py evaluation/isaac_bottle_place.py evaluation/isaac_bottle_v1.py evaluation/isaac_bottle_v1c.py evaluation/bottle_v2_ctrl.py evaluation/bottle_v1_cam.py evaluation/bottle_v1_est.py evaluation/bottle_v1_est_node.py evaluation/isaac_bottle_v2.py evaluation/bottle_v2_link.py evaluation/isaac_bottle_w1c.py evaluation/bottle_w1_cam.py evaluation/isaac_bottle_w1.py evaluation/bottle_w1_ctrl.py evaluation/isaac_bottle_w1p.py evaluation/bottle_w1_parked_ctrl.py evaluation/isaac_bottle_w1ho.py evaluation/bottle_w1_handover_ctrl.py evaluation/isaac_bottle_w1dyn.py evaluation/bottle_w1_nopred_ctrl.py evaluation/bottle_w1_dyn_case.py evaluation/isaac_bottle_w1dynm.py evaluation/bottle_monitor_progress.py evaluation/isaac_bottle_w1dynb.py evaluation/bottle_w1_box_ctrl.py evaluation/isaac_bottle_w1dynb2.py evaluation/bottle_w1_box2_ctrl.py evaluation/isaac_bottle_w1dynl.py evaluation/bottle_w1_liftb_ctrl.py evaluation/isaac_bottle_w1dynlc.py evaluation/bottle_w1_liftc_ctrl.py evaluation/bottle_w1_liftc_fake.py evaluation/isaac_bottle_w1dynsr.py evaluation/bottle_w1_shrow_ctrl.py evaluation/bottle_w1_shrow_fake.py evaluation/isaac_bottle_w1dynsr2.py evaluation/isaac_bottle_w1dynsr3.py evaluation/bottle_w1_shrow_np_ctrl.py evaluation/isaac_bottle_w1dynsr4.py evaluation/isaac_bottle_w1dynsr5.py evaluation/bottle_tracker_kfclock.py evaluation/bottle_tracker_stamped_kfc.py evaluation/bottle_tracker_stamped_kfcm.py evaluation/isaac_bottle_w1dynsr6.py evaluation/isaac_bottle_w1dynsr7.py "$ASSET" > "$DIR/versions.sha256"
case "${ISAAC_SCRIPT:-}${FAKE_SCRIPT:-}" in                      # 有第二桌的場景（放置、V1、V2、W1）：地圖必須含第二桌腳，否則追蹤器把桌腳當未知物（bottle_w1c_1 漏帶 ⇒ 假動態障礙）
  *place*|*_v1*|*_v2*|*_w1*)
    case "${MAP_ARGS:-}" in *--table2*) ;; *) say "場景有第二桌但 MAP_ARGS 未帶 --table2 ⇒ 拒跑"; echo 76 > "$DIR/exit_rc.txt"; exit 76;; esac;;
esac
python3 evaluation/bottle_map_write.py "$DIR/map" ${S1_ARGS:-} ${MAP_ARGS:-} | tee -a "$DIR/run.log"
PRM=$(python3 - <<'PY'
import sys; sys.path.insert(0,'evaluation')
import bottle_nav_chain as N
def f(d): return ' '.join(f"-p {k}:={str(v).lower() if isinstance(v,bool) else v}" for k,v in d.items())
print(f(dict(N.TRACKER, use_map_subtraction=True))); print(f(N.SHIELD))
PY
)
TR_P=$(sed -n 1p <<<"$PRM"); SH_P=$(sed -n 2p <<<"$PRM")
say "追蹤器參數 $TR_P；節點 ${TRACKER_SCRIPT:-evaluation/bottle_tracker_stamped.py}"; say "Shield 參數 $SH_P"
spawn rsp ros2 run robot_state_publisher robot_state_publisher "$URDF" --ros-args -p use_sim_time:=true
spawn tf_map ros2 run tf2_ros static_transform_publisher --x 0 --y 0 --z 0 --qx 0 --qy 0 --qz 0 --qw 1 --frame-id map --child-frame-id odom --ros-args -p use_sim_time:=true
spawn map ros2 run nav2_map_server map_server --ros-args -p use_sim_time:=true -p yaml_filename:="$DIR/map.yaml"
spawn maplife ros2 run nav2_lifecycle_manager lifecycle_manager --ros-args -r __node:=lifecycle_manager_map -p use_sim_time:=true -p autostart:=true -p node_names:="['map_server']"
spawn relay ros2 run ammr_bringup scan_relay --ros-args -p use_sim_time:=true
spawn tracker python3 -u "${TRACKER_SCRIPT:-evaluation/bottle_tracker_stamped.py}" --ros-args -p use_sim_time:=true -p scan_topic:=/scan -p map_topic:=/map $TR_P
spawn shield python3 -u -c "import sys; sys.path.insert(0,'src/ammr_wholebody_mpc'); from ammr_wholebody_mpc.scan_safety_shield import main; main()" --ros-args -p use_sim_time:=true -p scan_topic:=/scan -p cmd_in_topic:=/cmd_vel_pre_shield -p cmd_out_topic:=/cmd_vel $SH_P
PLAN_ARGS=""
if [ "${V1:-0}" = 1 ]; then                                    # V1 視覺旁路估計節點（桌子＝地圖先驗）
  V1_TABLE=$(python3 -c "import sys;sys.path.insert(0,'evaluation');import bottle_s1_geom as G;n,c,s=[b for b in G.table_boxes() if b[0]=='table_top'][0];print(f'{c[2]+s[2]/2},{c[0]-s[0]/2},{c[1]-s[1]/2},{c[0]+s[0]/2},{c[1]+s[1]/2}')")
  say "V1 桌子先驗 $V1_TABLE"
  spawn v1est python3 -u evaluation/bottle_v1_est_node.py --table "$V1_TABLE" --ros-args -p use_sim_time:=true
  if [ "${W1:-0}" = 1 ]; then                                  # W1 手腕相機估計節點：同估計器、/w1/* 話題，與底盤分開
    spawn w1est python3 -u evaluation/bottle_v1_est_node.py --table "$V1_TABLE" --ns w1 --ros-args -p use_sim_time:=true
  fi
fi
if [ "${PLAN:-0}" = 1 ]; then                                  # N2：planner_server 只供路徑（SmacPlanner2D），不起控制器
  spawn planner ros2 run nav2_planner planner_server --ros-args --params-file evaluation/bottle_n2_planner.yaml -p use_sim_time:=true
  spawn planlife ros2 run nav2_lifecycle_manager lifecycle_manager --ros-args -r __node:=lifecycle_manager_planner -p use_sim_time:=true -p autostart:=true -p node_names:="['planner_server']"
  PLAN_ARGS="--plan"
fi
if [ "$SIM_MODE" = fake ]; then
  spawn sim python3 -u "${FAKE_SCRIPT:-evaluation/bottle_ros_fake.py}" --out "$DIR" --mode "${FAKE_MODE:-full}" $PLAN_ARGS ${FAKE_GOAL:+--goal="$FAKE_GOAL"}
  say "起 sim PID=${PIDS[-1]}"
else
  spawn sim "$ISAAC_PY" -u "${ISAAC_SCRIPT:-evaluation/isaac_bottle_ros.py}" --asset "$ASSET" --out "$DIR" --headless false ${S1_ARGS:-}
  SIM=${PIDS[-1]}
  say "起 sim PID=$SIM"
  (
    for i in $(seq 1 90); do
      W=$(xdotool search --onlyvisible --name "Isaac Sim" 2>/dev/null | tail -1)
      if [ -n "$W" ]; then
        xdotool windowmap "$W" windowraise "$W" windowactivate "$W" 2>/dev/null
        say "視窗 $W 已置前（第 $i 秒）"
        break
      fi
      kill -0 "$SIM" 2>/dev/null || break
      read -t 1 < /dev/zero
    done
  ) &
fi
wait "${PIDS[-1]}"; RC=$?
RC2=$(cat "$DIR/exit_rc.txt" 2>/dev/null || echo 99)
say "程序離開碼 $RC；腳本結果碼 $RC2（0＝完整成功、3＝已停止且有效、4＝停止未驗／無效、70＝握手未過、72＝無有效路徑未起跑、73＝V1／W1 第一張影格未過、74＝V1／W1 採集不成立、76＝有第二桌的場景地圖未帶第二桌腳、71＝報告序列化失敗（另存 _diag）、99＝無結果檔）"
grep -aE "\[s1\]|\[fake\]" "$DIR/sim.log" | tee -a "$DIR/run.log"
[ "$RC2" = 99 ] && [ "$RC" -ne 0 ] && exit "$RC"
exit "$RC2"
