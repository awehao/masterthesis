#!/usr/bin/env bash
# 遠距實際取像趟（Codex 20261116）：Isaac 住宅場景＋原導航鏈真節點＋進線監督節點（取像模式）。不閉爪、不綁定、不開門。
#   模擬器＝evaluation/isaac_m8_house_far_capture.py（只提供感測與受控對象介面）；導航鏈＝omni_bot_dynamic.launch.py（與基準相同的開關，另設 GMPC_CMD_TOPIC 讓命令選擇器插在平滑器之前）；
#   監督＝evaluation/far_dry/merge_node.py obs（把手估計來自本趟實際影像；起點第一格有效估計定通過線）。
# 用法：RUN_ID=m8house_far_capture_1 ROS_DOMAIN_ID=74 FREEZE=<凍結清單> RUNNER=evaluation/run_m8_house_far_capture.sh bash evaluation/run_guarded_v3.sh
#   STATIC_S=<秒> ：靜態取像冒煙（不起導航與監督；機器人不動，只取像與偵測）
#   START="x,y,yaw" ARM="六關節" SIM_LIMIT=<模擬秒>
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WS"
: "${RUN_ID:?RUN_ID 未設定}"
: "${FREEZE:?FREEZE（凍結清單）未設定}"
: "${ROS_DOMAIN_ID:?ROS_DOMAIN_ID 未設定}"
export ROS_LOCALHOST_ONLY=1
ISAAC_PY="${ISAAC_PY:-$HOME/venvs/isaacsim-6.0.1/bin/python}"
DIR="$WS/evaluation/runs/$RUN_ID"; mkdir -p "$DIR"
MEM_MAX="${MEM_MAX:-24G}"
START="${START:-1.5,-2.8,1.5708}"; ARM="${ARM:--0.0128,-0.0519,0.086,0.0116,-1.6249,-0.0018}"
STATIC_S="${STATIC_S:-0}"; SIM_LIMIT="${SIM_LIMIT:-150}"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$DIR/run.log"; }
if ! sha256sum -c --quiet "$FREEZE" > "$DIR/freeze_check.log" 2>&1; then
  say "凍結清單核對不符（$FREEZE）⇒ 拒跑：$(head -3 "$DIR/freeze_check.log")"; exit 96
fi
cp "$FREEZE" "$DIR/versions.sha256"
set +u; source /opt/ros/jazzy/setup.bash; source install/setup.bash; set -u
PAT="[s]can_safety_shield|[w]heel_limit_guard|[v]elocity_smoother|[g]mpc_node|[g]mpc_merge_wrap|[p]lanner_server|[w]orld_node.py|[m]erge_node.py"
n=$(pgrep -fc "$PAT" || true)
if [ "$n" != "0" ]; then say "已有導航／乾跑節點在跑（$n 個）⇒ 拒跑"; exit 99; fi
say "=== 遠距取像：RUN_ID=$RUN_ID START=$START STATIC_S=$STATIC_S SIM_LIMIT=$SIM_LIMIT domain=$ROS_DOMAIN_ID；凍結 $FREEZE 全部相符；機器 $(hostname)／$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)／$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
PIDS=()
cleanup(){
  trap - EXIT
  for p in "${PIDS[@]}"; do kill -INT -- "-$p" 2>/dev/null; done
  pkill -INT -f "[g]mpc_merge_wrap" 2>/dev/null
  sleep 4
  for p in "${PIDS[@]}"; do kill -KILL -- "-$p" 2>/dev/null; done
  pkill -KILL -f "[g]mpc_merge_wrap" 2>/dev/null
  left=$(pgrep -fc "$PAT" || true); say "收尾完成；殘留導航節點 $left"
}
trap cleanup EXIT
setsid systemd-run --user --scope -q -p MemoryMax="$MEM_MAX" -p MemorySwapMax=0 env -u VIRTUAL_ENV "$ISAAC_PY" -u evaluation/isaac_m8_house_far_capture.py --out "$DIR" --start "$START" --arm="$ARM" --sim-limit-s "$SIM_LIMIT" --static-s "$STATIC_S" > "$DIR/sim.log" 2>&1 < /dev/null &
SIM=$!
sleep 1
P=$(pgrep -f "[i]saac_m8_house_far_capture.py.*runs/$RUN_ID" | head -1)
say "  起 sim PID=${P:-$SIM}"
trap 'say "收到 TERM"; [ -n "${P:-}" ] && kill -TERM "$P" 2>/dev/null; sleep 3; [ -n "${P:-}" ] && kill -KILL "$P" 2>/dev/null; exit 143' TERM
trap 'say "收到 INT"; [ -n "${P:-}" ] && kill -TERM "$P" 2>/dev/null; exit 130' INT
if [ "$STATIC_S" = "0" ]; then
  ok=0
  for i in $(seq 1 150); do
    kill -0 "$SIM" 2>/dev/null || { say "模擬器在 /clock 出現前已退出"; break; }
    s1=$(timeout 3 ros2 topic echo --once --field clock.sec /clock 2>/dev/null | head -1)
    if [ -n "${s1:-}" ]; then sleep 2; s2=$(timeout 3 ros2 topic echo --once --field clock.sec /clock 2>/dev/null | head -1); [ -n "${s2:-}" ] && [ "$s2" -gt "$s1" ] 2>/dev/null && { ok=1; break; }; fi
    sleep 1
  done
  if [ "$ok" = "1" ]; then
    say "  /clock 前進中；啟動導航鏈（與基準相同的開關＋GMPC_CMD_TOPIC=cmd_vel_nav_raw）"
    SX=$(echo "$START" | cut -d, -f1); SY=$(echo "$START" | cut -d, -f2); SYAW=$(echo "$START" | cut -d, -f3)
    PYS="env -u VIRTUAL_ENV /usr/bin/python3"
    NO_GZ=1 BIGARENA=1 SCENE=mission_house_m8 NO_TRAFFIC=1 HEADING=1 GUARD=1 SHIELD=1 GMPC_CMD_TOPIC=cmd_vel_nav_raw SPAWN_X="$SX" SPAWN_Y="$SY" \
      setsid ros2 launch my_omnibot_description omni_bot_dynamic.launch.py gui:=false use_arm:=true > "$DIR/launch.log" 2>&1 < /dev/null & PIDS+=( $! )
    setsid $PYS -u evaluation/far_dry/logger_node.py "$DIR/ros" > "$DIR/logger.log" 2>&1 < /dev/null & PIDS+=( $! )
    sleep 25
    QZ=$(python3 -c "import math;print(math.sin($SYAW/2))"); QW=$(python3 -c "import math;print(math.cos($SYAW/2))")
    timeout 10 ros2 topic pub -t 3 -r 1 /initialpose geometry_msgs/msg/PoseWithCovarianceStamped "{header: {frame_id: 'map'}, pose: {pose: {position: {x: $SX, y: $SY, z: 0.0}, orientation: {z: $QZ, w: $QW}}}}" > "$DIR/initpose.log" 2>&1 || true
    sleep 6
    mkdir -p "$DIR/ros.side"
    for nd in /gmpc_controller /scan_safety_shield /wheel_limit_guard /scan_obstacle_tracker /velocity_smoother /planner_server /amcl; do
      timeout 25 ros2 param dump "$nd" > "$DIR/ros.side/params$(echo $nd | tr '/' '_').yaml" 2> /dev/null || true
    done
    for nd in planner_server map_server amcl velocity_smoother; do echo "$nd $(timeout 6 ros2 lifecycle get /$nd 2>/dev/null | head -1)" >> "$DIR/ros.side/lifecycle.txt"; done
    if ! grep -q "velocity_smoother active" "$DIR/ros.side/lifecycle.txt" || ! grep -q "planner_server active" "$DIR/ros.side/lifecycle.txt"; then
      say "導航節點未就緒（$(tr '\n' ';' < "$DIR/ros.side/lifecycle.txt")）⇒ 不啟動監督節點，結束模擬"; kill -TERM "${P:-$SIM}" 2>/dev/null
    else
      setsid $PYS -u evaluation/far_dry/merge_node.py "$DIR/ros" obs > "$DIR/merge.log" 2>&1 < /dev/null & PIDS+=( $! )
      say "  監督節點（取像模式）已啟動"
      sleep 12
      setsid timeout $((SIM_LIMIT * 4 + 60)) ros2 topic echo /merge/gmpc/diag_v2 std_msgs/msg/String --field data > "$DIR/ros.merge_diag_v2.txt" 2>/dev/null < /dev/null & PIDS+=( $! )
      for t in /cmd_vel /cmd_vel_nav /cmd_vel_nav_raw /cmd_vel_merge /merge/plan /scan /far/arm_cmd /far/handle_obs; do { echo "== $t"; timeout 10 ros2 topic info -v "$t" 2>/dev/null | grep -E "Node name|Publisher count|Subscription count|Endpoint type"; } >> "$DIR/ros.side/topic_endpoints.txt"; done
    fi
  else
    say "/clock 未前進 ⇒ 結束模擬"; kill -TERM "${P:-$SIM}" 2>/dev/null
  fi
fi
wait "$SIM"; rc=$?
say "模擬器結束 rc=$rc"
grep -aE "^\[fc\] |拒跑|Traceback" "$DIR/sim.log" | tail -30 | cut -c1-600 | tee -a "$DIR/run.log"
grep -a "\[far\]" "$DIR/merge.log" 2>/dev/null | sed 's/.*\[far\]/[far]/' | cut -c1-400 | tee -a "$DIR/run.log"
exit "$rc"
