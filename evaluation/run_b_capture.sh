#!/usr/bin/env bash
# 主線 B 無接觸取像趟（Codex 20261117_034316／054848；一條候選、一趟；不閉爪、不綁定、不開門）。
#   模擬器＝evaluation/isaac_m8_house_b_capture.py（物理依牆鐘固定 RTF、不等計算；偵測在啟動前 fork 的獨立程序）
#   導航＝omni_bot_dynamic.launch.py：同基準開關（NO_GZ BIGARENA SCENE=mission_house_m8 NO_TRAFFIC HEADING GUARD SHIELD）＋GMPC_TASK=1（任務擴充子類別：朝向、速度框、區域模式、預測發布）
#         ＋GMPC_CMD_TOPIC=cmd_vel_nav_raw（監督節點轉送到 /cmd_vel_nav；受控停止時送零）；導航參數未改
#   監督＝evaluation/motm_b/b_capture_node2.py（走廊、觀察、線上重算過渡、每拍完整停止核對、45 ms 牆鐘期限）
# 用法：RUN_ID=m8house_b_capture_1 ROS_DOMAIN_ID=76 FREEZE=<凍結清單> RUNNER=evaluation/run_b_capture.sh bash evaluation/run_guarded_v3.sh
#   START="x,y,yaw" ARM="六關節" SIM_LIMIT=<模擬秒> RTF=<目標即時比>
# 排程（筆電 TUF A14：快核 0–3、10–13，慢核 4–9、14–19）：Isaac ISAAC_CPUS、導航 NAV_CPUS、監督工作程序＋池 SUP_CPUS、規劃程序＋池 B_PLANNER_CPUS
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
SIM_LIMIT="${SIM_LIMIT:-90}"; RTF="${RTF:-0.3}"
PLATFORM="${PLATFORM:-laptop}"
case "$PLATFORM" in                                                  # 平台排程（另登錄、凍結）：laptop＝TUF A14（快核 0–3、10–13）；desktop＝9800X3D（邏輯核 i 與 i+8 同實體核，lscpu -e 核過）
  desktop) D_ISAAC=0,8,1,9; D_NAV=2,10,7,15; D_SUP=3,11,4,12; D_PL=5,13,6,14 ;;
  *)       D_ISAAC=0,10,4,5,14,15; D_NAV=6,7,16,17; D_SUP=1,2,3,11; D_PL=12,13,8,9 ;;
esac
ISAAC_CPUS="${ISAAC_CPUS:-$D_ISAAC}"; NAV_CPUS="${NAV_CPUS:-$D_NAV}"; SUP_CPUS="${SUP_CPUS:-$D_SUP}"; export B_PLANNER_CPUS="${B_PLANNER_CPUS:-$D_PL}"
export B_RTF="$RTF" B_SWEEP_S="${B_SWEEP_S:-0.030}" B_POOL_N="${B_POOL_N:-3}" B_POOL_N_PL="${B_POOL_N_PL:-2}"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$DIR/run.log"; }
if ! sha256sum -c --quiet "$FREEZE" > "$DIR/freeze_check.log" 2>&1; then
  say "凍結清單核對不符（$FREEZE）⇒ 拒跑：$(head -3 "$DIR/freeze_check.log")"; exit 96
fi
cp "$FREEZE" "$DIR/versions.sha256"
set +u; source /opt/ros/jazzy/setup.bash; source install/setup.bash; set -u
PAT="[s]can_safety_shield|[w]heel_limit_guard|[v]elocity_smoother|[g]mpc_node|[g]mpc_task_node|[p]lanner_server|[w]orld_node.py|[m]erge_node.py|[b]_capture_node"
n=$(pgrep -fc "$PAT" || true)
if [ "$n" != "0" ]; then say "已有導航／乾跑節點在跑（$n 個）⇒ 拒跑"; exit 99; fi
say "=== 主線 B 取像：RUN_ID=$RUN_ID START=$START SIM_LIMIT=$SIM_LIMIT RTF=$RTF domain=$ROS_DOMAIN_ID；凍結 $FREEZE 全部相符；機器 $(hostname)／$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)／$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)；電源 $(powerprofilesctl get 2>/dev/null)"
say "平台 $PLATFORM 排程：Isaac $ISAAC_CPUS；導航 $NAV_CPUS；監督 $SUP_CPUS（池 $B_POOL_N）；規劃 $B_PLANNER_CPUS（池 $B_POOL_N_PL）；每拍期限 45 ms、掃掠截止 $B_SWEEP_S s"
PIDS=()
cleanup(){
  trap - EXIT
  for p in "${PIDS[@]}"; do kill -INT -- "-$p" 2>/dev/null; done
  sleep 4
  for p in "${PIDS[@]}"; do kill -KILL -- "-$p" 2>/dev/null; done
  left=$(pgrep -fc "$PAT" || true); say "收尾完成；殘留導航節點 $left"
}
trap cleanup EXIT
setsid systemd-run --user --scope -q -p MemoryMax="$MEM_MAX" -p MemorySwapMax=0 taskset -c "$ISAAC_CPUS" env -u VIRTUAL_ENV "$ISAAC_PY" -u evaluation/isaac_m8_house_b_capture.py --out "$DIR" --start "$START" --arm="$ARM" --sim-limit-s "$SIM_LIMIT" --rtf "$RTF" > "$DIR/sim.log" 2>&1 < /dev/null &
SIM=$!
sleep 1
P=$(pgrep -f "[i]saac_m8_house_b_capture.py.*runs/$RUN_ID" | head -1)
say "  起 sim PID=${P:-$SIM}"
trap 'say "收到 TERM"; [ -n "${P:-}" ] && kill -TERM "$P" 2>/dev/null; sleep 3; [ -n "${P:-}" ] && kill -KILL "$P" 2>/dev/null; exit 143' TERM
trap 'say "收到 INT"; [ -n "${P:-}" ] && kill -TERM "$P" 2>/dev/null; exit 130' INT
ok=0
for i in $(seq 1 150); do
  kill -0 "$SIM" 2>/dev/null || { say "模擬器在 /clock 出現前已退出"; break; }
  s1=$(timeout 3 ros2 topic echo --once --field clock.sec /clock 2>/dev/null | head -1)
  if [ -n "${s1:-}" ]; then sleep 2; s2=$(timeout 3 ros2 topic echo --once --field clock.sec /clock 2>/dev/null | head -1); [ -n "${s2:-}" ] && [ "$s2" -gt "$s1" ] 2>/dev/null && { ok=1; break; }; fi
  sleep 1
done
if [ "$ok" = "1" ]; then
  say "  /clock 前進中；啟動導航鏈"
  SX=$(echo "$START" | cut -d, -f1); SY=$(echo "$START" | cut -d, -f2); SYAW=$(echo "$START" | cut -d, -f3)
  PYS="env -u VIRTUAL_ENV PATH=/usr/local/bin:/usr/bin:/bin /usr/bin/python3"
  NO_GZ=1 BIGARENA=1 SCENE=mission_house_m8 NO_TRAFFIC=1 HEADING=1 GUARD=1 SHIELD=1 GMPC_TASK=1 GMPC_CMD_TOPIC=cmd_vel_nav_raw SPAWN_X="$SX" SPAWN_Y="$SY" \
    setsid taskset -c "$NAV_CPUS" ros2 launch my_omnibot_description omni_bot_dynamic.launch.py gui:=false use_arm:=true > "$DIR/launch.log" 2>&1 < /dev/null & PIDS+=( $! )
  setsid taskset -c "$NAV_CPUS" $PYS -u evaluation/far_dry/logger_node.py "$DIR/ros" > "$DIR/logger.log" 2>&1 < /dev/null & PIDS+=( $! )
  sleep 15
  for i in $(seq 40); do timeout 6 ros2 lifecycle get /amcl 2>/dev/null | grep -q "^active" && break; sleep 2; done
  QZ=$(python3 -c "import math;print(math.sin($SYAW/2))"); QW=$(python3 -c "import math;print(math.cos($SYAW/2))")
  timeout 10 ros2 topic pub -t 3 -r 1 /initialpose geometry_msgs/msg/PoseWithCovarianceStamped "{header: {frame_id: 'map'}, pose: {pose: {position: {x: $SX, y: $SY, z: 0.0}, orientation: {z: $QZ, w: $QW}}}}" > "$DIR/initpose.log" 2>&1 || true
  sleep 3
  mkdir -p "$DIR/ros.side"
  for nd in planner_server map_server amcl velocity_smoother; do echo "$nd $(timeout 6 ros2 lifecycle get /$nd 2>/dev/null | head -1)" >> "$DIR/ros.side/lifecycle.txt"; done
  for nd in /gmpc_controller /scan_safety_shield /wheel_limit_guard /velocity_smoother /planner_server /amcl; do
    timeout 25 ros2 param dump "$nd" > "$DIR/ros.side/params$(echo $nd | tr '/' '_').yaml" 2> /dev/null || true
  done
  if ! grep -q "velocity_smoother active" "$DIR/ros.side/lifecycle.txt" || ! grep -q "planner_server active" "$DIR/ros.side/lifecycle.txt" || ! grep -q "amcl active" "$DIR/ros.side/lifecycle.txt"; then
    say "導航節點未就緒（$(tr '\n' ';' < "$DIR/ros.side/lifecycle.txt")）⇒ 不啟動監督節點，結束模擬"; kill -TERM "${P:-$SIM}" 2>/dev/null
  else
    setsid taskset -c "$SUP_CPUS" $PYS -u evaluation/motm_b/b_capture_node2.py "$DIR/ros_bcap" > "$DIR/sup.log" 2>&1 < /dev/null & PIDS+=( $! )
    say "  監督節點已啟動"
    sleep 8
    for t in /cmd_vel /cmd_vel_nav /cmd_vel_nav_raw /plan /scan /far/arm_cmd /far/handle_obs /task/conditions /task/region_status /task/gmpc_pred; do { echo "== $t"; timeout 10 ros2 topic info -v "$t" 2>/dev/null | grep -E "Node name|Publisher count|Subscription count|Endpoint type"; } >> "$DIR/ros.side/topic_endpoints.txt"; done
    for pp in $(pgrep -f "[v]elocity_smoother|[s]can_safety_shield|[w]heel_limit_guard|[g]mpc_task_node|[p]lanner_server|[a]mcl|[b]_capture_node2"); do echo "$(taskset -cp $pp 2>/dev/null) $(ps -o comm= -p $pp)"; done > "$DIR/ros.side/affinity.txt" 2>/dev/null
  fi
else
  say "/clock 未前進 ⇒ 結束模擬"; kill -TERM "${P:-$SIM}" 2>/dev/null
fi
wait "$SIM"; prc=$?
if [ -f "$DIR/exit_code.txt" ]; then rc=$(cat "$DIR/exit_code.txt"); else rc=72; fi
[ "$prc" != "0" ] && [ "$rc" = "0" ] && rc="$prc"
say "模擬器結束 行程碼=$prc 執行器離開碼=$rc（0 正常；70 模擬工程中止／保護鎖存；71 例外 partial；72 無離開碼）"
grep -aE "^\[fc\] |拒跑|Traceback" "$DIR/sim.log" | tail -30 | cut -c1-500 | tee -a "$DIR/run.log"
grep -a "\[bcap2\]" "$DIR/sup.log" 2>/dev/null | grep -vE '"what": "retr"' | cut -c1-400 | tail -25 | tee -a "$DIR/run.log"
exit "$rc"
