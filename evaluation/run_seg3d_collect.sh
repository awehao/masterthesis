#!/usr/bin/env bash
# run_seg3d_collect：三維定位採集一個佈置（Isaac 開窗，isaac_bottle_seg3d_collect.py）＋兩個 bottle_seg3d_node（v1、w1，use_sim_time，控制不採用）。
# 由 run_guarded_v3.sh 包 92 °C 熱中止與單一 Isaac 互斥：
#   RUN_ID=seg3d_col_z045_p1 ROS_DOMAIN_ID=95 COLLECT_ARGS="--table-z 0.45 --bottle-x 0.0 --bottle-y 0.42" RUNNER=evaluation/run_seg3d_collect.sh bash evaluation/run_guarded_v3.sh
# 同網域已有任何節點就拒跑（孤兒節點會干擾）；收尾只動本趟程序群組。
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WS"
: "${RUN_ID:?RUN_ID 未設定}"
: "${ROS_DOMAIN_ID:?ROS_DOMAIN_ID 未設定}"
DIR="$WS/evaluation/runs/$RUN_ID"; mkdir -p "$DIR"
unset VIRTUAL_ENV
export PATH=/opt/ros/jazzy/bin:/usr/local/bin:/usr/bin:/bin
set +u; source /opt/ros/jazzy/setup.bash; [ -f "$WS/install/setup.bash" ] && source "$WS/install/setup.bash"; set -u
ISAAC_PY="${ISAAC_PY:-$HOME/venvs/isaacsim-6.0.1/bin/python}"
DL_PY="$WS/.venv-dl0/bin/python"
ASSET=src/my_omnibot_description/config/bottle_s0.yaml
WEIGHTS=evaluation/dl0_ckpt/bottle_seg_v1_continue3/best.pth
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
say "=== 三維定位採集 參數［${COLLECT_ARGS:-無}］ RUN_ID=$RUN_ID 網域 $ROS_DOMAIN_ID；git $(git rev-parse --short HEAD)"
_nodes=$(timeout 10 ros2 node list 2>/dev/null)
if [ -n "$_nodes" ]; then say "網域 $ROS_DOMAIN_ID 已有節點（$_nodes）⇒ 拒跑"; echo 77 > "$DIR/exit_rc.txt"; exit 77; fi
sha256sum evaluation/isaac_bottle_seg3d_collect.py evaluation/run_seg3d_collect.sh evaluation/bottle_seg3d_node.py evaluation/bottle_seg_depth3d.py evaluation/bottle_seg_infer.py \
  evaluation/bottle_v1_cam.py evaluation/bottle_w1_cam.py evaluation/bottle_ros_chain.py evaluation/results/vision/seg3d_collect_design.json "$WEIGHTS" "$ASSET" > "$DIR/versions.sha256"
spawn seg_v1 "$DL_PY" -u evaluation/bottle_seg3d_node.py --weights "$WEIGHTS" --ns v1 --ros-args -p use_sim_time:=true
spawn seg_w1 "$DL_PY" -u evaluation/bottle_seg3d_node.py --weights "$WEIGHTS" --ns w1 --ros-args -p use_sim_time:=true
for i in $(seq 1 60); do
  _n=$(timeout 5 ros2 node list 2>/dev/null | grep -c seg3d)
  [ "$_n" -ge 2 ] && break
  read -t 1 < /dev/zero
done
say "旁路節點就緒 ${_n:-0}／2"
if [ "${_n:-0}" -lt 2 ]; then echo 78 > "$DIR/exit_rc.txt"; say "旁路節點未就緒 ⇒ 拒跑"; exit 78; fi
spawn sim "$ISAAC_PY" -u evaluation/isaac_bottle_seg3d_collect.py --asset "$ASSET" --out "$DIR" --headless false ${COLLECT_ARGS:-}
say "起 sim PID=${PIDS[-1]}"
wait "${PIDS[-1]}"; RC=$?
RC2=$(cat "$DIR/exit_rc.txt" 2>/dev/null || echo 99)
say "模擬結束 rc=$RC exit_rc=$RC2"
exit "$RC2"
