#!/usr/bin/env bash
# 主線全開配置（heading＋GUARD＋Shield 新串接）在原大場地跑一個 seed（Codex 20261112_151745 條件授權）。
# 不改 run_bigarena_isaac.sh；另起側錄（補錄 /gmpc/obstacles、/shield/diag 等、各節點參數快照、/cmd_vel 發布者）。
# 用法：bash evaluation/run_bigarena_fullon_seed.sh <seed> <RUN_ID> <凍結清單>
set -u
cd /home/howardchen/masterthesis
SEED="$1"; RID="$2"; FREEZE="$3"
sha256sum -c --quiet "$FREEZE" || { echo '凍結清單不符 ⇒ 拒跑'; exit 96; }
[ -e /tmp/omni_bot_wb.urdf ] || { echo '/tmp/omni_bot_wb.urdf 不存在 ⇒ 拒跑（先跑 expand_wholebody_urdf.sh）'; exit 97; }
for f in /sys/class/hwmon/*/; do [ "$(cat $f/name 2>/dev/null)" = k10temp ] && T=$(cat $f/temp1_input); done
[ "${T:-0}" -gt 1000 ] || { echo '溫度來源無有效讀值 ⇒ 拒跑'; exit 92; }
exec 9>/run/user/1000/masterthesis-isaac-1000.lock
flock -n 9 || { echo 'Isaac 互斥鎖忙碌，未啟動'; exit 95; }
E=(HOME="$HOME" USER="$USER" DISPLAY="${DISPLAY:-:0}" XAUTHORITY="${XAUTHORITY:-}" XDG_RUNTIME_DIR="$XDG_RUNTIME_DIR" PATH=/usr/local/bin:/usr/bin:/bin ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=82)
D=evaluation/runs/isaac_bigarena_gmpc_scan_heading__seed${SEED}__${RID}
env -i "${E[@]}" RUN_DIR="$D" bash evaluation/nav_fullon_side_record.sh 9>&- &
SIDE=$!
env -i "${E[@]}" RUN_ID="$RID" \
  AMMR_OBSTACLE_MODE=scheduled AMMR_TRAJ_FILE="/home/howardchen/masterthesis/src/ammr_bringup/config/dynamic_trajectories_bigarena_traffic_v3.yaml" AMMR_PHASE_YAML="/home/howardchen/masterthesis/src/ammr_bringup/config/dynamic_trajectories_bigarena_traffic_v3.yaml" PHASE_DELTA=30.0 \
  GUARD=1 SHIELD=1 HEADING=1 PLANNER_SCAN=/scan REPLAN=1.0 \
  CPU_LIMIT=92 CPU_THREADS=8 HEADLESS=false CAMERA=false RENDER_HZ=12 VIEW_WIDTH=1600 VIEW_HEIGHT=900 \
  bash evaluation/run_bigarena_isaac.sh gmpc_scan_heading "$SEED" 180
rc=$?
kill -INT $SIDE 2>/dev/null; wait $SIDE 2>/dev/null
exit $rc
