#!/usr/bin/env bash
# run_m8_doorcap（2026-11-10；Codex 20261110_225452 核可）：門把手靜態 RGB-D 多視點擷取。只擺位＋算圖：不閉爪、不開門、不跑控制、不起 ROS。
#   場景＝m8（mission_room.sdf＋門 v3＋抽屜櫃＋未知箱體），行人清空（mission_obstacles_seed1_nowalkers_doorcap.json；只是感知擷取條件）。
#   由 run_guarded_v3.sh 包住（Isaac 互斥鎖＋92 °C 熱中止）；本腳本另加記憶體上限（systemd scope MemoryMax）與 nice 10；異常不自動重跑。
# 用法：RUN_ID=m8doorcap_1 ROS_DOMAIN_ID=95 VP=evaluation/results/vision/m8_door_viewpoints_r1.json RUNNER=evaluation/run_m8_doorcap.sh bash evaluation/run_guarded_v3.sh
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WS"
: "${RUN_ID:?RUN_ID 未設定}"
: "${VP:?VP（視點登錄檔）未設定}"
ISAAC_PY="${ISAAC_PY:-$HOME/venvs/isaacsim-6.0.1/bin/python}"
WORLD="$WS/src/ammr_bringup/worlds/mission_room.sdf"
DOOR3=src/my_omnibot_description/config/door_local_v3.yaml
DRAWER=src/my_omnibot_description/config/drawer_unit_bar26.yaml
OBS=evaluation/results/mission/mission_obstacles_seed1_nowalkers_doorcap.json
K2_READY='0,0,0.035,0,-1.5708,0'
DIR="$WS/evaluation/runs/$RUN_ID"; mkdir -p "$DIR"
MEM_MAX="${MEM_MAX:-20G}"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$DIR/run.log"; }
sha256sum evaluation/isaac_door_k2_sim_ev9r9d.py evaluation/wrist_door_capture.py evaluation/wrist_v0_capture.py evaluation/m8_door_viewpoints.py "$VP" "$OBS" "$DOOR3" evaluation/run_m8_doorcap.sh > "$DIR/versions.sha256"
say "=== 門把手靜態 RGB-D 擷取：RUN_ID=$RUN_ID；視點 $VP；記憶體上限 $MEM_MAX；nice 10；GUI 開"
setsid systemd-run --user --scope -q -p MemoryMax="$MEM_MAX" -p MemorySwapMax=0 nice -n 10 "$ISAAC_PY" -u evaluation/isaac_door_k2_sim_ev9r9d.py --headless false --world "$WORLD" --door-asset "$DOOR3" \
  --with-drawer --drawer-asset "$WS/$DRAWER" --finger-collision split --start-pose="0.5,-1.5,1.570796" --stow-q="$K2_READY" --run-id "$RUN_ID" --obstacles-file "$WS/$OBS" \
  --wrist-door "$WS/$VP" --wrist-res 640x480 --wrist-hfov 69.0 --wrist-hz 30 --out "$DIR" >> "$DIR/sim.log" 2>&1 &
SIM=$!
sleep 1
P=$(pgrep -f "[i]saac_door_k2_sim_ev9r9d.py.*$RUN_ID" | head -1)
say "  起 sim PID=${P:-$SIM}"
trap 'say "收到 TERM"; [ -n "${P:-}" ] && kill -TERM "$P" 2>/dev/null; sleep 3; [ -n "${P:-}" ] && kill -KILL "$P" 2>/dev/null; exit 143' TERM
trap 'say "收到 INT"; [ -n "${P:-}" ] && kill -TERM "$P" 2>/dev/null; exit 130' INT
wait "$SIM"; rc=$?
say "模擬器結束 rc=$rc"
grep -E "\[wdoor\] 完成|中止|Error|Traceback" "$DIR/sim.log" | tail -5 | tee -a "$DIR/run.log"
exit "$rc"
