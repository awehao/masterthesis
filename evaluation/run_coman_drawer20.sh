#!/usr/bin/env bash
# **20 mm 協同抽屜操作：主成果首測**
#
# 一趟走完：起動前檢查 → 接近 → 連接 → 拉開 20 mm → 保持 → 釋放 → 退出。
# 檢查不過或保護觸發才停；不拆成一串小型模擬。
#
# 命令鏈：求解節點 → 安全濾波 → adapter → E2 執行端（同一物理步套用 3+6 維）。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; WS="$(dirname "$HERE")"
cd "$WS"
[ -z "${ROS_DOMAIN_ID:-}" ] && { echo "**ROS_DOMAIN_ID 未設定，拒絕啟動**"; exit 64; }
export ROS_DOMAIN_ID
RUN_ID="${RUN_ID:-coman_drawer20_$(date +%H%M%S)}"
DIR="$WS/evaluation/runs/$RUN_ID"; mkdir -p "$DIR"
LOG="$DIR/run.log"; : > "$LOG"
ISAAC_PY="${ISAAC_PY:-$HOME/venvs/isaacsim-6.0.1/bin/python}"
URDF_WB="$WS/evaluation/models/omni_bot_wholebody_expanded.urdf"
SIM_LIMIT="${SIM_LIMIT:-120}"
PAIR_D0="${PAIR_D0:-uflite_finger1:0.010,uflite_finger2:0.010}"
STROKE="${STROKE:-0.020}"
PULL_S="${PULL_S:-4.0}"
PIDS=(); say(){ echo "[$(date +%T)] $*" | tee -a "$LOG"; }
cleanup(){ say "cleanup（只針對本趟 PID）..."
  for p in "${PIDS[@]:-}"; do kill -TERM -"$p" 2>/dev/null; kill -TERM "$p" 2>/dev/null; done
  sleep 3; for p in "${PIDS[@]:-}"; do kill -KILL "$p" 2>/dev/null; done; say "cleanup done"; }
trap cleanup EXIT
spawn(){ local n="$1"; shift; setsid "$@" >>"$LOG" 2>&1 </dev/null & PIDS+=($!); say "  起 $n PID=$!"; }

say "=== 20 mm 協同抽屜操作 主成果首測 RUN_ID=$RUN_ID domain=$ROS_DOMAIN_ID ==="
say "起跑前 CPU $(python3 evaluation/cpu_temp.py)"

say "[1/6] 起動前檢查（版本、規格、配對規則一致性）"
python3 - "$DIR" "$PAIR_D0" <<'PY' | tee -a "$LOG" || exit 2
import hashlib, json, os, sys, yaml
ws=os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if False else os.getcwd()
sha=lambda f: hashlib.sha256(open(f,'rb').read()).hexdigest()[:16]
out, pair = sys.argv[1], sys.argv[2]
sp='evaluation/results/specs'
v1=yaml.safe_load(open(f'{sp}/wb_coman_drawer20_criteria_v1.yaml',encoding='utf-8'))
s1=yaml.safe_load(open(f'{sp}/wb_coman_drawer20_supplement_s1.yaml',encoding='utf-8'))
fails=[]
if v1['status']!='frozen': fails.append('v1 未凍結')
if sha(f'{sp}/wb_coman_drawer20_criteria_v1.yaml')!=s1['references']['criteria_v1_sha256_16']:
    fails.append('v1 sha 與 S1 記錄不符（v1 可能被就地改寫）')
if s1['status']!='approved': fails.append(f"S1 補充規格未核准（status={s1['status']}）")
want={k:float(v) for k,v in s1['pair_avoidance']['override'].items()}
got={k:float(v) for k,v in (x.split(':') for x in pair.split(',') if x.strip())}
if want!=got: fails.append(f'pair_d0 與 S1 不符：{got} vs {want}')
for f_,rec in s1['checkers_sha256_16'].items():
    path=f'evaluation/{f_}' if not f_.startswith('wholebody_safety_filter') else \
         f'src/ammr_wholebody_mpc/ammr_wholebody_mpc/{f_}'
    if sha(path)!=rec: fails.append(f'{f_} sha 與 S1 記錄不符')
print(json.dumps({'checks_failed':fails}, ensure_ascii=False))
open(os.path.join(out,'preflight.json'),'w').write(json.dumps({'failed':fails},ensure_ascii=False))
sys.exit(1 if fails else 0)
PY
say "  起動前檢查通過"

say "[2/6] 啟動 Isaac 執行端（開放底盤、9 維命令、v1 資格、交接觸發連接）"
spawn isaac "$ISAAC_PY" -u evaluation/isaac_coman_drawer_sim.py \
  --out "$DIR/sim" --sim-limit "$SIM_LIMIT" \
  --free-base --cmd-source wb9 --machine --attach-on-handover \
  --post-stop-steps 60
say "[3/6] 等 /clock"
python3 evaluation/clock_advancing.py --discover 180 2>&1 | tee -a "$LOG" || exit 3

say "[4/6] 起感測與安全鏈（障礙物含櫃體與抽屜各部件，**橫桿不列入**）"
mapfile -t OBS < <(python3 evaluation/coman_obstacle_specs.py)
say "  障礙物 ${#OBS[@]} 個（橫桿為設計接觸對象，不列入）"
spawn dist ros2 run ammr_wholebody_mpc arm_link_distance --ros-args \
  -p use_sim_time:=true -p report_frame:=odom -p geometry:=links \
  -p wholebody_urdf:="$URDF_WB" -p obstacles:="[$(IFS=,; echo "${OBS[*]}")]"
spawn safety ros2 run ammr_wholebody_mpc wholebody_safety --ros-args \
  -p use_sim_time:=true -p report_frame:=odom -p base_frame:=base_link \
  -p wholebody_urdf:="$URDF_WB" -p freespace_confirmed:=false \
  -p pair_d0:="[$(echo "$PAIR_D0" | tr ',' ' ' | sed 's/ /,/g')]"
spawn adapter python3 -u evaluation/arm_vel_adapter.py --consumer-node /isaac_drawer_sim
sleep 5

say "[5/6] 起求解節點（接近→連接→拉開→保持→釋放→退出，一趟走完）"
python3 -u evaluation/coman_pull_solver_node.py --out "$DIR/solver_out.json" \
  --stroke-m "$STROKE" --pull-duration-s "$PULL_S" --pair-d0 "$PAIR_D0" \
  2>&1 | tee "$DIR/solver.log" | tee -a "$LOG" >/dev/null

say "[6/6] 等執行端收尾"
for i in $(seq 180); do pgrep -f isaac_coman_drawer_sim.py >/dev/null || break; sleep 1; done
say "收尾後 CPU $(python3 evaluation/cpu_temp.py)"
say "輸出目錄 $DIR"
