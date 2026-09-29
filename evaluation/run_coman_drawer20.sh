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
# **總靜態間距**（取代該配對的 d0+eps，不是在 30 mm 上再加 10 mm）。
# 一般規則 d0+eps = 80 mm；eps 單獨 30 mm 已大於設計間距 27.9 mm。
PAIR_GAP="${PAIR_GAP:-uflite_finger1:drawer_front_panel:0.010,uflite_finger2:drawer_front_panel:0.010}"
PAIR_D0="${PAIR_D0:-}"   # 不使用逐配對 d0 覆寫（與 PAIR_GAP 互斥）
CONTACT_PAIRS="${CONTACT_PAIRS:-uflite_finger1:handle_bar:engage|pull|hold|release,uflite_finger2:handle_bar:engage|pull|hold|release}"
# 一般列只留最近障礙物。橫桿被接觸例外刪列後，**任何**其他物件都可能完全沒有列
# （面板 28 mm 合格、支柱 40 mm 違規卻無列可約束），故對兩指補齊「其餘所有配對」。
PAIR_ROWS="${PAIR_ROWS:-uflite_finger1:*,uflite_finger2:*}"
# 免列＝接觸例外對象本身（它仍有一般的最近列，只在允許相位被濾掉）
PAIR_ROWS_EXEMPT="${PAIR_ROWS_EXEMPT:-uflite_finger1:handle_bar,uflite_finger2:handle_bar}"
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
python3 - "$DIR" "$PAIR_D0" "$CONTACT_PAIRS" "$PAIR_ROWS" "$PAIR_ROWS_EXEMPT" "$PAIR_GAP" <<'PY' | tee -a "$LOG" || exit 2
import hashlib, json, os, sys, yaml
ws=os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if False else os.getcwd()
sha=lambda f: hashlib.sha256(open(f,'rb').read()).hexdigest()[:16]
out, pair, cpairs, prows, pexempt, pgap = sys.argv[1:7]
sp='evaluation/results/specs'
v1=yaml.safe_load(open(f'{sp}/wb_coman_drawer20_criteria_v1.yaml',encoding='utf-8'))
s1=yaml.safe_load(open(f'{sp}/wb_coman_drawer20_supplement_s1.yaml',encoding='utf-8'))
fails=[]
if v1['status']!='frozen': fails.append('v1 未凍結')
if sha(f'{sp}/wb_coman_drawer20_criteria_v1.yaml')!=s1['references']['criteria_v1_sha256_16']:
    fails.append('v1 sha 與 S1 記錄不符（v1 可能被就地改寫）')
if s1['status']!='approved': fails.append(f"S1 補充規格未核准（status={s1['status']}）")
# 即使標為 approved，只要還有 blocking 的未決項就一律擋住
for _oi in s1.get('open_issues') or []:
    if str(_oi.get('severity'))=='blocking':
        fails.append(f"S1 未決項 {_oi.get('id')} 仍為 blocking：{_oi.get('title')}")
def _kv(spec):
    out={}
    for x in spec.split(','):
        if x.strip():
            lk,ob,v=x.split(':'); out[f'{lk}|{ob}']=float(v)
    return out
want={k:float(v) for k,v in (s1['pair_avoidance'].get('d0_by_pair') or {}).items()}
got=_kv(pair)
if want!=got: fails.append(f'pair_d0 與 S1 不符：{got} vs {want}')
wg={k:float(v) for k,v in (s1['pair_avoidance'].get('g_by_pair') or {}).items()}
gg=_kv(pgap)
if wg!=gg: fails.append(f'pair_gap 與 S1 不符：{gg} vs {wg}')
if set(wg)&set(want): fails.append(f'同一配對同時設了 d0 與總靜態間距：{sorted(set(wg)&set(want))}')
wcp={k:list(v) for k,v in s1['pair_avoidance']['contact_pairs'].items()}
gcp={}
for x in cpairs.split(','):
    if x.strip():
        lk,ob,ph=x.split(':'); gcp[f'{lk}|{ob}']=[y for y in ph.split('|') if y]
if wcp!=gcp: fails.append(f'contact_pairs 與 S1 不符：{gcp} vs {wcp}')
if not s1['pair_avoidance'].get('bar_stays_in_avoidance'):
    fails.append('S1 未聲明橫桿仍在避碰集合')
_rpr=s1['pair_avoidance']['required_pair_rows']
wpr=sorted(_rpr['pairs'])
gpr=sorted(x.strip() for x in prows.split(',') if x.strip())
if wpr!=gpr: fails.append(f'pair_rows 與 S1 必要配對列不符：{gpr} vs {wpr}')
wex=sorted(_rpr['exempt'])
gex=sorted(x.strip() for x in pexempt.split(',') if x.strip())
if wex!=gex: fails.append(f'pair_rows_exempt 與 S1 不符：{gex} vs {wex}')
# 免列對象必須就是接觸例外對象，不得放過任何沒有例外的物件
if {x.replace(':','|') for x in gex}!=set(wcp):
    fails.append(f'免列清單與 contact_pairs 不一致：{gex} vs {sorted(wcp)}')
_pm=s1['pair_avoidance']['phase_map']
_prod=set(_pm['contact_phases_produced'])
_used=set(p for v in wcp.values() for p in v)
if not _used <= _prod:
    fails.append(f'S1 接觸相位有無產生端者：{sorted(_used - _prod)}')
for f_,rec in s1['checkers_sha256_16'].items():
    path=(f'src/ammr_wholebody_mpc/ammr_wholebody_mpc/{f_}'
          if f_ in ('wholebody_safety_filter.py','arm_link_distance.py',
                    'wholebody_safety_node.py') else f'evaluation/{f_}')
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

say "[4/6] 起感測與安全鏈（障礙物含櫃體、橫桿與抽屜各部件）"
mapfile -t OBS < <(python3 evaluation/coman_obstacle_specs.py)
say "  障礙物 ${#OBS[@]} 個（**含橫桿**；接觸例外只給兩指且限定相位）"
spawn dist ros2 run ammr_wholebody_mpc arm_link_distance --ros-args \
  -p use_sim_time:=true -p report_frame:=odom -p geometry:=links \
  -p wholebody_urdf:="$URDF_WB" -p obstacles:="[$(IFS=,; echo "${OBS[*]}")]" \
  -p pair_rows:="[$(echo "$PAIR_ROWS" | sed 's/,/","/g; s/^/"/; s/$/"/')]" \
  -p pair_rows_exempt:="[$(echo "$PAIR_ROWS_EXEMPT" | sed 's/,/","/g; s/^/"/; s/$/"/')]"
spawn safety ros2 run ammr_wholebody_mpc wholebody_safety --ros-args \
  -p use_sim_time:=true -p report_frame:=odom -p base_frame:=base_link \
  -p wholebody_urdf:="$URDF_WB" -p freespace_confirmed:=false \
  -p pair_gap:="[$(echo "$PAIR_GAP" | sed 's/,/","/g; s/^/"/; s/$/"/')]" \
  -p contact_pairs:="[$(echo "$CONTACT_PAIRS" | sed 's/,/","/g; s/^/"/; s/$/"/')]"
spawn adapter python3 -u evaluation/arm_vel_adapter.py --consumer-node /isaac_drawer_sim
sleep 5

say "[5/6] 起求解節點（接近→連接→拉開→保持→釋放→退出，一趟走完）"
python3 -u evaluation/coman_pull_solver_node.py --out "$DIR/solver_out.json" \
  --stroke-m "$STROKE" --pull-duration-s "$PULL_S" \
  --pair-gap "$PAIR_GAP" --contact-pairs "$CONTACT_PAIRS" \
  2>&1 | tee "$DIR/solver.log" | tee -a "$LOG" >/dev/null

say "[6/6] 等執行端收尾"
for i in $(seq 180); do pgrep -f isaac_coman_drawer_sim.py >/dev/null || break; sleep 1; done
say "收尾後 CPU $(python3 evaluation/cpu_temp.py)"
say "輸出目錄 $DIR"
