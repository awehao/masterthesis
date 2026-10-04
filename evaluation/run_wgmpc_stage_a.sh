#!/usr/bin/env bash
# 階段 A：底盤與手臂同動接近抽屜，停在**接觸前暫停位姿**（退讓 0.120 m）。
# 設定見 src/ammr_wholebody_mpc/config/wgmpc_stage_a.yaml。
#
# **兩段分開記錄**
#   [A] 前置調姿  —— 從實際零位走 j3 梯形軌跡。這是**準備動作**，不是成果。
#   [B] 交棒後的階段 A —— 成果判定**從 W-GMPC 接管後的協同接近開始**。
# 兩段的命令都走同一條管線（安全層 → adapter → 執行端 E2），但用不同的
# run_id，所以事後能從封裝紀錄分辨每一筆來自誰。
#
# 清理只針對本趟記錄的 PID，每個子程序自成 process group；不用 pkill。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; WS="$(dirname "$HERE")"
cd "$WS"
set +u
. /opt/ros/jazzy/setup.bash
if [ -f "$WS/install/setup.bash" ]; then . "$WS/install/setup.bash"; else
  set -u; echo "**找不到 install/setup.bash —— 請先 colcon build**"; exit 65; fi
set -u
[ -z "${ROS_DOMAIN_ID:-}" ] && { echo "**ROS_DOMAIN_ID 未設定，拒絕啟動**"; exit 64; }
export ROS_DOMAIN_ID
RUN_ID="${RUN_ID:-wgmpc_stage_a_$(date +%H%M%S)}"
PREPOS_RUN_ID="${PREPOS_RUN_ID:-${RUN_ID}_prepos}"
DIR="$WS/evaluation/runs/$RUN_ID"; mkdir -p "$DIR"
LOG="$DIR/run.log"; : > "$LOG"
ISAAC_PY="${ISAAC_PY:-$HOME/venvs/isaacsim-6.0.1/bin/python}"
URDF_WB="$WS/evaluation/models/omni_bot_wholebody_expanded.urdf"
URDF_TF="$WS/evaluation/models/omni_bot_manip.urdf"
STAGE_CFG="$WS/src/ammr_wholebody_mpc/config/wgmpc_stage_a.yaml"
ASSET="$WS/src/my_omnibot_description/config/drawer_unit.yaml"

# ---- 階段 A 的定版配置（來源：wgmpc_stage_a.yaml，**不在這裡另立一份**）----
DRAWER_POSE="${DRAWER_POSE:-0.0,1.45}"
DRAWER_OPENING="${DRAWER_OPENING:-0.0}"
DRAWER_MODEL="${DRAWER_MODEL:-drawer_unit}"
BACKOFF="${BACKOFF:-0.120}"
MODE="solver_drawer"
JOINT_MARGIN="0.05"
N=5; RATE=20; SIM_LIMIT="${SIM_LIMIT:-180}"
REACH_P=0.005; REACH_R=0.02; HOLD_S=2.0
VMAX_BASE_LIN=0.035255; VMAX_BASE_ANG=0.199900; VMAX_ARM=0.999900
ARM_MODEL="${ARM_MODEL:-setpoint}"
ARM_IDENT="$WS/evaluation/results/wgmpc_arm_sp_ident_free4.json"
DELAY_STATE="${DELAY_STATE:-1.6}"; DELAY_CMD="${DELAY_CMD:-1.0}"
NEAR_GAMMA="${NEAR_GAMMA:-1.0}"
W_S="${W_S:-0.001}"; W_A="${W_A:-0.05}"; W_S_ARM="${W_S_ARM:-0.05}"
MARGIN_GUARD="${MARGIN_GUARD:-1}"
CMD_ENV="${CMD_ENV:-1}"
# **封裝模式要四段都開。** 先前只起了錄製器、卻沒把 --cmd-env／cmd_env:=true
# 傳給 isaac／safety／adapter／節點 ⇒ 前置調姿節點只發封裝，而安全層還在聽
# 舊的九維話題，那 113 筆命令沒有任何人消費、手臂從頭到尾沒動。
# 封裝紀錄當場就看得出來：只有 solver 113 筆，safety／adapter／endpoint 全 0。
if [ "$CMD_ENV" = "1" ]; then
  ENV_NODE="--cmd-env"; ENV_SAFETY="-p cmd_env:=true"
  ENV_ADAPTER="--cmd-env"; ENV_ISAAC="--cmd-env"
else
  ENV_NODE=""; ENV_SAFETY=""; ENV_ADAPTER=""; ENV_ISAAC=""
fi
TASK_SIM_S="${TASK_SIM_S:-60}"; TASK_WALL_S="${TASK_WALL_S:-900}"
# 前置調姿
PREPOS_VEL="${PREPOS_VEL:-0.35}"; PREPOS_ACC="${PREPOS_ACC:-0.7}"
PREPOS_TAIL_S="${PREPOS_TAIL_S:-1.0}"
# **閉迴路前置調姿**：到達容差比交棒的 PREPOS_GOAL_TOL 更緊，
# 這樣交棒那一關不會貼著邊緣過。
PREPOS_KP="${PREPOS_KP:-2.0}"
PREPOS_REACH_TOL="${PREPOS_REACH_TOL:-0.005}"
PREPOS_GOAL="${PREPOS_GOAL:-}"          # 空 = 有效限位中點
# ---- 起始構型怎麼來 ----
#   spawn : **機器人直接生成在起始構型**（--init-arm-q）⇒ **不做 j3 調姿**
#   move  : 從實際零位以閉迴路調姿到起始構型，再交棒（已驗證，保留）
# 為什麼預設 spawn：零位時 j3 向下只剩 0.0099 rad 的有效餘裕，而套用命令
# 落後兩週期 ⇒ 必須先調姿才跑得動。生成在可用構型就不需要那一段。
# **代價**：這樣就不再展示「機器人能自己從零位走到起始構型」——
# 那件事變成明列的**起始條件假設**，見 wgmpc_stage_a.yaml 的 start_config。
START_MODE="${START_MODE:-spawn}"
INIT_ARM_Q="${INIT_ARM_Q:-}"
if [ "$START_MODE" = "spawn" ] && [ -z "$INIT_ARM_Q" ]; then
  INIT_ARM_Q=$(python3 - "$URDF_WB" "$JOINT_MARGIN" <<'EOF'
import sys
import numpy as np
sys.path.insert(0, 'src/ammr_wholebody_mpc')
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics
K = WholeBodyKinematics.from_urdf_file(sys.argv[1])
lim = np.array(K.joint_limits()); m = float(sys.argv[2])
lo, hi = lim[0, 3:] + m, lim[1, 3:] - m
q = np.zeros(6)
# **只有 j3 需要離開限位**；其餘五軸維持零位。
# 與 wgmpc_wg2_zero_start_block.yaml 的起步 B 相同。
q[2] = 0.5 * (lo[2] + hi[2])
assert np.all(q >= lo) and np.all(q <= hi), '起始構型落在有效限位外'
print(','.join(f'{v:.9f}' for v in q))
EOF
) || exit 70
fi
# 交棒
QUIET_S="${QUIET_S:-0.4}"               # 必須 >= max_cmd_age_s 0.2
MAX_STATE_AGE_S="${MAX_STATE_AGE_S:-0.1}"
SP_MEAS_TOL="${SP_MEAS_TOL:-0.01}"
# **與 SP_MEAS_TOL 是不同判準**：前者核設定點與實測角彼此相符，
# 這一條核前置調姿是否真的到達宣告的目標構型。見 wgmpc_stage_a.yaml。
PREPOS_GOAL_TOL="${PREPOS_GOAL_TOL:-0.02}"
# 生成式起步的「手臂確實靜止」上限：套用回報任一分量的絕對值
REST_TOL="${REST_TOL:-0.001}"
HANDOVER_TIMEOUT_S="${HANDOVER_TIMEOUT_S:-60}"
# 錄影
REC_RES="${REC_RES:-1280x720}"; REC_FPS="${REC_FPS:-30}"
REC_AT="${REC_AT:-}"; REC_EYE="${REC_EYE:-}"; REC_FROM="${REC_FROM:-0}"
# 目標標記（**純視覺**，無碰撞體／剛體；模擬端會核對並在帶物理 API 時中止）。
# 空 = 不放。影片要能看出 TCP 有沒有停在接觸前暫停位姿，所以預設放。
REC_TARGET="${REC_TARGET:-}"
# **錄影可關**：REC_ENABLE=0 時完全不傳 --record-*，相機／RTX 那整段不執行。
# 首趟（021149）在相機路徑第一次用於抽屜場景時中止，原因未確立 ⇒
# 要能把「控制驗證」與「錄影」分開跑。
REC_ENABLE="${REC_ENABLE:-1}"
if [ "$REC_ENABLE" = "1" ]; then
  REC_ARGS="--record-frames $DIR/frames --record-res $REC_RES"
  REC_ARGS="$REC_ARGS --record-fps $REC_FPS --record-from $REC_FROM"
  [ -n "$REC_AT" ] && REC_ARGS="$REC_ARGS --record-at $REC_AT"
  [ -n "$REC_EYE" ] && REC_ARGS="$REC_ARGS --record-eye $REC_EYE"
  [ -n "$REC_TARGET" ] && REC_ARGS="$REC_ARGS --record-target $REC_TARGET"
else
  REC_ARGS=""
fi

PIDS=(); NAMES=()
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
cleanup(){ say "cleanup（只針對本趟 PID）..."
  for i in "${!PIDS[@]}"; do kill -TERM -"${PIDS[$i]}" 2>/dev/null || true
                             kill -TERM "${PIDS[$i]}" 2>/dev/null || true; done
  sleep 3
  for i in "${!PIDS[@]}"; do kill -KILL "${PIDS[$i]}" 2>/dev/null || true; done
  say "cleanup done"; }
trap cleanup EXIT
spawn(){ local n="$1"; shift; setsid "$@" >>"$LOG" 2>&1 </dev/null &
         PIDS+=($!); NAMES+=("$n"); say "  起 $n PID=$!"; }
kill_named(){ local n="$1"
  for i in "${!NAMES[@]}"; do
    if [ "${NAMES[$i]}" = "$n" ]; then
      kill -TERM -"${PIDS[$i]}" 2>/dev/null || true
      kill -TERM "${PIDS[$i]}" 2>/dev/null || true
      sleep 1
      kill -KILL -"${PIDS[$i]}" 2>/dev/null || true
      kill -KILL "${PIDS[$i]}" 2>/dev/null || true
      NAMES[$i]="${n}(已終止)"
    fi
  done; }

say "=== 階段 A RUN_ID=$RUN_ID domain=$ROS_DOMAIN_ID ==="
say "起跑前 CPU $(python3 evaluation/cpu_temp.py)"
if [ "$START_MODE" = "spawn" ]; then
  say "起始構型：**直接生成**（START_MODE=spawn）⇒ 起始位置**不去調 j3**"
  say "**成果判定從 W-GMPC 上線後的協同接近開始**"
else
  say "起始構型：從零位閉迴路調姿（START_MODE=move）"
  say "**成果判定從 W-GMPC 接管後的協同接近開始**；前置調姿是準備動作"
fi
say "抽屜 位姿=$DRAWER_POSE 開度=$DRAWER_OPENING 模型=$DRAWER_MODEL；退讓=$BACKOFF"
say "**抽屜位姿取真值 —— 這是階段假設，成果必須標明**"

# ---- 離線前提：先把能在不啟動任何東西的情況下核的都核完 ----
say "[0/9] 離線前提核對（不啟動 Isaac）"
SOLVER_JM=$(python3 -c "import sys;sys.path.insert(0,'src/ammr_wholebody_mpc');
from ammr_wholebody_mpc.wgmpc_core import WGMPCConfig;print(WGMPCConfig().joint_margin)")
# 位姿、開度、退讓、模式、餘裕都由**同一組變數**帶進去，不在這裡寫死：
# 寫死的話覆寫 DRAWER_POSE 時預覽會印出另一個場景的目標。
python3 - "$SOLVER_JM" "$JOINT_MARGIN" "$MODE" "$ASSET" \
         "$DRAWER_POSE" "$DRAWER_OPENING" "$BACKOFF" \
         <<'EOF' 2>&1 | tee -a "$LOG" || exit 70
import sys
sys.path.insert(0, 'evaluation')
import wgmpc_stage_a_entry as E
import drawer_asset as DA
solver_jm, ep_jm, mode, asset, pose_s, opening, backoff = sys.argv[1:8]
spec = DA.load(asset)
pose = tuple(float(v) for v in pose_s.split(','))
E.check_margin_consistency(float(solver_jm), float(ep_jm))
E.check_mode(mode)
p, R, meta = E.compute_pre_contact_pause(spec, pose, float(opening),
                                         float(backoff),
                                         source='離線前提核對')
print(f'  求解器與執行端 joint_margin 一致：{solver_jm} == {ep_jm}')
print(f'  模式 {mode} 已核')
print(f'  接觸前暫停位姿（離線預覽，位姿 {pose} 開度 {opening} 退讓 {backoff}）'
      f'{[round(v,5) for v in p]}')
EOF
# 距離節點的參數**由產生器輸出**，幾何只有一份來源。
# **逐引數讀成陣列**，不用 eval 文字：ros2 的參數陣列需要引號留在詞元裡
#（YAML 解析），eval 會把引號吃掉，續行符還會讓後續的 -p 帶上前導空白。
mapfile -t DIST_ARGS < <(python3 evaluation/gen_drawer_obstacle_params.py \
  --asset "$ASSET" --pose "$DRAWER_POSE" --opening "$DRAWER_OPENING" \
  --model "$DRAWER_MODEL" --argv) || exit 70
[ "${#DIST_ARGS[@]}" -eq 8 ] || {
  say "**距離節點參數引數數不對**：得 ${#DIST_ARGS[@]}，應為 8"; exit 70; }
say "  距離節點參數已由 gen_drawer_obstacle_params.py 產生"
say "  （幾何單一來源，${#DIST_ARGS[@]} 個引數，逐引數傳遞不經 eval）"

python3 - <<EOF | tee -a "$LOG"
import json
json.dump({
  'stage': 'A',
  'stage_config': '$STAGE_CFG',
  'scope': ('成果判定從 W-GMPC 接管後的協同接近開始；'
            '前置調姿是準備動作，單獨記錄'),
  'localisation': '抽屜位姿取真值（階段假設）',
  'occlusion_source': 'scene_truth（明確標記的模擬場景真值）',
  'drawer': {'pose_xy': [$(echo $DRAWER_POSE | tr ',' ',')],
             'opening_m': $DRAWER_OPENING, 'model': '$DRAWER_MODEL',
             'asset': '$ASSET'},
  'pre_contact_pause': {'approach_backoff_m': $BACKOFF,
     'note': '比既有 0.100 m 位姿再退 20 mm；由當步抽屜位姿算出，見 handover.json'},
  'mode': '$MODE', 'joint_margin': $JOINT_MARGIN,
  'solver': {'N': $N, 'rate_hz': $RATE, 'w_s': $W_S, 'w_a': $W_A,
             'w_s_arm': $W_S_ARM, 'near_target_gamma': $NEAR_GAMMA,
             'arm_model': '$ARM_MODEL', 'arm_ident': '$ARM_IDENT'},
  'delay': {'comp_state_cycles': $DELAY_STATE, 'comp_cmd_cycles': $DELAY_CMD},
  'start_mode': '$START_MODE',
  'init_arm_q': '$INIT_ARM_Q' or None,
  'start_mode_note': (
      '**生成式起始構型**：機器人直接生成在起始構型，不做 j3 調姿。'
      '本趟不展示「機器人自己從零位走到起始構型」，那是明列的起始條件假設。'
      '起始閘門是**起始構型核對**：判準組成與交棒判定不同 —— 無前置來源'
      '可核，但多核「起始構型真的生效」與「手臂確實靜止」；'
      'u_prev 與交棒路徑相同，取自實際套用回報。'
      if '$START_MODE' == 'spawn' else
      '前置調姿 → 交棒：u_prev 取自實際套用回報'),
  'prepos': {'vel_max_rps': $PREPOS_VEL, 'acc_max_rps2': $PREPOS_ACC,
             'tail_zero_rate_s': $PREPOS_TAIL_S,
             'goal': '$PREPOS_GOAL' or '有效限位中點',
             'run_id': '$PREPOS_RUN_ID',
             'pipeline': '與 W-GMPC 相同（安全層 → adapter → 執行端 E2）'},
  'handover': {'quiet_s': $QUIET_S, 'max_state_age_s': $MAX_STATE_AGE_S,
               'setpoint_meas_tol_rad': $SP_MEAS_TOL,
               'u_prev': '取自實際套用回報；**不傳 --assume-initial-rest**',
               'rest_tol': $REST_TOL},
  'reach': {'pos_m': $REACH_P, 'rot_rad': $REACH_R, 'hold_s': $HOLD_S},
  'low_speed_box_l1': {'base_lin': $VMAX_BASE_LIN, 'base_ang': $VMAX_BASE_ANG,
                       'arm': $VMAX_ARM},
  'task_sim_s': $TASK_SIM_S, 'task_wall_s': $TASK_WALL_S,
}, open('$DIR/run_config.json', 'w'), ensure_ascii=False, indent=1)
print('啟動配置已落盤 $DIR/run_config.json')
EOF

# **保存本輪的程式版本**：事後要能確認結果出自哪一份程式
{
  echo "git_head=$(git rev-parse HEAD 2>/dev/null)"
  echo "git_dirty=$(git status --porcelain 2>/dev/null | wc -l)"
  for f in evaluation/isaac_wholebody_sim_e2.py evaluation/wgmpc_wg2_node.py \
           evaluation/wgmpc_prepos_node.py evaluation/wgmpc_stage_a_handover.py \
           evaluation/wgmpc_stage_a_start_check.py \
           evaluation/wgmpc_stage_a_entry.py evaluation/wb_cmd_chain_e2.py \
           src/ammr_wholebody_mpc/ammr_wholebody_mpc/wgmpc_handover.py \
           src/ammr_wholebody_mpc/ammr_wholebody_mpc/arm_link_distance.py \
           src/ammr_wholebody_mpc/ammr_wholebody_mpc/wholebody_safety_filter.py \
           src/ammr_wholebody_mpc/config/wgmpc_stage_a.yaml; do
    echo "$(sha256sum "$f" 2>/dev/null | cut -c1-16)  $f"
  done
} > "$DIR/code_versions.txt"
say "  程式版本已保存 $DIR/code_versions.txt"
say "  錄影：$([ "$REC_ENABLE" = "1" ] && echo "啟用（$REC_RES @ ${REC_FPS}fps）" || echo "**關閉**")"

# **紀錄標籤依起始路徑分開**：spawn 趟次沒有前置調姿也沒有交棒，
# 沿用 move 的標籤會讓 wb_run.json 描述一件沒發生的事。
if [ "$START_MODE" = "spawn" ]; then
  RUN_LABEL="階段A（生成式起始構型，不調 j3）：W-GMPC 協同接近至接觸前暫停位姿（退讓 $BACKOFF m，抽屜位姿真值）"
else
  RUN_LABEL="階段A：前置調姿後交棒，W-GMPC 協同接近至接觸前暫停位姿（退讓 $BACKOFF m，抽屜位姿真值）"
fi
say "[1/9] 起 Isaac 執行端（--mode $MODE，**場景含抽屜**）"
# 場景守衛只放行機器人、地面與 /World/drawer 子樹；其他位置有任何碰撞體即中止。
spawn isaac "$ISAAC_PY" -u evaluation/isaac_wholebody_sim_e2.py \
  --out "$DIR/sim" --mode "$MODE" --sim-limit "$SIM_LIMIT" \
  --solver-label wgmpc_stage_a \
  $REC_ARGS \
  --drawer-asset "$ASSET" --drawer-pose "$DRAWER_POSE" \
  $([ -n "$INIT_ARM_Q" ] && echo --init-arm-q "$INIT_ARM_Q") \
  --joint-margin "$JOINT_MARGIN" $ENV_ISAAC \
  --run-label "$RUN_LABEL"
# **等一個只有本程式會印的標記。** 先前比對 'physics'，而 kit 自己的啟動參數
# 就含 `--/physics/cudaDevice=0` ⇒ 第 1 秒就假命中跳出，等待等於沒有等。
# 後果不只是等太短：階段 A 接著跑 /clock 檢查，於是把「場景沒建起來」
# **錯誤歸因**成「/clock 未前進」。
SCENE_WAIT_S="${SCENE_WAIT_S:-240}"
say "  等 Isaac 建好場景（最多 ${SCENE_WAIT_S} s；標記 '[wb] articulation root'）"
SCENE_OK=0
for i in $(seq "$SCENE_WAIT_S"); do
  if grep -q '^\[wb\] articulation root' "$LOG" 2>/dev/null; then SCENE_OK=1; break; fi
  # Isaac 自己中止會印 [wb] **…中止**；不必等滿
  if grep -q '^\[wb\].*中止' "$LOG" 2>/dev/null; then
    say "  **Isaac 自行中止**：$(grep -m1 '^\[wb\].*中止' "$LOG")"
    exit 83
  fi
  if ! kill -0 "${PIDS[0]}" 2>/dev/null; then
    say "  **Isaac 程序已結束**（未印出 articulation root）⇒ 場景未建起"
    exit 83
  fi
  if [ $((i % 30)) -eq 0 ]; then
    say "  …等場景 ${i}s（log $(wc -c <"$LOG") bytes）CPU $(python3 evaluation/cpu_temp.py 2>/dev/null | head -1)"
  fi
  sleep 1
done
if [ "$SCENE_OK" != "1" ]; then
  say "  **場景未在 ${SCENE_WAIT_S} s 內建起**（log 無 '[wb] articulation root'）"
  # **先採證再殺。** 這個卡死是間歇性的，而且 Isaac 連一個字都沒輸出 ——
  # 不留證據就只能一直猜。把程序狀態與 Isaac 自己的日誌抓下來。
  HD="$DIR/isaac_hang"; mkdir -p "$HD"
  IP="${PIDS[0]}"
  {
    echo "=== 時間 $(date -Is) ==="
    echo "=== isaac pid=$IP ==="
    cat "/proc/$IP/status" 2>/dev/null
    echo "=== wchan ==="; cat "/proc/$IP/wchan" 2>/dev/null; echo
    echo "=== stat ==="; cat "/proc/$IP/stat" 2>/dev/null
    echo "=== 執行緒數 ==="; ls "/proc/$IP/task" 2>/dev/null | wc -l
    echo "=== 各執行緒 wchan（前 30）==="
    for th in $(ls "/proc/$IP/task" 2>/dev/null | head -30); do
      printf '%s %s\n' "$th" "$(cat /proc/$IP/task/$th/wchan 2>/dev/null)"
    done
    echo "=== 開啟的檔案（末 25）==="
    ls -l "/proc/$IP/fd" 2>/dev/null | tail -25
    echo "=== 子程序 ==="; ps --ppid "$IP" -o pid,stat,etimes,cmd 2>/dev/null | head
  } > "$HD/proc_state.txt" 2>&1
  for L in "$HOME/.nvidia-omniverse/logs/Kit"/*/*/*.log; do
    [ -f "$L" ] && cp "$L" "$HD/" 2>/dev/null
  done
  ls "$HD" > "$HD/MANIFEST.txt" 2>/dev/null
  say "  已採證 ⇒ $HD（程序狀態、各執行緒 wchan、開啟檔案、Kit 日誌）"
  say "  ⇒ 以「場景未建起」中止，**不**往下跑 /clock 檢查而把問題錯誤歸因"
  exit 83
fi
say "  場景已建起：$(grep -m1 '^\[wb\] articulation root' "$LOG")"

say "[2/9] 等 /clock 前進"
timeout 200 python3 evaluation/clock_advancing.py --discover 180 \
  >>"$LOG" 2>&1 || { echo "**/clock 未前進**" | tee -a "$LOG"; exit 68; }

say "[3/9] 起 robot_state_publisher 與距離節點（**抽屜具名幾何 ＋ 場景真值**）"
spawn rsp ros2 run robot_state_publisher robot_state_publisher "$URDF_TF" \
  --ros-args -p use_sim_time:=true
spawn dist ros2 run ammr_wholebody_mpc arm_link_distance --ros-args \
  -p use_sim_time:=true -p report_frame:=odom -p geometry:=links \
  -p wholebody_urdf:="$URDF_WB" "${DIST_ARGS[@]}"
sleep 10

say "[4/9] 起安全層（**freespace_confirmed:=false —— 場景裡有櫃體與抽屜**）"
# **不要從自由空間趟次複製 freespace_confirmed:=true。**
spawn safety ros2 run ammr_wholebody_mpc wholebody_safety --ros-args \
  -p use_sim_time:=true -p report_frame:=odom -p base_frame:=base_link \
  -p wholebody_urdf:="$URDF_WB" \
  -p vmax_base_lin:="$VMAX_BASE_LIN" -p vmax_base_ang:="$VMAX_BASE_ANG" \
  -p vmax_arm:="$VMAX_ARM" \
  -p freespace_confirmed:=false $ENV_SAFETY
sleep 8

say "[5/9] **啟動核對（執行期讀回）**：真值模式、freespace=false、具名幾何、模式、餘裕"
python3 evaluation/wgmpc_stage_a_entry.py \
  --asset "$ASSET" --pose "$DRAWER_POSE" --opening "$DRAWER_OPENING" \
  --model "$DRAWER_MODEL" \
  --solver-joint-margin "$SOLVER_JM" --endpoint-joint-margin "$JOINT_MARGIN" \
  --mode "$MODE" --out "$DIR/entry_check.json" 2>&1 | tee -a "$LOG" || exit 70

say "[6/9] 起 adapter 與封裝錄製器"
spawn adapter python3 -u evaluation/arm_vel_adapter.py \
  --consumer-node /isaac_wholebody_sim $ENV_ADAPTER
[ "$CMD_ENV" = "1" ] && spawn envrec python3 -u evaluation/wgmpc_env_recorder.py \
  --out "$DIR/cmd_env.jsonl"
sleep 5

# ================= [A] 起始構型：spawn 直接生成 / move 前置調姿 =================
if [ "$START_MODE" = "spawn" ]; then
  say "[7/9] **[A] 起始構型由生成提供**（--init-arm-q=$INIT_ARM_Q）⇒ **不做 j3 調姿**"
  say "  代價：本趟**不展示**機器人自己從零位走到起始構型；那是明列的起始條件假設"
  say "[8/9] **起始構型核對**（**不是交棒判定**：無前置來源可核，但多核起始構型生效與手臂靜止）"
  python3 evaluation/wgmpc_stage_a_start_check.py \
    --urdf "$URDF_WB" --start-q "$INIT_ARM_Q" \
    --max-state-age-s "$MAX_STATE_AGE_S" \
    --start-tol-rad "$PREPOS_GOAL_TOL" \
    --setpoint-meas-tol-rad "$SP_MEAS_TOL" \
    --rest-tol "$REST_TOL" \
    --joint-margin "$JOINT_MARGIN" \
    --timeout-s "$HANDOVER_TIMEOUT_S" \
    --out "$DIR/handover.json" 2>&1 | tee -a "$LOG"
  HO_RC=${PIPESTATUS[0]}
  if [ "$HO_RC" != "0" ]; then
    say "**起始構型核對失敗（rc=$HO_RC）⇒ 不讓 W-GMPC 上線，走受控停止**"
    timeout 10 ros2 topic pub --once /wb_sim/stop_request std_msgs/msg/String \
      "{data: '起始構型核對失敗，請走受控停止並封存'}" >>"$LOG" 2>&1 || true
    sleep 20
    exit 72
  fi
  # **不傳 --assume-initial-rest**：adapter 一上線就持續送零命令，所以
  # u_prev 取得到實際套用回報，與 move 路徑相同，不需要初始靜止假設。
  UPREV_ARGS=()
else
  UPREV_ARGS=()
  say "[7/9] **[A] 前置調姿**（準備動作；run_id=$PREPOS_RUN_ID，走同一條管線）"
  spawn prepos python3 -u evaluation/wgmpc_prepos_node.py \
    --urdf "$URDF_WB" --run-id "$PREPOS_RUN_ID" \
    $([ -n "$PREPOS_GOAL" ] && echo --goal "$PREPOS_GOAL") \
    --vel-max "$PREPOS_VEL" --acc-max "$PREPOS_ACC" --rate "$RATE" \
    --kp "$PREPOS_KP" --reach-tol "$PREPOS_REACH_TOL" \
    --joint-margin "$JOINT_MARGIN" --tail-s "$PREPOS_TAIL_S" \
    --out "$DIR/prepos.json"
  say "  等前置調姿發出完成宣告（最多 120 s）"
  PREPOS_OK=0
  for i in $(seq 120); do
    [ -f "$DIR/prepos.json" ] && { PREPOS_OK=1; break; }
    sleep 1
  done
  if [ "$PREPOS_OK" != "1" ]; then
    say "**前置調姿未在 120 s 內完成** ⇒ 停止，不交棒"
    exit 71
  fi
  # **先解除前置來源，再做交棒判定**：不讓兩個來源同時控制手臂。
  # 記下 PID，交棒閘門會核對它已不存在（把「已解除」從斷言變成可驗證）。
  PREPOS_PID=""
  for i in "${!NAMES[@]}"; do
    [ "${NAMES[$i]}" = "prepos" ] && PREPOS_PID="${PIDS[$i]}"
  done
  say "  解除前置來源（終止 prepos 節點 PID=$PREPOS_PID）"
  kill_named prepos
  sleep 2

  say "[8/9] **交棒判定**（完成宣告 ＋ 安靜 ${QUIET_S}s ＋ 實測／設定點／套用回報有效）"
  python3 evaluation/wgmpc_stage_a_handover.py \
    --urdf "$URDF_WB" --quiet-s "$QUIET_S" \
    --done-file "$DIR/prepos.json" \
    ${PREPOS_PID:+--prepos-pid "$PREPOS_PID"} \
    --max-state-age-s "$MAX_STATE_AGE_S" \
    --setpoint-meas-tol-rad "$SP_MEAS_TOL" \
    --prepos-goal-tol-rad "$PREPOS_GOAL_TOL" \
    --joint-margin "$JOINT_MARGIN" \
    --timeout-s "$HANDOVER_TIMEOUT_S" \
    --out "$DIR/handover.json" 2>&1 | tee -a "$LOG"
  HO_RC=${PIPESTATUS[0]}
  if [ "$HO_RC" != "0" ]; then
    say "**交棒失敗（rc=$HO_RC）⇒ 不讓 W-GMPC 上線，走受控停止**"
    timeout 10 ros2 topic pub --once /wb_sim/stop_request std_msgs/msg/String \
      "{data: '交棒失敗，請走受控停止並封存'}" >>"$LOG" 2>&1 || true
    sleep 20
    exit 72
  fi
fi

# ---- 目標：由**當步抽屜位姿**算出，並記下來源與時間 ----
say "  由當步抽屜位姿算接觸前暫停位姿（六維），記來源與時間"
TARGET=$(python3 - <<EOF
import json, sys
sys.path.insert(0, 'evaluation')
import wgmpc_stage_a_entry as E, drawer_asset as DA
spec = DA.load('$ASSET')
ho = json.load(open('$DIR/handover.json'))
p, R, meta = E.compute_pre_contact_pause(
    spec, tuple(float(v) for v in '$DRAWER_POSE'.split(',')),
    $DRAWER_OPENING, $BACKOFF, stamp_sim_t=ho.get('handover_sim_t'),
    source='交棒時刻的抽屜位姿真值（$DRAWER_POSE，開度 $DRAWER_OPENING）')
E.check_target(p, R, spec, tuple(float(v) for v in '$DRAWER_POSE'.split(',')),
               $DRAWER_OPENING, $BACKOFF)
json.dump(meta, open('$DIR/target.json', 'w'), ensure_ascii=False, indent=1)
print(f'{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}')
EOF
) || exit 70
# **六維目標**：姿態用列優先旋轉矩陣傳，不用 rpy（R_DES 本來就是矩陣定義）
TARGET_ROT=$(python3 - <<'EOF'
import sys
sys.path.insert(0, 'evaluation')
from wgmpc_stage_a_entry import R_DES
print(' '.join(f'{v:.12f}' for v in R_DES.reshape(-1)))
EOF
) || exit 70
say "  目標 TCP = ($TARGET)；姿態 R_DES = ($TARGET_ROT)"
say "  目標來源與時間已落盤 $DIR/target.json"

# =========================== [B] 階段 A ===========================
if [ "$START_MODE" = "spawn" ]; then
  say "[9/9] **[B] 階段 A**：W-GMPC 協同接近（u_prev 取自實際套用回報）"
else
  say "[9/9] **[B] 交棒後的階段 A**：W-GMPC 協同接近（u_prev 取自實際套用回報）"
fi
# **兩條路徑都不傳 --assume-initial-rest**：第一輪的 u_prev 都來自實際套用
# 回報（UPREV_ARGS 在兩邊都是空陣列，留著是為了讓這個前提看得見）。
[ "$ARM_MODEL" = "setpoint" ] && { [ -f "$ARM_IDENT" ] || {
  echo "**找不到辨識檔 $ARM_IDENT**" | tee -a "$LOG"; exit 66; }; }
python3 -u evaluation/wgmpc_wg2_node.py \
  --arm-model "$ARM_MODEL" --arm-ident "$ARM_IDENT" \
  --delay-comp-state-cycles "$DELAY_STATE" \
  --delay-comp-cmd-cycles "$DELAY_CMD" \
  $([ "$CMD_ENV" = "1" ] && echo --use-applied-for-predict) \
  --near-target-gamma "$NEAR_GAMMA" --w-s "$W_S" --w-a "$W_A" \
  --w-s-arm "$W_S_ARM" \
  $([ "$MARGIN_GUARD" = "1" ] && echo --margin-guard) \
  $ENV_NODE --run-id "$RUN_ID" --N "$N" --rate "$RATE" \
  --target $TARGET --target-rot $TARGET_ROT \
  --reach-pos-m "$REACH_P" --reach-rot-rad "$REACH_R" --hold-s "$HOLD_S" \
  --duration-s "$TASK_WALL_S" --duration-sim-s "$TASK_SIM_S" \
  --u-prev-policy strict "${UPREV_ARGS[@]}" \
  --out "$DIR/wg2_out.json" 2>&1 | tee -a "$LOG"

# ---- 受控停止與落盤**先於** cleanup ----
ARCHIVE_WAIT_S="${ARCHIVE_WAIT_S:-180}"
isaac_alive(){ kill -0 "${PIDS[0]}" 2>/dev/null; }
say "[收尾 1/4] 請執行端受控停止（/wb_sim/stop_request）"
if isaac_alive; then
  timeout 10 ros2 topic pub --once /wb_sim/stop_request std_msgs/msg/String \
    "{data: '階段A 節點已結束，請走受控停止與停止觀察後封存'}" \
    >>"$LOG" 2>&1 || say "  （停止請求發布失敗，改等自然收尾）"
else
  # **Isaac 已退出 ⇒ 不徒等它接收停止請求。**
  say "  **Isaac 程序已退出**，不發停止請求也不等它回應"
fi

say "[收尾 2/4] 讓封裝錄製器受控結束並關檔"
if [ "$CMD_ENV" = "1" ]; then
  kill_named envrec
  for i in $(seq 30); do
    grep -q '"type": "summary"' "$DIR/cmd_env.jsonl" 2>/dev/null && break
    sleep 1
  done
  grep -q '"type": "summary"' "$DIR/cmd_env.jsonl" 2>/dev/null \
    && say "  已關檔（summary 已寫入）" \
    || say "  **未在 30 s 內關檔** ⇒ 封存核對會標為追蹤不完整"
fi

say "[收尾 3/4] 等執行端完成封存（上限 ${ARCHIVE_WAIT_S} s）"
for i in $(seq "$ARCHIVE_WAIT_S"); do
  [ -f "$DIR/sim/wb_run.json" ] && break
  if ! isaac_alive; then
    # Isaac 已結束：再等 5 s 讓已寫出的檔案落地，之後不再等
    sleep 5
    [ -f "$DIR/sim/wb_run.json" ] \
      && say "  Isaac 已退出，但 wb_run.json 已落盤" \
      || say "  **Isaac 已退出且無 wb_run.json** ⇒ 不再等待（徒等無意義）"
    break
  fi
  sleep 1
done
python3 evaluation/wgmpc_wg2_archive_check.py "$DIR" \
  $([ "$REC_ENABLE" = "1" ] && echo --expect-recording) \
  --expect-arm-model "$ARM_MODEL" \
  $([ "$CMD_ENV" = "1" ] && echo --expect-cmd-env "$DIR/cmd_env.jsonl") \
  --out "$DIR/archive_check.json" 2>&1 | tee -a "$LOG" || true

say "[收尾 4/4] 抽屜：位移與接觸**分開**判定"
# **門檻必須量過**：DRAWER_BASELINE 指向基線趟次產生的 drawer_threshold.json
#（由 run_wgmpc_stage_a_baseline.sh 依**執行前定版的規則**產出）。
# 沒有基線也沒有顯式門檻時，由檢查器自己拒絕下判定。
#
# **措辭**：超過開度門檻只能說「抽屜位移超過靜止基線」，
# **不能**單憑開度證明夾爪碰到或推了抽屜；接觸看獨立的接觸力／距離證據。
if [ -f "$DIR/sim/wb_run.json" ]; then
  python3 evaluation/wgmpc_stage_a_drawer_check.py "$DIR/sim/wb_run.json" \
    ${DRAWER_BASELINE:+--baseline "$DRAWER_BASELINE"} \
    ${DRAWER_TOL_M:+--tol-m "$DRAWER_TOL_M"} \
    ${DRAWER_CONTACT_TOL_N:+--contact-force-tol-n "$DRAWER_CONTACT_TOL_N"} \
    --out "$DIR/drawer_check.json" 2>&1 | tee -a "$LOG" || \
    say "  **抽屜判定未通過或未完成**（見上方原因；缺少不等於沒被碰動）"
fi
say "收尾前 CPU $(python3 evaluation/cpu_temp.py)"
say "=== 階段 A 結束 RUN_ID=$RUN_ID ==="
