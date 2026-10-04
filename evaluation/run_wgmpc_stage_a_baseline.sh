#!/usr/bin/env bash
# 階段 A 的**靜止基線**趟次：量抽屜在這次模擬配置下自己會漂多少。
#
# 取值規則**執行前已定版**：evaluation/results/wgmpc_stage_a_baseline_rule.yaml
# 跑完由 wgmpc_stage_a_baseline_derive.py 依規則產生門檻，**不事後調整規則**。
#
# 這趟**不驗證**：交棒流程（尚未實跑）、協同接近、到達保持、屏障在運動中的行為。
# 它只產生「這次模擬配置的工程偵測門檻」，不是所有情況的漂移上界。
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
RUN_ID="${RUN_ID:-wgmpc_stage_a_baseline_$(date +%H%M%S)}"
DIR="$WS/evaluation/runs/$RUN_ID"; mkdir -p "$DIR"
LOG="$DIR/run.log"; : > "$LOG"
ISAAC_PY="${ISAAC_PY:-$HOME/venvs/isaacsim-6.0.1/bin/python}"
URDF_WB="$WS/evaluation/models/omni_bot_wholebody_expanded.urdf"
ASSET="$WS/src/my_omnibot_description/config/drawer_unit.yaml"
RULE="$WS/evaluation/results/wgmpc_stage_a_baseline_rule.yaml"

# ---- 與階段 A 對齊的條件（來源：wgmpc_stage_a.yaml 與 baseline_rule.yaml）----
DRAWER_POSE="${DRAWER_POSE:-0.0,1.45}"
MODE="solver_drawer"
JOINT_MARGIN="0.05"
PHYSICS_DT="${PHYSICS_DT:-0.01}"
# **觀察時長與階段 A 的模擬時間預算相同**
OBSERVE_SIM_S="${OBSERVE_SIM_S:-60}"
SIM_LIMIT="$OBSERVE_SIM_S"
# 機器人停在**階段 A 的起始姿態**（交棒後那一段的起點）：
# j3 在有效限位中點、其餘五軸零位、底盤於原點。由求解器的限位推出，不寫死。
INIT_ARM_Q="${INIT_ARM_Q:-}"
# ---- 熱況條件（規則檔定版）----
# **起跑線不得調高**。預設取規則檔的上限；環境變數只能往下調，
# 往上調會被 wgmpc_stage_a_baseline_gate.py 擋住（不是靠這裡的預設值把關）。
START_TEMP_C="${START_TEMP_C:-42}"
STAGE_CFG="$WS/src/ammr_wholebody_mpc/config/wgmpc_stage_a.yaml"

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

say "=== 階段 A 靜止基線 RUN_ID=$RUN_ID domain=$ROS_DOMAIN_ID ==="
say "取值規則**執行前已定版**：$(basename "$RULE")"
say "這趟**不驗證**交棒流程、協同接近、到達保持或屏障在運動中的行為"

# ---- 可覆寫參數必須符合事前規則（含「起跑線不得調高」）----
say "[0/4] 參數核對（事前規則：$(basename "$RULE")）"
python3 evaluation/wgmpc_stage_a_baseline_gate.py \
  --rule "$RULE" --stage-config "$STAGE_CFG" \
  --start-temp-c "$START_TEMP_C" --observe-sim-s "$OBSERVE_SIM_S" \
  --physics-dt-s "$PHYSICS_DT" --mode "$MODE" \
  --joint-margin "$JOINT_MARGIN" --drawer-pose "$DRAWER_POSE" \
  --drawer-asset "$ASSET" --out "$DIR/param_gate.json" \
  2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" = "0" ] || { say "**參數核對未通過 ⇒ 不啟動**"; exit 80; }

# ---- 起跑溫度：**已解除為啟動條件**，但仍記錄 ----
# 解除的授權與日期寫在規則檔 start_temp_c 的修訂紀錄（enforced: false）。
# 執行中的 92 °C 熱中止線**不動** —— 那與起跑線是兩回事，且已實測在本支
# 模擬裡讀得到溫度；觸發時 stop_reason 為 monitor_failure，而驗收要求
# sim_limit ⇒ 熱中止的趟次不會產生門檻。
T_NOW=$(python3 evaluation/cpu_temp.py 2>/dev/null | grep -oE '[0-9]+(\.[0-9]+)?' | head -1)
TEMP_ENFORCED=$(python3 - "$RULE" <<'EOF'
import sys, yaml
r = yaml.safe_load(open(sys.argv[1]))
st = (r.get('可覆寫參數的約束') or {}).get('start_temp_c') or {}
print('0' if st.get('enforced') is False else '1')
EOF
)
if [ "$TEMP_ENFORCED" = "1" ]; then
  say "起跑前 CPU ${T_NOW}°C（門檻 ${START_TEMP_C}°C，熱中止線 92°C **不得提高**）"
  if ! python3 -c "import sys;sys.exit(0 if float('${T_NOW:-999}')<=float('$START_TEMP_C') else 1)"; then
    say "**起跑溫度 ${T_NOW}°C 高於門檻 ${START_TEMP_C}°C ⇒ 不啟動**（等機器降溫）"
    exit 78
  fi
else
  say "起跑前 CPU ${T_NOW}°C —— **起跑溫度已解除為啟動條件**（規則檔 enforced: false）；"
  say "  執行中的 92°C 熱中止線仍生效，觸發時該趟不會產生門檻"
fi

# ---- 起始手臂構型：由求解器的有效限位推出 ----
if [ -z "$INIT_ARM_Q" ]; then
  INIT_ARM_Q=$(python3 - "$URDF_WB" "$JOINT_MARGIN" <<'EOF'
import sys
import numpy as np
sys.path.insert(0, 'src/ammr_wholebody_mpc')
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics
K = WholeBodyKinematics.from_urdf_file(sys.argv[1])
lim = np.array(K.joint_limits()); m = float(sys.argv[2])
lo, hi = lim[0, 3:] + m, lim[1, 3:] - m
q = np.zeros(6)
# **只有 j3 需要抬離限位**（零位時向下只剩 0.0099 rad 有效餘裕）；
# 其餘五軸維持零位 —— 與 wgmpc_wg2_zero_start_block.yaml 的起步 B 相同。
q[2] = 0.5 * (lo[2] + hi[2])
assert np.all(q >= lo) and np.all(q <= hi), '起始構型落在有效限位外'
print(','.join(f'{v:.9f}' for v in q))
EOF
) || exit 70
fi
say "起始手臂構型（階段 A 的起點）= $INIT_ARM_Q"
say "**不發任何移動命令**：不起 W-GMPC、不起前置調姿、不起 adapter"

python3 - <<EOF | tee -a "$LOG"
import json
json.dump({
  'kind': 'stage_a_static_baseline',
  'purpose': '量抽屜在本模擬配置下的自身漂移與接觸力讀數，據以產生工程偵測門檻',
  'rule_file': 'evaluation/results/wgmpc_stage_a_baseline_rule.yaml',
  'rule_prereg': True,
  'aligned_with_stage_a': {
    'scene': '同一櫃體、抽屜與機器人；/World/drawer_unit 由 build_usd 建出',
    'drawer_pose_xy': [$(echo $DRAWER_POSE | tr ',' ',')],
    'physics_dt_s': $PHYSICS_DT,
    'observe_sim_s': $OBSERVE_SIM_S,
    'mode': '$MODE',
    'joint_margin': $JOINT_MARGIN,
    'opening_readout': 'opening = DY0 − 抽屜剛體世界 y（與階段 A 同式同時機）',
    'drawer_drive': 'stiffness／damping／max_force 皆 0，讀回核對',
  },
  'robot': {'init_arm_q': '$INIT_ARM_Q', 'base_xy_yaw': [0.0, 0.0, 0.0],
            'commands': '**不發任何移動命令**'},
  'not_validated': ['交棒流程（尚未實跑）', '協同接近', '到達與保持',
                    '屏障在運動中的行為'],
  'scope': '這次模擬配置的工程偵測門檻；**不是**所有情況的漂移上界',
  'thermal': {'start_temp_c': $START_TEMP_C, 'abort_c': 92,
              'note': '熱中止線不得提高；保護觸發即保留結果並停止，不自動重跑'},
}, open('$DIR/run_config.json', 'w'), ensure_ascii=False, indent=1)
print('基線配置已落盤 $DIR/run_config.json')
EOF

# **牆鐘上限在起 Isaac 之前算好，且用浮點安全的方式。**
# 先前寫 `WALL_LIMIT=$((OBSERVE_SIM_S * 20 + 300))` —— shell 的 $(( )) 只做
# 整數運算，合法的非整數加長（例如 90.5）會在**Isaac 已經起來之後**才語法錯誤，
# 而腳本沒有 set -e，接著 `seq "$WALL_LIMIT"` 又會在 set -u 下再炸一次。
WALL_LIMIT=$(python3 - "$OBSERVE_SIM_S" <<'EOF'
import math, sys
v = float(sys.argv[1])
if not math.isfinite(v) or v <= 0:
    raise SystemExit('觀察時長不是有限正數')
print(int(math.ceil(v * 20.0 + 300.0)))
EOF
) || { say "**牆鐘上限算不出來（觀察時長 $OBSERVE_SIM_S 不合法）⇒ 不啟動**"; exit 82; }
case "$WALL_LIMIT" in ''|*[!0-9]*)
  say "**牆鐘上限 '$WALL_LIMIT' 不是正整數 ⇒ 不啟動**"; exit 82;; esac
say "牆鐘上限 ${WALL_LIMIT}s（觀察 ${OBSERVE_SIM_S}s 模擬時間）"

say "[1/4] 起 Isaac（--mode $MODE，場景含抽屜；機器人停在階段 A 起始姿態）"
spawn isaac "$ISAAC_PY" -u evaluation/isaac_wholebody_sim_e2.py \
  --out "$DIR/sim" --mode "$MODE" --sim-limit "$SIM_LIMIT" \
  --physics-dt "$PHYSICS_DT" \
  --drawer-asset "$ASSET" --drawer-pose "$DRAWER_POSE" \
  --init-arm-q "$INIT_ARM_Q" \
  --joint-margin "$JOINT_MARGIN" \
  --solver-label stage_a_baseline \
  --run-label "階段A 靜止基線：抽屜自身漂移與接觸力（機器人停在起始姿態、不發命令）"
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
  say "  ⇒ 以「場景未建起」中止，**不**往下跑 /clock 檢查而把問題錯誤歸因"
  exit 83
fi
say "  場景已建起：$(grep -m1 '^\[wb\] articulation root' "$LOG")"

say "[2/4] 觀察 ${OBSERVE_SIM_S}s 模擬時間（不發命令）"
# **不起任何命令來源**。執行端收不到命令 ⇒ 回報 no_command，抽屜自由漂移。
for i in $(seq "$WALL_LIMIT"); do
  [ -f "$DIR/sim/wb_run.json" ] && break
  if [ $((i % 60)) -eq 0 ]; then
    say "  觀察中… ${i}s（牆鐘上限 ${WALL_LIMIT}s）CPU $(python3 evaluation/cpu_temp.py 2>/dev/null | head -1)"
  fi
  sleep 1
done
EARLY_STOP=0
if [ ! -f "$DIR/sim/wb_run.json" ]; then
  say "**基線未在牆鐘上限內落盤** ⇒ 請執行端受控停止"
  # 這條路徑**一定**是提早中止。仍然請求受控停止是為了把紀錄留下來，
  # **不是**為了接著產生門檻 —— 偏低的門檻比沒有門檻更糟。
  EARLY_STOP=1
  timeout 10 ros2 topic pub --once /wb_sim/stop_request std_msgs/msg/String \
    "{data: '基線觀察已達牆鐘上限，請受控停止並封存'}" >>"$LOG" 2>&1 || true
  for i in $(seq 180); do [ -f "$DIR/sim/wb_run.json" ] && break; sleep 1; done
fi
[ -f "$DIR/sim/wb_run.json" ] || { say "**基線無紀錄，不產生門檻**"; exit 79; }
if [ "$EARLY_STOP" = "1" ]; then
  say "**基線提早中止 ⇒ 不產生門檻**（紀錄已保留於 $DIR/sim/wb_run.json）"
  say "  偏低的門檻會讓階段 A 的任何微小數值都看起來超標；**不自動重跑**"
  exit 79
fi

say "[3/4] 驗收 ＋ 依**執行前定版的規則**產生門檻"
python3 evaluation/wgmpc_stage_a_baseline_derive.py "$DIR/sim/wb_run.json" \
  --rule "$RULE" --observe-sim-s "$OBSERVE_SIM_S" \
  --out "$DIR/drawer_threshold.json" 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" = "0" ] || {
  say "**門檻未產生**（見上方原因；驗收不通過的趟次不得產生門檻）"; exit 77; }

say "[4/4] 門檻已綁定配置（資產內容雜湊、擺放、步長、開度讀法）"
# **路徑用引數傳**，不靠 heredoc 展開：`<<'EOF'`（分隔符加引號）**不會**
# 展開 $DIR，先前這段會去開一個名為字面 "$DIR/..." 的檔案而失敗。
# 腳本沒有 set -e，所以失敗之後仍會往下印「基線結束」⇒ 退出碼要明確檢查。
python3 - "$DIR/drawer_threshold.json" <<'EOF' | tee -a "$LOG"
import json, sys
b = json.load(open(sys.argv[1]))
fp = b['config_fingerprint']
print(f'  資產雜湊 {fp["asset_sha256"][:16]}…')
print(f'  擺放 {fp["drawer_pose_xy"]}  步長 {fp["physics_dt_s"]}')
print(f'  schema {fp["drawer_schema"]}  名稱 {fp["drawer_name"]}')
print(f'  開度讀法 {fp["opening_formula"]}')
print('  階段 A 的判定器會重算這份指紋並逐項比對，不符即拒用')
EOF
[ "${PIPESTATUS[0]}" = "0" ] || {
  say "**門檻檔讀回失敗 ⇒ 不宣稱基線完成**（門檻檔可能損壞或缺指紋）"
  exit 81; }

say "收尾前 CPU $(python3 evaluation/cpu_temp.py 2>/dev/null | head -1)"
say "門檻檔：$DIR/drawer_threshold.json"
say "階段 A 用法：run_wgmpc_stage_a.sh 時設 DRAWER_BASELINE=$DIR/drawer_threshold.json"
say "=== 基線結束 RUN_ID=$RUN_ID ==="
