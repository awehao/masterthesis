#!/usr/bin/env bash
# 階段 A 的**有界夜間工作計畫**。
#
# 目標：驗證 前置調姿 → 交棒 → W-GMPC 協同接近 → 接觸前暫停並保持。
# 成功後確認可重現性，再補一支影片。
#
# **邊界**（硬性，不在過程中放寬）
#   總牆鐘預算   WALL_BUDGET_S（預設 8 小時）
#   Isaac 啟動   MAX_LAUNCH（預設 5 次）
#   固定不動     控制權重、視界、目標、退距、幾何、限位、驗收門檻
#   保留         92 °C 執行中熱中止線
#
# **階段**（符合條件自動往下；任一階段阻滯即轉入第 5 項並產出報告）
#   1 入口核對與殘留清理（**不重跑已完成的靜止基線**）
#   2 一趟不錄影的階段 A
#   3 成功則再兩趟相同配置、不錄影
#   4 三趟成功則：短相機診斷（10 s，無命令來源）→ 一趟錄影階段 A
#   5 阻滯時的離線診斷與隔日交付（OVERNIGHT_REPORT.md）
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; WS="$(dirname "$HERE")"
cd "$WS"
[ -z "${ROS_DOMAIN_ID:-}" ] && { echo "**ROS_DOMAIN_ID 未設定，拒絕啟動**"; exit 64; }
export ROS_DOMAIN_ID

SESSION="${SESSION:-stage_a_overnight_$(date +%m%d_%H%M%S)}"
OUT="$WS/evaluation/runs/$SESSION"; mkdir -p "$OUT"
SLOG="$OUT/session.log"; : > "$SLOG"
STATE="$OUT/state.json"
WALL_BUDGET_S="${WALL_BUDGET_S:-28800}"      # 8 小時
# **啟動次數上限已撤銷**（Howard 2026-10-03：那是管理限制，不是研究要求）。
# 0 = 不限。保留變數只為了讓紀錄看得出本輪的設定。
MAX_LAUNCH="${MAX_LAUNCH:-0}"
# **趟次之間的降溫門檻**：上一趟峰值 91.0 °C，離 92 °C 中止線只差 1 °C。
# 不以相同熱況立即重跑。中止線本身**不動**。
COOL_TEMP_C="${COOL_TEMP_C:-70}"
COOL_WAIT_S="${COOL_WAIT_S:-1200}"
T0=$(date +%s)
LAUNCHES=0
THRESH="$WS/evaluation/runs/wgmpc_stage_a_baseline_long120_020143/drawer_threshold.json"
ISAAC_PY="${ISAAC_PY:-$HOME/venvs/isaacsim-6.0.1/bin/python}"
ASSET="$WS/src/my_omnibot_description/config/drawer_unit.yaml"
# 錄影取景（只在第 4 階段使用）
REC_AT_V="0.10,0.60,0.50"; REC_EYE_V="2.40,-1.20,1.50"
REC_TARGET_V="0.00000,1.05970,0.55000"

say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$SLOG"; }
elapsed(){ echo $(( $(date +%s) - T0 )); }
left(){ echo $(( WALL_BUDGET_S - $(elapsed) )); }
budget_ok(){ [ "$(left)" -gt "${1:-600}" ]; }
launch_ok(){ [ "$MAX_LAUNCH" -eq 0 ] || [ "$LAUNCHES" -lt "$MAX_LAUNCH" ]; }

cool_down(){
  # 等 CPU 降到門檻以下再起下一趟。**不動 92 °C 中止線**，這只是起跑前的
  # 熱餘裕：上一趟峰值 91.0 °C，立即重跑等於用同一個熱況再賭一次。
  local t n=0
  while :; do
    t=$(python3 evaluation/cpu_temp.py 2>/dev/null | cut -d' ' -f1)
    python3 -c "import sys;sys.exit(0 if float('${t:-999}')<=float('$COOL_TEMP_C') else 1)"       && { say "  降溫完成 ${t}°C <= ${COOL_TEMP_C}°C（等了 ${n}s）"; return 0; }
    if [ "$n" -ge "$COOL_WAIT_S" ]; then
      say "  **等 ${COOL_WAIT_S}s 仍未降到 ${COOL_TEMP_C}°C（目前 ${t}°C）**"
      say "  仍然起跑（92°C 中止線會在執行中保護），但這一點記進紀錄"
      return 0
    fi
    [ $((n % 120)) -eq 0 ] && say "  降溫中… ${t}°C（門檻 ${COOL_TEMP_C}）已等 ${n}s"
    sleep 15; n=$((n+15))
  done
}

# 階段結果累積（給報告用）
RUNS=(); RESULTS=(); BLOCK_STAGE=""; BLOCK_WHY=""

note_state(){
  python3 - "$STATE" "$SESSION" "$(elapsed)" "$LAUNCHES" "$MAX_LAUNCH" \
           "$WALL_BUDGET_S" "${BLOCK_STAGE:-}" "${BLOCK_WHY:-}" \
           "${RUNS[*]:-}" "${RESULTS[*]:-}" <<'EOF'
import json, sys
p, ses, el, la, ml, wb, bs, bw, runs, res = sys.argv[1:11]
json.dump({'session': ses, 'elapsed_s': int(el), 'launches': int(la),
           'max_launch': int(ml), 'wall_budget_s': int(wb),
           'block_stage': bs or None, 'block_why': bw or None,
           'runs': runs.split() if runs else [],
           'results': res.split() if res else []},
          open(p, 'w'), ensure_ascii=False, indent=1)
EOF
}

kill_leftovers(){
  # **只殺本工作階段域內的殘留**，用 /proc/<pid>/environ 核對，不用 pkill 字串比對
  local n=0
  for pat in isaac_wholebody_sim_e2 wgmpc_wg2_node wgmpc_prepos_node \
             wgmpc_stage_a_handover arm_vel_adapter wgmpc_env_recorder \
             arm_link_distance wholebody_safety; do
    for pid in $(pgrep -f "$pat" 2>/dev/null); do
      [ "$pid" = "$$" ] && continue
      if tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null \
           | grep -q "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"; then
        kill -TERM "$pid" 2>/dev/null && n=$((n+1))
      fi
    done
  done
  [ "$n" -gt 0 ] && { sleep 3
    for pat in isaac_wholebody_sim_e2 wgmpc_wg2_node wgmpc_prepos_node; do
      for pid in $(pgrep -f "$pat" 2>/dev/null); do
        [ "$pid" = "$$" ] && continue
        tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null \
          | grep -q "ROS_DOMAIN_ID=$ROS_DOMAIN_ID" && kill -KILL "$pid" 2>/dev/null
      done
    done; }
  echo "$n"
}

# ---- 一趟階段 A。回傳 0 = 達成成功判準 ----
stage_a_run(){
  local tag="$1" rec="$2"
  if ! launch_ok; then BLOCK_STAGE="budget"; BLOCK_WHY="已達 Isaac 啟動上限 $MAX_LAUNCH"; return 90; fi
  :
  if ! budget_ok 1200; then BLOCK_STAGE="budget"; BLOCK_WHY="牆鐘預算剩 $(left)s 不足一趟"; return 91; fi
  local rid="${SESSION}_${tag}"
  cool_down
  LAUNCHES=$((LAUNCHES+1))
  :
  say "── 趟次 $tag（錄影=$rec，第 $LAUNCHES/$MAX_LAUNCH 次啟動，剩 $(left)s）"
  local n; n=$(kill_leftovers); [ "$n" -gt 0 ] && say "  清掉 $n 個殘留程序"
  (
    export RUN_ID="$rid" DRAWER_BASELINE="$THRESH" REC_ENABLE="$rec"
    [ "$rec" = "1" ] && export REC_AT="$REC_AT_V" REC_EYE="$REC_EYE_V" \
                               REC_TARGET="$REC_TARGET_V" REC_RES=1280x720 REC_FPS=10
    bash evaluation/run_wgmpc_stage_a.sh
  ) >>"$SLOG" 2>&1
  local rc=$?
  # **碼 83 = 場景未建起**：Isaac 連一個字都沒輸出，那一趟沒有產生任何資料。
  # 這是間歇性的啟動卡死（最初那次有相機、這次沒有 ⇒ **相機不是原因**）。
  # 重試它不是「重跑失敗的實驗」，是重試一個沒有資料的啟動；
  # 每次嘗試都另存證據於該趟的 isaac_hang/。
  local try=1
  while [ "$rc" = "83" ] && [ "$try" -lt "${SCENE_RETRY:-3}" ]; do
    try=$((try+1))
    say "  **場景未建起（碼 83，第 $((try-1)) 次）** ⇒ 採證已存，重試第 $try 次"
    mv "$WS/evaluation/runs/$rid" "$WS/evaluation/runs/${rid}_hang$((try-1))" \
      2>/dev/null || true
    kill_leftovers >/dev/null; cool_down
    LAUNCHES=$((LAUNCHES+1))
    (
      export RUN_ID="$rid" DRAWER_BASELINE="$THRESH" REC_ENABLE="$rec"
      [ "$rec" = "1" ] && export REC_AT="$REC_AT_V" REC_EYE="$REC_EYE_V" \
                                 REC_TARGET="$REC_TARGET_V" REC_RES=1280x720 REC_FPS=10
      bash evaluation/run_wgmpc_stage_a.sh
    ) >>"$SLOG" 2>&1
    rc=$?
  done
  RUNS+=("$rid")
  local d="$WS/evaluation/runs/$rid"
  local verdict; verdict=$(python3 evaluation/wgmpc_stage_a_verdict.py "$d" \
                             --quiet 2>/dev/null || echo fail)
  RESULTS+=("$tag=$verdict(rc=$rc)")
  say "  趟次 $tag 結束：腳本 rc=$rc、判定 $verdict"
  note_state
  [ "$verdict" = "pass" ] && return 0 || return 1
}

trap 'say "收到中斷訊號，清理本輪程序"; kill_leftovers >/dev/null; note_state' INT TERM

say "=== 階段 A 夜間工作計畫 SESSION=$SESSION domain=$ROS_DOMAIN_ID ==="
say "預算 牆鐘 ${WALL_BUDGET_S}s、Isaac 啟動上限 $([ "$MAX_LAUNCH" -eq 0 ] && echo '**已撤銷**' || echo "$MAX_LAUNCH")"
say "趟次間降溫門檻 ${COOL_TEMP_C}°C（上限等 ${COOL_WAIT_S}s）；92°C 中止線不動"
say "門檻檔 $(basename "$(dirname "$THRESH")")/$(basename "$THRESH")"
say "**控制權重／視界／目標／退距／幾何／限位／驗收門檻固定；92°C 熱中止線保留**"
note_state

# ============================ 1 入口核對 ============================
say "[階段 1] 入口核對與殘留清理（**不重跑靜止基線**）"
n=$(kill_leftovers); say "  殘留程序清理：$n 個"
[ -f "$THRESH" ] || { BLOCK_STAGE="entry"; BLOCK_WHY="找不到門檻檔 $THRESH"; }
if [ -z "$BLOCK_STAGE" ]; then
  if python3 evaluation/wgmpc_stage_a_preflight.py --out "$OUT/preflight.json" \
       2>&1 | tee -a "$SLOG"; then
    say "  入口核對通過"
  else
    # **只有「套件未重建」這一類會自動修**：那是建置產物，不是實驗參數。
    # 前一輪就是因為 install/ 是舊版（scene_truth 參數沒宣告）而在 35 秒內
    # 中止，然後整夜沒有任何進展。這一類不該再吃掉時間。
    ONLY_BUILD=$(python3 - "$OUT/preflight.json" <<'EOF'
import json, sys
try:
    v = json.load(open(sys.argv[1])).get('violations') or []
except Exception:
    print('0'); raise SystemExit
print('1' if v and all(x.startswith('installed_matches_src') for x in v)
      else '0')
EOF
)
    if [ "$ONLY_BUILD" = "1" ]; then
      say "  入口核對只差「install 與 src 不一致」⇒ **自動 colcon build 一次**"
      say "  （依既定約定**不用** --symlink-install）"
      set +u; . /opt/ros/jazzy/setup.bash; set -u
      if timeout 1800 colcon build --packages-select ammr_wholebody_mpc \
           >>"$SLOG" 2>&1; then
        set +u; . "$WS/install/setup.bash"; set -u
        say "  重建完成，重核入口"
        if python3 evaluation/wgmpc_stage_a_preflight.py \
             --out "$OUT/preflight.json" 2>&1 | tee -a "$SLOG"; then
          say "  入口核對通過（重建後）"
        else
          BLOCK_STAGE="entry"; BLOCK_WHY="重建後入口核對仍未通過"
        fi
      else
        BLOCK_STAGE="entry"; BLOCK_WHY="colcon build 失敗（見 session.log）"
      fi
    else
      BLOCK_STAGE="entry"; BLOCK_WHY="入口核對未通過（見 preflight.json）"
    fi
  fi
fi
note_state

# ============================ 2 首趟（不錄影）============================
if [ -z "$BLOCK_STAGE" ]; then
  say "[階段 2] 一趟不錄影的階段 A"
  if stage_a_run r1 0; then
    say "  **首趟達成成功判準**"
  else
    BLOCK_STAGE="run_r1"; BLOCK_WHY="首趟未達成功判準（見該趟 verdict.json）"
    say "  首趟未達成功判準 ⇒ **停止後續 Isaac 趟次**，轉入階段 5"
  fi
fi
note_state

# ============================ 3 再兩趟（不錄影）============================
if [ -z "$BLOCK_STAGE" ]; then
  say "[階段 3] 相同配置再兩趟、不錄影"
  for tag in r2 r3; do
    if ! stage_a_run "$tag" 0; then
      BLOCK_STAGE="run_$tag"; BLOCK_WHY="$tag 未達成功判準 ⇒ 停止實跑"
      say "  $tag 未達成功判準 ⇒ 停止實跑，轉入階段 5"; break
    fi
  done
fi
note_state

# ============================ 4 錄影分支 ============================
REC_DIAG="skipped"; REC_RUN="skipped"
if [ -z "$BLOCK_STAGE" ]; then
  say "[階段 4] 錄影分支：先短相機診斷（10 s 模擬時間、無命令來源）"
  if launch_ok && budget_ok 900; then
    cool_down
    LAUNCHES=$((LAUNCHES+1))
    CD="$WS/evaluation/runs/${SESSION}_camdiag"; mkdir -p "$CD"
    say "  相機診斷（第 $LAUNCHES/$MAX_LAUNCH 次啟動）"
    timeout 900 "$ISAAC_PY" -u evaluation/isaac_wholebody_sim_e2.py \
      --out "$CD/sim" --mode solver_drawer --sim-limit 10 --physics-dt 0.01 \
      --drawer-asset "$ASSET" --drawer-pose 0.0,1.45 \
      --init-arm-q 0,0,1.437278401,0,0,0 --joint-margin 0.05 \
      --record-frames "$CD/frames" --record-res 1280x720 --record-fps 10 \
      --record-from 0 --record-at "$REC_AT_V" --record-eye "$REC_EYE_V" \
      --record-target "$REC_TARGET_V" --solver-label camdiag \
      --run-label "相機診斷：抽屜場景、無命令來源、10 s" \
      >>"$CD/run.log" 2>&1
    crc=$?
    if python3 evaluation/wgmpc_camdiag_check.py "$CD" --out "$CD/camdiag.json" \
         2>&1 | tee -a "$SLOG"; then
      REC_DIAG="pass"; say "  相機診斷通過"
    else
      REC_DIAG="fail(rc=$crc)"
      say "  **相機診斷未通過**（rc=$crc）⇒ 結束錄影分支，不影響前三趟結果"
    fi
  else
    REC_DIAG="skipped(budget)"
  fi
  if [ "$REC_DIAG" = "pass" ]; then
    say "[階段 4b] 一趟錄影階段 A（1280x720 @ 10 fps）"
    if stage_a_run rec 1; then REC_RUN="pass"; else REC_RUN="fail"; fi
  fi
fi
note_state

# ============================ 5 報告 ============================
say "[階段 5] 產出隔日交付 OVERNIGHT_REPORT.md"
python3 evaluation/wgmpc_stage_a_report.py \
  --session-dir "$OUT" --runs "${RUNS[*]:-}" \
  --rec-diag "$REC_DIAG" --rec-run "$REC_RUN" \
  --block-stage "${BLOCK_STAGE:-}" --block-why "${BLOCK_WHY:-}" \
  --elapsed-s "$(elapsed)" --launches "$LAUNCHES" \
  --out "$OUT/OVERNIGHT_REPORT.md" 2>&1 | tee -a "$SLOG"

n=$(kill_leftovers); say "收尾清理：$n 個殘留程序"
say "=== 結束（用時 $(elapsed)s、啟動 $LAUNCHES 次）==="
say "交付：$OUT/OVERNIGHT_REPORT.md"
note_state
