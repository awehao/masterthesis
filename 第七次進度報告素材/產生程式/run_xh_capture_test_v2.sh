#!/usr/bin/env bash
# 【attempt2，2026-10-21】run_xh_capture_test.sh（凍結 r5，不改）的複本，只供 xh_open_test_v2.py 呼叫。差異：
#   ①開封紀錄由環境變數 XH_OPENING 明示（不寫死 attempt1 的檔名），須 state=opened、attempt=2；
#   ②趟次根目錄取自該紀錄的 runs_root，且必須在 evaluation/runs/ 之下的新子目錄；RUN_ID＝<子目錄>/xh_<序列名>；
#   ③守護改 run_guarded_v2.sh（k10temp → coretemp）；④另核對 XH_FREEZE（r6）。
# 原：【XH3 測試版】run_xh_capture.sh 的複本，只供 xh_open_test.sh 呼叫：只允許 test 資產，且須有狀態 opened 的開封紀錄。
# 原：XH2 只渲染擷取的批次驅動：bash evaluation/run_xh_capture.sh <序列名> [<序列名> ...]
# 每條前核對 freeze_xh2_capture.sha256；RUN_ID＝xh_<序列名>，已存在拒跑；任一條非 0 結束即停批（保留紀錄）。
# 擷取由 run_guarded.sh（92 °C 熱中止）包 run_wrist_xh.sh；序列與時間窗取自 XH2_paths_registration.yaml。test 資產拒跑。
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"; cd "$WS"
REG=evaluation/results/vision/XH2_paths_registration.yaml
FRZ=evaluation/results/vision/freeze_xh2_capture.sha256
: "${XH_OPENING:?XH_OPENING 未設定（attempt2 開封紀錄路徑）}"
: "${XH_FREEZE:?XH_FREEZE 未設定}"
ROOT=$(python3 - "$XH_OPENING" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
ok = r.get('state') == 'opened' and r.get('attempt') == 2 and str(r.get('runs_root', '')).startswith('evaluation/runs/')
sub = r['runs_root'][len('evaluation/runs/'):] if ok else ''
print(sub if ok and sub and '/' not in sub and sub not in ('.', '..') else '')
PY
)
[ -n "$ROOT" ] || { echo "**開封紀錄不是 opened／attempt 2，或 runs_root 不合法** ⇒ 拒跑"; exit 93; }
mkdir -p "evaluation/runs/$ROOT"
LOG=evaluation/runs/$ROOT/test_capture.log
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
for SEQ in "$@"; do
  sha256sum -c --quiet "$FRZ" >> "$LOG" 2>&1 || { say "**凍結不符**（$SEQ 前）⇒ 停批"; exit 95; }
  sha256sum -c --quiet "$XH_FREEZE" >> "$LOG" 2>&1 || { say "**r6 凍結不符**（$SEQ 前）⇒ 停批"; exit 95; }
  read -r FILE T0 T1 ASSET SPLIT < <(python3 - "$REG" "$SEQ" <<'PY'
import sys, yaml
r = yaml.safe_load(open(sys.argv[1]))['sequences'][sys.argv[2]]
print(r['file'], r['sim_t_window'][0], r['sim_t_window'][1], r['asset'], r['split'])
PY
)
  [ "$SPLIT" = test ] || { say "**$SEQ 不是 test 資產** ⇒ 拒跑"; exit 94; }
  python3 -c "import json,sys;r=json.load(open(sys.argv[1]));sys.exit(0 if r.get('state')=='opened' and r.get('attempt')==2 and r.get('runs_root')=='evaluation/runs/'+sys.argv[2] else 1)" "$XH_OPENING" "$ROOT" 2>/dev/null || { say "**開封紀錄已非 opened 或根目錄改變** ⇒ 拒跑"; exit 93; }
  RID="$ROOT/xh_$SEQ"
  [ -e "evaluation/runs/$RID" ] && { say "**$RID 已存在** ⇒ 拒跑（不覆寫）"; exit 91; }
  say "--- $RID 起跑（資產 $ASSET、$SPLIT、窗 $T0–$T1 s）"
  ROS_DOMAIN_ID=95 RUN_ID="$RID" RUNNER=evaluation/run_wrist_xh.sh \
    DRAWER_ASSET="$WS/src/my_omnibot_description/config/xh/drawer_unit_xh_${ASSET}.yaml" \
    WRIST_EXTRA="--wrist-replay $WS/$FILE --wrist-replay-t0 $T0 --wrist-replay-t1 $T1" \
    bash evaluation/run_guarded_v2.sh > "evaluation/runs/${RID}.guard.out" 2>&1 < /dev/null
  rc=$?
  say "    $RID 結束 rc=$rc；$(tail -1 evaluation/runs/$RID/guard.log 2>/dev/null)"
  [ "$rc" = 0 ] || { say "**$RID 非 0 結束** ⇒ 停批（保留紀錄）"; exit "$rc"; }
done
say "=== 本批結束 ==="
