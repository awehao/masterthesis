#!/usr/bin/env bash
# 【XH3 測試版】run_xh_capture.sh 的複本，只供 xh_open_test.sh 呼叫：只允許 test 資產，且須有狀態 opened 的開封紀錄。
# 原：XH2 只渲染擷取的批次驅動：bash evaluation/run_xh_capture.sh <序列名> [<序列名> ...]
# 每條前核對 freeze_xh2_capture.sha256；RUN_ID＝xh_<序列名>，已存在拒跑；任一條非 0 結束即停批（保留紀錄）。
# 擷取由 run_guarded.sh（92 °C 熱中止）包 run_wrist_xh.sh；序列與時間窗取自 XH2_paths_registration.yaml。test 資產拒跑。
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"; cd "$WS"
REG=evaluation/results/vision/XH2_paths_registration.yaml
FRZ=evaluation/results/vision/freeze_xh2_capture.sha256
LOG=evaluation/runs/xh3_test_capture.log
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
for SEQ in "$@"; do
  sha256sum -c --quiet "$FRZ" >> "$LOG" 2>&1 || { say "**凍結不符**（$SEQ 前）⇒ 停批"; exit 95; }
  read -r FILE T0 T1 ASSET SPLIT < <(python3 - "$REG" "$SEQ" <<'PY'
import sys, yaml
r = yaml.safe_load(open(sys.argv[1]))['sequences'][sys.argv[2]]
print(r['file'], r['sim_t_window'][0], r['sim_t_window'][1], r['asset'], r['split'])
PY
)
  [ "$SPLIT" = test ] || { say "**$SEQ 不是 test 資產** ⇒ 拒跑"; exit 94; }
  python3 -c "import json,sys;sys.exit(0 if json.load(open('evaluation/results/vision/XH_test_opening.json')).get('state')=='opened' else 1)" 2>/dev/null || { say "**沒有 opened 開封紀錄** ⇒ 拒跑"; exit 93; }
  RID="xh_$SEQ"
  [ -e "evaluation/runs/$RID" ] && { say "**$RID 已存在** ⇒ 拒跑（不覆寫）"; exit 91; }
  say "--- $RID 起跑（資產 $ASSET、$SPLIT、窗 $T0–$T1 s）"
  ROS_DOMAIN_ID=95 RUN_ID="$RID" RUNNER=evaluation/run_wrist_xh.sh \
    DRAWER_ASSET="$WS/src/my_omnibot_description/config/xh/drawer_unit_xh_${ASSET}.yaml" \
    WRIST_EXTRA="--wrist-replay $WS/$FILE --wrist-replay-t0 $T0 --wrist-replay-t1 $T1" \
    bash evaluation/run_guarded.sh > "evaluation/runs/${RID}.guard.out" 2>&1 < /dev/null
  rc=$?
  say "    $RID 結束 rc=$rc；$(tail -1 evaluation/runs/$RID/guard.log 2>/dev/null)"
  [ "$rc" = 0 ] || { say "**$RID 非 0 結束** ⇒ 停批（保留紀錄）"; exit "$rc"; }
done
say "=== 本批結束 ==="
