#!/usr/bin/env bash
# 【D：跨機器／跨次渲染敏感性對照，2026-10-21】run_xh_capture.sh（凍結，不改）的複本。差異：
#   ①只允許 dev 資產（不只拒 test，也拒 train）；②守護改 run_guarded_v2.sh（k10temp → coretemp）；
#   ③趟次放在新根目錄 runs/xh_rerender_msi/xh_<序列名>（RUN_ID 含子目錄），不碰原錄；④另核對 v2 工具凍結。
# 原：XH2 只渲染擷取的批次驅動：bash evaluation/run_xh_capture.sh <序列名> [<序列名> ...]
# 每條前核對 freeze_xh2_capture.sha256；RUN_ID＝xh_<序列名>，已存在拒跑；任一條非 0 結束即停批（保留紀錄）。
# 擷取由 run_guarded.sh（92 °C 熱中止）包 run_wrist_xh.sh；序列與時間窗取自 XH2_paths_registration.yaml。test 資產拒跑。
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"; cd "$WS"
REG=evaluation/results/vision/XH2_paths_registration.yaml
FRZ=evaluation/results/vision/freeze_xh2_capture.sha256
FRZ2=evaluation/results/vision/freeze_xh_capture_v2_tools.sha256
ROOT=xh_rerender_msi
mkdir -p evaluation/runs/$ROOT
LOG=evaluation/runs/$ROOT/capture_batch.log
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
for SEQ in "$@"; do
  sha256sum -c --quiet "$FRZ" >> "$LOG" 2>&1 || { say "**凍結不符**（$SEQ 前）⇒ 停批"; exit 95; }
  sha256sum -c --quiet "$FRZ2" >> "$LOG" 2>&1 || { say "**v2 工具凍結不符**（$SEQ 前）⇒ 停批"; exit 95; }
  read -r FILE T0 T1 ASSET SPLIT < <(python3 - "$REG" "$SEQ" <<'PY'
import sys, yaml
r = yaml.safe_load(open(sys.argv[1]))['sequences'][sys.argv[2]]
print(r['file'], r['sim_t_window'][0], r['sim_t_window'][1], r['asset'], r['split'])
PY
)
  [ "$SPLIT" = dev ] || { say "**$SEQ 不是 dev 資產（$SPLIT）** ⇒ 拒跑"; exit 94; }
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
