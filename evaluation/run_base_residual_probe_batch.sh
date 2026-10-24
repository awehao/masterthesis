#!/usr/bin/env bash
# B0 六條件依序執行（各自 run_guarded_v2.sh、各自 RUN_ID）；任一條件非 0 ⇒ 停批（不補跑、不覆寫）。未核可前不得執行。
#   env -u VIRTUAL_ENV PATH=<去掉 .venv-dl0> TAG=1 bash evaluation/run_base_residual_probe_batch.sh
set -u
WS="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WS"
TAG="${TAG:?TAG 未設定}"
for COND in $(python3 -c "import sys; sys.path.insert(0,'evaluation'); import base_residual_probe_core as C; print(' '.join(C.CONDITIONS))"); do
  RID="b0_${COND}_${TAG}"
  COND="$COND" RUN_ID="$RID" ROS_DOMAIN_ID=97 RUNNER=evaluation/run_base_residual_probe.sh bash evaluation/run_guarded_v2.sh
  rc=$?
  echo "=== $RID rc=$rc"
  [ "$rc" = 0 ] || { echo "**停批** $RID"; exit "$rc"; }
done
echo ALLDONE
