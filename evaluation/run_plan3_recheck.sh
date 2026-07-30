#!/usr/bin/env bash
# Re-run plan3 in the 20 m room on the CURRENT perception code.
#
# archive_plan3 (deg/m 43.8, 0.6 reversals/run) was recorded before two fixes
# landed. Evidence they matter: in that batch the published obstacle radius had
# a median of exactly 0.250 -- the r_prior default -- meaning a large share of
# clusters took the degenerate path where the circle fit is rejected and the
# extent floor was not applied, so the disc did not even cover the visible face
# (14% of returned points inside it, measured synthetically). The current code
# reports a median of 0.309, matching the true 0.30 m cylinders.
#
# Without this the big-room and arena numbers cannot be compared: the perception
# layer changed between them.
cd /home/howardchen/masterthesis
[ -d evaluation/bags/archive_plan3b ] && { echo "archive_plan3b exists"; exit 1; }
PLANNER_SCAN=/scan_filtered PLAN_BLEND=1.0 CBF_ALPHA=1.5 \
  ./evaluation/run_omnibot_dynamic.sh 10 250 17 17 gmpc_scan
cp evaluation/results/omnibot_dynamic_gmpc_scan.csv evaluation/results/final_plan3b.csv
mkdir -p evaluation/bags/archive_plan3b
for i in $(seq 1 10); do
  [ -d "evaluation/bags/gmpc_cbf__scan_seed$i" ] && \
    cp -r "evaluation/bags/gmpc_cbf__scan_seed$i" evaluation/bags/archive_plan3b/
done
echo "=== PLAN3B DONE ==="
