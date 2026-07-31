#!/usr/bin/env bash
# The three-item action list, on top of the circle fit:
#   B  three-way split: unknown statics stay OUT of the scan mask (already) and
#      now also go INTO the static-CBF at v=0, so they have an owner even when
#      the planner is decoupled
#   C1 reference blending over 1.0 s, smoothstep so it is C1 at both ends
#   D3 cbf_alpha 3.0 -> 1.5, re-verified on the new tracker
#
# Compare against archive_circfit (circle fit + decoupling, none of the above):
#   pillars 0/10 already, dynamic collisions 3/10, dynamic clearance median
#   +0.087, corridor dwell 21.2 s, arrival 140.3 s.
cd /home/howardchen/masterthesis
rm -f /tmp/omnibot_dynamic.pid
PLANNER_SCAN=/scan_filtered DETOUR=0 PLAN_BLEND=1.0 CBF_ALPHA=1.5 \
  ./evaluation/run_omnibot_dynamic.sh 10 250 17 17 gmpc_scan
cp evaluation/results/omnibot_dynamic_gmpc_scan.csv evaluation/results/final_hm_bigroom.csv
mkdir -p evaluation/bags/archive_hm_bigroom
for i in $(seq 1 10); do
  [ -d "evaluation/bags/gmpc_cbf__scan_seed$i" ] && \
    cp -r "evaluation/bags/gmpc_cbf__scan_seed$i" evaluation/bags/archive_hm_bigroom/
done
echo "=== HM_BIGROOM DONE ==="
