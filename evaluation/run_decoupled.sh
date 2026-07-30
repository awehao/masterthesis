#!/usr/bin/env bash
# Decoupled perception, with the tracker fixes:
#   - the scan mask and the CBF publication now use ONE shared mover decision
#     (they used to disagree, so the planner was blinded to objects the CBF was
#      not protecting against)
#   - the net-displacement gate is ON (min_net_speed 0.0 -> 0.05, window 2.0 s);
#     with it off, static cylinders reached /gmpc/obstacles in 10-32% of frames
#
# Compare against archive_base015 (PLANNER_SCAN=/scan, DETOUR=0, same speed),
# BUT note that group ran on the OLD tracker -- the control needs a rerun for a
# clean single-variable comparison.
cd /home/howardchen/masterthesis
rm -f /tmp/omnibot_dynamic.pid
PLANNER_SCAN=/scan_filtered DETOUR=0 ./evaluation/run_omnibot_dynamic.sh 10 250 17 17 gmpc_scan
cp evaluation/results/omnibot_dynamic_gmpc_scan.csv evaluation/results/final_decoup.csv
mkdir -p evaluation/bags/archive_decoup
for i in $(seq 1 10); do
  [ -d "evaluation/bags/gmpc_cbf__scan_seed$i" ] && \
    cp -r "evaluation/bags/gmpc_cbf__scan_seed$i" evaluation/bags/archive_decoup/
done
echo "=== DECOUP DONE ==="
