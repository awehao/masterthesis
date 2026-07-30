#!/usr/bin/env bash
# Cluster centres from a circle fit instead of the arc mean.
# Synthetic check: circling a stationary 0.30 m cylinder, the arc mean drifts
# 0.43 m (apparent speed 0.114 m/s, ABOVE the 0.10 mover gate) while the fitted
# centre drifts 0.03 m (0.016 m/s). That drift is why stationary pillars were
# published as dynamic in 10-32% of frames and why the CBF was protecting a
# point 0.28 m off the real obstacle centre.
cd /home/howardchen/masterthesis
rm -f /tmp/omnibot_dynamic.pid
PLANNER_SCAN=/scan_filtered DETOUR=0 ./evaluation/run_omnibot_dynamic.sh 10 250 17 17 gmpc_scan
cp evaluation/results/omnibot_dynamic_gmpc_scan.csv evaluation/results/final_circfit.csv
mkdir -p evaluation/bags/archive_circfit
for i in $(seq 1 10); do
  [ -d "evaluation/bags/gmpc_cbf__scan_seed$i" ] && \
    cp -r "evaluation/bags/gmpc_cbf__scan_seed$i" evaluation/bags/archive_circfit/
done
echo "=== CIRCFIT DONE ==="
