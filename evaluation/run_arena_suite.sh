#!/usr/bin/env bash
# The arena suite under the hm_bigroom configuration.
#
# hm_bigroom = three-way perception split + reference blending + cbf_alpha 1.5, on
# top of the circle-fit tracker. In the 20 m room it was the first configuration
# all night with zero collisions of either kind (dynamic 0/10, unknown static
# 0/10, arrival 10/10), 42 s faster than the baseline with 58% less time stuck
# in the corridor.
#
# Seven scenarios, each isolating one failure mode; ~50 s per trial instead of
# ~170, which is what makes n=10 mean something for collision rate.
cd /home/howardchen/masterthesis
for S in none crossing gapblock corridor converge overtake parked; do
  rm -f /tmp/omnibot_dynamic.pid
  TAG="arena_$S"
  [ -d "evaluation/bags/archive_$TAG" ] && { echo "SKIP $TAG"; continue; }
  echo "=== SCENARIO $S ==="
  ARENA=1 TRAJ="arena_$S" PLANNER_SCAN=/scan_filtered PLAN_BLEND=1.0 CBF_ALPHA=1.5 \
    ./evaluation/run_omnibot_dynamic.sh 10 90 7.5 7.5 gmpc_scan
  cp evaluation/results/omnibot_dynamic_gmpc_scan.csv "evaluation/results/final_$TAG.csv"
  mkdir -p "evaluation/bags/archive_$TAG"
  for i in $(seq 1 10); do
    [ -d "evaluation/bags/gmpc_cbf__scan_seed$i" ] && \
      cp -r "evaluation/bags/gmpc_cbf__scan_seed$i" "evaluation/bags/archive_$TAG/"
  done
done
echo "=== ARENA SUITE DONE ==="
