#!/usr/bin/env bash
# All 20 arena scenarios under hm_bigroom.
#
# hm_bigroom = circle-fit tracker + three-way perception split + 1.0 s smoothstep
# reference blending + cbf_alpha 1.5. In the 20 m room it was the first
# configuration with zero collisions of either kind (dynamic 0/10, unknown
# static 0/10, 10/10 arrivals, 42 s faster than baseline).
#
# The arena is placed around the launch's hardcoded (0,0) spawn -- an earlier
# layout put the wall exactly one robot radius away, so every trial reported a
# minimum clearance of +0.000 taken at t=0 and the metric measured parking
# rather than avoidance.
cd /home/howardchen/masterthesis
for S in none crossing gapblock corridor converge overtake parked \
         shapes occlude stopgo small ell dense \
         diagonal headon bothgaps fast chase merge wide; do
  TAG="arena_$S"
  [ -d "evaluation/bags/archive_$TAG" ] && { echo "SKIP $TAG"; continue; }
  echo "=== SCENARIO $S ==="
  ARENA=1 TRAJ="arena_$S" PLANNER_SCAN=/scan_filtered PLAN_BLEND=1.0 CBF_ALPHA=1.5 \
    ./evaluation/run_omnibot_dynamic.sh 10 90 6.6 6.6 gmpc_scan
  cp evaluation/results/omnibot_dynamic_gmpc_scan.csv "evaluation/results/final_$TAG.csv"
  mkdir -p "evaluation/bags/archive_$TAG"
  for i in $(seq 1 10); do
    [ -d "evaluation/bags/gmpc_cbf__scan_seed$i" ] && \
      cp -r "evaluation/bags/gmpc_cbf__scan_seed$i" "evaluation/bags/archive_$TAG/"
  done
done
echo "=== ARENA20 DONE ==="
