#!/usr/bin/env bash
# All 20 scenarios, surface-point CBF, on the corrected arena.
#
# Everything here was rebuilt after three defects invalidated the first sweep:
#   * record.sh listed only dyn_obs_{0,1,2,5}, so the movers in small, ell,
#     merge and one each in bothgaps and dense were never recorded and their
#     "zero collisions" measured nothing but walls
#   * ten scenarios put their mover on the gap the robot does not use -- the two
#     routes differ by 0.1 m and the planner picks LEFT 10 times out of 10, so
#     corridor's mover stayed 2.92 m away all run
#   * long ping-pong lanes made the encounter depend on phase
# Lanes are now short, centred on the recorded route, and shrunk to fit each
# obstacle's own radius; every scenario is checked for encounter, wall and
# pillar clearance before it runs.
#
# Configuration: surface-point CBF (no shape fitting anywhere), three-way
# perception split, smoothstep reference blending, cbf_alpha 1.5, static margin
# 0.38.
cd /home/howardchen/masterthesis
for S in none crossing gapblock corridor converge overtake parked \
         shapes occlude stopgo small ell dense \
         diagonal headon bothgaps fast chase merge wide; do
  TAG="s20_$S"
  [ -d "evaluation/bags/archive_$TAG" ] && { echo "SKIP $TAG"; continue; }
  echo "=== SCENARIO $S ==="
  rm -f /tmp/omnibot_dynamic.pid
  ARENA=1 TRAJ="arena_$S" PLANNER_SCAN=/scan_filtered PLAN_BLEND=1.0 CBF_ALPHA=1.5 \
    ./evaluation/run_omnibot_dynamic.sh 10 120 6.6 6.6 gmpc_scan
  cp evaluation/results/omnibot_dynamic_gmpc_scan.csv "evaluation/results/final_$TAG.csv"
  mkdir -p "evaluation/bags/archive_$TAG"
  for i in $(seq 1 10); do
    [ -d "evaluation/bags/gmpc_cbf__scan_seed$i" ] && \
      cp -r "evaluation/bags/gmpc_cbf__scan_seed$i" "evaluation/bags/archive_$TAG/"
  done
done
echo "=== S20 DONE ==="
