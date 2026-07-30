#!/usr/bin/env bash
# Second wave: the scenarios and obstacle shapes the first wave could not test.
#
#   none     the control that failed to launch at all -- 'dynamic_obstacles:'
#            with nothing under it parses to None, and the get() default only
#            covers a MISSING key, so five call sites crashed on it
#   shapes   0.7x0.4 box + 1.2x0.3 cart: circle fitting degenerates on a flat
#            face, and one lidar view cannot see an object's depth
#   occlude  mover hidden behind the divider until it is close -- the only
#            scenario where PERCEPTION fails first
#   stopgo   reverses every ~2 s, so constant-velocity extrapolation, which the
#            CBF and every prediction here rests on, is wrong most of the time
#   small    0.15 m body near the clustering floor: may vanish, not just blur
#   ell      L-shape: every covering scheme here assumes convexity
#   dense    three movers of mixed size and speed
cd /home/howardchen/masterthesis
for S in none shapes occlude stopgo small ell dense; do
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
echo "=== ARENA2 DONE ==="
