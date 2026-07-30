#!/usr/bin/env bash
# All 14 arena scenarios under plan3, on the repositioned arena.
#
# The first attempt had to be discarded: the launch spawns the robot at a
# hardcoded (0, 0), and the arena's corner sat at -0.5 with 0.20 m walls, so the
# inner face was exactly the robot radius away. Every trial then reported a
# minimum clearance of precisely +0.000 taken at t = 0, which drowned out
# everything that happened during the run. The arena is now placed around the
# spawn, leaving 0.85 m there.
#
# Scenarios, each isolating one thing the big room could not:
#   none      no movers -- the cost of geometry and unknown statics alone
#   crossing  perpendicular, prediction valid
#   gapblock  mover patrolling the direct gap: too narrow to pass, so wait or
#             re-route -- a homotopy decision, not a local dodge
#   corridor  mover sweeping ALONG the approach: nothing to go around
#   converge  two movers, opposite sides
#   overtake  slower mover, same direction (_blocking needs fx > 0)
#   parked    speed 0: the static/dynamic split itself
#   shapes    0.7x0.4 box + 1.2x0.3 cart
#   occlude   mover hidden behind the divider until close: PERCEPTION fails first
#   stopgo    reverses every ~2 s: constant-velocity extrapolation mostly wrong
#   small     0.15 m body, near the clustering floor
#   ell       L-shape: breaks the convexity every covering scheme assumes
#   dense     three movers, mixed size and speed
set -u
cd /home/howardchen/masterthesis
for S in none crossing gapblock corridor converge overtake parked \
         shapes occlude stopgo small ell dense; do
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
echo "=== ARENA ALL DONE ==="
