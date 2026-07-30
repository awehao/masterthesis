#!/usr/bin/env bash
# Decoupled perception: should the PLANNER see dynamic obstacles at all?
#
# /scan          the planner marks movers at their CURRENT position. Measured:
#                24% of replans move the path ahead of the robot by >0.25 m
#                (max 1.82 m, heading up to 63 deg), and in the 0.3 s after such
#                a jump the controller runs at 0.96 of a_max and saturates 88.5%
#                of the time against 13.5% otherwise.
# /scan_filtered movers masked out by scan_obstacle_tracker, so the planner's
#                cost field is static and the plan stops flipping. Dynamic
#                avoidance becomes entirely the CBF's job.
#
# Watch two things that pull in opposite directions:
#   jitter   -- should drop, the step inputs are gone
#   corridor -- may get WORSE: with the planner blind, nothing stops it routing
#               straight down the corridor dyn_obs_1 sweeps, where the robot
#               already spends 43.5 s (26% of a run) and backs up 4x as often.
#
# CBF stays on in both. dyn_obs_1 = 0.15, matching the N=35 (98%, 1/35) run.
cd /home/howardchen/masterthesis
for S in /scan /scan_filtered; do
  rm -f /tmp/omnibot_dynamic.pid
  TAG="pscan$(basename $S)"
  [ -d "evaluation/bags/archive_$TAG" ] && { echo "SKIP $TAG (exists)"; continue; }
  echo "=== GROUP planner_scan=$S -> $TAG ==="
  PLANNER_SCAN=$S DETOUR=0 ./evaluation/run_omnibot_dynamic.sh 10 250 17 17 gmpc_scan
  cp evaluation/results/omnibot_dynamic_gmpc_scan.csv "evaluation/results/final_$TAG.csv"
  mkdir -p "evaluation/bags/archive_$TAG"
  for i in $(seq 1 10); do
    [ -d "evaluation/bags/gmpc_cbf__scan_seed$i" ] && \
      cp -r "evaluation/bags/gmpc_cbf__scan_seed$i" "evaluation/bags/archive_$TAG/"
  done
done
echo "=== PLANNER SCAN AB DONE ==="
