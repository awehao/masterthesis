#!/usr/bin/env bash
# Is what the GUI shows the same system the headless numbers describe?
#
# Watched twice through the GUI, the robot turned 565 degrees in the first 15 s
# while covering 1.35 m, wz pinned at its +-0.8 limit and reversing every three
# seconds, and one run ended in a 0.428 m penetration. The headless run between
# those two, same configuration, reached the goal in 130 s with no contact at
# all. Rendering competes with gz physics, the simulated lidar and the 20 Hz
# control loop for CPU -- the run script's own comment notes it steals time from
# the gpu_lidar, and GUI solve times measured 3.27 ms against 2.51 headless.
#
# Either the GUI is showing a CPU-starved system, in which case it cannot be
# used to judge the method; or the method is fragile to timing, in which case a
# real robot will show the same. Both matter, and the loop timing separates them.
cd /home/howardchen/masterthesis
for G in 1 0; do
  TAG="gh_$G"
  [ -d "evaluation/bags/archive_$TAG" ] && { echo "SKIP $TAG"; continue; }
  echo "=== GUI=$G ==="
  rm -f /tmp/omnibot_dynamic.pid
  GUI=$G PLANNER_SCAN=/scan_filtered PLAN_BLEND=1.0 CBF_ALPHA=1.5 \
    ./evaluation/run_omnibot_dynamic.sh 2 300 17 17 gmpc_scan
  cp evaluation/results/omnibot_dynamic_gmpc_scan.csv "evaluation/results/final_$TAG.csv"
  mkdir -p "evaluation/bags/archive_$TAG"
  for i in 1 2; do
    [ -d "evaluation/bags/gmpc_cbf__scan_seed$i" ] && \
      cp -r "evaluation/bags/gmpc_cbf__scan_seed$i" "evaluation/bags/archive_$TAG/"
  done
done
echo "=== GH DONE ==="
