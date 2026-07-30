#!/usr/bin/env bash
# The control that was missing: everything OFF, at dyn_obs_1 = 0.15, on TODAY's
# code. The 1/35 reference is from 2026-06-30, and a great deal has changed
# since (ROS upgrade of 337 packages, arm integration, gmpc.py split-slack /
# spacetime / margin-growth / prog-weight code paths, the non-convergence
# feasibility guard). Every recent group has been blamed on the detour, but
# nothing has re-measured the BASELINE on current code at this speed.
#
# If this comes back ~1/10, the detour really does erode clearance.
# If it comes back ~4/10, the regression is in the baseline and the detour was
# never the cause.
cd /home/howardchen/masterthesis
rm -f /tmp/omnibot_dynamic.pid
R=evaluation/results; B=evaluation/bags
[ -d "$B/archive_base015" ] && { echo "REFUSING: archive_base015 exists"; exit 1; }
source /opt/ros/jazzy/setup.bash; source install/setup.bash
DETOUR=0 ./evaluation/run_omnibot_dynamic.sh 10 250 17 17 gmpc_scan
cp "$R/omnibot_dynamic_gmpc_scan.csv" "$R/final_base015.csv"
mkdir -p "$B/archive_base015"
for i in $(seq 1 10); do
  [ -d "$B/gmpc_cbf__scan_seed$i" ] && cp -r "$B/gmpc_cbf__scan_seed$i" "$B/archive_base015/"
done
echo "=== BASE015 DONE ==="
