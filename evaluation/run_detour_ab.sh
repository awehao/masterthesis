#!/usr/bin/env bash
# Committed-detour benchmark. The baseline it is compared against is the
# already-measured archive_w0 (10/10 reached, 7.7 reversals/run, 90.5 deg/m),
# so only the detour group is run here.
#
# Offset 0.35 rather than the 0.60 default: the wider swing lost runs in the 2D
# screen, and 0.35 is already above the robot radius.
#
# Usage:  ./evaluation/run_detour_ab.sh <tag>       e.g. detour2
# Writes: results/final_<tag>.csv and bags/archive_<tag>/
#
# Bag directories are named by seed and REUSED across experiments, so the
# archive copy below is what makes a batch's data survive the next run. Always
# use a fresh tag -- overwriting an archive silently mixes configurations, which
# is how archive_w0 ended up holding 40 directories for a 10-trial batch.
# No `set -u`: the ROS setup scripts read unbound variables (AMENT_TRACE_SETUP_FILES).
TAG="${1:-detour2}"
cd /home/howardchen/masterthesis || exit 1
source /opt/ros/jazzy/setup.bash
source install/setup.bash
R=evaluation/results; B=evaluation/bags

if [ -d "$B/archive_$TAG" ]; then
  echo "REFUSING: $B/archive_$TAG already exists -- pick a fresh tag." >&2
  exit 1
fi

# Overridable per experiment: e.g. DETOUR_VX_FLOOR=0 ./run_detour_ab.sh noFloor
DETOUR=1 \
  DETOUR_OFFSET="${DETOUR_OFFSET:-0.35}" \
  DETOUR_VX_FLOOR="${DETOUR_VX_FLOOR:-0.10}" \
  DETOUR_CLEAR_REF="${DETOUR_CLEAR_REF:-1}" \
  DETOUR_CLEAR_PAD="${DETOUR_CLEAR_PAD:-0.18}" \
  ./evaluation/run_omnibot_dynamic.sh 10 250 17 17 gmpc_scan
echo "config: offset=${DETOUR_OFFSET:-0.35} vx_floor=${DETOUR_VX_FLOOR:-0.10}" \
     "clear_ref=${DETOUR_CLEAR_REF:-1} pad=${DETOUR_CLEAR_PAD:-0.18}" 

cp "$R/omnibot_dynamic_gmpc_scan.csv" "$R/final_$TAG.csv"
mkdir -p "$B/archive_$TAG"
for i in $(seq 1 10); do
  [ -d "$B/gmpc_cbf__scan_seed$i" ] && cp -r "$B/gmpc_cbf__scan_seed$i" "$B/archive_$TAG/"
done
echo "=== DONE $TAG ==="
