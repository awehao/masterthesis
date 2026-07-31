#!/usr/bin/env bash
# Reference heading from a look-ahead chord instead of the local path segment.
#
# Watched symptom, confirmed in the log: in the first 15 s the robot turned 565
# degrees -- one and a half full revolutions -- while covering 1.35 m, with wz
# pinned at the +-0.8 limit and reversing sign every three seconds. The
# reference heading is the tangent of a single ~0.15 m path segment, and while
# the robot is barely moving its closest-point projection jitters along the
# path, so that tangent swings wildly and the robot chases it.
#
# Synthetic prediction: a chord from s to s + L cuts the heading demand over a
# traverse by 76% (594 -> 140 degrees) while tracking the TRUE tangent more
# closely (5.7 -> 2.6 degrees median error).
cd /home/howardchen/masterthesis
for Y in 0 0.7; do
  TAG="yaw_${Y/./p}"
  [ -d "evaluation/bags/archive_$TAG" ] && { echo "SKIP $TAG"; continue; }
  echo "=== yaw_lookahead=$Y ==="
  rm -f /tmp/omnibot_dynamic.pid
  PLANNER_SCAN=/scan_filtered PLAN_BLEND=1.0 CBF_ALPHA=1.5 YAW_LOOKAHEAD=$Y \
    ./evaluation/run_omnibot_dynamic.sh 5 300 17 17 gmpc_scan
  cp evaluation/results/omnibot_dynamic_gmpc_scan.csv "evaluation/results/final_$TAG.csv"
  mkdir -p "evaluation/bags/archive_$TAG"
  for i in $(seq 1 5); do
    [ -d "evaluation/bags/gmpc_cbf__scan_seed$i" ] && \
      cp -r "evaluation/bags/gmpc_cbf__scan_seed$i" "evaluation/bags/archive_$TAG/"
  done
done
echo "=== YAW AB DONE ==="
