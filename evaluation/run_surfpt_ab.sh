#!/usr/bin/env bash
# Surface-point CBF: does dropping shape fitting altogether hold up?
#
# Every obstacle -- known wall, unknown static, mover -- is now handed to the
# controller as a few of its own laser returns, at v = 0 or the track's velocity.
# A lidar return IS a point on the surface, so nothing is assumed about shape.
# That removes the one part of the pipeline that was shape-specific: circle
# fitting is exact for a cylinder (0.008 m centre error) but no better than the
# raw centroid for anything longer than about 2:1 (0.332 vs 0.287 on a
# 1.2 x 0.3 m body), which made it look like a fixture of the test props rather
# than a method.
#
# Run on the scenarios that stress geometry (shapes, ell, wide), one that
# stresses timing (stopgo), and two that must not regress (none, crossing).
cd /home/howardchen/masterthesis
for S in shapes ell wide stopgo none crossing; do
  TAG="surf_$S"
  [ -d "evaluation/bags/archive_$TAG" ] && { echo "SKIP $TAG"; continue; }
  echo "=== SURFPT $S ==="
  rm -f /tmp/omnibot_dynamic.pid
  ARENA=1 TRAJ="arena_$S" PLANNER_SCAN=/scan_filtered PLAN_BLEND=1.0 \
    CBF_ALPHA=1.5 STATIC_MARGIN=0.38 \
    ./evaluation/run_omnibot_dynamic.sh 10 120 6.6 6.6 gmpc_scan
  cp evaluation/results/omnibot_dynamic_gmpc_scan.csv "evaluation/results/final_$TAG.csv"
  mkdir -p "evaluation/bags/archive_$TAG"
  for i in $(seq 1 10); do
    [ -d "evaluation/bags/gmpc_cbf__scan_seed$i" ] && \
      cp -r "evaluation/bags/gmpc_cbf__scan_seed$i" "evaluation/bags/archive_$TAG/"
  done
done
echo "=== SURFPT DONE ==="
