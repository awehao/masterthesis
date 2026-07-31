#!/usr/bin/env bash
# Re-run the five scenarios whose mover was never recorded.
#
# record.sh listed /model/dyn_obs_{0,1,2,5}/pose only, so dyn_obs_3, 4 and 6 --
# the 0.15 m cylinder, the L-shape and the second small cylinder -- produced no
# ground-truth track. analyze.py takes its dynamic clearance from those topics,
# so for small, ell and merge the obstacle under test was invisible to the
# metric and their "zero collisions" measured nothing but walls; bothgaps and
# dense were each missing one of their movers.
cd /home/howardchen/masterthesis
for S in small ell merge bothgaps dense; do
  TAG="arena_$S"
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
echo "=== RERECORD DONE ==="
