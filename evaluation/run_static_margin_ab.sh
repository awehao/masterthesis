#!/usr/bin/env bash
# Does the unknown-static keep-out explain every collision in the arena?
#
# Across 187 arena trials EVERY collision was a graze of an unknown static
# pillar -- never a wall, never a mover. The arithmetic says why: the CBF treats
# the robot as a point, so the 0.30 m robot radius lives INSIDE the margin.
#   static_cbf_safe_margin 0.33 -> keep-out 0.30 + 0.33 = 0.63 from a pillar
#   centre, against a 0.60 m contact distance = 3 cm of true buffer
#   cbf_safe_margin        0.38 -> 8 cm for movers
# Measured pillar clearance in the failing scenarios: -0.006 to +0.011, i.e.
# that 3 cm target minus slack.
#
# 0.38 gives statics the same 8 cm. It costs 5% of navigable cells (94.8% ->
# 89.5% of free cells still admit the robot centre), so narrow passages should
# survive -- shapes and stopgo are run because they are where it failed, and
# none/crossing because they are where nothing must break.
cd /home/howardchen/masterthesis
for S in shapes stopgo none crossing; do
  for M in 0.33 0.38; do
    TAG="sm_${S}_${M/./p}"
    [ -d "evaluation/bags/archive_$TAG" ] && { echo "SKIP $TAG"; continue; }
    echo "=== $S  static_margin=$M ==="
    rm -f /tmp/omnibot_dynamic.pid
    ARENA=1 TRAJ="arena_$S" PLANNER_SCAN=/scan_filtered PLAN_BLEND=1.0 \
      CBF_ALPHA=1.5 STATIC_MARGIN=$M \
      ./evaluation/run_omnibot_dynamic.sh 10 120 6.6 6.6 gmpc_scan
    cp evaluation/results/omnibot_dynamic_gmpc_scan.csv "evaluation/results/final_$TAG.csv"
    mkdir -p "evaluation/bags/archive_$TAG"
    for i in $(seq 1 10); do
      [ -d "evaluation/bags/gmpc_cbf__scan_seed$i" ] && \
        cp -r "evaluation/bags/gmpc_cbf__scan_seed$i" "evaluation/bags/archive_$TAG/"
    done
  done
done
echo "=== STATIC MARGIN AB DONE ==="
