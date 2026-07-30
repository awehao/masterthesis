#!/usr/bin/env bash
# Non-circular movers: a 0.7x0.4 box and a 1.2x0.3 cart.
#
# Everything the perception stack does with an obstacle starts by fitting a
# CIRCLE to its laser cluster, so a world containing only cylinders can never
# show what that costs. Synthetic arcs said: the fit degenerates on a flat face
# (falls back), a single view cannot see an object's depth at all, and the
# covering-disc decomposition splits the cart into 3 discs when seen side-on,
# taking its keep-out from 1.27 m down to 0.67 m -- the difference between
# fitting through a 1.4 m gap and not.
#
# Clearance here is measured against the true rectangle, not a 0.25 m circle,
# which for this box differs by up to 0.10 m either way.
cd /home/howardchen/masterthesis
rm -f /tmp/omnibot_dynamic.pid
[ -d evaluation/bags/archive_arena_shapes ] && { echo "exists"; exit 1; }
ARENA=1 TRAJ=arena_shapes PLANNER_SCAN=/scan_filtered PLAN_BLEND=1.0 CBF_ALPHA=1.5 \
  ./evaluation/run_omnibot_dynamic.sh 10 90 7.5 7.5 gmpc_scan
cp evaluation/results/omnibot_dynamic_gmpc_scan.csv evaluation/results/final_arena_shapes.csv
mkdir -p evaluation/bags/archive_arena_shapes
for i in $(seq 1 10); do
  [ -d "evaluation/bags/gmpc_cbf__scan_seed$i" ] && \
    cp -r "evaluation/bags/gmpc_cbf__scan_seed$i" evaluation/bags/archive_arena_shapes/
done
echo "=== SHAPES DONE ==="
