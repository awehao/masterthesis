#!/usr/bin/env bash
# Wait for batch L, then build and run M and N back to back.
#
# Ordering matters and is not negotiable:
#   1. L must be finished. run_omnibot_dynamic.sh is read incrementally by bash,
#      so editing it mid-run shifts byte offsets and corrupts execution. The
#      colcon build swaps install/ under a running node for the same reason.
#   2. Strip the dead YAW_* names from the env forwarding list (the launch no
#      longer reads them).
#   3. Build. Plain colcon build -- NEVER --symlink-install on the ammr python
#      packages: it corrupts entry-point metadata and every node dies with
#      PackageNotFoundError.
#   4. M = no heading reference, otherwise identical to K.
#   5. N = M plus PLAN_BLEND=0.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

LOCKFILE=/tmp/omnibot_dynamic.pid
echo "[$(date +%T)] waiting for batch L ..."
while [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE" 2>/dev/null)" 2>/dev/null; do
    sleep 60
done
echo "[$(date +%T)] batch L finished"

# L's results live in the shared CSV until its own script copies them out; give
# that copy a moment to land before anything else touches the file.
sleep 20
if [ ! -s evaluation/results/slowL/batch.csv ]; then
    cp evaluation/results/omnibot_dynamic_gmpc_scan.csv \
       evaluation/results/slowL/batch.csv 2>/dev/null || true
fi
echo "[$(date +%T)] L archived: $(( $(wc -l < evaluation/results/slowL/batch.csv 2>/dev/null || echo 1) - 1 )) trials"

# 2. dead env names
sed -i 's/ PLAN_BLEND YAW_LOOKAHEAD YAW_RATE_MAX / PLAN_BLEND /; s/ HORIZON YAW_TRACK / HORIZON /' \
    evaluation/run_omnibot_dynamic.sh
bash -n evaluation/run_omnibot_dynamic.sh || { echo "FATAL: run script broken by sed"; exit 1; }
echo "[$(date +%T)] stripped dead YAW_* env names"

# 3. build
echo "[$(date +%T)] colcon build ..."
source /opt/ros/jazzy/setup.bash
colcon build --packages-select ammr_wholebody_mpc my_omnibot_description \
    > evaluation/logs/build_noyaw.log 2>&1 || {
        echo "FATAL: build failed, see evaluation/logs/build_noyaw.log"; exit 1; }
echo "[$(date +%T)] build ok"

# 4/5. the two batches
echo "[$(date +%T)] === batch M (no heading reference) ==="
./evaluation/run_noyaw.sh         > evaluation/logs/noyawM.log 2>&1
echo "[$(date +%T)] === batch N (M + PLAN_BLEND=0) ==="
./evaluation/run_noyaw_noblend.sh > evaluation/logs/noblendN.log 2>&1
echo "[$(date +%T)] chain complete"
