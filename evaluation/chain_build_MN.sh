#!/usr/bin/env bash
# Build, then run batches M and N back to back.
#
# NO `set -u`. The previous version had it and died silently at
# `source /opt/ros/jazzy/setup.bash`: ROS's setup scripts read variables that
# are legitimately unset, which under -u is a fatal error, so the colcon line
# never ran and no build log was ever created. `set -e` is also wrong here --
# a failed batch should not prevent the next one from running.
#
# Steps already done by the previous chain and NOT repeated: batch L archived,
# dead YAW_* names stripped from run_omnibot_dynamic.sh.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

LOCKFILE=/tmp/omnibot_dynamic.pid
if [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE" 2>/dev/null)" 2>/dev/null; then
    echo "ERROR: a batch is already running (pid $(cat "$LOCKFILE"))."; exit 1
fi

echo "[$(date +%T)] colcon build ..."
source /opt/ros/jazzy/setup.bash
# Plain build: --symlink-install corrupts entry-point metadata on the ammr
# python packages and every node then dies with PackageNotFoundError.
colcon build --packages-select ammr_wholebody_mpc my_omnibot_description \
    > evaluation/logs/build_noyaw.log 2>&1
rc=$?
echo "[$(date +%T)] colcon exit $rc"

# Trust the installed FILES, not colcon's exit code: a stale install/ is what
# would silently invalidate both batches.
PP=install/ammr_wholebody_mpc/lib/python3.12/site-packages/ammr_wholebody_mpc
if grep -q "ref_yaw\|yaw_lookahead\|Q_yaw" "$PP/gmpc_node.py" 2>/dev/null; then
    echo "FATAL: install/ gmpc_node.py still carries yaw parameters"; exit 1
fi
if ! grep -q "No heading reference" "$PP/path_processor.py" 2>/dev/null; then
    echo "FATAL: install/ path_processor.py is stale"; exit 1
fi
echo "[$(date +%T)] install/ verified: no heading reference"

echo "[$(date +%T)] === batch M (no heading reference) ==="
./evaluation/run_noyaw.sh         > evaluation/logs/noyawM.log 2>&1
echo "[$(date +%T)] M exit $?"

echo "[$(date +%T)] === batch N (M + PLAN_BLEND=0) ==="
./evaluation/run_noyaw_noblend.sh > evaluation/logs/noblendN.log 2>&1
echo "[$(date +%T)] N exit $?"

echo "[$(date +%T)] chain complete"
