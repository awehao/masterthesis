#!/usr/bin/env bash
# Symmetric vx: vx_min -0.20 -> -0.35, then re-run pose sets A and B.
#
# The chassis is omnidirectional (mecanum), so forward and reverse thrust are
# symmetric; vx_min = -0.20 has no physical basis and looks inherited from a
# differential-drive default. It is also the binding constraint on every CBF
# encounter, because backing off is how the barrier buys distance.
#
# What it buys, from the feasibility boundary (alpha 0.5, margin 0.60):
#
#   net escape speed vs dyn_obs_5 (0.10 m/s):  0.10 -> 0.25 m/s   (2.5x)
#   minimum maintainable distance d*:          1.53 -> 1.29 m     (-16%)
#   d* against dyn_obs_3 (fastest):            1.01 -> 0.76 m     (-24%)
#
# On the deepest contact of the final benchmark (A seed17, -0.076 m against
# dyn_obs_5) the robot was already commanding reverse and simply could not pull
# away: recovering the 0.78 m of overlap needed 7.8 s at 0.10 m/s of net escape,
# against an encounter lasting a few seconds. At 0.25 m/s it needs 3.1 s, which
# is inside the encounter.
#
# Waits for the final benchmark: changing gmpc_params.yaml mid-run would make
# the D pose set inconsistent with A-C, which are already recorded.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

LOCKFILE=/tmp/omnibot_dynamic.pid
echo "[$(date +%T)] waiting for the final benchmark ..."
while ! grep -qE "final benchmark complete|ABORTED" evaluation/logs/chainFinal.log 2>/dev/null; do sleep 120; done
while [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE" 2>/dev/null)" 2>/dev/null; do sleep 60; done
sleep 30

Y=src/ammr_wholebody_mpc/config/gmpc_params.yaml
if grep -q '^ *vx_min: *-0\.20' "$Y"; then
    sed -i 's/^\( *vx_min: *\)-0\.20/\1-0.35/' "$Y"
    grep -q '^ *vx_min: *-0.35' "$Y" || { echo "FATAL: vx_min edit failed"; exit 1; }
    echo "[$(date +%T)] vx_min -0.20 -> -0.35"
fi
source /opt/ros/jazzy/setup.bash
colcon build --packages-select ammr_wholebody_mpc > evaluation/logs/build_vxsym.log 2>&1
grep -q '\-0.35' install/ammr_wholebody_mpc/share/ammr_wholebody_mpc/config/gmpc_params.yaml \
    || { echo "FATAL: install/ still has the old vx_min"; exit 1; }
echo "[$(date +%T)] build ok"

CSV=evaluation/results/omnibot_dynamic_gmpc_scan.csv
run_one () {                     # $1 outdir  $2 poses file
    local out="$1" poses="$2"
    mkdir -p "evaluation/results/$out"; rm -rf "evaluation/bags/archive_$out"
    echo "[$(date +%T)] === $out ==="
    env POSES_CSV="$PWD/$poses" BIGARENA=1 TRAJ=bigarena_traffic GUI=0 \
        PLANNER_SCAN=/scan_filtered PLAN_BLEND=0.0 \
        CBF_ALPHA=0.5 CBF_SAFE_MARGIN=0.60 \
        CBF_PRUNE_RANGE=1.2 CBF_FAR_STRIDE=2 \
        ./evaluation/run_omnibot_dynamic.sh 30 250 0 0 gmpc_scan \
        > "evaluation/logs/${out}.log" 2>&1 &
    local bpid=$!
    ./evaluation/guard_first_trial.sh "$CSV" || { wait $bpid 2>/dev/null; return 1; }
    wait $bpid
    cp "$CSV" "evaluation/results/$out/batch.csv"
    local ad="evaluation/bags/archive_$out"; mkdir -p "$ad"
    for d in evaluation/bags/gmpc_cbf__scan_seed*; do
        case "$d" in *__prev_*) continue;; esac
        [ -f "$d/metadata.yaml" ] || continue
        cp -r "$d" "$ad/$(basename "$d")" 2>/dev/null
    done
    cp "$CSV" "$ad/results.csv"
    echo "[$(date +%T)] $out done (bags $(ls -d "$ad"/*_seed* 2>/dev/null | wc -l))"
}

run_one vxsym_A evaluation/results/bigarena_poses.csv
run_one vxsym_B evaluation/results/bigarena_poses_b.csv
echo "[$(date +%T)] vx symmetric sweep complete"
