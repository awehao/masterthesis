#!/usr/bin/env bash
# min_cluster_pts test, after the poses-B sweep.
#
# The residual contacts are a PERCEPTION limit, not a controller one. Measured
# over the 3 s before closest approach (robot within 3 m, union of the dynamic
# and static constraint topics -- counting only one of them halves every number
# and produced a spurious flat 50%):
#
#   dyn_obs_3  r=0.15   coverage 83%  (q1 75%, worst 12%)
#   dyn_obs_9  r=0.20   coverage 100% (q1 88%)
#   everything r>=0.25  coverage 100% (q1 100%)
#
# dyn_obs_3 is also 4 of the 8 contacts in 172 trials. At 360 beams over 360 deg
# the beams are 5.2 cm apart at 3 m, so a 0.15 m body returns one or two points
# and min_cluster_pts = 2 discards it.
#
# Judged on COVERAGE, not contacts: coverage is measured on every encounter of
# every trial, so 15 trials settle it, while a contact rate of 2/29 needs
# hundreds of trials to move detectably.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

LOCKFILE=/tmp/omnibot_dynamic.pid
echo "[$(date +%T)] waiting for the poses-B sweep ..."
while ! grep -qE "poses-B sweep complete|ABORT" evaluation/logs/chainB.log 2>/dev/null; do sleep 120; done
while [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE" 2>/dev/null)" 2>/dev/null; do sleep 60; done
sleep 30

# Only now: the forwarding list lives in a script bash is reading line by line
# while a batch runs, and colcon swaps install/ under live nodes.
if ! grep -q MIN_CLUSTER_PTS evaluation/run_omnibot_dynamic.sh; then
    sed -i 's/ STATIC_WINDOW MIN_NET_SPEED STATIC_KEEP_VEL; do/ STATIC_WINDOW MIN_NET_SPEED STATIC_KEEP_VEL MIN_CLUSTER_PTS; do/' \
        evaluation/run_omnibot_dynamic.sh
    bash -n evaluation/run_omnibot_dynamic.sh || { echo "FATAL: sed broke the run script"; exit 1; }
fi
source /opt/ros/jazzy/setup.bash
colcon build --packages-select my_omnibot_description > evaluation/logs/build_mincluster.log 2>&1
grep -q MIN_CLUSTER_PTS \
    install/my_omnibot_description/share/my_omnibot_description/launch/omni_bot_dynamic.launch.py \
    || { echo "FATAL: launch hook not installed"; exit 1; }
echo "[$(date +%T)] build ok"

CSV=evaluation/results/omnibot_dynamic_gmpc_scan.csv
run_one () {                     # $1 label  $2 outdir  $3.. env
    local label="$1" out="$2"; shift 2
    mkdir -p "evaluation/results/$out"
    echo "[$(date +%T)] === $label ==="
    env "$@" \
        POSES_CSV="$PWD/evaluation/results/bigarena_poses.csv" \
        BIGARENA=1 TRAJ=bigarena_traffic GUI=0 \
        PLANNER_SCAN=/scan_filtered PLAN_BLEND=0.0 \
        CBF_ALPHA=0.5 CBF_SAFE_MARGIN=0.60 \
        CBF_PRUNE_RANGE=1.2 CBF_FAR_STRIDE=2 \
        ./evaluation/run_omnibot_dynamic.sh 15 250 0 0 gmpc_scan \
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
    echo "[$(date +%T)] $label done (bags $(ls "$ad" | wc -l))"
}

run_one "MC1  min_cluster_pts 1" mincl1 MIN_CLUSTER_PTS=1
echo "[$(date +%T)] min_cluster test complete"
