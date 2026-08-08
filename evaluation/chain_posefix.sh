#!/usr/bin/env bash
# Arm F: take the controller's pose from the EKF topic instead of TF.
#
# Runs AFTER chain_overnight finishes, so the rebuild cannot swap the binaries
# under a batch that is still going. Written as a separate file for the same
# reason: editing a running script rewrites it by byte offset and corrupts the
# part bash has not read yet.
#
# Why this arm exists. On hardC/seed1 the pose the GMPC used sat a median
# 0.176 m and a peak 0.826 m away from /odometry/filtered, while five
# neighbouring trials agreed with it to a median of 0.000 m -- intermittent, not
# a bias. AMCL and the EKF were both within 0.02 m of ground truth throughout,
# so localisation was healthy; what failed was lookup_transform(map,
# base_footprint, Time()), which composes the newest map->odom with the newest
# odom->base_footprint and does not require them to be the same instant.
#
# The consequence is that a hard barrier can be satisfied and still be hit: the
# CBF placed a wall 1.02 m away that was really 0.25 m away, so no amount of
# slack pricing, hierarchy or shielding would have helped. That makes this the
# first thing to fix, ahead of the QP work.
#
# Paired against hardC and softD on the same 40 routes.
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
exec 2>&1

LOCKFILE=/tmp/omnibot_dynamic.pid
CSV=evaluation/results/omnibot_dynamic_gmpc_scan.csv
N=40
say () { echo "[$(date +%H:%M:%S)] $*"; }

say "waiting for chain_overnight ..."
while ps -eo cmd --no-headers | grep -q "[c]hain_overnight"; do sleep 60; done
while [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE" 2>/dev/null)" 2>/dev/null; do sleep 30; done
sleep 20
say "overnight chain finished"

say "rebuilding with pose_source support"
source /opt/ros/jazzy/setup.bash
colcon build --packages-select ammr_wholebody_mpc my_omnibot_description 2>&1 | tail -2
python3 - <<'PY'
import subprocess
out = subprocess.run(['grep', '-c', 'pose_source',
                      'install/my_omnibot_description/share/my_omnibot_description/'
                      'launch/omni_bot_dynamic.launch.py'],
                     capture_output=True, text=True).stdout.strip()
print(f"  install carries pose_source: {out}")
PY

run_arm () {
    local out="$1"; shift
    mkdir -p "evaluation/results/$out"; rm -rf "evaluation/bags/archive_$out"
    rm -f "$CSV"
    say "=== $out ==="
    env "$@" MASK_HW=10.0 POSES_CSV="$PWD/evaluation/results/bigarena_poses_big.csv" \
        BIGARENA=1 TRAJ=bigarena_traffic GUI=0 \
        MARGIN_MODE=fixed CBF_SAFE_MARGIN=0.60 \
        PLANNER_SCAN=/scan_filtered PLAN_BLEND=0.0 \
        CBF_ALPHA=0.5 CBF_PRUNE_RANGE=1.2 CBF_FAR_STRIDE=2 \
        ./evaluation/run_omnibot_dynamic.sh "$N" 250 0 0 gmpc_scan \
        > "evaluation/logs/${out}.log" 2>&1 &
    local bpid=$!
    ./evaluation/guard_first_trial.sh "$CSV" || { wait $bpid 2>/dev/null; say "$out ABORTED"; return 1; }
    wait $bpid
    cp "$CSV" "evaluation/results/$out/batch.csv"
    local ad="evaluation/bags/archive_$out"; mkdir -p "$ad"
    for d in evaluation/bags/gmpc_cbf__scan_seed*; do
        case "$d" in *__prev_*) continue;; esac
        [ -f "$d/metadata.yaml" ] || continue
        local s; s=$(basename "$d" | sed 's/.*seed//'); [ "$s" -le "$N" ] || continue
        cp -r "$d" "$ad/$(basename "$d")" 2>/dev/null
    done
    cp "$CSV" "$ad/results.csv"
    local neg; neg=$(awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="min_clearance_m") k=i; next}
                              $k+0 < 0 {n++} END{print n+0}' "$CSV")
    say "$out done: $(( $(wc -l < "$CSV") - 1 )) trials, $neg negative"
}

# Soft slack, as softD, so the ONLY difference from softD is where the pose
# comes from. Hardening the barrier on top of a wrong pose was what made hardC
# worse, so that combination is deliberately not repeated here.
run_arm poseF POSE_SOURCE=odom HARD_K0_STATIC=0

say "writing summary"
python3 evaluation/summarise_overnight.py > evaluation/results/OVERNIGHT_SUMMARY.md 2>&1
say "ALL DONE"
