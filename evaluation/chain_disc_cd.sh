#!/usr/bin/env bash
# The N configuration with the chassis collision body corrected to a disc.
#
# gz simulated a 0.45 x 0.45 box (half-extent 0.225) while the analysis judged
# clearance against the real chassis, a disc of radius 0.300 measured off the
# visual mesh. The simulated robot was therefore 7.5 cm smaller than the real
# one along its axes, and any reported contact shallower than that had not
# physically happened. Across ~230 trials of the final configuration the two
# measures disagreed about 11 of the 15 reported contacts: only 4 were deeper
# than 7.5 cm.
#
# With the collision body a 0.300 m cylinder, physics and measurement finally
# agree and "contact" means the robot actually hit something. Earlier batches
# are NOT comparable to this one: the robot is now larger on its axes (0.300 vs
# 0.225) and slightly smaller on its diagonals (0.300 vs 0.318), so it will
# touch things it used to squeeze past.
#
# Both pose sets are run, giving n = 60 on the corrected geometry.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

LOCKFILE=/tmp/omnibot_dynamic.pid
if [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE" 2>/dev/null)" 2>/dev/null; then
    echo "ERROR: a batch is already running (pid $(cat "$LOCKFILE"))."; exit 1
fi
CSV=evaluation/results/omnibot_dynamic_gmpc_scan.csv

run_one () {                     # $1 label  $2 outdir  $3 poses file
    local label="$1" out="$2" poses="$3"
    mkdir -p "evaluation/results/$out"
    rm -rf "evaluation/bags/archive_$out"
    echo "[$(date +%T)] === $label ==="
    POSES_CSV="$PWD/$poses" \
    BIGARENA=1 TRAJ=bigarena_traffic GUI=0 \
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
    echo "[$(date +%T)] $label done (bags $(ls "$ad" | wc -l))"
}

run_one "DISC-C  poses C" discC evaluation/results/bigarena_poses_c.csv
run_one "DISC-D  poses D" discD evaluation/results/bigarena_poses_d.csv
echo "[$(date +%T)] disc CD sweep complete"
