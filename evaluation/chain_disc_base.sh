#!/usr/bin/env bash
# MPPI and RPP with the corrected disc collision body.
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

run_base () {                    # $1 method  $2 outdir  $3 poses file
    local m="$1" out="$2" poses="$3"
    local CSV="evaluation/results/omnibot_dynamic_${m}.csv"
    mkdir -p "evaluation/results/$out"
    rm -rf "evaluation/bags/archive_$out"
    echo "[$(date +%T)] === $m -> $out ==="
    POSES_CSV="$PWD/$poses" BIGARENA=1 TRAJ=bigarena_traffic GUI=0 \
        ./evaluation/run_omnibot_dynamic.sh 30 250 0 0 "$m" \
        > "evaluation/logs/${out}.log" 2>&1
    cp "$CSV" "evaluation/results/$out/batch.csv"
    local ad="evaluation/bags/archive_$out"; mkdir -p "$ad"
    for d in evaluation/bags/${m}__${m}_seed*; do
        case "$d" in *__prev_*) continue;; esac
        [ -f "$d/metadata.yaml" ] || continue
        cp -r "$d" "$ad/$(basename "$d")" 2>/dev/null
    done
    echo "[$(date +%T)] $out done (bags $(ls "$ad" | wc -l))"
}

run_base mppi discmppiA evaluation/results/bigarena_poses.csv
run_base mppi discmppiB evaluation/results/bigarena_poses_b.csv
run_base rpp  discrppA  evaluation/results/bigarena_poses.csv
run_base rpp  discrppB  evaluation/results/bigarena_poses_b.csv
echo "[$(date +%T)] disc baselines complete"
