#!/usr/bin/env bash
# Second pose set for all three methods, to double n.
#
# The headline numbers currently rest on 24 paired seeds: GMPC 2 contacts, MPPI
# 1, RPP 18. At n=24 a 2/24 contact rate has a 95% interval of roughly 1-27%,
# which is too wide to claim anything close to zero. Running the same three
# methods over a second, independent set of 30 random start/goal pairs takes
# each to n~=54 and roughly halves that interval.
#
# The poses are a fresh draw (random_poses.py --seed 7). Every method sees the
# same 30 traverses -- paired, exactly as before.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

LOCKFILE=/tmp/omnibot_dynamic.pid
if [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE" 2>/dev/null)" 2>/dev/null; then
    echo "ERROR: a batch is already running (pid $(cat "$LOCKFILE"))."; exit 1
fi
POSES="$PWD/evaluation/results/bigarena_poses_b.csv"

run_gmpc () {
    local out=gmpcB CSV=evaluation/results/omnibot_dynamic_gmpc_scan.csv
    mkdir -p "evaluation/results/$out"
    echo "[$(date +%T)] === GMPC N-config, poses B ==="
    POSES_CSV="$POSES" BIGARENA=1 TRAJ=bigarena_traffic GUI=0 \
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
    echo "[$(date +%T)] GMPC poses B done (bags $(ls "$ad" | wc -l))"
}

run_base () {                  # $1 = mppi | rpp
    local m="$1" CSV="evaluation/results/omnibot_dynamic_$1.csv"
    mkdir -p "evaluation/results/base_${m}B"
    echo "[$(date +%T)] === $m, poses B ==="
    POSES_CSV="$POSES" BIGARENA=1 TRAJ=bigarena_traffic GUI=0 \
        ./evaluation/run_omnibot_dynamic.sh 30 250 0 0 "$m" \
        > "evaluation/logs/base_${m}B.log" 2>&1
    cp "$CSV" "evaluation/results/base_${m}B/batch.csv"
    local ad="evaluation/bags/archive_base_${m}B"; mkdir -p "$ad"
    for d in evaluation/bags/${m}__${m}_seed*; do
        case "$d" in *__prev_*) continue;; esac
        [ -f "$d/metadata.yaml" ] || continue
        cp -r "$d" "$ad/$(basename "$d")" 2>/dev/null
    done
    echo "[$(date +%T)] $m poses B done (bags $(ls "$ad" | wc -l))"
}

# The B file reuses seed numbers 1-30 with different coordinates, so 30 trials
# cover it exactly. Bags and CSVs would collide with the A batches, which is why
# each run archives into its own directory immediately afterwards.
run_gmpc
run_base mppi
run_base rpp
echo "[$(date +%T)] poses-B sweep complete"
