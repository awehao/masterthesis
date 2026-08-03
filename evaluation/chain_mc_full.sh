#!/usr/bin/env bash
# min_cluster_pts = 1 over the full 30-pose set, plus a paired mcp = 2 repeat.
#
# The 15-trial probe was encouraging but far too small to act on: dyn_obs_3's
# median coverage went 92% -> 100% while its q1 stayed at 84% and its worst case
# fell 83% -> 59%, all on 4-5 encounters. Contacts over the same 15 seeds went
# 1 -> 0 and the worst penetration -0.019 -> 0.000, which is one event either
# way. Thirty trials roughly triples the encounter count per obstacle.
#
# The mcp = 2 repeat is not redundant with batch N. N is the only sample of this
# configuration, so a difference against it could just as easily be run-to-run
# variation; running both arms now, back to back on the same machine and the
# same 30 poses, makes the comparison paired in time as well as in route.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

LOCKFILE=/tmp/omnibot_dynamic.pid
if [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE" 2>/dev/null)" 2>/dev/null; then
    echo "ERROR: a batch is already running (pid $(cat "$LOCKFILE"))."; exit 1
fi
CSV=evaluation/results/omnibot_dynamic_gmpc_scan.csv

run_one () {                     # $1 label  $2 outdir  $3.. env
    local label="$1" out="$2"; shift 2
    mkdir -p "evaluation/results/$out"
    rm -rf "evaluation/bags/archive_$out"
    echo "[$(date +%T)] === $label ==="
    env "$@" \
        POSES_CSV="$PWD/evaluation/results/bigarena_poses.csv" \
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

run_one "MCP1  min_cluster_pts 1" mcp1_full MIN_CLUSTER_PTS=1
run_one "MCP2  min_cluster_pts 2" mcp2_full MIN_CLUSTER_PTS=2
echo "[$(date +%T)] min_cluster full sweep complete"
