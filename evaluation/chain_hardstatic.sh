#!/usr/bin/env bash
# A/B on the door-post grazes: shared soft slack vs hard k=0 for walls only.
#
# The residual static contacts on bigarena are all door posts, and the per-cycle
# diagnostic showed the wall constraint IS built and IS violated -- the data path
# is intact, so what remains is how the QP prices a violation. With one epsilon
# shared by every obstacle at a step, a single row needing relaxation loosens all
# the others, and tracking can outbid safety.
#
#   A  as-is (shared soft slack)
#   B  cbf_hard_k0_static: walls at k=0 get their own epsilon, pinned to zero.
#      The dynamic block keeps its slack, unlike the earlier HARD_K0 which
#      hardened everything at k=0 and produced 13-31 infeasible events a trial.
#
# Judged on PER-CYCLE counts, not contacts: at n=4 the contact count has no
# resolution (report S6.6), but "cycles where A z - l < 0 while eps > 0" has
# thousands of samples inside a single trial.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

LOCKFILE=/tmp/omnibot_dynamic.pid
if [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE" 2>/dev/null)" 2>/dev/null; then
    echo "ERROR: a batch is already running."; exit 1
fi
CSV=evaluation/results/omnibot_dynamic_gmpc_scan.csv
N=4

run_arm () {                      # $1 outdir   $2.. extra env
    local out="$1"; shift
    mkdir -p "evaluation/results/$out"; rm -rf "evaluation/bags/archive_$out"
    rm -f "$CSV"
    echo "[$(date +%T)] === $out ==="
    env "$@" MASK_HW=10.0 POSES_CSV="$PWD/evaluation/results/bigarena_poses_big.csv" \
        BIGARENA=1 TRAJ=bigarena_traffic GUI=0 \
        MARGIN_MODE=fixed CBF_SAFE_MARGIN=0.60 \
        PLANNER_SCAN=/scan_filtered PLAN_BLEND=0.0 \
        CBF_ALPHA=0.5 CBF_PRUNE_RANGE=1.2 CBF_FAR_STRIDE=2 \
        ./evaluation/run_omnibot_dynamic.sh "$N" 250 0 0 gmpc_scan \
        > "evaluation/logs/${out}.log" 2>&1
    cp "$CSV" "evaluation/results/$out/batch.csv"
    local ad="evaluation/bags/archive_$out"; mkdir -p "$ad"
    for d in evaluation/bags/gmpc_cbf__scan_seed*; do
        case "$d" in *__prev_*) continue;; esac
        [ -f "$d/metadata.yaml" ] || continue
        s=$(basename "$d" | sed 's/.*seed//'); [ "$s" -le "$N" ] || continue
        cp -r "$d" "$ad/$(basename "$d")" 2>/dev/null
    done
    cp "$CSV" "$ad/results.csv"
    echo "[$(date +%T)] $out done"
}

run_arm slackA HARD_K0_STATIC=0
run_arm slackB HARD_K0_STATIC=1
echo "[$(date +%T)] A/B complete"
