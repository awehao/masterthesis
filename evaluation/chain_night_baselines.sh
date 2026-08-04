#!/usr/bin/env bash
# Bring the baselines up to the same route coverage as GMPC, on the corrected
# disc collision geometry.
#
# GMPC has pose sets A-F (n ~= 177); MPPI and RPP only have A and B (n ~= 55).
# The precision of a THREE-WAY comparison is set by the smallest arm, so adding
# more GMPC trials buys nothing -- the baselines are what need trials. After
# this run all three methods have run the same six sets, 180 routes each, and
# every route is paired across all three.
#
# Order is interleaved by pose set rather than by method so that if the night is
# cut short, what exists is still balanced: a complete set for both baselines
# rather than all of MPPI and none of RPP.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

LOCKFILE=/tmp/omnibot_dynamic.pid
if [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE" 2>/dev/null)" 2>/dev/null; then
    echo "ERROR: a batch is already running (pid $(cat "$LOCKFILE"))."; exit 1
fi

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

for s in c d e f; do
    P="evaluation/results/bigarena_poses_${s}.csv"
    run_base mppi "discmppi${s^^}" "$P"
    run_base rpp  "discrpp${s^^}"  "$P"
done
echo "[$(date +%T)] night baselines complete"
