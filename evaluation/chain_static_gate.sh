#!/usr/bin/env bash
# Four batches attacking the residual contacts, all on top of the N config.
#
# Diagnosis (batch N, strict analysis): 2/29 contacts, and the near misses are
# dominated by the SLOWEST movers. The net-displacement gate asks for
# min_net_speed averaged over static_window_s = 0.10 m over 2 s at the defaults;
# a 0.10 m/s mover covers 0.20 m, only 2x the threshold, so occlusion tips it
# either way. dyn_obs_2 and dyn_obs_8 (0.10-0.11 m/s) were routed static at 2 of
# their closest approaches each. Static routing publishes v = 0, so over the
# 1.0 s horizon the CBF is wrong by ~0.10 m -- the same order as the residual
# penetrations (-0.019, -0.039, -0.063 m).
#
#   P  static_window_s 4.0        classify more reliably (0.40 m vs 0.10 m gate)
#   Q  static_keep_velocity 1     make a misclassification cost nothing
#   R  both
#   S  min_net_speed 0.03         lower the bar instead (risks ghost movers)
#
# Everything else is exactly the N configuration, so each batch differs from N
# by one thing (R by two, deliberately: it is the combination of P and Q).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

LOCKFILE=/tmp/omnibot_dynamic.pid
if [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE" 2>/dev/null)" 2>/dev/null; then
    echo "ERROR: a batch is already running (pid $(cat "$LOCKFILE"))."; exit 1
fi

CSV=evaluation/results/omnibot_dynamic_gmpc_scan.csv

run_one () {                    # $1 label  $2 outdir   $3.. extra env
    local label="$1" out="$2"; shift 2
    mkdir -p "evaluation/results/$out"
    echo "[$(date +%T)] === $label ==="
    [ -s "$CSV" ] && cp "$CSV" "evaluation/results/archived_$(date +%Y%m%d_%H%M%S).csv"

    env "$@" \
        POSES_CSV="$PWD/evaluation/results/bigarena_poses.csv" \
        BIGARENA=1 TRAJ=bigarena_traffic GUI=0 \
        PLANNER_SCAN=/scan_filtered PLAN_BLEND=0.0 \
        CBF_ALPHA=0.5 CBF_SAFE_MARGIN=0.60 \
        CBF_PRUNE_RANGE=1.2 CBF_FAR_STRIDE=2 \
        ./evaluation/run_omnibot_dynamic.sh 30 250 0 0 gmpc_scan \
        > "evaluation/logs/${out}.log" 2>&1 &
    local bpid=$!

    # Abort immediately if the first trial shows a robot that never moved.
    ./evaluation/guard_first_trial.sh "$CSV" || { wait $bpid 2>/dev/null; return 1; }
    wait $bpid

    cp "$CSV" "evaluation/results/$out/batch.csv"
    # Archive the bags: only CLOSED ones (metadata.yaml present), and never the
    # __prev copies, which belong to the previous batch.
    local ad="evaluation/bags/archive_$out"; mkdir -p "$ad"
    for d in evaluation/bags/gmpc_cbf__scan_seed*; do
        case "$d" in *__prev_*) continue;; esac
        [ -f "$d/metadata.yaml" ] || continue
        cp -r "$d" "$ad/$(basename "$d")" 2>/dev/null
    done
    echo "[$(date +%T)] $label done -> evaluation/results/$out/batch.csv  (bags $(ls "$ad" | wc -l))"
}

run_one "P  static_window 4.0"        staticP  STATIC_WINDOW=4.0
run_one "Q  keep velocity"            keepvelQ STATIC_KEEP_VEL=1
run_one "R  window 4.0 + keep vel"    bothR    STATIC_WINDOW=4.0 STATIC_KEEP_VEL=1
run_one "S  min_net_speed 0.03"       netS     MIN_NET_SPEED=0.03

echo "[$(date +%T)] chain complete"
