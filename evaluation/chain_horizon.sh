#!/usr/bin/env bash
# Horizon sweep on the current (N) configuration.
#
# Pooled over 172 trials the residual contact rate is 4.7% (8 contacts), and
# dyn_obs_3 alone accounts for 4 of the 8 while being 1 of 10 movers. It is the
# fastest (0.18 m/s) and smallest (r = 0.15 m). At the measured feasibility
# boundary -- escape speed 0.20 m/s, alpha 0.5, r_eff 0.90 -- it needs the
# largest minimum maintainable distance of any mover, 0.86 m, and avoiding it
# proactively needs roughly 2 s of warning. The horizon is 1.0 s.
#
# A longer horizon was tested before and was monotonically WORSE (N=40, N=60),
# but that was measured while the heading reference was still driving the base:
# solve time exceeded the 50 ms period in 1.3-9% of cycles and N=60 hit the OSQP
# iteration limit 22-52 times per run. With the heading reference gone, solve
# over 50 ms is 0.0% of cycles. The constraint that defeated the long horizon no
# longer exists, so the old result does not carry over and is re-tested here.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

LOCKFILE=/tmp/omnibot_dynamic.pid
if [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE" 2>/dev/null)" 2>/dev/null; then
    echo "ERROR: a batch is already running (pid $(cat "$LOCKFILE"))."; exit 1
fi
CSV=evaluation/results/omnibot_dynamic_gmpc_scan.csv

run_one () {                    # $1 label  $2 outdir  $3.. extra env
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

run_one "H30  horizon 30 (1.5 s)" horiz30 HORIZON=30
run_one "H40  horizon 40 (2.0 s)" horiz40 HORIZON=40
echo "[$(date +%T)] horizon sweep complete"
