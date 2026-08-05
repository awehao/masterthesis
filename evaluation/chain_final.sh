#!/usr/bin/env bash
# Final benchmark: the locked navigation configuration, three methods, four
# pose sets (120 routes each), everything re-run on the same code.
#
# Re-running the baselines is not optional. The EKF fix (pose0_rejection_
# threshold disabled) lives in ekf_fusion.yaml, which MPPI and RPP load too, so
# every earlier baseline number was measured with the divergent filter. Mixing
# those with post-fix GMPC numbers would compare two different robots.
#
# Sizing: the GMPC contact rate to beat is 8/177 = 4.5%. Separating that from
# near-zero needs about 100 trials; 120 gives margin for the occasional
# bring-up failure. The localisation claim (25/175 = 14% divergence, 0/26 after
# the fix) is already at p ~= 0.03 and does not drive the sizing.
#
# Interleaved by pose set, not by method: if the run is cut short, what exists
# is balanced across all three methods rather than complete for one.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

LOCKFILE=/tmp/omnibot_dynamic.pid
if [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE" 2>/dev/null)" 2>/dev/null; then
    echo "ERROR: a batch is already running (pid $(cat "$LOCKFILE"))."; exit 1
fi

run_one () {                     # $1 method  $2 outdir  $3 poses file
    local m="$1" out="$2" poses="$3"
    local CSV="evaluation/results/omnibot_dynamic_${m}.csv"
    local bagpat tag
    case "$m" in
        gmpc_scan) bagpat='gmpc_cbf__scan_seed' ;;
        *)         bagpat="${m}__${m}_seed" ;;
    esac
    mkdir -p "evaluation/results/$out"
    rm -rf "evaluation/bags/archive_$out"
    echo "[$(date +%T)] === $m -> $out ==="

    local envs=(POSES_CSV="$PWD/$poses" BIGARENA=1 TRAJ=bigarena_traffic GUI=0)
    if [ "$m" = gmpc_scan ]; then
        envs+=(PLANNER_SCAN=/scan_filtered PLAN_BLEND=0.0
               CBF_ALPHA=0.5 CBF_SAFE_MARGIN=0.60
               CBF_PRUNE_RANGE=1.2 CBF_FAR_STRIDE=2)
    fi
    env "${envs[@]}" ./evaluation/run_omnibot_dynamic.sh 30 250 0 0 "$m" \
        > "evaluation/logs/${out}.log" 2>&1 &
    local bpid=$!
    ./evaluation/guard_first_trial.sh "$CSV" || { wait $bpid 2>/dev/null; return 1; }
    wait $bpid

    cp "$CSV" "evaluation/results/$out/batch.csv"
    local ad="evaluation/bags/archive_$out"; mkdir -p "$ad"
    for d in evaluation/bags/${bagpat}*; do
        case "$d" in *__prev_*) continue;; esac
        [ -f "$d/metadata.yaml" ] || continue
        cp -r "$d" "$ad/$(basename "$d")" 2>/dev/null
    done
    # The batch's own CSV, so the analyser never falls back to the shared file
    # and reads another batch's arrival times.
    cp "$CSV" "$ad/results.csv"
    echo "[$(date +%T)] $out done (bags $(ls -d "$ad"/*_seed* 2>/dev/null | wc -l))"
}

for s in "" _b _c _d; do
    tag=$([ -z "$s" ] && echo A || echo "${s#_}")
    tag=$(echo "$tag" | tr 'a-z' 'A-Z')
    P="evaluation/results/bigarena_poses${s}.csv"
    run_one gmpc_scan "fin_gmpc_${tag}" "$P"
    run_one mppi      "fin_mppi_${tag}" "$P"
    run_one rpp       "fin_rpp_${tag}"  "$P"
done
echo "[$(date +%T)] final benchmark complete"
