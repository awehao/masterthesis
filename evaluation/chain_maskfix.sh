#!/usr/bin/env bash
# OBSOLETE (2026-08-08): this batch ran under motion limits that do not match
# the hardware -- vx +-0.35 exceeds the chassis by 26%, acceleration was 1/8 of
# what the wheels allow, and velocity_smoother clamped reverse to 0.20 so the
# vx_min arm measured nothing. Kept for provenance; its numbers are void and it
# must not be re-run as written. See the fourth report, section 0.
#
# The mask fix, then the acceleration that was never actually applied.
#
# Two independent defects were found by inspection, and each gets its own arm so
# the effects can be told apart. Both run the SAME 80 routes as new_fixed
# (7/71 contacts), so every arm is paired against it and against each other.
#
#   mask10   blocked_halfwidth_deg 15 -> 10
#            21 trials at different positions show only 34 beams are genuinely
#            self-occluded (four sectors of 8-9 deg at +-45 and +-135), while
#            the mask discarded 120. Worse, the 60 deg clear gaps were narrower
#            than a large mover's visible arc -- r=0.82 at 0.5 m subtends
#            60.8 deg -- so 47 of 60 close encounters had the arc cut in two,
#            splitting the cluster, jumping the 0.80 m association gate and
#            resetting the track. Coverage for the large movers at 0.5-1.5 m was
#            31-53%, against 82-100% for the small ones. At 10 deg the gaps are
#            70 deg and neither large mover straddles them inside 1 m.
#
#   accel    ax/ay/az 0.8/0.6/1.2 -> 1.5/1.0/2.0
#            gmpc_params.yaml already says 1.5/1.0/2.0, but cbf_overrides in the
#            launch is applied AFTER the yaml and defaults to 0.8/0.6/1.2, so
#            the yaml value has never taken effect. This also means the earlier
#            "stage 4: full acceleration -> contacts 1 -> 0" result is void:
#            acceleration never changed, and that difference was obstacle phase.
#
#   both     mask10 + accel
#
# 40 trials per arm. At the 10% contact rate of new_fixed the 95% interval at
# n=40 is roughly 3-24%, so this sizes for "did it move a lot", not for a
# precise rate.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

LOCKFILE=/tmp/omnibot_dynamic.pid
if [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE" 2>/dev/null)" 2>/dev/null; then
    echo "ERROR: a batch is already running."; exit 1
fi
CSV=evaluation/results/omnibot_dynamic_gmpc_scan.csv
P=evaluation/results/bigarena_poses_big.csv

run_one () {                     # $1 outdir  $2.. env
    local out="$1"; shift
    mkdir -p "evaluation/results/$out"; rm -rf "evaluation/bags/archive_$out"
    echo "[$(date +%T)] === $out ==="
    env "$@" POSES_CSV="$PWD/$P" BIGARENA=1 TRAJ=bigarena_traffic GUI=0 \
        MARGIN_MODE=fixed CBF_SAFE_MARGIN=0.60 \
        PLANNER_SCAN=/scan_filtered PLAN_BLEND=0.0 \
        CBF_ALPHA=0.5 CBF_PRUNE_RANGE=1.2 CBF_FAR_STRIDE=2 \
        ./evaluation/run_omnibot_dynamic.sh 40 250 0 0 gmpc_scan \
        > "evaluation/logs/${out}.log" 2>&1 &
    local bpid=$!
    ./evaluation/guard_first_trial.sh "$CSV" || { wait $bpid 2>/dev/null; return 1; }
    wait $bpid
    cp "$CSV" "evaluation/results/$out/batch.csv"
    local ad="evaluation/bags/archive_$out"; mkdir -p "$ad"
    for d in evaluation/bags/gmpc_cbf__scan_seed*; do
        case "$d" in *__prev_*) continue;; esac
        [ -f "$d/metadata.yaml" ] || continue
        s=$(basename "$d" | sed 's/.*seed//'); [ "$s" -le 40 ] || continue
        cp -r "$d" "$ad/$(basename "$d")" 2>/dev/null
    done
    cp "$CSV" "$ad/results.csv"
    local neg
    neg=$(awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="min_clearance_m") k=i; next}
                   $k+0 < 0 {n++} END{print n+0}' "$CSV")
    echo "[$(date +%T)] $out done: $(( $(wc -l < "$CSV") - 1 )) trials, $neg negative"
}

run_one mask10 MASK_HW=10.0                       # mask fix only
run_one accel  MASK_HW=15.0 AX_MAX=1.5 AY_MAX=1.0 AZ_MAX=2.0   # accel only
run_one both   MASK_HW=10.0 AX_MAX=1.5 AY_MAX=1.0 AZ_MAX=2.0   # both
echo "[$(date +%T)] mask/accel sweep complete"
