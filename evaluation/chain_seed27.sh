#!/usr/bin/env bash
# seed27, ten times per configuration.
#
# seed27 is the only failure that outlived the noise floor: contact in all four
# overnight arms (-0.130, -0.066, -0.063, -0.300) while the same-route
# same-config spread is 0.043 m. Everything else that repeated was a sub-2 cm
# graze. A reproducible case is worth ten replays far more than a fresh 40-route
# batch is worth two hours.
#
# Mechanism, established from the bag: the goal at (0.79, 10.37) sits on the
# path of dyn_obs_5, a 1.6 x 0.4 m box. On final approach the centre distance
# falls to 0.62 m -- inside the box's own 0.80 m half-length -- so the visible
# arc spans every masked sector, the cluster breaks into 7 fragments, each jumps
# the 0.80 m association gate, all the new tracks sit below min_track_age, and
# /gmpc/obstacles publishes NOTHING for 1.5 s while the box drives over the
# robot. min_h_dynamic stayed at +47 throughout: the CBF never knew.
#
# Four arms, one change at a time, so a regression can be attributed:
#   T0  as-is
#   T1  + associate against the KF prediction, Mahalanobis gate
#   T2  + one track may absorb several fragments
#   T3  + keep publishing a coasted track, radius grown with its age
#
# Judged on per-cycle coverage and dropout, not on 10 contact counts.
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
exec 2>&1

LOCKFILE=/tmp/omnibot_dynamic.pid
if [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE" 2>/dev/null)" 2>/dev/null; then
    echo "ERROR: a batch is already running."; exit 1
fi
CSV=evaluation/results/omnibot_dynamic_gmpc_scan.csv
REPS=10
say () { echo "[$(date +%H:%M:%S)] $*"; }

# Route 27 only. run_omnibot_dynamic walks POSES_CSV from the top, so a
# one-row file replayed REPS times gives the same route with fresh obstacle
# phase each run -- which is the point: the mechanism must survive phase, not
# be a single lucky alignment.
P27=evaluation/results/poses_seed27.csv
head -1 evaluation/results/bigarena_poses_big.csv > "$P27"
awk 'NR==28' evaluation/results/bigarena_poses_big.csv >> "$P27"
say "route file: $(wc -l < "$P27") lines"

run_arm () {
    local out="$1"; shift
    mkdir -p "evaluation/results/$out"; rm -rf "evaluation/bags/archive_$out"
    rm -f "$CSV"
    say "=== $out ==="
    for i in $(seq 1 "$REPS"); do
        env "$@" MASK_HW=10.0 POSES_CSV="$PWD/$P27" \
            BIGARENA=1 TRAJ=bigarena_traffic GUI=0 \
            MARGIN_MODE=fixed CBF_SAFE_MARGIN=0.60 \
            PLANNER_SCAN=/scan_filtered PLAN_BLEND=0.0 \
            CBF_ALPHA=0.5 CBF_PRUNE_RANGE=1.2 CBF_FAR_STRIDE=2 \
            POSE_SOURCE=odom \
            ./evaluation/run_omnibot_dynamic.sh 1 250 0 0 gmpc_scan \
            >> "evaluation/logs/${out}.log" 2>&1
        local ad="evaluation/bags/archive_$out"; mkdir -p "$ad"
        for d in evaluation/bags/gmpc_cbf__scan_seed1; do
            [ -f "$d/metadata.yaml" ] || continue
            rm -rf "$ad/rep$i"; cp -r "$d" "$ad/rep$i" 2>/dev/null
        done
    done
    cp "$CSV" "evaluation/results/$out/batch.csv" 2>/dev/null
    local neg; neg=$(awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="min_clearance_m") k=i; next}
                              $k+0 < 0 {n++} END{print n+0}' "$CSV" 2>/dev/null)
    say "$out done: $(( $(wc -l < "$CSV") - 1 )) runs, $neg negative"
}

run_arm T0
run_arm T1 ASSOC_PREDICT=1 ASSOC_MAHA=1
run_arm T2 ASSOC_PREDICT=1 ASSOC_MAHA=1 FRAG_MERGE=1
run_arm T3 ASSOC_PREDICT=1 ASSOC_MAHA=1 FRAG_MERGE=1 COAST_S=1.0 COAST_GROWTH=0.15

say "analysing"
python3 evaluation/summarise_seed27.py > evaluation/results/SEED27_SUMMARY.md 2>&1
say "ALL DONE -> evaluation/results/SEED27_SUMMARY.md"
