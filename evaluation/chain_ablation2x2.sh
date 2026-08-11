#!/usr/bin/env bash
# The 2x2: predictive CBF against reactive shield.
#
#   A  GMPC alone                what pure trajectory tracking gives
#   B  GMPC + CBF                what the barrier adds on its own
#   C  GMPC + shield             what the reactive layer gives on its own
#   D  GMPC + CBF + shield       the full stack
#
#   B - A   predictive avoidance
#   C - A   reactive protection alone
#   D - B   what the shield adds where perception or the CBF misses
#   D - C   what the barrier adds in early avoidance and efficiency
#
# D already exists as gmpc100 (96 trials, same routes, same settings), so only
# A, B and C run here.
#
# The question this is really for is whether the shield makes the CBF redundant.
# The cycle-level attribution already says no -- the barrier constrains 90% of
# cycles while the shield touches 1.8% -- but that is an internal measure. C
# answers it from the outside: if C alone were as good as D, the barrier would
# have nothing left to justify.
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
exec 2>&1

LOCKFILE=/tmp/omnibot_dynamic.pid
if [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE" 2>/dev/null)" 2>/dev/null; then
    echo "ERROR: a batch is already running."; exit 1
fi
N=100
POSES="$PWD/evaluation/results/bigarena_poses_big.csv"
say () { echo "[$(date +%H:%M:%S)] $*"; }

run_arm () {              # $1 outdir  $2 method  $3 csv-tag  $4 prefix  $5.. env
    local out="$1" method="$2" tag="$3" prefix="$4"; shift 4
    local csv="evaluation/results/omnibot_dynamic_${tag}.csv"
    mkdir -p "evaluation/results/$out"; rm -rf "evaluation/bags/archive_$out"
    rm -f "$csv"
    # Clear this arm's bag pattern BEFORE running. A and C share the
    # gmpc_nocbf method and therefore the same bag names, as do B and D;
    # a trial that fails bring-up leaves the previous arm's bag in place,
    # and the archive step would then file another arm's run under this
    # one -- the contamination that put foreign bags in all three of the
    # three-method archives.
    rm -rf evaluation/bags/${prefix}seed*
    say "=== $out ($method, $N routes) ==="
    env "$@" MASK_HW=10.0 POSES_CSV="$POSES" \
        BIGARENA=1 TRAJ=bigarena_traffic GUI=0 \
        MARGIN_MODE=fixed CBF_SAFE_MARGIN=0.60 \
        PLANNER_SCAN=/scan_filtered PLAN_BLEND=0.0 \
        CBF_ALPHA=0.5 CBF_PRUNE_RANGE=1.2 CBF_FAR_STRIDE=2 \
        POSE_SOURCE=odom HARD_K0_STATIC=0 \
        ./evaluation/run_omnibot_dynamic.sh "$N" 250 0 0 "$method" \
        > "evaluation/logs/${out}.log" 2>&1 &
    local bpid=$!
    ./evaluation/guard_first_trial.sh "$csv" || {
        wait $bpid 2>/dev/null; say "$out ABORTED"; return 1; }
    wait $bpid
    cp "$csv" "evaluation/results/$out/batch.csv"
    # Only this arm's own bags -- the directory holds every earlier batch, and a
    # bag from another scenario scored against this map reports -0.300 for a
    # trial that touched nothing.
    local ad="evaluation/bags/archive_$out"; mkdir -p "$ad"
    for d in evaluation/bags/${prefix}seed*; do
        case "$d" in *__prev_*) continue;; esac
        [ -f "$d/metadata.yaml" ] || continue
        local s; s=$(basename "$d" | sed 's/.*seed//'); [ "$s" -le "$N" ] || continue
        cp -r "$d" "$ad/$(basename "$d")" 2>/dev/null
    done
    cp "$csv" "$ad/results.csv"
    local neg; neg=$(awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="min_clearance_m") k=i; next}
                              $k+0 < 0 {n++} END{print n+0}' "$csv")
    say "$out done: $(( $(wc -l < "$csv") - 1 )) trials, $neg negative"
}

# A and C use gmpc_nocbf (cbf:=false); they differ only in SHIELD.
run_arm ablA gmpc_nocbf nocbf 'gmpc_cbf__nocbf_' SHIELD=0
run_arm ablC gmpc_nocbf nocbf 'gmpc_cbf__nocbf_' SHIELD=1
run_arm ablB gmpc_scan  gmpc_scan 'gmpc_cbf__scan_' SHIELD=0

say "analysing"
python3 evaluation/summarise_ablation2x2.py > evaluation/results/ABLATION2X2_SUMMARY.md 2>&1
say "ALL DONE -> evaluation/results/ABLATION2X2_SUMMARY.md"
