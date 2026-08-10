#!/usr/bin/env bash
# Routes 1-20 with the corrected shield fallback, paired against shield20.
#
# shield20 ran the same twenty routes with the OLD fallback, which clamped the
# relaxed bound with np.minimum instead of np.maximum. That turned a distant
# return's generous limit into "do not approach at all", collapsed the feasible
# set when returns surrounded the robot, and fired the last-resort zero -- on
# seed70 it converted a GMPC retreat command into a dead stop while a mover
# closed. The single-trial replay after the fix gave fallback 82 -> 0 cycles and
# clearance -0.029 -> +0.132.
#
# The question this batch answers is NOT whether the fix helps safety; one
# reproducible case already showed that. It is whether it costs distance.
# That same replay's path went 24.7 m -> 47.2 m, nearly double. If that is
# typical rather than particular, the fix has to be reworked, so path length is
# the primary metric here and contacts are secondary.
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
exec 2>&1

LOCKFILE=/tmp/omnibot_dynamic.pid
if [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE" 2>/dev/null)" 2>/dev/null; then
    echo "ERROR: a batch is already running."; exit 1
fi
CSV=evaluation/results/omnibot_dynamic_gmpc_scan.csv
N=20
say () { echo "[$(date +%H:%M:%S)] $*"; }

out=shieldfix20
mkdir -p "evaluation/results/$out"; rm -rf "evaluation/bags/archive_$out"
rm -f "$CSV"
say "=== $out : corrected shield fallback, routes 1-20 ==="
env SHIELD=1 MASK_HW=10.0 \
    POSES_CSV="$PWD/evaluation/results/bigarena_poses_big.csv" \
    BIGARENA=1 TRAJ=bigarena_traffic GUI=0 \
    MARGIN_MODE=fixed CBF_SAFE_MARGIN=0.60 \
    PLANNER_SCAN=/scan_filtered PLAN_BLEND=0.0 \
    CBF_ALPHA=0.5 CBF_PRUNE_RANGE=1.2 CBF_FAR_STRIDE=2 \
    POSE_SOURCE=odom HARD_K0_STATIC=0 \
    ./evaluation/run_omnibot_dynamic.sh "$N" 250 0 0 gmpc_scan \
    > "evaluation/logs/${out}.log" 2>&1 &
bpid=$!
./evaluation/guard_first_trial.sh "$CSV" || { wait $bpid 2>/dev/null; say "ABORTED"; exit 1; }
wait $bpid

cp "$CSV" "evaluation/results/$out/batch.csv"
ad="evaluation/bags/archive_$out"; mkdir -p "$ad"
for d in evaluation/bags/gmpc_cbf__scan_seed*; do
    case "$d" in *__prev_*) continue;; esac
    [ -f "$d/metadata.yaml" ] || continue
    s=$(basename "$d" | sed 's/.*seed//'); [ "$s" -le "$N" ] || continue
    cp -r "$d" "$ad/$(basename "$d")" 2>/dev/null
done
cp "$CSV" "$ad/results.csv"
say "$out done: $(( $(wc -l < "$CSV") - 1 )) trials"
say "analysing"
python3 evaluation/summarise_shieldfix.py > evaluation/results/SHIELDFIX_SUMMARY.md 2>&1
say "ALL DONE -> evaluation/results/SHIELDFIX_SUMMARY.md"
