#!/usr/bin/env bash
# Does the shield hold up on the full scenario, not just seed27?
#
# 20 routes with the shield ON and everything else identical to poseF, which
# already ran the same route file with the same pose source, the same soft
# slack and the same min_track_speed. The shield is therefore the only
# variable, and poseF supplies the control without spending another 1.1 h.
#
# 20 rather than 40 because the useful question tonight is directional. poseF
# contacted on 6 of 35 (17%), so 20 routes should carry 3-4; if the shield
# takes that to zero, McNemar sits around p = 0.06-0.13 -- clear in direction,
# short of the sample a headline number would need. seed27 went 9/9 -> 0/10, so
# an effect of that size is resolvable here; a small one would not be, and this
# batch should not be read as if it were (report S6.6).
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

out=shield20
mkdir -p "evaluation/results/$out"; rm -rf "evaluation/bags/archive_$out"
rm -f "$CSV"
say "=== $out : shield ON, otherwise identical to poseF ==="
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
neg=$(awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="min_clearance_m") k=i; next}
               $k+0 < 0 {n++} END{print n+0}' "$CSV")
say "$out done: $(( $(wc -l < "$CSV") - 1 )) trials, $neg negative"

say "analysing"
python3 evaluation/summarise_shield20.py > evaluation/results/SHIELD20_SUMMARY.md 2>&1
say "ALL DONE -> evaluation/results/SHIELD20_SUMMARY.md"
