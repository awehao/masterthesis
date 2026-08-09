#!/usr/bin/env bash
# Extend the shield's paired comparison from 20 routes to 30.
#
# shield20 covered routes 1-20; poseF (the control) already ran 1-35. Routes
# 21-30 are the ten the shield has not seen, so running just those extends the
# pairing without repeating work.
#
# The route file relabels seeds 21-30 as 1-10 because pose_for_seed() matches
# column 1 against the TRIAL index, not against the route id -- leaving the
# original numbering makes the lookup miss and the harness silently falls back
# to a (0,0) spawn with a (0,0) goal, which is how two earlier attempts ended
# up measuring a robot that never left the start.
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
exec 2>&1

LOCKFILE=/tmp/omnibot_dynamic.pid
if [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE" 2>/dev/null)" 2>/dev/null; then
    echo "ERROR: a batch is already running."; exit 1
fi
CSV=evaluation/results/omnibot_dynamic_gmpc_scan.csv
P=evaluation/results/poses_21_30.csv
N=10
say () { echo "[$(date +%H:%M:%S)] $*"; }

grep -q '^1,' "$P" || { say "FATAL: route file not relabelled"; exit 1; }
say "routes 21-30 relabelled 1-10"

out=shield30
mkdir -p "evaluation/results/$out"; rm -rf "evaluation/bags/archive_$out"
rm -f "$CSV"
say "=== $out : shield ON, routes 21-30 ==="
env SHIELD=1 MASK_HW=10.0 POSES_CSV="$PWD/$P" \
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
    # Rename back to the ORIGINAL route id so the archive can be paired with
    # poseF without carrying the offset around in every analysis.
    cp -r "$d" "$ad/gmpc_cbf__scan_seed$((s + 20))" 2>/dev/null
done
neg=$(awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="min_clearance_m") k=i; next}
               $k+0 < 0 {n++} END{print n+0}' "$CSV")
say "$out done: $(( $(wc -l < "$CSV") - 1 )) trials, $neg negative"
say "ALL DONE"
