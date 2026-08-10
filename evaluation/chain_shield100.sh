#!/usr/bin/env bash
# 100 routes with the shield, to get a contact rate that can be quoted.
#
# 27 paired routes gave 5 -> 0, which is the right direction but a 95% upper
# bound of 11% on zero events -- too loose to write down. 100 clean runs put
# that bound at 3%, which is a claim rather than an observation:
#
#     zero contacts in N trials  ->  95% upper bound ~ 3/N
#       27  ->  11%
#      100  ->   3%
#      300  ->   1%
#
# Routes 1-100 of bigarena_poses_big.csv. poseF and shield20/shield30 already
# cover 1-35 of them, so the first third is a re-run under identical settings --
# useful in itself, since the same-route same-config spread is the noise floor
# every other comparison is measured against.
#
# Guarded: if trial 1 comes back with path_length = 0 the batch aborts rather
# than spending five hours on a broken build.
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
exec 2>&1

LOCKFILE=/tmp/omnibot_dynamic.pid
if [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE" 2>/dev/null)" 2>/dev/null; then
    echo "ERROR: a batch is already running."; exit 1
fi
CSV=evaluation/results/omnibot_dynamic_gmpc_scan.csv
N=100
say () { echo "[$(date +%H:%M:%S)] $*"; }

out=shield100
mkdir -p "evaluation/results/$out"; rm -rf "evaluation/bags/archive_$out"
rm -f "$CSV"
say "=== $out : shield ON, $N routes ==="
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
python3 evaluation/summarise_shield100.py > evaluation/results/SHIELD100_SUMMARY.md 2>&1
say "ALL DONE -> evaluation/results/SHIELD100_SUMMARY.md"
