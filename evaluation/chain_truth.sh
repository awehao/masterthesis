#!/usr/bin/env bash
# Ground-truth perception ablation, after the mask/accel sweep.
#
# gmpc_truth swaps scan_obstacle_tracker for obstacle_aggregator: the CBF gets
# the movers' true poses instead of lidar-derived surface points. The
# controller, the barrier, the planner and the routes are identical, so the
# difference is the perception pipeline and nothing else.
#
# This is the ablation the fourth report needs. Right now the evidence shows
# the CBF does all of the dynamic avoidance (turning it off gives 10 contacts
# in 14 trials against 0 in 15), but it cannot say whether the residual ~10%
# contact rate is the CONTROLLER's limit or PERCEPTION's. With true poses:
#
#   contacts fall to near zero  -> the controller is sound and perception is the
#                                  bottleneck, which the two defects already
#                                  found (over-wide mask splitting large movers'
#                                  clusters, 31-53% coverage at 0.5-1.5 m)
#                                  independently support
#   contacts stay at ~10%       -> the limit is the controller or the feasibility
#                                  boundary, and better perception will not help
#
# Same 40 routes and the same fixed margins as the mask/accel arms, with the
# corrected 10 deg mask, so it pairs directly against `both`.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

LOCKFILE=/tmp/omnibot_dynamic.pid
echo "[$(date +%T)] waiting for the mask/accel sweep ..."
while ! grep -qE "mask/accel sweep complete|ABORT" evaluation/logs/chainMask.log 2>/dev/null; do sleep 120; done
while [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE" 2>/dev/null)" 2>/dev/null; do sleep 60; done
sleep 30

CSV=evaluation/results/omnibot_dynamic_gmpc_truth.csv
out=truth40
mkdir -p "evaluation/results/$out"; rm -rf "evaluation/bags/archive_$out"
echo "[$(date +%T)] === $out ==="
env MASK_HW=10.0 AX_MAX=1.5 AY_MAX=1.0 AZ_MAX=2.0 \
    POSES_CSV="$PWD/evaluation/results/bigarena_poses_big.csv" \
    BIGARENA=1 TRAJ=bigarena_traffic GUI=0 \
    MARGIN_MODE=fixed CBF_SAFE_MARGIN=0.60 \
    PLANNER_SCAN=/scan_filtered PLAN_BLEND=0.0 \
    CBF_ALPHA=0.5 CBF_PRUNE_RANGE=1.2 CBF_FAR_STRIDE=2 \
    ./evaluation/run_omnibot_dynamic.sh 40 250 0 0 gmpc_truth \
    > "evaluation/logs/${out}.log" 2>&1 &
bpid=$!
./evaluation/guard_first_trial.sh "$CSV" || { wait $bpid 2>/dev/null; exit 1; }
wait $bpid

cp "$CSV" "evaluation/results/$out/batch.csv"
ad="evaluation/bags/archive_$out"; mkdir -p "$ad"
for d in evaluation/bags/gmpc_cbf__truth_seed*; do
    case "$d" in *__prev_*) continue;; esac
    [ -f "$d/metadata.yaml" ] || continue
    s=$(basename "$d" | sed 's/.*seed//'); [ "$s" -le 40 ] || continue
    cp -r "$d" "$ad/$(basename "$d")" 2>/dev/null
done
cp "$CSV" "$ad/results.csv"
neg=$(awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="min_clearance_m") k=i; next}
               $k+0 < 0 {n++} END{print n+0}' "$CSV")
echo "[$(date +%T)] $out done: $(( $(wc -l < "$CSV") - 1 )) trials, $neg negative"
echo "[$(date +%T)] truth ablation complete"
