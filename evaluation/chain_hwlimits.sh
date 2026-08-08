#!/usr/bin/env bash
# GMPC under the chassis's real motion limits, paired against `both`.
#
# Every earlier batch ran on limits that do not match the hardware: vx +-0.35
# exceeds the wheels by 26%, acceleration was 1/8 of what they allow, and
# velocity_smoother clamped reverse to 0.20 so the vx_min arm measured nothing.
# The wheel Jacobian (r=0.05, L=0.245, w_max=5.55, a_max=125) gives
# 0.2775 m/s per axis, 1.1327 rad/s, 6.25 m/s^2, 25.51 rad/s^2.
#
# Same first 20 routes and same everything else as `both` (3/31 negative at
# 1.5/1.0/2.0), so the only variable is the motion box. The prior is that this
# helps: on 31 paired routes, 0.8 -> 1.5 already took contacts 6 -> 3 and jerk
# 0.701 -> 0.591, so more authority bought BOTH safety and smoothness.
#
# A smoke trial confirmed the limits reach the wheels: vx, vy and dvx/dt all
# saturate at the new values and reverse reaches -0.2775. That check is the one
# the two earlier "parameter changed" experiments skipped.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

LOCKFILE=/tmp/omnibot_dynamic.pid
if [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE" 2>/dev/null)" 2>/dev/null; then
    echo "ERROR: a batch is already running."; exit 1
fi
CSV=evaluation/results/omnibot_dynamic_gmpc_scan.csv
out=hwlimits
mkdir -p "evaluation/results/$out"; rm -rf "evaluation/bags/archive_$out"
echo "[$(date +%T)] === $out : hardware limits, 20 routes paired with both ==="
env MASK_HW=10.0 POSES_CSV="$PWD/evaluation/results/bigarena_poses_big.csv" \
    BIGARENA=1 TRAJ=bigarena_traffic GUI=0 \
    MARGIN_MODE=fixed CBF_SAFE_MARGIN=0.60 \
    PLANNER_SCAN=/scan_filtered PLAN_BLEND=0.0 \
    CBF_ALPHA=0.5 CBF_PRUNE_RANGE=1.2 CBF_FAR_STRIDE=2 \
    ./evaluation/run_omnibot_dynamic.sh 20 250 0 0 gmpc_scan \
    > "evaluation/logs/${out}.log" 2>&1 &
bpid=$!
./evaluation/guard_first_trial.sh "$CSV" || { wait $bpid 2>/dev/null; exit 1; }
wait $bpid

cp "$CSV" "evaluation/results/$out/batch.csv"
ad="evaluation/bags/archive_$out"; mkdir -p "$ad"
for d in evaluation/bags/gmpc_cbf__scan_seed*; do
    case "$d" in *__prev_*) continue;; esac
    [ -f "$d/metadata.yaml" ] || continue
    s=$(basename "$d" | sed 's/.*seed//'); [ "$s" -le 20 ] || continue
    cp -r "$d" "$ad/$(basename "$d")" 2>/dev/null
done
cp "$CSV" "$ad/results.csv"
neg=$(awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="min_clearance_m") k=i; next}
               $k+0 < 0 {n++} END{print n+0}' "$CSV")
echo "[$(date +%T)] $out done: $(( $(wc -l < "$CSV") - 1 )) trials, $neg negative"
