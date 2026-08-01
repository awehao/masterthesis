#!/usr/bin/env bash
# Scan the CBF slack weight: how hard does the barrier have to be?
#
# Measured over 146 trials, the barrier is violated while the robot is CRUISING:
# at min_h < 0 it was using 0.77 of its speed limit (0.74 when safe), 0.04 of
# its turn rate, and the QP solved in 0.18 ms. It is not failing to avoid --
# it is choosing to, because a violation costs less than the detour. The weight
# is 500 against a node default of 10000, lowered 20x as an "ANTI-FREEZE"
# measure back when the robot preferred u = 0 forever.
#
# That freeze risk is real, so this measures BOTH ends rather than assuming the
# old setting is simply wrong:
#   safety   -- true surface clearance, min_h < 0 fraction, contacts
#   mobility -- arrival rate, time to goal, fraction of time stalled
#
# Three trials per value: enough to see the SHAPE of the trade-off and where the
# knee is, not enough to quote a rate. Follow the knee up with a proper n.
#
#   ./evaluation/run_slack_sweep.sh [N_PER_VALUE] [DURATION_S]
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
N="${1:-3}"
DUR="${2:-250}"
CSV="${HERE}/results/omnibot_dynamic_gmpc_scan.csv"

for W in 500 1500 5000 15000 50000; do
  out="${HERE}/results/slack_${W}"
  echo "=========================================================="
  echo "[$(date +%T)] CBF_SLACK_W=${W}, ${N} trials, cap ${DUR}s"
  echo "=========================================================="
  mkdir -p "$out"
  export BIGARENA=1 TRAJ=bigarena_traffic GUI=0
  export PLANNER_SCAN=/scan_filtered PLAN_BLEND=1.0 CBF_ALPHA=1.5
  export YAW_LOOKAHEAD=1.2 YAW_RATE_MAX=0.8 CBF_SAFE_MARGIN=0.45
  export CBF_SLACK_W="$W"
  "${HERE}/run_omnibot_dynamic.sh" "$N" "$DUR" 17 17 gmpc_scan
  cp "$CSV" "${out}/results.csv" 2>/dev/null
  for d in "${HERE}"/bags/gmpc_cbf__scan_seed*; do
    [ -d "$d" ] || continue
    case "$d" in *__prev_*) continue ;; esac
    mv "$d" "${out}/$(basename "$d")" 2>/dev/null
  done
  echo "[$(date +%T)] W=${W} done -> ${out}"
done

echo
echo "[$(date +%T)] === SWEEP DONE ==="
