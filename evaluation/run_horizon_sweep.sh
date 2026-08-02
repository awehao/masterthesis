#!/usr/bin/env bash
# Scan the CBF horizon: can it see the encounter coming, or only react to it?
#
# Of 24 contacts over 146 trials, 23 were with dyn_obs_1 -- the fastest mover
# (0.30 m/s, 9.3 m traverse, 62 s period, so it is met twice per run) -- and at
# the moment of contact the ROBOT was doing 0.03-0.09 m/s. It had already
# braked. Visibility was 87% median in the 3 s before, so it saw the obstacle
# throughout. It stopped, and was driven into.
#
# That is a horizon problem, not a perception or a cost problem. At N=20 and
# dt=0.05 the barrier sees 1.0 s; against a 0.52 m/s closing speed that is
# 0.52 m of approach, so the constraint only activates ~0.97 m out, 1.3 s
# before contact. In 1.3 s braking is the only available response, and braking
# does not increase the distance to something that keeps coming -- the CBF
# condition needs the robot to open the gap, which needs it to move earlier.
#
# N = 20 / 40 / 60 covers 1.0 / 2.0 / 3.0 s. The QP grows with N; measured
# solve time at N=20 is 0.18 ms, so there is room, but this records it.
#
#   ./evaluation/run_horizon_sweep.sh [N_PER_VALUE] [DURATION_S]
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
N="${1:-3}"
DUR="${2:-250}"
CSV="${HERE}/results/omnibot_dynamic_gmpc_scan.csv"

for H in 20 40 60; do
  out="${HERE}/results/horizon_${H}"
  echo "=========================================================="
  echo "[$(date +%T)] HORIZON=${H} (${H}x0.05 = $(python3 -c "print(f'{$H*0.05:.1f}')") s), ${N} trials"
  echo "=========================================================="
  mkdir -p "$out"
  export BIGARENA=1 TRAJ=bigarena_traffic GUI=0
  export PLANNER_SCAN=/scan_filtered PLAN_BLEND=1.0 CBF_ALPHA=1.5
  export CBF_SAFE_MARGIN=0.45
  export HORIZON="$H"
  "${HERE}/run_omnibot_dynamic.sh" "$N" "$DUR" 17 17 gmpc_scan
  cp "$CSV" "${out}/results.csv" 2>/dev/null
  for d in "${HERE}"/bags/gmpc_cbf__scan_seed*; do
    [ -d "$d" ] || continue
    case "$d" in *__prev_*) continue ;; esac
    mv "$d" "${out}/$(basename "$d")" 2>/dev/null
  done
  echo "[$(date +%T)] HORIZON=${H} done -> ${out}"
done

echo
echo "[$(date +%T)] === HORIZON SWEEP DONE ==="
