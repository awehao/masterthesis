#!/usr/bin/env bash
# Scan the CBF class-K gain on the 20 m floor: how EARLY does the barrier act?
#
# (run_alpha_sweep.sh is the older 20 m random-room version, held at the detour
# config with PLAN_BLEND and YAW_LOOKAHEAD off. This one uses the bigarena
# configuration and is not comparable with it.)
#
# Why alpha and not a longer horizon. Of 23 contacts measured over 146 trials,
# the robot commanded retreat in 23/23 -- vx went negative and its velocity had
# a positive away-component at some point -- and the distance still never
# opened, in 0/23. The reason is kinematic, not a tuning failure:
#
#     robot reverse limit   vx_min -0.20 / vy_min -0.25  ->  0.25 m/s
#     dyn_obs_1 approach                                     0.30 m/s
#
# It backs away more slowly than the obstacle closes, so h_dot cannot be made
# positive and `h_dot + alpha*h >= 0` is infeasible; the QP can only absorb it
# in slack. Sidestepping does not save it either: at 1 m separation and 0.52
# m/s closing there are 1.3 s, which buys 0.33 m of lateral offset against the
# 0.65 m needed to clear a 0.7 m body. Inside ~1 m the encounter is decided.
#
# So avoidance has to begin earlier. The horizon is the obvious lever and it
# fails: N=40/60 made clearance and arrival monotonically worse and produced
# 22-52 OSQP iteration-limit bailouts per trial, because the constraint count
# is n_obs x N. Alpha moves the engagement point WITHOUT adding a decision
# variable or a row -- smaller alpha starts the barrier pushing further out,
# at the cost of being more conservative in open space. That cost is what the
# arrival time and path length in this sweep are there to measure.
#
#   ./evaluation/run_alpha_sweep_bigarena.sh [N_PER_VALUE] [DURATION_S]
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
N="${1:-3}"
DUR="${2:-250}"
CSV="${HERE}/results/omnibot_dynamic_gmpc_scan.csv"

for A in 1.5 0.9 0.5 0.3; do
  tag="${A/./p}"
  out="${HERE}/results/alpha_${tag}"
  echo "=========================================================="
  echo "[$(date +%T)] CBF_ALPHA=${A}, ${N} trials, cap ${DUR}s"
  echo "=========================================================="
  mkdir -p "$out"
  export BIGARENA=1 TRAJ=bigarena_traffic GUI=0
  export PLANNER_SCAN=/scan_filtered PLAN_BLEND=1.0
  export YAW_LOOKAHEAD=1.2 YAW_RATE_MAX=0.8 CBF_SAFE_MARGIN=0.45
  export CBF_ALPHA="$A"
  "${HERE}/run_omnibot_dynamic.sh" "$N" "$DUR" 17 17 gmpc_scan
  cp "$CSV" "${out}/results.csv" 2>/dev/null
  for d in "${HERE}"/bags/gmpc_cbf__scan_seed*; do
    [ -d "$d" ] || continue
    case "$d" in *__prev_*) continue ;; esac
    mv "$d" "${out}/$(basename "$d")" 2>/dev/null
  done
  echo "[$(date +%T)] alpha=${A} done -> ${out}"
done

echo
echo "[$(date +%T)] === ALPHA SWEEP DONE ==="
