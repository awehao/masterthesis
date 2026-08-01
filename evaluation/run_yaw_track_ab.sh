#!/usr/bin/env bash
# Two matched batches on the same traverse: heading tracked, then heading off.
#
# (Not to be confused with run_yaw_ab.sh, which tested the look-ahead CHORD the
# heading reference is built from. This tests whether the heading should be
# tracked AT ALL.)
#
# The base is omnidirectional and the lidar is 360 deg, so nothing about the
# task requires the robot to face where it is going -- vx and vy do the driving.
# Slaving yaw to the path therefore buys nothing while handing the QP a nearly
# free degree of freedom, and whenever the CBF tightens the position constraints
# it spends that freedom on rotation. Measured once, on a hard traverse: the
# heading reference asked for 1355 deg over 100 s, the robot turned 4981 deg,
# wz saturated 58% of the time, and it made no progress. With yaw off, the same
# traverse produced 6 deg of rotation and 0% saturation -- but still did not
# reach the goal, for an unrelated reason (the planner oscillating between two
# doorways), so this is one route with one confound, not a result.
#
# Hence a paired run: same world, same traffic, same seeds, one variable.
#
# Each arm's CSV and bags are archived separately, because the batch script
# writes one CSV per method and names bags by seed, so the second arm would
# otherwise overwrite the first.
#
#   ./evaluation/run_yaw_track_ab.sh [N_PER_ARM] [DURATION_S]
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
N="${1:-75}"
DUR="${2:-250}"
CSV="${HERE}/results/omnibot_dynamic_gmpc_scan.csv"

arm() {                      # $1 = name, $2 = YAW_TRACK value
  local name="$1" yaw="$2"
  local out="${HERE}/results/abyaw_${name}"
  echo "=========================================================="
  echo "[$(date +%T)] ARM ${name}: YAW_TRACK=${yaw}, ${N} trials, cap ${DUR}s"
  echo "=========================================================="
  mkdir -p "$out"
  export BIGARENA=1 TRAJ=bigarena_traffic GUI=0
  export PLANNER_SCAN=/scan_filtered PLAN_BLEND=1.0 CBF_ALPHA=1.5
  export YAW_LOOKAHEAD=1.2 CBF_SAFE_MARGIN=0.45
  # YAW_RATE_MAX only means anything while the heading is being tracked.
  if [ "$yaw" = "1" ]; then export YAW_RATE_MAX=0.8; else unset YAW_RATE_MAX; fi
  export YAW_TRACK="$yaw"
  "${HERE}/run_omnibot_dynamic.sh" "$N" "$DUR" 17 17 gmpc_scan
  cp "$CSV" "${out}/results.csv" 2>/dev/null
  for d in "${HERE}"/bags/gmpc_cbf__scan_seed*; do
    [ -d "$d" ] || continue
    case "$d" in *__prev_*) continue ;; esac
    mv "$d" "${out}/$(basename "$d")" 2>/dev/null
  done
  echo "[$(date +%T)] ARM ${name} done -> ${out}"
}

arm yawon  1
arm yawoff 0

echo
echo "[$(date +%T)] === BOTH ARMS DONE ==="
echo "  ${HERE}/results/abyaw_yawon/results.csv"
echo "  ${HERE}/results/abyaw_yawoff/results.csv"
