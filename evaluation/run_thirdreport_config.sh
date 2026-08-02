#!/usr/bin/env bash
# Batch O -- the third progress report's validated configuration, on bigarena.
#
# That configuration is the only one ever measured at 0 dynamic collisions
# (39/40 arrival, 1/40 contacts and that one a shallow wall graze). Everything
# running today is that set after a series of one-at-a-time changes, each with
# its own reason, none of which was ever validated as a COMBINATION:
#
#   cbf_alpha        3.0  -> 0.5    (alpha sweep on the old scenario)
#   dynamic margin   0.38 -> 0.60   (to cut penetration depth)
#   static margin    0.33 -> 0.38
#   inflation        0.45 -> 0.70   (planner/CBF were fighting at 0.45 vs 0.45)
#   velocity_smoother OFF -> ON     (no recorded reason -- a regression)
#
# This batch puts all five back at once. That is deliberate: it tests a known
# -good SET, not one variable. If it wins, bisect afterwards; if it loses, the
# current set is vindicated and the difficulty is the scenario, not the tuning.
#
# The heading reference stays REMOVED -- batch M cut contacts from 10/28 to
# 1/28 with it gone, so it is not up for re-litigation.
#
# smoother OFF is expressed by METHOD=gmpc_scan_nosm, which also renames the
# output: results/omnibot_dynamic_gmpc_scan_nosm.csv and bags gmpc_cbf__scan_nosm_*.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

LOCKFILE=/tmp/omnibot_dynamic.pid
if [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE" 2>/dev/null)" 2>/dev/null; then
    echo "ERROR: a batch is already running (pid $(cat "$LOCKFILE"))."; exit 1
fi

OUT=evaluation/results/thirdcfgO
mkdir -p "$OUT"
CSV=evaluation/results/omnibot_dynamic_gmpc_scan_nosm.csv
[ -s "$CSV" ] && cp "$CSV" "evaluation/results/archived_$(date +%Y%m%d_%H%M%S).csv"

POSES_CSV="$HERE/results/bigarena_poses.csv" \
BIGARENA=1 TRAJ=bigarena_traffic GUI=0 \
PLANNER_SCAN=/scan_filtered PLAN_BLEND=1.0 \
CBF_ALPHA=3.0 CBF_SAFE_MARGIN=0.38 STATIC_MARGIN=0.33 \
INFLATION=0.45 \
CBF_PRUNE_RANGE=1.2 CBF_FAR_STRIDE=2 \
    ./evaluation/run_omnibot_dynamic.sh 30 250 0 0 gmpc_scan_nosm

cp "$CSV" "$OUT/batch.csv"
echo "batch O done -> $OUT/batch.csv"
