#!/usr/bin/env bash
# Sequential batch runner — METHODS × SEEDS, one trial at a time.
#
# Designed to be launched in the background, e.g.:
#   nohup ./run_all_trials.sh > batch.log 2>&1 &
#   tail -f batch.log
#
# Resumes safely:  if a bag for a (METHOD, SEED) pair already exists, the
# trial is skipped, so re-running the script after a crash continues where
# it left off.
#
# Configure with env vars:
#   METHODS  : space-separated list                  (default: all four)
#   SEEDS    : space-separated list                  (default: 0 1 2)
#   DURATION : seconds per trial                     (default: 250)
#   GAP      : seconds to sleep between trials       (default: 10)

set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

METHODS="${METHODS:-rpp mppi gmpc gmpc_cbf}"
SEEDS="${SEEDS:-0 1 2}"
DURATION="${DURATION:-250}"
GAP="${GAP:-10}"

mkdir -p "${HERE}/bags" "${HERE}/logs"

echo "=== batch runner ==="
echo "  methods : $METHODS"
echo "  seeds   : $SEEDS"
echo "  duration: ${DURATION}s per trial"
echo "  total   : $(echo $METHODS $SEEDS | wc -w) potential trials"
echo "  output  : ${HERE}/bags  +  ${HERE}/logs"
echo

TOTAL=0
RAN=0
SKIPPED=0
for METHOD in $METHODS; do
    for SEED in $SEEDS; do
        TOTAL=$((TOTAL+1))
        BAG_DIR="${HERE}/bags/${METHOD}__seed${SEED}"
        if [[ -d "$BAG_DIR" ]]; then
            echo "[$(date +%T)] [skip] $METHOD seed=$SEED  (bag already exists at $BAG_DIR)"
            SKIPPED=$((SKIPPED+1))
            continue
        fi
        echo "[$(date +%T)] === [$TOTAL] launching trial: $METHOD seed=$SEED ==="
        "${HERE}/run_one_trial.sh" "$METHOD" "$SEED" "$DURATION" \
            || echo "[$(date +%T)] [warn] trial returned non-zero — continuing"
        RAN=$((RAN+1))
        echo "[$(date +%T)] sleeping ${GAP}s before next trial ..."
        sleep "$GAP"
    done
done

echo
echo "=== batch done.  ran=$RAN  skipped=$SKIPPED  total=$TOTAL ==="
echo "next:  python3 ${HERE}/analyze.py ${HERE}/bags/<each>  --method <m> --run <r>"
