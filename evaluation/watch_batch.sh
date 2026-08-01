#!/usr/bin/env bash
# Analyse trials AS THEY LAND, so a systematic failure is caught in minutes
# rather than at the end of a two-hour batch.
#
# Twice today a batch ran to completion producing data that was worthless --
# once because localisation was wrong from the first trial, once because a fifth
# of the spawns were inside obstacles. Both were visible in the first few trials.
#
# Prints a running tally and flags anything that needs attention: a contact, a
# trial that did not arrive, degraded data coverage, or a spawn that never moved.
#
#   ./evaluation/watch_batch.sh [SECONDS_BETWEEN_CHECKS]
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVERY="${1:-120}"
POSES="${POSES:-${HERE}/results/bigarena_poses.csv}"
seen=0
while true; do
    n=$(ls -d "${HERE}"/bags/gmpc_cbf__scan_seed* 2>/dev/null | grep -vc __prev_ || echo 0)
    running=$(ps -eo cmd --no-headers | grep -cE '[r]un_omnibot_dynamic' || true)
    if [ "$n" -gt "$seen" ]; then
        seen=$n
        echo "=========== $(date +%H:%M)  ${n} trials on disk ==========="
        python3 -u "${HERE}/analyze_trials.py" "${HERE}/bags" \
            --poses "$POSES" 2>/dev/null \
            | sed -n '/^=== /,/^  run /p' | grep -vE '^  run |^=== '
        # anything that needs a human
        python3 -u "${HERE}/analyze_trials.py" "${HERE}/bags" \
            --poses "$POSES" 2>/dev/null \
            | awk '/^  seed/ && ($3!="arrived" || $4+0<0)' | head -8
    fi
    [ "$running" -eq 0 ] && { echo "[$(date +%H:%M)] batch finished"; break; }
    sleep "$EVERY"
done
