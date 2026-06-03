#!/usr/bin/env bash
# Walk every bag in bags/, run analyze.py on it, append rows to a single CSV.
#
# Expected bag naming (set by run_one_trial.sh):  bags/<METHOD>__<RUN_TAG>/
# e.g. bags/gmpc_cbf__seed3/ -> --method gmpc_cbf --run seed3
#
# Usage:  ./analyze_all_bags.sh [OUT_CSV]
#         (default OUT_CSV = results/runs.csv)

set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_CSV="${1:-${HERE}/results/runs.csv}"

mkdir -p "$(dirname "$OUT_CSV")"
# Start fresh so re-running is idempotent (avoids stale rows from prior runs).
rm -f "$OUT_CSV"

count=0
fail=0
for bag in "${HERE}"/bags/*/; do
    bag="${bag%/}"
    name="$(basename "$bag")"
    if [[ "$name" != *"__"* ]]; then
        echo "[skip] $name (not in METHOD__RUN format)"
        continue
    fi
    METHOD="${name%%__*}"
    RUN_TAG="${name#*__}"
    # Skip legacy ad-hoc bags (corner_goal_*, horizon_*, etc.) so the CSV
    # only contains the controlled batch trials (seed0 .. seed9).
    if [[ "$RUN_TAG" != seed* ]]; then
        echo "[skip ] $METHOD / $RUN_TAG (not a seed* run)"
        continue
    fi
    # Skip apparatus failures (test-harness bugs, not algorithm). Standard
    # scientific practice: bag retained as evidence, excluded from stats.
    # ALGO_FAIL trials (e.g. gmpc_cbf seed 4 Start-occupied lockout) are
    # NOT skipped because they are real algorithmic outcomes already
    # documented as a known limitation in the report.
    #
    # Only apply this filter to GMPC+CBF runs -- classify_failures.py uses
    # patterns specific to that stack (gmpc_controller / goal_to_plan_relay
    # / scan_relay log strings) and would mis-classify RPP/MPPI logs.
    if [[ "$METHOD" == "gmpc_cbf" && -f "${HERE}/logs/${name}.log" ]]; then
        cat=$(python3 -c "
import sys; sys.path.insert(0, '${HERE}')
from classify_failures import classify_log
from pathlib import Path
print(classify_log(Path('${HERE}/logs/${name}.log'))[0])
" 2>/dev/null)
        if [[ "$cat" == "APPARATUS_FAIL" ]]; then
            echo "[skip ] $METHOD / $RUN_TAG (APPARATUS_FAIL excluded)"
            continue
        fi
    fi
    echo "[analyze] $METHOD / $RUN_TAG"
    if python3 "${HERE}/analyze.py" "$bag" \
            --method "$METHOD" --run "$RUN_TAG" \
            --out "$OUT_CSV" > /dev/null; then
        count=$((count+1))
    else
        echo "  [fail] analyze.py returned non-zero for $name"
        fail=$((fail+1))
    fi
done

echo
echo "=== done.  analyzed=$count  failed=$fail  csv=$OUT_CSV ==="
echo "next:  python3 ${HERE}/plot.py $OUT_CSV"
