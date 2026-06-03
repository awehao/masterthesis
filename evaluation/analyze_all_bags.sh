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
