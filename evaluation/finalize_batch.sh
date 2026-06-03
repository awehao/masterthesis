#!/usr/bin/env bash
# Run this AFTER the gmpc_cbf 400s batch finishes. One command, three steps:
#
#   1. Classify trial outcomes (SUCCESS / ALGO_FAIL / APPARATUS_FAIL).
#   2. Re-analyse every bag through analyze.py with the new controller-log
#      override (recovers the false-failure trials like seed 2).
#   3. Re-plot the comparison figures.
#
# Idempotent: safe to re-run.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo " STEP 1 — classify trial outcomes (apparatus vs algorithm)"
echo "============================================================"
python3 "${HERE}/classify_failures.py"
echo

echo "============================================================"
echo " STEP 2 — re-analyse every bag (uses controller_log override)"
echo "============================================================"
"${HERE}/analyze_all_bags.sh"
echo

echo "============================================================"
echo " STEP 3 — re-plot comparison figures (no collision, no jerk)"
echo "============================================================"
python3 "${HERE}/plot.py" \
    --csv "${HERE}/results/runs.csv" \
    --out "${HERE}/results/figs/"
echo

echo "============================================================"
echo " STEP 4 — per-seed gmpc_cbf summary with new success_source"
echo "============================================================"
python3 - <<'PY'
import csv
from pathlib import Path
csv_path = Path(__file__).resolve().parent / 'results' / 'runs.csv' \
        if False else Path('results/runs.csv')
rows = [r for r in csv.DictReader(open(csv_path))
        if r['run'].startswith('seed') and r['method'] == 'gmpc_cbf']
rows.sort(key=lambda r: int(r['run'].replace('seed', '')))
print(f'  {"seed":>6s}  {"success":>8s}  {"source":>16s}  '
      f'{"ctrl_dist":>10s}  {"arrival_s":>10s}')
print('  ' + '-' * 60)
for r in rows:
    d = r['controller_arrival_dist_m'] or 'nan'
    t = r['arrival_time_s'] or 'nan'
    try: d = f'{float(d):.3f}'
    except Exception: pass
    try: t = f'{float(t):.1f}'
    except Exception: pass
    print(f'  {r["run"]:>6s}  {r["success"]:>8s}  '
          f'{r["success_source"]:>16s}  {d:>10s}  {t:>10s}')

ok = sum(1 for r in rows if r['success'].lower() == 'true')
print()
print(f'  GMPC+CBF raw success: {ok}/{len(rows)} '
      f'({100*ok/max(len(rows),1):.0f}%)')
print('  (excluding apparatus failures: see Step 1 output)')
PY
echo

echo "Output files:"
echo "  CSV  : ${HERE}/results/runs.csv"
echo "  Figs : ${HERE}/results/figs/"
echo "Done."
