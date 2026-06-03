#!/usr/bin/env python3
"""Triage GMPC+CBF trial outcomes into:
   - SUCCESS                        controller logged 'Goal reached'
   - ALGO_FAIL (Start-occupied)     planner kept failing, no Goal reached
   - APPARATUS_FAIL (scan_relay)    scan_relay node crashed during trial
   - APPARATUS_FAIL (goal_relay)    goal_to_plan_relay never processed goal
   - UNKNOWN                        none of the above

This is the honest accounting for the thesis: we report success rate over
ALGORITHMICALLY-VALID trials, and document apparatus failures separately so
the advisor sees both the headline number and the test-harness limits.

Usage:
    python3 classify_failures.py [logs_dir]    (default = ./logs)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


def classify_log(log_path: Path) -> tuple[str, str]:
    """Return (category, one-line reason).

    Priority (top wins):
      1. Goal reached → SUCCESS.
      2. Algorithm got a fair shake (≥10 plan requests issued) but never
         reached goal → ALGO_FAIL. We classify here BEFORE checking the
         scan_relay crash, because Start-occupied lockouts can cascade into
         a downstream scan_relay shutdown after several minutes of stress.
      3. scan_relay traceback present + few plan requests → genuine
         APPARATUS_FAIL (node died during stack bringup, not from long-run
         stress).
      4. No plan requests at all → goal_to_plan_relay never accepted the
         goal (QoS mismatch or subscriber not ready) → APPARATUS_FAIL.
      5. Otherwise UNKNOWN.
    """
    try:
        text = log_path.read_text(errors='replace')
    except OSError as e:
        return ('UNKNOWN', f'cannot read log: {e}')

    if re.search(r'gmpc_controller.*Goal reached', text):
        return ('SUCCESS', 'controller logged Goal reached')

    plan_requests  = len(re.findall(r'goal_to_plan_relay.*Plan request', text))
    start_occupied = len(re.findall(r'GridBased plugin failed.*Start occupied', text))
    has_relay_crash = bool(re.search(r'\[scan_relay.*Traceback', text))

    if plan_requests >= 10:
        # System was alive long enough for the controller to be exercised.
        # Goal not reached → real algorithmic limitation (almost certainly
        # the Start-occupied lockout we documented in P14 future work).
        return ('ALGO_FAIL',
                f'planner Start-occupied lockout '
                f'(plans={plan_requests}, start_occupied={start_occupied})')

    if has_relay_crash:
        return ('APPARATUS_FAIL', 'scan_relay crashed during stack bringup')

    if plan_requests == 0:
        return ('APPARATUS_FAIL', 'goal_to_plan_relay never processed /goal_pose')

    return ('UNKNOWN', f'plans={plan_requests}, '
                      f'start_occupied={start_occupied}, '
                      f'relay_crash={has_relay_crash}')


def main():
    logs_dir = Path(sys.argv[1] if len(sys.argv) > 1 else 'logs')
    if not logs_dir.is_dir():
        print(f'no such dir: {logs_dir}', file=sys.stderr); sys.exit(1)

    trials = sorted(p for p in logs_dir.glob('gmpc_cbf__seed*.log'))
    if not trials:
        print('no gmpc_cbf__seed*.log files found'); return

    by_cat: dict[str, list[str]] = {}
    for log in trials:
        seed = log.stem.replace('gmpc_cbf__', '')
        cat, why = classify_log(log)
        by_cat.setdefault(cat, []).append(seed)
        print(f'  {seed:>10s}  {cat:<15s}  {why}')

    print()
    print('=' * 60)
    n_total = len(trials)
    n_success = len(by_cat.get('SUCCESS', []))
    n_algo    = len(by_cat.get('ALGO_FAIL', []))
    n_appar   = len(by_cat.get('APPARATUS_FAIL', []))
    n_unknown = len(by_cat.get('UNKNOWN', []))
    n_valid   = n_total - n_appar - n_unknown

    print(f'  Total trials      : {n_total}')
    print(f'  Apparatus failures: {n_appar}  -> EXCLUDED from algo success')
    print(f'  Unknown           : {n_unknown}')
    print(f'  Algorithm-valid N : {n_valid}')
    print(f'  Algorithm success : {n_success}/{n_valid} = '
          f'{100*n_success/max(n_valid,1):.0f}%')
    print(f'  Algorithm fail    : {n_algo}/{n_valid} (Start-occupied lockout)')


if __name__ == '__main__':
    main()
