#!/usr/bin/env python3
"""One targeted nominal check for the W1 parked-grasp strategy.

Uses the existing fake plant and synthetic visual observations. This verifies
logic and wiring, not physical grasp, contact detection or placement capability.
"""
import json
from pathlib import Path

import numpy as np

import test_bottle_w1 as T
import bottle_w1_parked_ctrl as P


def main():
    original = T.W1.W1PlaceController
    T.W1.W1PlaceController = P.W1PlaceController
    try:
        ctrl, plant = T.build()
    finally:
        T.W1.W1PlaceController = original
    report, _ = T.run(ctrl, plant, t_end=85.0)
    T.show('P matched visual nominal (fake plant)', report)
    checks = dict(complete=bool(report['mission_done']),
                  finite_commands=bool(T.finite_cmds(plant)),
                  wrist_fix=report['w1']['n_groups'] >= 1,
                  wrist_locked=bool(report['w1']['locked']),
                  goal_reached=bool(np.allclose(report['w1']['cur'],
                                              report['w1']['goal'], atol=1e-6)))
    inserts = [e['t'] for e in report['events']
               if e.get('what') == 'phase' and e.get('to') == 'INSERT']
    if inserts:
        t_insert = inserts[0]
        times = np.array([r['t'] for r in plant.hist])
        positions = np.array([r['q'][:2] for r in plant.hist])
        speeds = np.linalg.norm(np.diff(positions, axis=0), axis=1) / np.diff(times)
        mask = ((times[1:] >= t_insert - 0.20 - 1e-8)
                & (times[1:] < t_insert))
        checks['measured_base_still_before_insert'] = (
            int(mask.sum()) >= 19 and float(speeds[mask].max()) < 0.01)
    else:
        checks['measured_base_still_before_insert'] = False
    checks['rate_limit'] = bool(T.rate_ok(ctrl)[0])
    result = dict(kind='fake_plant_not_physics', checks=checks, report=report)
    path = Path(__file__).parent / 'results/bottle/w1_parked_offline_nominal.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1,
                              default=lambda x: x.tolist() if isinstance(x, np.ndarray)
                              else float(x)), encoding='utf-8')
    print(json.dumps(checks, ensure_ascii=False), flush=True)
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
