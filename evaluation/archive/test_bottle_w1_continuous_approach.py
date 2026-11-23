#!/usr/bin/env python3
"""連續展臂候選：小型排程核對；--nominal 額外執行一趟假受控對象。

假受控對象的接觸／視覺是合成資料，不能據此宣稱物理或渲染已通過。
"""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'src' / 'ammr_wholebody_mpc'))
import bottle_ref as BR
import bottle_w1_continuous_approach_ctrl as CA
import bottle_w1_shrow_ctrl as SR


class ScheduleTests(unittest.TestCase):
    def test_progress_and_endpoints(self):
        for x, expected in ((-2.0, 0.0), (-1.55, 0.0), (-1.075, 0.5), (-0.6, 1.0), (0.0, 1.0)):
            self.assertAlmostEqual(CA.approach_progress(x, -1.55, -0.6), expected)

    def test_invalid(self):
        for args in ((float('nan'), -1.55, -0.6), (0, 1, 0), (0, 0, 0)):
            with self.assertRaises(ValueError):
                CA.approach_progress(*args)

    def test_profile_has_no_interior_plateau(self):
        p = np.linspace(0, 1, 101)
        s = np.array([BR.min_jerk(x) for x in p])
        self.assertTrue(np.all(np.diff(s) > 0))
        self.assertEqual(s[0], 0.0)
        self.assertEqual(s[-1], 1.0)

    def test_does_not_finish_from_elapsed_time(self):
        c = CA.W1PlaceController.__new__(CA.W1PlaceController)
        c.continuous_unfold, c.q_tuck, c.t_u0 = True, np.zeros(6), 0.0
        c.t, c.t_unfold, c.approach_measured = 99.0, 3.4, 0.8
        self.assertFalse(c._unfolded())
        c.approach_measured = 1.0
        self.assertTrue(c._unfolded())

    def test_disabled_uses_parent_completion(self):
        c = CA.W1PlaceController.__new__(CA.W1PlaceController)
        c.continuous_unfold, c.q_tuck = False, np.zeros(6)
        c.t_u0, c.t, c.t_unfold = 1.0, 5.0, 3.4
        self.assertEqual(c._unfolded(), SR.W1PlaceController._unfolded(c))

    def test_disabled_target_matches_parent(self):
        c = CA.W1PlaceController.__new__(CA.W1PlaceController)
        c.continuous_unfold, c.q_tuck, c.q_arm_ready = False, np.zeros(6), np.arange(6, dtype=float)
        c.t_u0, c.t_unfold = 1.0, 3.4
        c.K = SimpleNamespace(fk=lambda q, link: q.copy())
        for tau in (0.0, 2.0, 9.0):
            np.testing.assert_array_equal(c._rel_target(tau), SR.W1PlaceController._rel_target(c, tau))

    def test_prediction_uses_sigma_to_world_x(self):
        c = CA.W1PlaceController.__new__(CA.W1PlaceController)
        c.continuous_unfold, c.q_tuck, c.q_arm_ready = True, np.zeros(6), np.ones(6)
        c.t_u0, c.t, c.s_ref, c.v_b = 0.0, 2.0, -4.0, 0.1
        c.x_unfold, c.x_blend, c.approach_measured = -1.55, -0.6, 0.0
        c.path = SimpleNamespace(xy=lambda sig: np.array([[-1.075, 0.0]]))
        c.K = SimpleNamespace(fk=lambda q, link: q.copy())
        np.testing.assert_allclose(c._rel_target(2.0)[3:], np.full(6, 0.5))
        c.approach_measured = 0.9
        np.testing.assert_allclose(c._rel_target(2.0)[3:], np.full(6, BR.min_jerk(0.9)))


def nominal():
    import test_bottle_w1 as TW
    original = TW.W1.W1PlaceController
    TW.W1.W1PlaceController = CA.W1PlaceController
    try:
        c, f = TW.build()
    finally:
        TW.W1.W1PlaceController = original
    r, _ = TW.run(c, f)
    TW.show('連續展臂（合成視覺與接觸，非物理）', r)
    transitions = {e.get('what'): e['t'] for e in r['events']}
    log = np.array(c.approach_log, float).reshape(-1, 3)
    assert len(log) and np.all(np.diff(log[:, 2]) >= -1e-12)
    assert r['mission_done'] and TW.finite_cmds(f), r['stop']
    assert r['w1']['n_groups'] >= 1 and r['w1']['locked']
    assert c.t_close_req is not None
    assert transitions.get('blend_start') is not None
    # Scheduling itself does not complete early and then hold at READY.
    ready = log[log[:, 2] >= 1.0 - 1e-12]
    scheduled_hold = None if not len(ready) else transitions['blend_start'] - float(ready[0, 0])
    assert scheduled_hold is not None and scheduled_hold <= c.nav.dt + 1e-8, scheduled_hold
    # Only audit the changed pre-close trajectory. The old geometry helper's
    # attached-bottle approximation is not valid after release at table two.
    import bottle_s0b_path_check as PC
    mins, self_min, _, rate, margin = PC.check(SimpleNamespace(hist=[h for h in f.hist if h['t'] <= c.t_close_req]))
    geometry = dict(minima={k: list(v) for k, v in mins.items()}, self_min=list(self_min),
                    setpoint_rate_max=rate, joint_margin_min=margin,
                    scope='變更的抓取前模型軌跡；點雲近似，非接觸物理或形式保證')
    assert self_min[0] > 0 and margin > 0
    assert all(v[0] >= 0 for v in mins.values()), geometry
    result = dict(
        mode='offline_synthetic_not_physical', mission_done=r['mission_done'], stop=r['stop'],
        events=r['events'], scheduled_ready_hold_s=scheduled_hold,
        wrist_groups=r['w1']['n_groups'], wrist_locked=r['w1']['locked'],
        t_close=c.t_close_req, clearance=r['clear_bottle'], forbidden=r['forbidden_pulses'],
        continuous_approach=r['continuous_approach'],
        preclose_geometry=geometry,
        remaining='模型抓取前幾何已核；實際手腕可見度與物理閉迴路尚未驗證',
    )
    target = HERE / 'results' / 'bottle' / 'w1_continuous_approach_offline.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=lambda x: np.asarray(x).tolist()) + '\n')
    print(f'排程備妥等待 {scheduled_hold:.3f} s；結果 {target}', flush=True)


if __name__ == '__main__':
    use_nominal = '--nominal' in sys.argv
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ScheduleTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
    if use_nominal:
        nominal()
