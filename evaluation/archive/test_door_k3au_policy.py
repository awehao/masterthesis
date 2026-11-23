#!/usr/bin/env python3
"""door_k3au_policy 的離線反例（沿用 test_door_k3a_policy 的合成世界；TCP＝目標＋固定偏移）。

    python3 evaluation/test_door_k3au_policy.py
"""
import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3au_policy as U                          # noqa: E402
import test_door_k3a_policy as TP                     # noqa: E402

DT = TP.DT


def mk_sim(inject=None, off=(0.0, -0.00105, 0.0)):
    s = TP.Sim(inject, tcp_offset=off)
    s.cfg = U.K3AUConfig()
    s.pol = U.K3AUPolicy(s.cfg, TP.T_HG())
    return s


def events(pol, key):
    return [e for e in pol.events if key in e]


def phase_t(pol, ph):
    return next((e['t'] for e in pol.events if e.get('to') == ph), None)


class Unload(unittest.TestCase):
    def test_nominal_capture_ramp_hold_open(self):
        s = mk_sim()
        pol = s.run()
        self.assertEqual(pol.phase, 'DONE', pol.events[-3:])
        cap = events(pol, 'unload_capture')
        self.assertEqual(len(cap), 1)
        ev = cap[0]['unload_capture']
        self.assertAlmostEqual(ev['delta'], -0.00105, places=12)
        t_hold, t_open = phase_t(pol, 'HOLD'), phase_t(pol, 'OPEN')
        self.assertGreaterEqual(ev['t'] - t_hold, 0.5 - 1e-9)
        self.assertLess(ev['t'] - t_hold, 0.5 + DT + 1e-9)
        self.assertGreaterEqual(t_open - ev['t'], 1.0 - 1e-9)
        self.assertGreaterEqual(t_open - t_hold, 1.5 - 1e-9)
        # 斜坡路徑與姿態：已發布目標＝p0＋aδσ，姿態＝T_ref0
        T0, a = np.array(ev['T_ref0']), np.array(ev['a'])
        for t, ph, act, _ in s.acts:
            if ph == 'HOLD' and t >= ev['t'] and act['target'] is not None:
                exp = T0[:3, 3] + a * ev['delta'] * U.smoothstep((t - ev['t']) / 0.5)
                np.testing.assert_allclose(act['target'][:3, 3], exp, atol=1e-12)
                np.testing.assert_allclose(act['target'][:3, :3], T0[:3, :3], atol=0)
        np.testing.assert_allclose(a, [0.0, 1.0, 0.0], atol=1e-12)               # 工具 z（指向門內），非退讓軸
        k = next(i for i, x in enumerate(s.acts) if x[1] == 'OPEN')
        T_last_hold = s.acts[k - 1][2]['target']
        np.testing.assert_allclose(s.acts[k + 1][2]['target'], T_last_hold, atol=1e-12)   # OPEN 首筆＝卸載後目標
        np.testing.assert_allclose(T_last_hold[:3, 3], T0[:3, 3] + a * ev['delta'], atol=1e-12)

    def test_delayed_capture_full_ramp_and_hold(self):
        def inj(sim, m):
            if sim.pol.phase == 'HOLD' and m.t - sim.pol.t_phase < 1.2:
                m.qdot_door = 0.003                                               # 不滿擷取條件
        s = mk_sim(inj)
        pol = s.run()
        self.assertEqual(pol.phase, 'DONE')
        ev = events(pol, 'unload_capture')[0]['unload_capture']
        t_hold, t_open = phase_t(pol, 'HOLD'), phase_t(pol, 'OPEN')
        self.assertGreaterEqual(ev['t'] - t_hold, 1.2 - 1e-9)
        self.assertGreaterEqual(t_open - ev['t'], 1.0 - 1e-9)                       # 不被父類 hold_s（1.0 s）提前切 OPEN
        self.assertGreater(t_open - t_hold, 2.2 - 1e-9)

    def test_capture_timeout(self):
        def inj(sim, m):
            if sim.pol.phase == 'HOLD':
                m.qdot_door = 0.003
        s = mk_sim(inj)
        pol = s.run()
        self.assertEqual(pol.abort, 'unload_capture_timeout')
        self.assertIsNone(phase_t(pol, 'OPEN'))
        self.assertTrue(all(a['target'] is None for t, p, a, _ in s.acts if t >= pol.t_abort))
        self.assertTrue(all(a['grip'] <= 1e-12 for t, p, a, _ in s.acts if t >= pol.t_abort))

    def test_positive_delta_aborts(self):
        s = mk_sim(off=(0.0, +0.0008, 0.0))
        pol = s.run()
        self.assertEqual(pol.abort, 'unload_offset_positive')
        self.assertEqual(len(events(pol, 'unload_capture')), 1)                    # 記錄後中止
        self.assertIsNone(phase_t(pol, 'OPEN'))

    def test_out_of_range_aborts(self):
        s = mk_sim(off=(0.0, -0.0035, 0.0))
        pol = s.run()
        self.assertEqual(pol.abort, 'unload_offset_out_of_range')
        self.assertIsNone(phase_t(pol, 'OPEN'))

    def test_zero_delta_zero_intervention(self):
        s = mk_sim(off=(0.0, 0.0, 0.0))
        pol = s.run()
        self.assertEqual(pol.phase, 'DONE')
        ev = events(pol, 'unload_capture')[0]['unload_capture']
        self.assertEqual(ev['delta'], 0.0)
        T0 = np.array(ev['T_ref0'])
        for t, ph, act, _ in s.acts:
            if ph == 'HOLD' and act['target'] is not None:
                np.testing.assert_allclose(act['target'], T0, atol=0)
        self.assertGreaterEqual(phase_t(pol, 'OPEN') - ev['t'], 1.0 - 1e-9)

    def test_abort_mid_ramp_no_open_no_targets(self):
        def inj(sim, m):
            cap = [e for e in sim.pol.events if 'unload_capture' in e]
            if cap and 0.2 <= m.t - cap[0]['t'] < 0.3:
                m.wrist_max = 31.0
        s = mk_sim(inj)
        pol = s.run()
        self.assertEqual(pol.abort, 'wrist_force')
        self.assertIsNone(phase_t(pol, 'OPEN'))
        self.assertTrue(all(a['target'] is None for t, p, a, _ in s.acts if t >= pol.t_abort))

    def test_monitor_priority_over_capture(self):
        def inj(sim, m):
            if sim.pol.phase == 'HOLD' and m.t - sim.pol.t_phase >= 0.5:
                m.T_tcp = m.T_tcp.copy()
                m.T_tcp[0, 3] = float('nan')
        s = mk_sim(inj)
        pol = s.run()
        self.assertEqual(pol.abort, 'monitor_nonfinite_or_invalid')
        self.assertEqual(events(pol, 'unload_capture'), [])

    def test_capture_only_once_even_if_conditions_flip(self):
        def inj(sim, m):
            cap = [e for e in sim.pol.events if 'unload_capture' in e]
            if cap and sim.pol.phase == 'HOLD':
                m.T_tcp = m.T_tcp.copy()
                m.T_tcp[1, 3] -= 0.0005                                             # 擷取後實測再變，參考不得跟著改
        s = mk_sim(inj)
        pol = s.run()
        self.assertEqual(len(events(pol, 'unload_capture')), 1)
        ev = events(pol, 'unload_capture')[0]['unload_capture']
        self.assertAlmostEqual(ev['delta'], -0.00105, places=12)

    def test_parent_k3a_unchanged(self):
        s = TP.Sim()
        pol = s.run()
        self.assertEqual(pol.phase, 'DONE')
        self.assertLess(phase_t(pol, 'OPEN') - phase_t(pol, 'HOLD'), 1.0 + DT + 1e-9)


if __name__ == '__main__':
    unittest.main(verbosity=2)
