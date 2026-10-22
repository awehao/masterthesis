#!/usr/bin/env python3
"""door_k3ax_policy 的離線反例（沿用 test_door_k3a_policy 的合成世界；TCP＝目標＋固定偏移）。
含**旋轉把手座標**：暫時把合成世界的把手位姿繞 z 轉 30°，確認軸＝H_frozen 第一欄（不寫死世界 x），位移只沿該軸。

    python3 evaluation/test_door_k3ax_policy.py
"""
import contextlib
import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3ax_policy as X                          # noqa: E402
import test_door_k3a_policy as TP                     # noqa: E402
from door_target import rotate_about_hinge            # noqa: E402

DT = TP.DT


@contextlib.contextmanager
def rotated_handle(yaw_deg):
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    H0r = np.eye(4)
    H0r[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    H0r[:3, 3] = [0.8, -0.04, 0.85]
    old_H0, old_Hof = TP.H0, TP.H_of
    TP.H0 = lambda: H0r.copy()
    TP.H_of = lambda q: rotate_about_hinge(H0r, np.zeros(3), TP.AX, q)
    try:
        yield H0r
    finally:
        TP.H0, TP.H_of = old_H0, old_Hof


def mk_sim(inject=None, off=None, axis_world=(1.0, 0.0, 0.0), d_mm=-1.16):
    off = tuple(np.asarray(axis_world) * d_mm * 1e-3) if off is None else off
    s = TP.Sim(inject, tcp_offset=off)
    s.cfg = X.K3AXConfig()
    s.pol = X.K3AXPolicy(s.cfg, TP.T_HG())
    return s


def cap(pol):
    return next(e['unload_capture'] for e in pol.events if 'unload_capture' in e)


def phase_t(pol, ph):
    return next((e['t'] for e in pol.events if e.get('to') == ph), None)


class BarAxis(unittest.TestCase):
    def check_path(self, s, H0r):
        pol = s.pol
        ev = cap(pol)
        T0, a = np.array(ev['T_ref0']), np.array(ev['a'])
        np.testing.assert_allclose(a, H0r[:3, 0], atol=1e-12)                       # 軸＝H_frozen 第一欄
        R_h = H0r[:3, :3]
        for t, ph, act, _ in s.acts:
            if ph == 'HOLD' and act['target'] is not None:
                d = act['target'][:3, 3] - T0[:3, 3]
                dh = R_h.T @ d                                                     # 把手座標
                self.assertLess(abs(dh[1]), 1e-12)                                 # 無接近軸（y）位移
                self.assertLess(abs(dh[2]), 1e-12)
                np.testing.assert_allclose(act['target'][:3, :3], T0[:3, :3], atol=0)
                if t >= ev['t']:
                    self.assertAlmostEqual(dh[0], ev['delta'] * X.smoothstep((t - ev['t']) / 0.5), places=12)
        return ev

    def test_nominal(self):
        s = mk_sim()
        pol = s.run()
        self.assertEqual(pol.phase, 'DONE', pol.events[-3:])
        ev = self.check_path(s, TP.H0())
        self.assertAlmostEqual(ev['delta'], -0.00116, places=12)
        self.assertEqual(ev['axis_mode'], 'bar')
        self.assertGreaterEqual(phase_t(pol, 'OPEN') - ev['t'], 1.0 - 1e-9)
        k = next(i for i, x in enumerate(s.acts) if x[1] == 'OPEN')
        np.testing.assert_allclose(s.acts[k + 1][2]['target'], s.acts[k - 1][2]['target'], atol=1e-12)

    def test_rotated_handle_frame(self):
        with rotated_handle(30.0) as H0r:
            s = mk_sim(axis_world=H0r[:3, 0])
            pol = s.run()
            self.assertEqual(pol.phase, 'DONE', pol.events[-3:])
            ev = self.check_path(s, H0r)
            self.assertAlmostEqual(ev['delta'], -0.00116, places=12)
            self.assertGreater(abs(ev['a'][1]), 0.4)                               # 確實不是世界 x

    def test_y_offset_alone_is_zero_intervention(self):
        s = mk_sim(off=(0.0, -0.00105, 0.0))                                        # 只有接近軸偏移 ⇒ 沿桿 δ＝0
        pol = s.run()
        self.assertEqual(pol.phase, 'DONE')
        self.assertAlmostEqual(cap(pol)['delta'], 0.0, places=15)

    def test_delayed_capture(self):
        def inj(sim, m):
            if sim.pol.phase == 'HOLD' and m.t - sim.pol.t_phase < 1.2:
                m.qdot_door = 0.003
        s = mk_sim(inj)
        pol = s.run()
        self.assertEqual(pol.phase, 'DONE')
        self.assertGreaterEqual(cap(pol)['t'] - phase_t(pol, 'HOLD'), 1.2 - 1e-9)
        self.assertGreaterEqual(phase_t(pol, 'OPEN') - cap(pol)['t'], 1.0 - 1e-9)

    def test_positive_and_out_of_range(self):
        self.assertEqual(mk_sim(d_mm=+0.8).run().abort, 'unload_offset_positive')
        self.assertEqual(mk_sim(d_mm=-3.5).run().abort, 'unload_offset_out_of_range')

    def test_abort_mid_ramp(self):
        def inj(sim, m):
            c = [e for e in sim.pol.events if 'unload_capture' in e]
            if c and 0.2 <= m.t - c[0]['t'] < 0.3:
                m.wrist_max = 31.0
        s = mk_sim(inj)
        pol = s.run()
        self.assertEqual(pol.abort, 'wrist_force')
        self.assertIsNone(phase_t(pol, 'OPEN'))
        self.assertTrue(all(a['target'] is None for t, p, a, _ in s.acts if t >= pol.t_abort))

    def test_axis_mode_must_be_bar(self):
        cfg = X.K3AXConfig()
        cfg.axis_mode = 'approach'
        with self.assertRaises(ValueError):
            X.K3AXPolicy(cfg, TP.T_HG())


if __name__ == '__main__':
    unittest.main(verbosity=2)
