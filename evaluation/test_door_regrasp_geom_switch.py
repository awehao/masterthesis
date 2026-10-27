#!/usr/bin/env python3
"""door_regrasp_geom_switch 反例（Codex r60 §4）。後端以合成距離替身（只模擬「目標改變使手指類組成改變」）：
手指類＝min(門板距離, 外側橫桿距離（目標非 out 時才計入）)。
  S1 靜止切換不誤停：門板 60 mm、外側橫桿 30 mm 不動，目標 out → None：包裝後不停（接近率 0）；未包裝（直接切）會停（對照，證明假象存在）；
  S2 切換同時真實接近仍會停：切換當輪外側橫桿由 40 降到 5 mm（真實接近）⇒ 停；
  S3 前態失效拒絕：前態 qf 含 NaN（替身對 NaN 拋例外）⇒ geometry_switch_prev_invalid；
  S4 事件記錄前態舊／新距離、重算耗時，且重算耗時進入 state_age（T_pred 變大）。

    python3 evaluation/test_door_regrasp_geom_switch.py
"""
import math
import os
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import door_k3m_geom_monitor as GM                     # noqa: E402
import door_regrasp_geom_switch as GS                  # noqa: E402


class Fake:
    """qf[0]＝門板距離、qf[1]＝外側橫桿距離（m）；其他類固定。"""

    def __init__(self, delay=0.0):
        self.tgt, self.delay = 'out', delay

    def set_target(self, t):
        self.tgt = t

    def __call__(self, qf, q, qfi):
        if not np.all(np.isfinite(qf)):
            raise ValueError('nan')
        if self.delay:
            time.sleep(self.delay)
        leaf, bar = float(qf[0]), float(qf[1])
        f = leaf if self.tgt == 'out' else min(leaf, bar)
        return {'nonfinger': (0.1, 'x'), 'finger': (f, 'x'), 'base': (0.5, 'x'), 'self': (0.1, 'x')}


def qf(leaf, bar):
    return np.r_[leaf, bar, np.zeros(7)]


def mon(fn):
    return GM.GeometryMonitor(GM.GeomMonitorConfig(age_max_s=10.0), fn)


class TS(unittest.TestCase):
    def test_s1(self):
        fn = Fake()
        g = GS.SwitchAwareGeom(mon(fn), fn)
        self.assertEqual(g.step(0.0, qf(0.06, 0.03), 0.8, 0.0, 0.01, 'out')[0], 'ok')
        dec, det = g.step(0.05, qf(0.06, 0.03), 0.8, 0.0, 0.01, None)
        self.assertNotEqual(dec, 'stop', det)
        self.assertEqual(det['classes']['finger']['closing_rate'], 0.0)
        fn2 = Fake()                                                    # 對照：未包裝
        m2 = mon(fn2)
        fn2.set_target('out')
        m2.step(0.0, qf(0.06, 0.03), 0.8, 0.0, 0.01)
        fn2.set_target(None)
        dec2, det2 = m2.step(0.05, qf(0.06, 0.03), 0.8, 0.0, 0.01)
        self.assertEqual(dec2, 'stop')                                 # 靜止、30 mm：只因組成改變被誤判接近
        self.assertGreater(det2['classes']['finger']['closing_rate'], 0.5)

    def test_s2(self):
        fn = Fake()
        g = GS.SwitchAwareGeom(mon(fn), fn)
        g.step(0.0, qf(0.06, 0.04), 0.8, 0.0, 0.01, 'out')
        dec, det = g.step(0.05, qf(0.06, 0.005), 0.8, 0.0, 0.01, None)
        self.assertEqual(dec, 'stop')
        self.assertGreater(det['classes']['finger']['closing_rate'], 0.5)

    def test_s3(self):
        fn = Fake()
        g = GS.SwitchAwareGeom(mon(fn), fn)
        g.step(0.0, qf(0.06, 0.04), 0.8, 0.0, 0.01, 'out')
        g.prev_state = (0.0, np.r_[np.nan, np.zeros(8)], 0.8, 0.0)
        dec, det = g.step(0.05, qf(0.06, 0.04), 0.8, 0.0, 0.01, None)
        self.assertEqual((dec, det['why']), ('stop', 'geometry_switch_prev_invalid'))

    def test_s4(self):
        fn = Fake(delay=0.02)
        g = GS.SwitchAwareGeom(mon(fn), fn)
        g.step(0.0, qf(0.06, 0.04), 0.8, 0.0, 0.01, 'out')
        dec, det = g.step(0.05, qf(0.06, 0.04), 0.8, 0.0, 0.01, None)
        ev = g.events[-1]
        self.assertEqual(ev['switch'], ['out', None])
        self.assertAlmostEqual(ev['prev_old']['finger'], 0.06)
        self.assertAlmostEqual(ev['prev_new']['finger'], 0.04)
        self.assertGreater(ev['recompute_s'], 0.015)
        self.assertGreater(det['T_pred_s'], 0.05 + 0.01 + 0.015 + 0.02)
        self.assertTrue(math.isfinite(det['switch_recompute_s']))


if __name__ == '__main__':
    unittest.main(verbosity=1)
