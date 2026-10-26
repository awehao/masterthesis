#!/usr/bin/env python3
"""door_k3b_release5 反例（終點收斂窗修正；**合成資料**）。
  V1 不規則取樣（任務端週期 0.06／0.09 s 交替，終點靜止）：r4 窗內跨度最多 0.45 s ⇒ RETREAT_DONE 不成立；
     r5 於手臂到達終點後 ≤ 0.6 s 內 RETREAT_DONE，並 COMPLETE。
  V2 不規則取樣下終點振盪（x ±2.5 mm）⇒ r5 仍不進 RETREAT_DONE（窗略長於 0.5 s，保守方向）。
  R  回歸：test_door_k3b_release4 全部反例改以 release5 執行。

    python3 evaluation/test_door_k3b_release5.py
"""
import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3b_release4 as R4                         # noqa: E402
import door_k3b_release5 as R5                         # noqa: E402
import test_door_k3b_release4 as T4                    # noqa: E402

PER = (0.06, 0.09)                                   # 連續區間和只會是 0.45 或 0.51 s，不會恰為 0.50


def run_irregular(m, t_end=16.0, mod=None):
    m.open(0.0, 0, T4.Q_REL, T4.E_REL)
    s, t, i, t_arrive = 0.0, 0.0, 0, None
    while t < t_end:
        dt = PER[i % 2]
        t = round(t + dt, 4)
        i += 1
        es = 'B_CONFIRMED' if t >= 0.81 else 'A_OPENING'
        if m.state in ('NEAR', 'C_CLEAR') or (m.state in ('RETREAT_DONE', 'COMPLETE') and s < 0.10):
            s = min(0.10, s + 0.02 * dt)
        if s >= 0.10 and t_arrive is None:
            t_arrive = t
        kw = (mod(t, s, m) if mod else None) or {}
        x = T4.tsample(i * 7, t, s, es, **kw)
        m.step(x)
        if m.state == 'FAILED':
            break
    return m, t_arrive


class TV(unittest.TestCase):
    def test_v1_irregular_converges(self):
        m4, ta4 = run_irregular(R4.TaskReleaseMonitor())
        m5, ta5 = run_irregular(R5.TaskReleaseMonitor())
        self.assertIn('RETREAT_DONE', m5.t, (m5.state, m5.why))
        self.assertLessEqual(m5.t['RETREAT_DONE'] - ta5, 0.6 + 1e-9)
        self.assertEqual(m5.state, 'COMPLETE')
        late4 = m4.t.get('RETREAT_DONE')
        self.assertTrue(late4 is None or late4 - ta4 > 1.5, (late4, ta4))

    def test_v2_oscillation_still_rejected(self):
        def osc(t, s, mm):
            if s >= 0.095:
                v = np.zeros(3)
                v[0] = 0.0025 * math.sin(2 * math.pi * 1.5 * t)
                return {'E_extra': v}
            return None
        m, _ = run_irregular(R5.TaskReleaseMonitor(), mod=osc)
        self.assertNotIn('RETREAT_DONE', m.t, (m.state, m.why))


class RR(unittest.TestCase):
    def test_release4_cases_on_r5(self):
        orig = T4.R
        T4.R = R5
        try:
            res = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w')).run(unittest.defaultTestLoader.loadTestsFromModule(T4))
        finally:
            T4.R = orig
        self.assertTrue(res.wasSuccessful(), [str(x[1])[-300:] for x in res.failures + res.errors])
        self.assertEqual(res.testsRun, 17)


if __name__ == '__main__':
    unittest.main(verbosity=1)
