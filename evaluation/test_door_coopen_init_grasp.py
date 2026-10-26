#!/usr/bin/env python3
"""door_coopen_init_grasp（指定抓取初態 INIT_CHECK 相位）測試（合成世界沿用 test_door_coopen_core_policy.EdgeSim；策略＝init_grasp(gated2(K3APolicy))）。
  I1 正常：相位含 UNFOLD → INIT_CHECK → ENGAGE → HOLD → OPEN → SETTLE → END_HOLD → DONE，**從未出現 ALIGN_PRE／ALIGN_HOLD／APPROACH**；
     INIT_CHECK 目標＝H·T_HG（s＝0）；INIT_CHECK 至 ENGAGE ≥ 0.5 s；events 有 init_check enter／pass。
  I2 TCP 偏 3 mm（> grip_tol 2 mm）1.0 s ⇒ 偏移期間不進 ENGAGE；ENGAGE ≥ 偏移結束 ＋ 0.5 s。
  I3 TCP 持續偏 3 mm ⇒ timeout:INIT_CHECK（距進入 15 s）；不進 ENGAGE。
  I4 監看在 INIT_CHECK 生效：手指—橫桿 0.2 N ⇒ finger_bar_outside_allowed_phase；牆 0.2 N ⇒ wall_contact；門角 3° ⇒ door_moving；
     狀態過期 ⇒ monitor_stale_or_missing。
  I5 執行後 door_k3a_policy.ACTIVE_MONITOR 與原值相同（暫時擴充有還原；含中止路徑）。
  I6 cfg.timeouts 缺 INIT_CHECK ⇒ 建構拒絕（ValueError）。
  I7 UNFOLD 完成當輪即有意外接觸 ⇒ 中止於 UNFOLD，不進 INIT_CHECK。
  I8 反證（必要性）：若拿掉 (I3) 的監看擴充（monitor 直接呼叫父類），同樣的手指—橫桿 0.2 N 注入**不會**中止 ⇒ 證明 (I3) 是必要的。

    python3 evaluation/test_door_coopen_init_grasp.py
"""
import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_approach_gate as AG                 # noqa: E402
import door_coopen_approach_gate2 as AG2               # noqa: E402
import door_coopen_init_grasp as IG                    # noqa: E402
import door_k3a_policy as P                            # noqa: E402
import test_door_coopen_core_policy as TC              # noqa: E402
import test_door_k3a_policy as TK                      # noqa: E402

ORIG_AM = tuple(P.ACTIVE_MONITOR)


class InitSim(TC.EdgeSim):
    def __init__(self, inject=None, timeout=15.0):
        super().__init__(inject)
        T_HG, a_H = TC.edge()
        if timeout is not None:
            self.cfg.timeouts['INIT_CHECK'] = timeout
        self.pol = IG.init_grasp(AG2.gated2(P.K3APolicy))(self.cfg, T_HG, a_H=a_H)
        self.pol.gate_init(AG.ApproachGateConfig())
        self.T_HG = T_HG
        self.tcp = TK.H0() @ T_HG                         # 指定初態：TCP 在 s＝0 抓取位姿
        self.t_inj0 = None


def enter_t(pol):
    return next(e['t'] for e in pol.events if e.get('to') == 'INIT_CHECK')


def offset_inj(dur, mm=3.0, extra=None):
    def inj(sim, m):
        if sim.pol.phase == 'INIT_CHECK':
            sim.t_inj0 = sim.t_inj0 if sim.t_inj0 is not None else m.t
            if dur is None or m.t - sim.t_inj0 < dur:
                m.T_tcp = m.T_tcp.copy()
                m.T_tcp[:3, 3] += np.array([0.0, 0.0, mm / 1000.0])
        if extra:
            extra(sim, m)
    return inj


class I(unittest.TestCase):
    def tearDown(self):
        self.assertEqual(tuple(P.ACTIVE_MONITOR), ORIG_AM)                    # I5（每項都核）

    def test_i1_nominal(self):
        s = InitSim()
        pol = s.run(t_max=260.0)
        self.assertEqual(pol.phase, 'DONE', pol.events[-3:])
        ph = s.phases()
        for p in ('UNFOLD', 'INIT_CHECK', 'ENGAGE', 'HOLD', 'OPEN', 'SETTLE', 'END_HOLD', 'DONE'):
            self.assertIn(p, ph)
        for p in ('ALIGN_PRE', 'ALIGN_HOLD', 'APPROACH'):
            self.assertNotIn(p, ph)
            self.assertFalse(any(e.get('to') == p for e in pol.events))
        self.assertLess(ph.index('INIT_CHECK'), ph.index('ENGAGE'))
        for (_, p, a, m) in s.acts:
            if p == 'INIT_CHECK' and a['target'] is not None:
                np.testing.assert_allclose(a['target'], m.H @ s.T_HG, atol=1e-12)
        t_eng = next(e['t'] for e in pol.events if e.get('to') == 'ENGAGE')
        self.assertGreaterEqual(t_eng - enter_t(pol), 0.5 - 1e-9)
        self.assertEqual([e['init_check'] for e in pol.events if 'init_check' in e], ['enter', 'pass'])

    def test_i2_offset_blocks_engage(self):
        s = InitSim(offset_inj(1.0))
        pol = s.run(t_max=260.0)
        self.assertEqual(pol.phase, 'DONE', pol.events[-3:])
        t_eng = next(e['t'] for e in pol.events if e.get('to') == 'ENGAGE')
        self.assertGreaterEqual(t_eng, s.t_inj0 + 1.0 + 0.5 - 1e-6)

    def test_i3_timeout(self):
        s = InitSim(offset_inj(None))
        pol = s.run(t_max=260.0)
        self.assertEqual(pol.abort, 'timeout:INIT_CHECK')
        t_ab = next(e['t'] for e in pol.events if e.get('abort'))
        self.assertAlmostEqual(t_ab - enter_t(pol), 15.0, delta=0.06)
        self.assertNotIn('ENGAGE', s.phases())

    def test_i4_monitor_active(self):
        cases = (('finger_bar_outside_allowed_phase', lambda m: setattr(m, 'finger_bar_max', (0.2, 0.0))),
                 ('wall_contact', lambda m: setattr(m, 'wall_max', 0.2)),
                 ('door_moving', lambda m: setattr(m, 'q_door', math.radians(3.0))),
                 ('monitor_stale_or_missing', lambda m: setattr(m, 'state_age', 5.0)))
        for why, f in cases:
            def inj(sim, m, f=f):
                if sim.pol.phase == 'INIT_CHECK':
                    f(m)
            s = InitSim(offset_inj(None, extra=inj))
            pol = s.run(t_max=260.0)
            self.assertEqual(pol.abort, why)
            ab = next(e for e in pol.events if e.get('abort'))
            self.assertEqual(ab['phase'], 'INIT_CHECK')

    def test_i6_missing_timeout(self):
        with self.assertRaises(ValueError):
            InitSim(timeout=None)

    def test_i7_unfold_abort(self):
        def inj(sim, m):
            if sim.pol.phase == 'UNFOLD' and m.unfold_done:
                m.unexpected_max = 0.5
        s = InitSim(inj)
        pol = s.run(t_max=260.0)
        self.assertEqual(pol.abort, 'unexpected_contact')
        self.assertNotIn('INIT_CHECK', s.phases())

    def test_i8_without_scope_no_monitor(self):
        s = InitSim(offset_inj(None, extra=lambda sim, m: setattr(m, 'finger_bar_max', (0.2, 0.0)) if sim.pol.phase == 'INIT_CHECK' else None))
        Base = type(s.pol).__mro__[1]
        s.pol.monitor = lambda m: Base.monitor(s.pol, m)
        pol = s.run(t_max=40.0)
        self.assertNotEqual(pol.abort, 'finger_bar_outside_allowed_phase')


if __name__ == '__main__':
    unittest.main(verbosity=2)
