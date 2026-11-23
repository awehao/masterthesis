#!/usr/bin/env python3
"""door_coopen_init_grasp2（INIT_CHECK 連續確認 r2）測試（Codex r16 §2、§3）。
K3A 合成世界（test_door_coopen_core_policy.EdgeSim；策略 init_grasp2(gated2(K3APolicy))）：
  J1 正常：UNFOLD → INIT_CHECK → ENGAGE → … → DONE，無 ALIGN／APPROACH；ENGAGE 前 0.5 s 內每輪 act['init'] valid 且 good、win_t0 不變。
  J2 兩筆好資料中間空白 0.6 s（期間狀態時間不前進＝重讀）⇒ 重讀輪無效、確認窗清空；之後第一筆記 gap_reset；ENGAGE ≥ 空白結束 ＋ 0.5 s。
  J3 位置合格但姿態 0.03 rad（> 0.02）1.0 s ⇒ 不進 ENGAGE、n_rot_block > 0；移除後 ≥ 0.5 s 才 ENGAGE。
  J4 確認期間資料失效（確認窗累積 0.3 s 後重讀 3 輪）⇒ 確認窗清空；ENGAGE ≥ 失效結束 ＋ 0.5 s。
  J5 倒退時間樣本 ⇒ 無效、不累積。
  J6 持續偏 3 mm ⇒ timeout:INIT_CHECK；缺 INIT_CHECK 逾時 ⇒ 建構拒絕；INIT_CHECK 中 gcfg 缺 ⇒ init_rot_tol_missing 中止。
  J7 監看（K3A 組合）：手指—橫桿 0.2 N、牆 0.2 N、門角 3° ⇒ 各自中止於 INIT_CHECK；ACTIVE_MONITOR 每項後還原。
K3M 實際組合（test_door_k3m_settle_p2.P2Sim；策略 init_grasp2(gated2(K3MP2Policy))，同任務節點 v6）：
  M1 正常 ⇒ INIT_CHECK → ENGAGE → HOLD → OPEN …（到 DONE），無 ALIGN／APPROACH。
  M2 INIT_CHECK 中手指—橫桿 0.2 N ⇒ finger_bar_outside_allowed_phase；牆 0.2 N ⇒ wall_contact；門角 3° ⇒ door_moving（中止相位皆 INIT_CHECK）。
  M3 反證：拿掉監看擴充 ⇒ 同樣手指—橫桿注入不中止。

    python3 evaluation/test_door_coopen_init_grasp2.py
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
import door_coopen_init_grasp2 as IG2                  # noqa: E402
import door_k3a_policy as P                            # noqa: E402
import door_k3m_geom_monitor as GM                     # noqa: E402
import door_k3m_policy_p2 as KP2                       # noqa: E402
import test_door_coopen_core_policy as TC              # noqa: E402
import test_door_k3a_policy as TK                      # noqa: E402
import test_door_k3m_policy as TM                      # noqa: E402
import test_door_k3m_settle_p2 as TS                   # noqa: E402

ORIG_AM = tuple(P.ACTIVE_MONITOR)


def rot_z(a):
    R = np.eye(4)
    R[:2, :2] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
    return R


class InitSim(TC.EdgeSim):
    def __init__(self, inject=None, timeout=15.0, gate=True):
        super().__init__(inject)
        T_HG, a_H = TC.edge()
        if timeout is not None:
            self.cfg.timeouts['INIT_CHECK'] = timeout
        self.pol = IG2.init_grasp2(AG2.gated2(P.K3APolicy))(self.cfg, T_HG, a_H=a_H)
        if gate:
            self.pol.gate_init(AG.ApproachGateConfig())
        self.T_HG = T_HG
        self.tcp = TK.H0() @ T_HG
        self.t_inj0 = None
        self.frozen = None


def win_inj(start_after, dur, f):
    """INIT_CHECK 進入後 start_after 秒起、持續 dur 秒對量測施加 f(sim, m)。"""
    def inj(sim, m):
        if sim.pol.phase == 'INIT_CHECK':
            sim.t_inj0 = sim.t_inj0 if sim.t_inj0 is not None else m.t
            if start_after <= m.t - sim.t_inj0 < start_after + dur:
                f(sim, m)
    return inj


def dup(sim, m):
    sim.frozen = sim.frozen if sim.frozen is not None else m.t
    m.t = sim.frozen


def t_engage(pol):
    return next(e['t'] for e in pol.events if e.get('to') == 'ENGAGE')


class J(unittest.TestCase):
    def tearDown(self):
        self.assertEqual(tuple(P.ACTIVE_MONITOR), ORIG_AM)

    def test_j1_nominal(self):
        s = InitSim()
        pol = s.run(t_max=260.0)
        self.assertEqual(pol.phase, 'DONE', pol.events[-3:])
        ph = s.phases()
        for p in ('UNFOLD', 'INIT_CHECK', 'ENGAGE', 'HOLD', 'OPEN', 'SETTLE', 'END_HOLD', 'DONE'):
            self.assertIn(p, ph)
        for p in ('ALIGN_PRE', 'ALIGN_HOLD', 'APPROACH'):
            self.assertNotIn(p, ph)
        te = t_engage(pol)
        rows = [a['init'] for (t, p, a, m) in s.acts if p in ('INIT_CHECK', 'ENGAGE') and 'init' in a and te - 0.5 - 1e-9 <= t]
        self.assertTrue(rows)
        self.assertTrue(all(r['valid'] and r['good'] for r in rows))
        self.assertEqual(len({r['win_t0'] for r in rows}), 1)

    def test_j2_gap(self):
        s = InitSim(win_inj(0.2, 0.6, dup))
        pol = s.run(t_max=260.0)
        self.assertEqual(pol.phase, 'DONE', pol.events[-3:])
        self.assertGreaterEqual(pol.init_stats['n_invalid'], 1)
        self.assertGreaterEqual(pol.init_stats['n_gap_reset'], 1)
        self.assertGreaterEqual(t_engage(pol), s.t_inj0 + 0.8 + 0.5 - 1e-6)

    def test_j3_rot_block(self):
        def f(sim, m):
            m.T_tcp = m.T_tcp @ rot_z(0.03)
        s = InitSim(win_inj(0.0, 1.0, f))
        pol = s.run(t_max=260.0)
        self.assertEqual(pol.phase, 'DONE', pol.events[-3:])
        self.assertGreater(pol.init_stats['n_rot_block'], 0)
        self.assertGreaterEqual(t_engage(pol), s.t_inj0 + 1.0 + 0.5 - 1e-6)

    def test_j4_invalid_in_window(self):
        s = InitSim(win_inj(0.3, 0.15, dup))
        pol = s.run(t_max=260.0)
        self.assertEqual(pol.phase, 'DONE', pol.events[-3:])
        self.assertGreaterEqual(pol.init_stats['n_window_reset'], 1)
        self.assertGreaterEqual(t_engage(pol), s.t_inj0 + 0.45 + 0.5 - 1e-6)

    def test_j5_backward(self):
        def back(sim, m):
            m.t = m.t - 0.5
        s = InitSim(win_inj(0.3, 0.05, back))
        pol = s.run(t_max=260.0)
        self.assertEqual(pol.phase, 'DONE', pol.events[-3:])
        self.assertGreaterEqual(pol.init_stats['n_invalid'], 1)
        self.assertGreaterEqual(t_engage(pol), s.t_inj0 + 0.35 + 0.5 - 1e-6)

    def test_j6_timeout_and_config(self):
        def off(sim, m):
            m.T_tcp = m.T_tcp.copy()
            m.T_tcp[2, 3] += 0.003
        s = InitSim(win_inj(0.0, 1e9, off))
        pol = s.run(t_max=260.0)
        self.assertEqual(pol.abort, 'timeout:INIT_CHECK')
        with self.assertRaises(ValueError):
            InitSim(timeout=None)
        s = InitSim(win_inj(0.0, 1e9, lambda sim, m: setattr(sim.pol, 'gcfg', None)))   # 進入 INIT_CHECK 後才失去姿態容許（未 gate_init 時閘控更早即失敗）
        pol = s.run(t_max=60.0)
        self.assertEqual(pol.abort, 'init_rot_tol_missing')

    def test_j7_monitor_k3a(self):
        for why, f in (('finger_bar_outside_allowed_phase', lambda sim, m: setattr(m, 'finger_bar_max', (0.2, 0.0))),
                       ('wall_contact', lambda sim, m: setattr(m, 'wall_max', 0.2)),
                       ('door_moving', lambda sim, m: setattr(m, 'q_door', math.radians(3.0)))):
            s = InitSim(win_inj(0.1, 1e9, f))
            pol = s.run(t_max=60.0)
            self.assertEqual(pol.abort, why)
            self.assertEqual(next(e for e in pol.events if e.get('abort'))['phase'], 'INIT_CHECK')


class InitP2Sim(TS.P2Sim):
    def __init__(self, inject=None, **kw):
        super().__init__(**kw)
        gfn = (lambda q, a, b: {k: (v, 'x') for k, v in TM.NOM.items()})
        self.cfg = KP2.K3MP2Config()
        self.cfg.timeouts['INIT_CHECK'] = 15.0
        self.pol = IG2.init_grasp2(AG2.gated2(KP2.K3MP2Policy))(self.cfg, TK.T_HG(), TM.RUN, TM.COR, GM.GeometryMonitor(GM.GeomMonitorConfig(), gfn))
        self.pol.gate_init(AG.ApproachGateConfig())
        self.tcp = TK.H0() @ TK.T_HG()
        self.t_inj0 = None
        self.inject_init = inject

    def meas(self):
        m = super().meas()
        if self.inject_init:
            self.inject_init(self, m)
        return m


class M(unittest.TestCase):
    def tearDown(self):
        self.assertEqual(tuple(P.ACTIVE_MONITOR), ORIG_AM)

    def test_m1_nominal(self):
        s = InitP2Sim()
        pol = s.run(t_max=200.0)
        ph = s.phases()
        for p in ('INIT_CHECK', 'ENGAGE', 'HOLD', 'OPEN'):
            self.assertIn(p, ph)
        for p in ('ALIGN_PRE', 'ALIGN_HOLD', 'APPROACH'):
            self.assertNotIn(p, ph)
        self.assertEqual(pol.phase, 'DONE', pol.events[-4:])

    def test_m2_monitor_k3mp2(self):
        for why, f in (('finger_bar_outside_allowed_phase', lambda sim, m: setattr(m, 'finger_bar_max', (0.2, 0.0))),
                       ('wall_contact', lambda sim, m: setattr(m, 'wall_max', 0.2)),
                       ('door_moving', lambda sim, m: setattr(m, 'q_door', math.radians(3.0)))):
            s = InitP2Sim(win_inj(0.1, 1e9, f))
            pol = s.run(t_max=60.0)
            self.assertEqual(pol.abort, why)
            self.assertEqual(next(e for e in pol.events if e.get('abort'))['phase'], 'INIT_CHECK')

    def test_m3_without_scope(self):
        s = InitP2Sim(win_inj(0.1, 1e9, lambda sim, m: setattr(m, 'finger_bar_max', (0.2, 0.0))))
        Base = type(s.pol).__mro__[1]
        s.pol.monitor = lambda m: Base.monitor(s.pol, m)
        pol = s.run(t_max=30.0)
        self.assertNotEqual(pol.abort, 'finger_bar_outside_allowed_phase')


if __name__ == '__main__':
    unittest.main(verbosity=2)
