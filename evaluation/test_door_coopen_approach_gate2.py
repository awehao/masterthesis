#!/usr/bin/env python3
"""接近進度閘控 r2（door_coopen_approach_gate2）的反例（Codex 20261027_004139 §1、§2）。
r1 的 G1–G8 全部以 r2 重跑（子類換 gated2），另加 Codex 重現的三個缺口與恢復窗缺資料：
  R1 垂直 0、沿軸落後 10 mm（量測 TCP 沿 +接近軸偏 10 mm）⇒ 注入期間 s 不推進（每輪 s 不變）；移除後推進並 DONE。
  R2 s 推到 0、位置到抓取點、姿態 0.03 rad（量測繞工具 z 轉 0.03 rad）⇒ 注入期間不進 ENGAGE（engage_veto 或從未嘗試），移除後 ENGAGE；
     且 ENGAGE 前 0.5 s 內每輪 advance_ok 皆為真（閉爪前穩定窗在許可下累積）。
  R3 ALIGN_HOLD 只有兩筆有效資料、間隔 0.6 s（其間重讀同一狀態）⇒ 不放行；連續有效 0.5 s 後才放行。
  R4 暫停後恢復窗中出現 0.6 s 資料空白（重讀）⇒ 恢復窗清空，resume 時刻 ≥ 空白結束 ＋ 0.5 s；n_gap_reset ≥ 1。
  R5 無效量測（重讀）不推進：注入重讀的輪次 s 不變。

    python3 evaluation/test_door_coopen_approach_gate2.py
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
import door_k3a_policy as P                            # noqa: E402
import test_door_coopen_approach_gate as TG            # noqa: E402


class GSim2(TG.GSim):
    def __init__(self, inject=None, gate=True):
        super().__init__(inject, gate=False)
        if gate:
            T_HG, a_H = TG.TC.edge()
            self.pol = AG2.gated2(P.K3APolicy)(self.cfg, T_HG, a_H=a_H)
            self.pol.gate_init(AG.ApproachGateConfig())


class GOld(TG.G):
    """r1 的 G1–G8，以 r2 重跑。"""
    def setUp(self):
        self._orig = TG.GSim
        TG.GSim = GSim2

    def tearDown(self):
        TG.GSim = self._orig


def s_series(sim, t0, t1):
    return [(t, m_s) for (t, ph, a, m), m_s in zip(sim.acts, sim.s_log) if ph == 'APPROACH' and t0 <= t < t1]


class LogSim(GSim2):
    def __init__(self, inject=None):
        super().__init__(inject)
        self.s_log = []

    def run(self, t_max=260.0):
        pol = self.pol
        orig = pol.step

        def step(m, dt):
            a = orig(m, dt)
            self.s_log.append(getattr(pol, 's', None))
            return a
        pol.step = step
        return super().run(t_max)


class R(unittest.TestCase):
    def test_r1_along_lag_blocks_advance(self):
        def inj(sim, m):
            if sim.pol.phase == 'APPROACH' and sim.pol.s < 0.06:
                sim.t_inj0 = sim.t_inj0 if sim.t_inj0 is not None else m.t
                if m.t - sim.t_inj0 < 1.0:
                    ax = m.H[:3, :3] @ sim.a_H
                    m.T_tcp = m.T_tcp.copy()
                    m.T_tcp[:3, 3] += ax * 0.010
        s = LogSim(inj)
        pol = s.run()
        self.assertEqual(pol.phase, 'DONE', pol.events[-3:])
        ss = [v for (t, v) in s_series(s, s.t_inj0 + 0.051, s.t_inj0 + 1.0)]
        self.assertTrue(ss)
        self.assertEqual(len(set(round(v, 12) for v in ss)), 1, ss[:5])
        self.assertEqual(pol.gate_stats['n_pause'], 0)
        self.assertGreater(pol.gate_stats['n_hold_along'], 0)

    def test_r2_rot_blocks_engage(self):
        def inj(sim, m):
            if sim.pol.phase == 'APPROACH' and sim.pol.s <= 0.001:
                sim.t_inj0 = sim.t_inj0 if sim.t_inj0 is not None else m.t
                if m.t - sim.t_inj0 < 1.5:
                    c, s_ = math.cos(0.03), math.sin(0.03)
                    Rz = np.array([[c, -s_, 0, 0], [s_, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])
                    m.T_tcp = m.T_tcp @ Rz
        s = GSim2(inj)
        pol = s.run()
        self.assertEqual(pol.phase, 'DONE', pol.events[-3:])
        t_eng = next(e['t'] for e in pol.events if e.get('to') == 'ENGAGE')
        self.assertGreaterEqual(t_eng, s.t_inj0 + 1.5 + 0.5 - 1e-9)
        win = [a['gate']['advance_ok'] for (t, ph, a, m) in s.acts if ph == 'APPROACH' and t_eng - 0.5 + 1e-9 <= t < t_eng]
        self.assertTrue(win and all(win))

    def test_r3_align_two_samples_gap_not_confirmed(self):
        def inj(sim, m):
            if sim.pol.phase == 'ALIGN_HOLD':
                sim.ah0 = getattr(sim, 'ah0', None) or m.t
                if 0.05 <= m.t - sim.ah0 < 0.65:
                    m.t = sim.ah0 + 0.05                         # 重讀同一狀態 0.6 s
        s = GSim2(inj)
        s.run()
        t_hold = next(e['t'] for e in s.pol.events if e.get('to') == 'ALIGN_HOLD')
        t_appr = next(e['t'] for e in s.pol.events if e.get('to') == 'APPROACH')
        self.assertGreaterEqual(t_appr - t_hold, 0.65 + 0.5 - 0.06)
        self.assertGreaterEqual(s.pol.gate_stats['n_gap_reset'], 1)

    def test_r4_resume_window_gap(self):
        def inj(sim, m):
            if sim.pol.phase == 'APPROACH' and sim.pol.s < 0.06:
                sim.t_inj0 = sim.t_inj0 if sim.t_inj0 is not None else m.t
                dtt = m.t - sim.t_inj0
                if dtt < 1.0:
                    m.T_tcp = m.T_tcp.copy()
                    m.T_tcp[:3, 3] += sim.perp_dir(m.H) * 0.004
                elif 1.2 <= dtt < 1.8:
                    m.t = sim.t_inj0 + 1.2                       # 恢復窗中 0.6 s 重讀
        s = GSim2(inj)
        pol = s.run()
        rs = [e for e in pol.events if 'approach_gate_resume' in e]
        self.assertTrue(rs)
        self.assertGreaterEqual(rs[0]['t'], s.t_inj0 + 1.8 + 0.5 - 0.06)
        self.assertGreaterEqual(pol.gate_stats['n_gap_reset'], 1)

    def test_r5_invalid_no_advance(self):
        def inj(sim, m):
            if sim.pol.phase == 'APPROACH' and sim.pol.s < 0.08:
                sim.t_inj0 = sim.t_inj0 if sim.t_inj0 is not None else m.t
                if 0.0 <= sim.t - sim.t_inj0 < 0.5:
                    m.t = sim.t_inj0                             # 重讀
        s = LogSim(inj)
        s.run()
        ss = [v for (t, v) in s_series(s, -1, 1e9)]
        idx = [i for i, ((t, ph, a, m)) in enumerate(s.acts) if ph == 'APPROACH' and s.t_inj0 is not None and 0.05 <= t - s.t_inj0 < 0.45]
        vals = [s.s_log[i] for i in idx]
        self.assertTrue(vals)
        self.assertEqual(len(set(round(v, 12) for v in vals)), 1)
        self.assertTrue(ss)


if __name__ == '__main__':
    unittest.main(verbosity=2)
