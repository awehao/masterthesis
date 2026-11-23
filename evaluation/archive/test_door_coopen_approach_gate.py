#!/usr/bin/env python3
"""接近進度閘控（door_coopen_approach_gate）的假世界反例（Codex 20261027_003024 §3-2）。合成世界沿用 test_door_coopen_core_policy.EdgeSim
（繞緣 T_HG＋a_H、門速 0.6、θ 48），策略換成 gated(K3APolicy)；擾動以注入「量測 TCP 垂直接近軸偏移」模擬追蹤偏差。
  G1 偏離後凍結 s、恢復後續進：s < 0.06 起注入 4 mm 垂直偏移 1.5 s ⇒ 有 pause 事件、注入期間 s 不變；移除後 ≥ 0.5 s 才 resume；最後 DONE。
  G2 一直不收斂 ⇒ timeout:APPROACH（逾時自進入 APPROACH 起算、未重置）；s 未到 0、無 resume。
  G3 失效資料不累積恢復窗：偏移移除後 1.0 s 內量測狀態時間不前進（重讀同一狀態）⇒ resume 時刻 ≥ 移除 ＋ 1.0 ＋ 0.5 s。
  G4 暫停中提前碰桿（finger_bar_max 2 N）⇒ 仍立即 finger_bar_outside_allowed_phase。
  G5 對準確認：ALIGN_HOLD 起至 APPROACH ≥ 0.5 s；誤差每 4 輪閃爍一次超過 pre_tol ⇒ 不放行，閃爍停止後 ≥ 0.5 s 才放行。
  G6 暫停期間每輪仍發布目標（非 None）且＝T(s 凍結值)、solver_start＝True（控制照常、目標停在當前位置）。
  G7 無擾動：照常 DONE、無 pause 事件、APPROACH 時長 ≈ 原版＋0.5 s 對準確認。
  G8 回歸：未疊閘控的 K3APolicy 不受影響（同一合成世界照常 DONE）。

    python3 evaluation/test_door_coopen_approach_gate.py
"""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_approach_gate as AG                 # noqa: E402
import door_k3a_policy as P                            # noqa: E402
import test_door_coopen_core_policy as TC              # noqa: E402


class GSim(TC.EdgeSim):
    def __init__(self, inject=None, gate=True):
        super().__init__(inject)
        T_HG, a_H = TC.edge()
        if gate:
            self.pol = AG.gated(P.K3APolicy)(self.cfg, T_HG, a_H=a_H)
            self.pol.gate_init(AG.ApproachGateConfig())
        self.a_H = np.asarray(a_H)
        self.T_HG = T_HG
        self.t_inj0 = None

    def perp_dir(self, H):
        y = H[:3, :3] @ self.T_HG[:3, 1]                       # 夾爪開合方向（垂直於接近軸）
        ax = H[:3, :3] @ self.a_H
        y = y - (y @ ax) * ax
        return y / np.linalg.norm(y)


def run(sim, t_max=260.0):
    pol = sim.run(t_max=t_max)
    return pol, [e for e in pol.events if 'approach_gate_pause' in e], [e for e in pol.events if 'approach_gate_resume' in e]


def offset_inj(dur, mm=4.0, extra=None):
    def inj(sim, m):
        if sim.pol.phase == 'APPROACH' and sim.pol.s < 0.06:
            sim.t_inj0 = sim.t_inj0 if sim.t_inj0 is not None else m.t
            if dur is None or m.t - sim.t_inj0 < dur:
                m.T_tcp = m.T_tcp.copy()
                m.T_tcp[:3, 3] += sim.perp_dir(m.H) * mm / 1000.0
        if extra:
            extra(sim, m)
    return inj


class G(unittest.TestCase):
    def test_g1_pause_freeze_resume(self):
        s = GSim(offset_inj(1.5))
        pol, pz, rs = run(s)
        self.assertEqual(pol.phase, 'DONE', pol.events[-3:])
        self.assertGreaterEqual(len(pz), 1)
        self.assertGreaterEqual(len(rs), 1)
        t_end_inj = s.t_inj0 + 1.5
        self.assertGreaterEqual(rs[0]['t'], t_end_inj + 0.5 - 1e-9)
        svals = [(t, a.get('gate', {}).get('paused'), a['target']) for (t, ph, a, m) in s.acts if ph == 'APPROACH']
        paused_targets = [a['target'][:3, 3] for (t, ph, a, m) in s.acts if ph == 'APPROACH' and a.get('gate', {}).get('paused')]
        self.assertTrue(paused_targets)
        for p in paused_targets[1:]:
            np.testing.assert_allclose(p, paused_targets[0], atol=1e-12)
        self.assertTrue(svals)

    def test_g2_never_converges_timeout(self):
        pol, pz, rs = run(GSim(offset_inj(None)))
        self.assertEqual(pol.abort, 'timeout:APPROACH')
        self.assertGreater(pol.s, 0.0)
        self.assertEqual(rs, [])

    def test_g3_invalid_data_no_accumulate(self):
        def rep(sim, m):
            if sim.t_inj0 is not None and 1.5 <= m.t - sim.t_inj0 < 2.5:
                sim.t_rep = getattr(sim, 't_rep', None) or m.t
                m.t = sim.t_rep                                    # 重讀同一狀態：狀態時間不前進
        s = GSim(offset_inj(1.5, extra=rep))
        pol, pz, rs = run(s)
        self.assertTrue(rs)
        self.assertGreaterEqual(rs[0]['t'], s.t_inj0 + 1.5 + 1.0 + 0.5 - 0.06)

    def test_g4_bar_contact_still_aborts_when_paused(self):
        def bar(sim, m):
            if sim.pol.phase == 'APPROACH' and sim.pol.gate['paused'] and sim.t_inj0 is not None and m.t - sim.t_inj0 > 0.5:
                m.finger_bar_max = (2.0, 0.0)
        pol, pz, rs = run(GSim(offset_inj(None, extra=bar)))
        self.assertTrue(pz)
        self.assertEqual(pol.abort, 'finger_bar_outside_allowed_phase')

    def test_g5_align_confirm(self):
        s = GSim()
        s.run(t_max=260.0)
        t_hold = next(e['t'] for e in s.pol.events if e.get('to') == 'ALIGN_HOLD')
        t_appr = next(e['t'] for e in s.pol.events if e.get('to') == 'APPROACH')
        self.assertGreaterEqual(t_appr - t_hold, 0.5 - 1e-9)

        def flicker(sim, m):
            if sim.pol.phase == 'ALIGN_HOLD':
                sim.k = getattr(sim, 'k', 0) + 1
                if sim.k <= 40 and sim.k % 4 == 0:
                    m.T_tcp = m.T_tcp.copy()
                    m.T_tcp[:3, 3] += np.array([0.0, 0.0, 0.006])
        s2 = GSim(flicker)
        s2.run(t_max=260.0)
        t_hold2 = next(e['t'] for e in s2.pol.events if e.get('to') == 'ALIGN_HOLD')
        t_appr2 = next(e['t'] for e in s2.pol.events if e.get('to') == 'APPROACH')
        self.assertGreaterEqual(t_appr2 - t_hold2, 40 * 0.05 + 0.5 - 0.11)
        self.assertGreater(s2.pol.gate_stats['n_align_wait'], 0)

    def test_g6_control_continues_when_paused(self):
        s = GSim(offset_inj(1.5))
        s.run(t_max=260.0)
        paused = [(a, m) for (t, ph, a, m) in s.acts if ph == 'APPROACH' and a.get('gate', {}).get('paused')]
        self.assertTrue(paused)
        for a, m in paused:
            self.assertIsNotNone(a['target'])
            self.assertTrue(a['solver_start'])
            np.testing.assert_allclose(a['target'], s.pol.target(m.H, a['gate']['info']['s']), atol=1e-12)

    def test_g7_nominal(self):
        s = GSim()
        pol, pz, rs = run(s)
        self.assertEqual(pol.phase, 'DONE')
        self.assertEqual(pz, [])
        t_a = next(e['t'] for e in pol.events if e.get('to') == 'APPROACH')
        t_e = next(e['t'] for e in pol.events if e.get('to') == 'ENGAGE')
        self.assertLess(t_e - t_a, pol.c.s_pre / pol.c.approach_rate + pol.c.grip_settle_s + 0.3)

    def test_g8_ungated_unchanged(self):
        s = GSim(gate=False)
        self.assertEqual(s.run(t_max=260.0).phase, 'DONE')
        self.assertFalse(hasattr(s.pol, 'gate'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
