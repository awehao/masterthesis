#!/usr/bin/env python3
"""door_coopen_bench_grasp2（台架起始相位 r2）測試（Codex r21 §2）；合成世界 test_door_coopen_init_grasp2.InitP2Sim；策略＝bench_grasp2(init_grasp2(gated2(K3MP2Policy)))。
凍結：m.state_step＝0 且 m.t＝t0；前進：state_step 沿用合成世界步號（≥ 1）、m.t 前進。bench_state 依情境給。
  U1 正常（凍結 20 輪後同輪 released 且前進）⇒ BENCH_START → INIT_CHECK → … → DONE；相位無 WAIT_HANDOVER／UNFOLD／ALIGN／APPROACH；凍結輪不監看、前進輪有監看。
  U2 通知延遲（物理已前進 0.10 s 仍 frozen，之後 released）⇒ 延遲期間每輪監看（monitored_cycles ≥ 1）、之後照常 INIT_CHECK → DONE。
  U3 通知遺失（前進後一直 frozen）⇒ 0.2 s 後 bench_release_unconfirmed（相位 BENCH_START）。
  U4 第一筆前進資料已有手指—橫桿 0.2 N（通知尚未到）⇒ finger_bar_outside_allowed_phase，中止相位 BENCH_START。
  U5 凍結期注入手指—橫桿 0.2 N ⇒ 不中止（物理不演化，跳過監看並計數）；之後前進無接觸 ⇒ 正常。
  U6 ACTIVE_MONITOR 每項後還原。

    python3 evaluation/test_door_coopen_bench_grasp2.py
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_approach_gate as AG                 # noqa: E402
import door_coopen_approach_gate2 as AG2               # noqa: E402
import door_coopen_bench_grasp2 as BG2                 # noqa: E402
import door_coopen_init_grasp2 as IG2                  # noqa: E402
import door_k3a_policy as P                            # noqa: E402
import door_k3m_geom_monitor as GM                     # noqa: E402
import door_k3m_policy_p2 as KP2                       # noqa: E402
import test_door_coopen_init_grasp2 as T2              # noqa: E402
import test_door_k3a_policy as TK                      # noqa: E402
import test_door_k3m_policy as TM                      # noqa: E402

ORIG_AM = tuple(P.ACTIVE_MONITOR)


class BSim(T2.InitP2Sim):
    def __init__(self, frozen_n=20, notice_delay_n=0, notice_lost=False, inject=None):
        super().__init__()
        gfn = (lambda q, a, b: {k: (v, 'x') for k, v in TM.NOM.items()})
        self.pol = BG2.bench_grasp2(IG2.init_grasp2(AG2.gated2(KP2.K3MP2Policy)))(self.cfg, TK.T_HG(), TM.RUN, TM.COR,
                                                                                GM.GeometryMonitor(GM.GeomMonitorConfig(), gfn))
        self.pol.gate_init(AG.ApproachGateConfig())
        self.n, self.t0, self.owner = 0, None, 'wholebody'
        self.frozen_n, self.delay_n, self.lost, self.inj = frozen_n, notice_delay_n, notice_lost, inject

    def meas(self):
        m = super().meas()
        self.n += 1
        self.t0 = m.t if self.t0 is None else self.t0
        if self.n <= self.frozen_n:
            m.t, m.state_step, m.bench_state = self.t0, 0, 'frozen'
        else:
            m.state_step = max(1, int(m.state_step))                       # 前進：沿用合成世界步號（≥ 1；K3-M 模式請求依它）
            m.bench_state = 'frozen' if (self.lost or self.n <= self.frozen_n + self.delay_n) else 'released'
        m.bench_t0 = self.t0
        if self.inj:
            self.inj(self, m)
        return m


class U(unittest.TestCase):
    def tearDown(self):
        self.assertEqual(tuple(P.ACTIVE_MONITOR), ORIG_AM)

    def test_u1_nominal(self):
        s = BSim()
        pol = s.run(t_max=200.0)
        ph = s.phases()
        self.assertEqual(ph[0], 'BENCH_START')
        for p in ('INIT_CHECK', 'ENGAGE', 'HOLD', 'OPEN'):
            self.assertIn(p, ph)
        for p in ('WAIT_HANDOVER', 'UNFOLD', 'ALIGN_PRE', 'ALIGN_HOLD', 'APPROACH'):
            self.assertNotIn(p, ph)
        self.assertEqual(pol.phase, 'DONE', pol.events[-4:])
        self.assertEqual(pol.bench_stats['monitor_skipped_cycles'], 20)
        self.assertGreaterEqual(pol.bench_stats['monitored_cycles'], 1)

    def test_u2_notice_delayed(self):
        s = BSim(notice_delay_n=5)
        pol = s.run(t_max=200.0)
        self.assertEqual(pol.phase, 'DONE', pol.events[-4:])
        self.assertGreaterEqual(pol.bench_stats['monitored_cycles'], 6)

    def test_u3_notice_lost(self):
        s = BSim(notice_lost=True)
        pol = s.run(t_max=60.0)
        self.assertEqual(pol.abort, 'bench_release_unconfirmed')
        ab = next(e for e in pol.events if e.get('abort'))
        self.assertEqual(ab['phase'], 'BENCH_START')
        self.assertLessEqual(ab['t'] - pol.bench_stats['t_first_advanced'], 0.2 + 0.06)

    def test_u4_contact_first_advanced(self):
        def inj(s, m):
            if s.n == s.frozen_n + 1:
                m.finger_bar_max = (0.2, 0.0)
        s = BSim(notice_delay_n=3, inject=inj)
        pol = s.run(t_max=60.0)
        self.assertEqual(pol.abort, 'finger_bar_outside_allowed_phase')
        self.assertEqual(next(e for e in pol.events if e.get('abort'))['phase'], 'BENCH_START')

    def test_u5_frozen_injection(self):
        def inj(s, m):
            if s.n <= s.frozen_n:
                m.finger_bar_max = (0.2, 0.0)
        s = BSim(inject=inj)
        pol = s.run(t_max=200.0)
        self.assertEqual(pol.phase, 'DONE', pol.events[-4:])
        self.assertEqual(pol.bench_stats['monitor_skipped_cycles'], 20)


if __name__ == '__main__':
    unittest.main(verbosity=2)
