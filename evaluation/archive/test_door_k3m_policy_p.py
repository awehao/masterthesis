#!/usr/bin/env python3
"""door_k3m_policy_p（K3-M＋受控暫停）的離線反例（沿用 test_door_k3m_policy 的合成世界；暫停握手以 door_k3m_pause_mode 權威類別模擬）：
  正常：完整 SETTLE 後轉 PAUSE（不進 END_HOLD）、請求帶最後已發布目標位置、PAUSE 期間目標保持、求解端 DONE ⇒ DONE；
  求解端 FAIL ⇒ 中止 pause_failed:<why>；執行端 nack ⇒ pause_switch:executor_nack；無求解端 ack ⇒ pause_switch 逾時；
  PAUSE 期間非設計接觸 ⇒ 既有監看中止（紀錄相位為 PAUSE）；暫停狀態一直未完成 ⇒ pause_total_timeout；已中止 ⇒ 暫停請求清除不發。

    python3 evaluation/test_door_k3m_policy_p.py
"""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_pause_mode as PMODE                    # noqa: E402
import door_k3m_policy_p as KP                         # noqa: E402
import door_k3m_geom_monitor as GM                     # noqa: E402
import test_door_k3a_policy as TP                      # noqa: E402
import test_door_k3m_policy as TM                      # noqa: E402


class PSim(TM.MSim):
    """status_after：暫停確認後幾輪給求解端狀態（None ⇒ 不給）；status_state：'DONE'／'FAIL'；pause_exec_nack；pause_solver（是否回求解端 ack）。"""
    def __init__(self, status_after=5, status_state='DONE', pause_exec_nack=None, pause_solver=True, hook=None, **kw):
        super().__init__(hook=hook, **kw)
        self.cfg = KP.K3MPConfig()
        gfn = (lambda q, a, b: {k: (v, 'x') for k, v in TM.NOM.items()})
        self.pol = KP.K3MPPolicy(self.cfg, TP.T_HG(), TM.RUN, TM.COR, GM.GeometryMonitor(GM.GeomMonitorConfig(), gfn))
        self.PE, self.PS = PMODE.ExecutorPauseAuthority(TM.RUN), PMODE.SolverPauseSwitch(TM.RUN)
        self.preq, self.pn = None, 0
        self.status_after, self.status_state = status_after, status_state
        self.pause_exec_nack, self.pause_solver = pause_exec_nack, pause_solver
        self.pause_rows, self.n_conf = [], None
        self.n_pub = 0

    def meas(self):
        m = super().meas()
        if self.preq is not None:
            self.pn += 1
            if self.pn == 1:
                if self.pause_exec_nack:
                    self.pol.on_pause_executor({'run_id': TM.RUN, 'seq': 2, 'why': self.pause_exec_nack}, self.t, self.wall)
                else:
                    k, ack = self.PE.on_request(self.preq, m.state_step, self.t, aborted=False, base_mode_free=True)
                    self.pol.on_pause_executor(ack, self.t, self.wall)
                    self.PS.on_executor_ack(ack, stopped=False, free_active=True)
            if self.pn == 2 and self.pause_solver and self.PS.pending is not None:
                self.PS.apply(self.preq['frozen_target_T'], snap_t=self.t)
                self.pol.on_pause_solver(self.PS.note_first_solution(9, self.t, True, self.t), self.t, self.wall)
            if self.pol.pause_state == 'active':
                self.n_conf = self.n_conf if self.n_conf is not None else self.pn
                if self.status_after is not None and self.pn - self.n_conf >= self.status_after:
                    o = {'state': self.status_state, 'fail_why': None if self.status_state == 'DONE' else 'moved_after_stop',
                         'command_stop_confirmed': self.status_state == 'DONE', 'physical_stop_confirmed': self.status_state == 'DONE'}
                    self.pol.on_pause_status({'outcome': o})
        return m

    def run(self, t_max=160.0):
        orig = self.pol.step

        def wrapped(m, dt):
            self.pol.pub_seq = self.n_pub
            a = orig(m, dt)
            for r in self.pol.take_pause_outbox():
                self.preq = r
            if a['target'] is not None:
                self.n_pub += 1
                self.last_pub = a['target']
            if self.pol.phase == 'PAUSE':
                self.pause_rows.append((m.t, a['target']))
            return a
        self.pol.step = wrapped
        return super().run(t_max)


class T(unittest.TestCase):
    def test_nominal_pause_done(self):
        s = PSim()
        pol = s.run()
        self.assertEqual(pol.phase, 'DONE', pol.events[-4:])
        phases = [e.get('to') for e in pol.events if 'to' in e]
        self.assertIn('PAUSE', phases)
        self.assertNotIn('END_HOLD', [e.get('to') for e in pol.events if e.get('from') == 'PAUSE'])
        req = next(e['k3m_pause_request'] for e in pol.events if 'k3m_pause_request' in e)
        T0 = s.pause_rows[0][1]
        np.testing.assert_allclose(req['frozen_target_xyz'], T0[:3, 3], atol=1e-12)
        np.testing.assert_allclose(np.asarray(req['frozen_target_T']).reshape(4, 4), T0, atol=1e-12)   # 位置＋旋轉
        for t, T in s.pause_rows:
            np.testing.assert_array_equal(T, T0)                              # PAUSE 期間目標保持
        self.assertEqual(pol.pause_outcome['state'], 'DONE')

    def test_status_fail_aborts(self):
        pol = PSim(status_state='FAIL').run()
        self.assertEqual(pol.abort, 'pause_failed:moved_after_stop')

    def test_handshake_failures(self):
        self.assertEqual(PSim(pause_exec_nack='not_in_free').run().abort, 'pause_switch:executor_nack:not_in_free')
        self.assertEqual(PSim(pause_solver=False).run().abort, 'pause_switch:pause_switch_timeout')

    def test_monitor_active_during_pause(self):
        def hook(sim, m):
            if sim.pol.phase == 'PAUSE':
                m.unexpected_max = 0.5
        pol = PSim(status_after=None, hook=hook).run()
        self.assertNotEqual(pol.phase, 'DONE')
        ab = next(e for e in pol.events if e.get('abort'))
        self.assertEqual(ab['phase'], 'PAUSE', ab)

    def test_total_timeout(self):
        pol = PSim(status_after=None).run()
        self.assertEqual(pol.abort, 'pause_total_timeout')


if __name__ == '__main__':
    unittest.main(verbosity=2)
