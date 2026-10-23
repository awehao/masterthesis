#!/usr/bin/env python3
"""door_k3m_policy 的離線反例（沿用 test_door_k3a_policy 的合成世界；執行端／求解器以 door_k3m_mode 的權威類別模擬，延遲可調）。

    python3 evaluation/test_door_k3m_policy.py
"""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_policy as KM                           # noqa: E402
import door_k3m_mode as MODE                           # noqa: E402
import door_k3m_corridor as CO                         # noqa: E402
import door_k3m_geom_monitor as GM                     # noqa: E402
import test_door_k3a_policy as TP                      # noqa: E402

RUN = 'door_k3m_test'
ANCHOR = (0.6, -0.53, 1.5708)
COR = CO.Corridor([{'q_door_deg': 0.0, 'fwd_m': 0.0, 'lat_m': 0.0, 'dyaw_deg': 0.0},
                   {'q_door_deg': 8.0, 'fwd_m': 0.0, 'lat_m': 0.0, 'dyaw_deg': 0.0}], ANCHOR)
NOM = {'nonfinger': 0.0072, 'finger': 0.035, 'base': 0.2, 'self': 0.0098}


class MSim(TP.Sim):
    """exec_delay／solver_delay：收到請求後幾輪才回 ack（None ⇒ 不回）；exec_nack：回 nack；hook(sim, m)：注入。"""

    def __init__(self, exec_delay=1, solver_delay=2, exec_nack=None, geom=None, hook=None, inject=None):
        super().__init__(inject)
        self.cfg = KM.K3MConfig()
        gfn = (lambda q, a, b: {k: (v, 'x') for k, v in (geom or NOM).items()})
        self.pol = KM.K3MPolicy(self.cfg, TP.T_HG(), RUN, COR, GM.GeometryMonitor(GM.GeomMonitorConfig(), gfn))
        self.E, self.S = MODE.ExecutorModeAuthority(RUN), MODE.SolverModeSwitch(RUN)
        self.exec_delay, self.solver_delay, self.exec_nack = exec_delay, solver_delay, exec_nack
        self.hook = hook
        self.req = None
        self.n_since = 0
        self.base_pose = ANCHOR
        self.rows = []
        self.wall = 100.0

    def meas(self):
        m = super().meas()
        m.state_step = int(round(self.t * 100))
        m.wall = self.wall
        m.base_pose = self.base_pose
        m.qf = np.zeros(9)
        m.q_finger = 0.00166
        m.executor_violation = None
        if self.req is not None:
            self.n_since += 1
            if self.exec_delay is not None and self.n_since == self.exec_delay:
                if self.exec_nack:
                    self.pol.on_executor_ack({'run_id': RUN, 'seq': 1, 'why': self.exec_nack}, self.t)
                else:
                    k, ack = self.E.on_request(self.req, m.state_step, self.t, aborted=False, owner_wholebody=True)
                    self.pol.on_executor_ack(ack, self.t)
                    self.S.on_executor_ack(ack, stopped=False, park_hold_active=True)
            if self.solver_delay is not None and self.n_since == self.solver_delay and self.S.pending is not None:
                cfg = type('C', (), {'base_hold': (0, 0, 0), 'base_fixed': False})()
                self.S.apply(cfg, shaping_enabled=False, snap_t=self.t)
                self.pol.on_solver_ack(self.S.note_first_solution(7, self.t, [0.0] * 9, [0.0] * 9, published=True, publish_sim_t=self.t), self.t)
        if self.hook:
            self.hook(self, m)
        return m

    def run(self, t_max=120.0):
        orig = self.pol.step

        def wrapped(m, dt):
            ph0 = self.pol.phase
            a = orig(m, dt)
            for r in self.pol.outbox:
                self.req = r
            self.pol.outbox.clear()
            self.rows.append((m.t, ph0, self.pol.phase, self.pol.mode_state, a['target'], self.pol.dtk is not None))
            self.wall += 0.05
            return a
        self.pol.step = wrapped
        return super().run(t_max)


class Policy(unittest.TestCase):
    def test_nominal_done_and_target_held_while_waiting(self):
        s = MSim(exec_delay=1, solver_delay=3)
        pol = s.run()
        self.assertEqual(pol.phase, 'DONE', pol.events[-3:])
        i = next(i for i, r in enumerate(s.rows) if r[1] == 'OPEN')
        T_hold = s.rows[i - 1][4]
        waiting = [r for r in s.rows if r[1] == 'OPEN' and r[3] == 'waiting']
        self.assertGreaterEqual(len(waiting), 2)
        for r in waiting:
            np.testing.assert_array_equal(r[4], T_hold)                         # 等待中保持最後目標
            self.assertFalse(r[5])                                              # 未建立 DoorTargetK3
        first_free = next(r for r in s.rows if r[1] == 'OPEN' and r[3] == 'free')
        np.testing.assert_allclose(first_free[4], T_hold, atol=1e-12)          # 首筆仍連續
        self.assertTrue(any('k3m_mode_confirmed' in e for e in pol.events))

    def test_timeout_without_solver_ack(self):
        pol = MSim(exec_delay=1, solver_delay=None).run()
        self.assertEqual(pol.abort, 'mode_switch:mode_switch_timeout')

    def test_executor_nack(self):
        pol = MSim(exec_nack='request_stale').run()
        self.assertEqual(pol.abort, 'mode_switch:executor_nack:request_stale')

    def test_wall_timeout_when_sim_stalls(self):
        def hook(sim, m):
            if sim.pol.mode_state == 'waiting':
                sim.wall += 1.0                                                  # 牆鐘照走、模擬時間幾乎不前進的情形
        pol = MSim(exec_delay=1, solver_delay=None, hook=hook).run()
        self.assertIn(pol.abort, ('mode_switch:mode_switch_wall_timeout', 'mode_switch:mode_switch_timeout'))

    def test_base_drift_replaced_only_after_exec_ack(self):
        def before(sim, m):
            if sim.pol.phase == 'HOLD':
                m.base_drift_m = 0.03
        self.assertEqual(MSim(hook=before).run().abort, 'base_drift')
        def after(sim, m):
            if sim.pol.exec_switched:
                m.base_drift_m = 0.03                                            # 相對錨點 30 mm：走廊內（參考表也在錨點）⇒ 但走廊判定
        s = MSim(hook=after)
        s.base_pose = ANCHOR
        pol = s.run()
        self.assertEqual(pol.phase, 'DONE', pol.events[-3:])                    # 20 mm 漂移檢查已被取代

    def test_corridor_violation_aborts(self):
        def hook(sim, m):
            if sim.pol.mode_state == 'free' and sim.pol.q_ref is not None and sim.pol.q_ref > 0.02:
                sim.base_pose = CO.se2_compose(ANCHOR, (0.021, 0.0, 0.0))
        pol = MSim(hook=hook).run()
        self.assertEqual(pol.abort, 'base_corridor')

    def test_geometry_stop_aborts_and_invalid_input_stops(self):
        pol = MSim(geom=dict(NOM, self=0.004)).run()
        self.assertEqual(pol.abort, 'geometry:geometry_clearance_self')
        def hook(sim, m):
            if sim.pol.exec_switched:
                m.qf = np.zeros(8)
        self.assertEqual(MSim(hook=hook).run().abort, 'geometry:geometry_input_invalid')

    def test_executor_violation(self):
        def hook(sim, m):
            if sim.pol.mode_state == 'free':
                m.executor_violation = {'why': 'corridor'}
        self.assertEqual(MSim(hook=hook).run().abort, 'executor_violation')

    def test_parent_safety_during_waiting(self):
        def inj(sim, m):
            if sim.pol.mode_state == 'waiting':
                m.wrist_max = 31.0
        s = MSim(exec_delay=None, solver_delay=None, inject=inj)
        self.assertEqual(s.run().abort, 'wrist_force')

    def test_run_id_required(self):
        with self.assertRaises(ValueError):
            KM.K3MPolicy(KM.K3MConfig(), TP.T_HG(), '', COR, None)


if __name__ == '__main__':
    unittest.main(verbosity=2)
