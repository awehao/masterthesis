#!/usr/bin/env python3
"""door_k2_policy 的離線反例（合成量測；TCP 理想跟隨上一輪目標）。不啟動模擬器、不用 ROS。

    python3 evaluation/test_door_k2_policy.py
"""
import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k2_policy as P                             # noqa: E402

DT = 0.05
R_G = np.array([[-1.0, 0, 0], [0, 0, 1.0], [0, 1.0, 0]])


def T_HG():
    T = np.eye(4)
    T[:3, :3] = R_G
    T[:3, 3] = [0.0, 0.0068, 0.0]
    return T


def H0():
    T = np.eye(4)
    T[:3, 3] = [0.8, -0.04, 0.85]
    return T


class Sim:
    """合成世界：TCP 跟隨上一輪目標；故障由 inject(t, meas) 注入。"""

    def __init__(self, inject=None, solver_up_at=0.5):
        self.cfg = P.K2Config()
        self.pol = P.K2Policy(self.cfg, T_HG())
        self.t = 0.0
        self.H = H0()
        self.tcp = H0() @ np.array([[1, 0, 0, 0], [0, 1, 0, -0.15], [0, 0, 1, 0], [0, 0, 0, 1.0]]) @ T_HG()
        self.owner = 'nav'
        self.inject = inject
        self.solver_up_at = solver_up_at
        self.fq = (0.0089, 0.0089)
        self.acts = []
        self.handover_req_t = None

    def meas(self):
        ph = self.pol.phase
        closed = self.pol.grip_cmd < 0.004
        fbar = (2.0, 2.0) if closed else (0.0, 0.0)
        m = P.Meas(t=self.t, owner=self.owner, unfold_done=self.t > 1.0, solver_up=self.t > self.solver_up_at,
                   solver_failed=None, solver_status_age=0.05, state_age=0.02, contacts_age=0.02, q_door=0.0, qdot_door=0.0,
                   H=self.H.copy(), T_tcp=self.tcp.copy(), finger_bar_min=fbar, finger_bar_max=fbar, unexpected_max=0.0,
                   unexpected_who=None, wall_max=0.0, wrist_max=1.0, contacts_invalid=0,
                   contact_evidence_ok=closed and ph in ('ENGAGE',), finger_q=self.fq, base_drift_m=0.001, base_drift_yaw=0.0)
        if self.inject:
            self.inject(self, m)
        return m

    def run(self, t_max=80.0):
        while self.t < t_max:
            m = self.meas()
            a = self.pol.step(m, DT)
            self.acts.append((self.t, self.pol.phase, a))
            if a['request_handover'] and self.handover_req_t is None:
                self.handover_req_t = self.t
            if self.handover_req_t is not None and self.t - self.handover_req_t > 0.2:
                self.owner = 'wholebody'
            if a['target'] is not None:
                self.tcp = a['target'].copy()
            self.fq = (a['grip'], a['grip'])
            if a['finish']:
                break
            self.t = round(self.t + DT, 6)
        return self.pol

    def phases(self):
        out = []
        for _, ph, _ in self.acts:
            if not out or out[-1] != ph:
                out.append(ph)
        return out


class Normal(unittest.TestCase):
    def test_full_sequence_done(self):
        s = Sim()
        pol = s.run()
        self.assertEqual(pol.phase, 'DONE', pol.events)
        self.assertEqual(s.phases(), ['WAIT_HANDOVER', 'UNFOLD', 'ALIGN_PRE', 'ALIGN_HOLD', 'APPROACH', 'ENGAGE', 'HOLD', 'RELEASE',
                                      'RETREAT', 'DONE'])
        self.assertTrue(any('checkpoint_s' in e for e in pol.events))
        self.assertIsNone(pol.abort)

    def test_approach_and_retreat_rate(self):
        s = Sim()
        s.run()
        tg = [(t, a['target']) for t, ph, a in s.acts if ph == 'APPROACH' and a['target'] is not None]
        for (t1, a1), (t2, a2) in zip(tg, tg[1:]):
            self.assertLessEqual(np.linalg.norm(a2[:3, 3] - a1[:3, 3]), 0.020 * DT + 1e-9)

    def test_hold_target_frozen_even_if_handle_moves(self):
        def inj(sim, m):
            if sim.pol.phase in ('HOLD', 'ENGAGE'):
                m.H = m.H.copy()
                m.H[1, 3] -= 0.001 * (sim.t % 1)               # 實測把手在動（< 門檻），目標不得跟
        s = Sim(inj)
        s.run()
        tg = [a['target'] for t, ph, a in s.acts if ph in ('HOLD',) and a['target'] is not None]
        self.assertTrue(len(tg) > 5)
        for x in tg:
            np.testing.assert_allclose(x, tg[0], atol=1e-12)


class Aborts(unittest.TestCase):
    def common(self, s, why_prefix, allow_retreat=False):
        pol = s.run()
        self.assertEqual(pol.phase, 'ABORTED')
        self.assertTrue(pol.abort.startswith(why_prefix), pol.abort)
        after = [a for t, ph, a in s.acts if ph == 'ABORTED']
        self.assertTrue(all(a['target'] is None for a in after))         # 中止後不發新目標
        self.assertTrue(all(a['stop_solver'] for a in after))
        self.assertTrue(after[-1]['finish'])
        self.assertGreaterEqual(s.acts[-1][0] - pol.t_abort, 3.0 - 1e-9)   # 觀察 3 s
        if not allow_retreat:
            self.assertNotIn('RETREAT', s.phases())
        return pol, after

    def test_monitor_active_during_unfold(self):
        def inj(sim, m):
            if sim.pol.phase == 'UNFOLD':
                m.q_door = math.radians(3.0)
        self.common(Sim(inj), 'door_moving')

    def test_door_moving_in_hold_no_release(self):
        def inj(sim, m):
            if sim.pol.phase == 'HOLD' and sim.t - sim.pol.t_phase > 1.0:
                m.qdot_door = 0.08
        s = Sim(inj)
        pol, after = self.common(s, 'door_moving')
        self.assertTrue(all(a['grip'] == 0.0 for a in after))           # 保持閉爪，不放爪
        self.assertNotIn('RELEASE', s.phases())

    def test_abort_during_close_freezes_grip(self):
        def inj(sim, m):
            if sim.pol.phase == 'ENGAGE' and sim.t - sim.pol.t_phase > 0.4:
                m.unexpected_max, m.unexpected_who = 0.5, 'uflite_finger1'   # 手指碰門板
        pol, after = self.common(Sim(inj), 'unexpected_contact')
        g = {a['grip'] for a in after}
        self.assertEqual(len(g), 1)
        self.assertTrue(0.0 < g.pop() < 0.0089)                          # 停在閉爪斜坡中途，不續閉、不放

    def test_finger_bar_before_engage(self):
        def inj(sim, m):
            if sim.pol.phase == 'APPROACH':
                m.finger_bar_max = (0.3, 0.0)
        self.common(Sim(inj), 'finger_bar_outside_allowed_phase')

    def test_finger_bar_after_release_confirmed(self):
        def inj(sim, m):
            if sim.pol.phase == 'RETREAT':
                m.finger_bar_max = (0.0, 0.2)
        self.common(Sim(inj), 'finger_bar_outside_allowed_phase', allow_retreat=True)

    def test_solver_failed_stops_targets(self):
        def inj(sim, m):
            if sim.pol.phase == 'APPROACH':
                m.solver_failed = 'qp_infeasible'
        self.common(Sim(inj), 'solver_failed')

    def test_solver_status_stale(self):
        def inj(sim, m):
            if sim.pol.phase == 'ALIGN_HOLD':
                m.solver_status_age = 2.0
        self.common(Sim(inj), 'solver_status_stale')

    def test_missing_contacts(self):
        def inj(sim, m):
            if sim.pol.phase == 'ALIGN_PRE':
                m.contacts_age = None
        self.common(Sim(inj), 'monitor_stale_or_missing')

    def test_nan_door_angle(self):
        def inj(sim, m):
            if sim.pol.phase == 'HOLD':
                m.q_door = float('nan')
        self.common(Sim(inj), 'monitor_nonfinite_or_invalid')

    def test_contacts_invalid_steps(self):
        def inj(sim, m):
            if sim.pol.phase == 'APPROACH':
                m.contacts_invalid = 1
        self.common(Sim(inj), 'monitor_nonfinite_or_invalid')

    def test_wall_contact(self):
        def inj(sim, m):
            if sim.pol.phase == 'UNFOLD':
                m.wall_max = 0.5
        self.common(Sim(inj), 'wall_contact')

    def test_wrist_force(self):
        def inj(sim, m):
            if sim.pol.phase == 'HOLD':
                m.wrist_max = 31.0
        self.common(Sim(inj), 'wrist_force')

    def test_base_drift(self):
        def inj(sim, m):
            if sim.pol.phase == 'APPROACH':
                m.base_drift_m = 0.025
        self.common(Sim(inj), 'base_drift')

    def test_grasp_drift(self):
        def inj(sim, m):
            if sim.pol.phase == 'HOLD':
                m.T_tcp = m.T_tcp.copy()
                m.T_tcp[2, 3] += 0.01
        self.common(Sim(inj), 'grasp_drift')

    def test_finger_contact_lost(self):
        def inj(sim, m):
            if sim.pol.phase == 'HOLD' and sim.t - sim.pol.t_phase > 1.0:
                m.finger_bar_min = (2.0, 0.1)
        self.common(Sim(inj), 'finger_contact_lost')

    def test_timeout_align_hold_without_solver(self):
        self.common(Sim(solver_up_at=1e9), 'timeout:ALIGN_HOLD')

    def test_release_needs_open_fingers(self):
        def inj(sim, m):
            if sim.pol.phase == 'RELEASE':
                m.finger_q = (0.0089, 0.003)                            # 一指卡住未張開
        self.common(Sim(inj), 'timeout:RELEASE')


if __name__ == '__main__':
    unittest.main(verbosity=2)
