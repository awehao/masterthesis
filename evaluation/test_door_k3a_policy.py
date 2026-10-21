#!/usr/bin/env python3
"""door_k3a_policy 的離線反例（合成量測；夾持後門理想跟隨參考角，TCP 理想跟隨上一輪目標）。不啟動模擬器、不用 ROS。

    python3 evaluation/test_door_k3a_policy.py
"""
import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3a_policy as P                            # noqa: E402
from door_target import rotate_about_hinge            # noqa: E402

DT = 0.05
R_G = np.array([[-1.0, 0, 0], [0, 0, 1.0], [0, 1.0, 0]])
AX = np.array([0.0, 0.0, -1.0])
D2R = math.radians


def T_HG():
    T = np.eye(4)
    T[:3, :3] = R_G
    T[:3, 3] = [0.0, 0.0068, 0.0]
    return T


def H0():
    T = np.eye(4)
    T[:3, 3] = [0.8, -0.04, 0.85]
    return T


def H_of(q):
    return rotate_about_hinge(H0(), np.zeros(3), AX, q)


class Sim:
    """合成世界。夾持前門 0°；夾持後門角＝上一輪參考角（door_follow），TCP＝上一輪目標。故障由 inject(sim, meas) 注入。"""

    def __init__(self, inject=None, tcp_offset=(0.0, 0.0, 0.0)):
        self.cfg = P.K3AConfig()
        self.pol = P.K3APolicy(self.cfg, T_HG())
        self.t = 0.0
        self.q = 0.0
        self.q_prev = 0.0
        self.tcp = H0() @ np.array([[1, 0, 0, 0], [0, 1, 0, -0.15], [0, 0, 1, 0], [0, 0, 0, 1.0]]) @ T_HG()
        self.tcp_offset = np.array(tcp_offset)
        self.owner = 'nav'
        self.inject = inject
        self.fq = (0.0089, 0.0089)
        self.acts = []
        self.handover_req_t = None
        self.override_tcp = None

    def meas(self):
        ph = self.pol.phase
        closed = self.pol.grip_cmd < 0.004
        fbar = (2.0, 2.0) if closed else (0.0, 0.0)
        qdot = (self.q - self.q_prev) / DT
        m = P.Meas(t=self.t, owner=self.owner, unfold_done=self.t > 1.0, solver_up=self.t > 0.5,
                   solver_failed=None, solver_status_age=0.05, state_age=0.02, contacts_age=0.02, q_door=self.q, qdot_door=qdot,
                   H=H_of(self.q), T_tcp=self.tcp.copy(), finger_bar_min=fbar, finger_bar_max=fbar, unexpected_max=0.0,
                   unexpected_who=None, wall_max=0.0, wrist_max=1.0, contacts_invalid=0,
                   contact_evidence_ok=closed and ph == 'ENGAGE', finger_q=self.fq, base_drift_m=0.001, base_drift_yaw=0.0)
        if self.inject:
            self.inject(self, m)
        return m

    def run(self, t_max=120.0):
        while self.t < t_max:
            m = self.meas()
            a = self.pol.step(m, DT)
            self.acts.append((self.t, self.pol.phase, a, m))
            if a['request_handover'] and self.handover_req_t is None:
                self.handover_req_t = self.t
            if self.handover_req_t is not None and self.t - self.handover_req_t > 0.2:
                self.owner = 'wholebody'
            if a['target'] is not None:
                self.tcp = a['target'].copy()
                if self.pol.phase in ('ENGAGE', 'HOLD', 'OPEN', 'SETTLE', 'END_HOLD'):
                    self.tcp[:3, 3] += self.tcp_offset                       # 閉爪起才出現的追蹤偏差（夾持成立時已量入 G_T_H_grasp）
            self.q_prev = self.q
            if self.pol.q_ref is not None and self.pol.phase in ('OPEN', 'SETTLE', 'END_HOLD'):
                self.q = self.pol.q_ref                      # 門理想跟隨
            self.fq = (a['grip'], a['grip'])
            if a['finish']:
                break
            self.t = round(self.t + DT, 6)
        return self.pol

    def phases(self):
        out = []
        for _, ph, _, _ in self.acts:
            if not out or out[-1] != ph:
                out.append(ph)
        return out


class K3A(unittest.TestCase):
    def test_nominal_done_without_release(self):
        s = Sim()
        pol = s.run()
        self.assertEqual(pol.phase, 'DONE', pol.events[-3:])
        ph = s.phases()
        for p in ('ENGAGE', 'HOLD', 'OPEN', 'SETTLE', 'END_HOLD', 'DONE'):
            self.assertIn(p, ph)
        self.assertNotIn('RELEASE', ph)
        self.assertNotIn('RETREAT', ph)
        i_hold = ph.index('HOLD')
        for t, p, a, _ in s.acts:
            if p in ph[i_hold:]:
                self.assertLessEqual(a['grip'], 1e-12)                       # 夾持後全程閉爪命令
        self.assertLessEqual(max(pol.dtk.last.get('q_ref', 0.0), pol.q_ref), D2R(8.0) + 1e-12)
        self.assertAlmostEqual(pol.q_ref, D2R(8.0), places=12)

    def test_first_open_target_equals_last_hold_target(self):
        s = Sim(tcp_offset=(0.0012, -0.0008, 0.0015))                       # 實測 TCP 與目標差數 mm
        s.run()
        k = next(i for i, (_, p, _, _) in enumerate(s.acts) if p == 'OPEN')
        T_hold = s.acts[k - 1][2]['target']                                  # 轉入 OPEN 那輪仍是 HOLD 目標
        T_open = s.acts[k + 1][2]['target']                                  # OPEN 第一輪
        np.testing.assert_allclose(T_open, T_hold, atol=1e-12)
        np.testing.assert_allclose(s.pol.open_entry['T_last'], T_hold, atol=1e-12)

    def test_grasp_relation_not_overwritten(self):
        s = Sim(tcp_offset=(0.0012, -0.0008, 0.0015))
        s.run(t_max=30.0)
        G0 = s.pol.G_T_H.copy()
        s.run()
        np.testing.assert_allclose(s.pol.G_T_H, G0, atol=0)
        self.assertGreater(np.abs(s.pol.dtk.G_T_H - G0).max(), 1e-4)        # G_T_H_open 另存，與 grasp 不同

    def test_reference_rate_and_goal_every_step(self):
        s = Sim()
        s.run()
        rows = [x for x in s.acts if x[1] in ('OPEN', 'SETTLE')]
        self.assertTrue(rows)
        # 由已發布目標反推 q_ref：目標＝繞鉸鏈轉 q_ref 的 T_last
        T0 = np.asarray(s.pol.open_entry['T_last'])
        p0 = T0[:3, 3]
        qs = []
        for _, _, a, _ in rows:
            if a['target'] is None:
                continue
            p = a['target'][:3, 3]
            ang = math.atan2(p0[0] * p[1] - p0[1] * p[0], p0[0] * p[0] + p0[1] * p[1])
            qs.append(-ang)                                                  # 繞 −z
        for a_, b_ in zip(qs, qs[1:]):
            self.assertLessEqual(abs(b_ - a_), s.cfg.target_cfg().rate(a_) * DT + 1e-9)
        self.assertLessEqual(max(qs), D2R(8.0) + 1e-9)

    def test_door_over_goal_aborts_without_release(self):
        def inj(sim, m):
            if sim.pol.phase == 'SETTLE':
                m.q_door = D2R(9.2)
                m.qdot_door = 0.0
        s = Sim(inj)
        pol = s.run()
        self.assertEqual(pol.abort, 'door_angle_out_of_bounds')
        self.assertTrue(all(a['grip'] <= 1e-12 for t, p, a, _ in s.acts if t >= pol.t_abort))
        self.assertTrue(all(a['target'] is None for t, p, a, _ in s.acts if t >= pol.t_abort))

    def test_codex_counterexample_conflict(self):
        """門被外力慢慢推過目標（TCP 跟著實測門走 ⇒ 無漂移）：q_meas 8.51°、q_ref 8° ⇒ 參考衝突中止（不輸出 > 8°）。"""
        def inj(sim, m):
            if sim.pol.phase == 'SETTLE':
                sim.extra = getattr(sim, 'extra', 0.0) + 0.0008
                q = D2R(8.0) + sim.extra
                m.q_door, m.qdot_door = q, 0.0008 / DT
                m.H = H_of(q)
                m.T_tcp = H_of(q) @ np.linalg.inv(sim.pol.G_T_H)
        s = Sim(inj)
        pol = s.run()
        self.assertEqual(pol.abort, 'door_target_conflict', pol.events[-2:])
        self.assertLessEqual(pol.q_ref, D2R(8.0) + 1e-12)

    def test_door_stuck_drift_abort(self):
        def inj(sim, m):
            if sim.pol.phase in ('OPEN', 'SETTLE'):
                m.q_door, m.qdot_door, m.H = 0.0, 0.0, H_of(0.0)
        s = Sim(inj)
        pol = s.run()
        self.assertIn(pol.abort, ('grasp_drift', 'door_target_conflict'))
        self.assertLessEqual(pol.q_ref, D2R(0.5) + 1e-12)

    def test_settle_never_quiet_times_out(self):
        def inj(sim, m):
            if sim.pol.phase == 'SETTLE':
                m.qdot_door = 0.003
        s = Sim(inj)
        pol = s.run()
        self.assertEqual(pol.abort, 'timeout:SETTLE')
        self.assertNotIn('END_HOLD', s.phases())

    def test_settle_needs_continuous_1s(self):
        def inj(sim, m):
            if sim.pol.phase == 'SETTLE':
                k = getattr(sim, 'k', 0)
                sim.k = k + 1
                if k % 15 == 14:                                             # 每 0.75 s 一步不靜止
                    m.qdot_door = 0.003
        s = Sim(inj)
        pol = s.run()
        self.assertEqual(pol.abort, 'timeout:SETTLE')

    def test_finger_loss_during_open(self):
        def inj(sim, m):
            if sim.pol.phase == 'OPEN' and sim.pol.q_ref is not None and sim.pol.q_ref > D2R(3.0):
                m.finger_bar_min = (0.2, 2.0)
        s = Sim(inj)
        pol = s.run()
        self.assertEqual(pol.abort, 'finger_contact_lost')

    def test_short_finger_dip_tolerated(self):
        def inj(sim, m):
            if sim.pol.phase == 'OPEN' and 3.0 < m.t - sim.pol.t_phase < 3.12:
                m.finger_bar_min = (0.2, 2.0)
        s = Sim(inj)
        self.assertEqual(s.run().phase, 'DONE')

    def test_rotational_drift_abort(self):
        def inj(sim, m):
            if sim.pol.phase == 'OPEN':
                R = np.eye(4)
                c, s_ = math.cos(D2R(2.5)), math.sin(D2R(2.5))
                R[:3, :3] = [[c, -s_, 0], [s_, c, 0], [0, 0, 1]]
                m.T_tcp = m.T_tcp @ R
        s = Sim(inj)
        self.assertEqual(s.run().abort, 'grasp_drift')

    def test_negative_start_angle_ok(self):
        def inj(sim, m):
            if sim.pol.phase in ('HOLD',) or (sim.pol.phase == 'OPEN' and sim.pol.dtk is None):
                m.q_door, m.H = D2R(-0.05), H_of(D2R(-0.05))
        s = Sim(inj)
        self.assertEqual(s.run().phase, 'DONE')

    def test_door_rate_abort(self):
        def inj(sim, m):
            if sim.pol.phase == 'OPEN':
                m.qdot_door = 0.06
        self.assertEqual(Sim(inj).run().abort, 'door_rate')

    def test_pre_open_door_bound_still_2deg(self):
        def inj(sim, m):
            if sim.pol.phase == 'HOLD':
                m.q_door = D2R(2.1)
        self.assertEqual(Sim(inj).run().abort, 'door_moving')

    def test_finger_bar_allowed_in_opening_phases(self):
        s = Sim()
        pol = s.run()
        self.assertIsNone(pol.abort)

    def test_prim_pose_noise_not_in_target(self):
        """實測把手 prim 位姿帶 ±0.3 mm 雜訊：目標仍恰為上一筆已發布目標繞鉸鏈轉 (q_ref − q_entry)。"""
        rng = np.random.default_rng(1)

        def inj(sim, m):
            if sim.pol.phase in ('OPEN', 'SETTLE', 'END_HOLD'):
                m.H = m.H.copy()
                m.H[:3, 3] += rng.uniform(-0.0003, 0.0003, 3)
        s = Sim(inj)
        pol = s.run()
        self.assertEqual(pol.phase, 'DONE', pol.events[-2:])
        T0, q0 = np.asarray(pol.open_entry['T_last']), pol.open_entry['q_meas']
        n = 0
        for k, (_, p, a, m) in enumerate(s.acts):
            if p in ('OPEN', 'SETTLE') and a['target'] is not None and k > 0 and s.acts[k - 1][1] in ('OPEN', 'SETTLE'):
                n += 1
        self.assertGreater(n, 100)
        # 由最後一筆反推：目標與剛體旋轉一致
        Tf = s.acts[[i for i, x in enumerate(s.acts) if x[1] == 'SETTLE'][-2]][2]['target']
        np.testing.assert_allclose(Tf, rotate_about_hinge(T0, np.zeros(3), AX, pol.q_ref - q0), atol=1e-12)

    def test_open_timeout(self):
        cfg = P.K3AConfig()
        cfg.timeouts['OPEN'] = 2.0

        def inj(sim, m):
            pass
        s = Sim(inj)
        s.cfg = cfg
        s.pol = P.K3APolicy(cfg, T_HG())
        self.assertEqual(s.run().abort, 'timeout:OPEN')


if __name__ == '__main__':
    unittest.main(verbosity=2)
