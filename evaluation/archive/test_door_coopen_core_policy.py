#!/usr/bin/env python3
"""核心段：繞緣夾法（T_HG_full＋a_H）與門速 0.6 °/s 下的策略反例（合成世界沿用 test_door_k3a_policy.Sim；不啟動模擬器、不用 ROS）。
核心段任務節點（door_coopen_core_task_node.py）只改抓取基準、接近方向與門速；接近／夾持／開門／中止邏輯在 K2／K3A 層（本檔以 K3APolicy 驗），
K3M 層的走廊、幾何（geometry_*）與狀態過期（stale）中止與抓取基準無關，由既有 test_door_k3m_* 覆蓋（不另造）。
  E1 正常：UNFOLD → … → OPEN → SETTLE → END_HOLD → DONE，不進 RELEASE／RETREAT；ALIGN_HOLD 目標＝H·Trans(s_pre·a_H)·T_HG（沿 a_H 退開 s_pre；ALIGN_PRE 以限速斜坡逼近）；
     OPEN 中門角參考每輪增量 ≤ 0.6 °/s × dt；q_ref 終值＝48°。
  E2 OPEN 中（q_ref > 3°）單指 0.2 N ⇒ finger_contact_lost；E3 OPEN 中夾爪繞 z 偏 2.5° ⇒ grasp_drift；E4 OPEN 中門速 0.06 rad/s ⇒ door_rate。
  E5 回歸：K3APolicy 不帶 a_H 時 a_H＝(0, −1, 0)（正面夾法行為不變）。

    python3 evaluation/test_door_coopen_core_policy.py
"""
import json
import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3a_policy as P                            # noqa: E402
import test_door_k3a_policy as TK                      # noqa: E402

GB = os.path.join(HERE, 'results', 'door', 'coopen_core_grasp_basis_r4.json')
PRM = os.path.join(HERE, 'results', 'door', 'coopen_core_params_48_r4.json')
D2R = math.radians


def edge():
    a = json.load(open(GB))['args']
    return np.asarray(a['T_HG_full'], float), tuple(a['a_H'])


class EdgeSim(TK.Sim):
    def __init__(self, inject=None):
        super().__init__(inject)
        T_HG, a_H = edge()
        tp = json.load(open(PRM))['task_param_extension']
        self.cfg = P.K3AConfig(theta_goal_deg=float(tp['theta_goal_deg']), r_max_dps=float(tp['r_max_dps']))
        self.cfg.timeouts['OPEN'] = float(tp['open_timeout_s'])
        self.pol = P.K3APolicy(self.cfg, T_HG, a_H=a_H)
        off = np.eye(4)
        off[:3, 3] = 0.15 * np.asarray(a_H)
        self.tcp = TK.H0() @ off @ T_HG


class CoreEdge(unittest.TestCase):
    def test_e1_nominal(self):
        s = EdgeSim()
        pol = s.run(t_max=260.0)
        self.assertEqual(pol.phase, 'DONE', pol.events[-3:])
        ph = s.phases()
        for p in ('ALIGN_PRE', 'APPROACH', 'ENGAGE', 'HOLD', 'OPEN', 'SETTLE', 'END_HOLD', 'DONE'):
            self.assertIn(p, ph)
        self.assertNotIn('RELEASE', ph)
        self.assertNotIn('RETREAT', ph)
        T_HG, a_H = edge()
        hold = [i for i, (_, p, a, _) in enumerate(s.acts) if p == 'ALIGN_HOLD' and a['target'] is not None]
        _, _, a, m = s.acts[hold[len(hold) // 2]]                            # ALIGN_PRE 目標以限速斜坡逼近，ALIGN_HOLD 時即預抓取位姿
        off = np.eye(4)
        off[:3, 3] = s.cfg.s_pre * np.asarray(a_H)
        np.testing.assert_allclose(a['target'], m.H @ off @ T_HG, atol=1e-12)
        qs = [a['target'] is not None and pol.dtk is not None for (_, p, a, _) in s.acts if p == 'OPEN']
        self.assertTrue(any(qs))
        refs = [mm for (_, p, _, mm) in s.acts if p in ('OPEN', 'SETTLE')]
        q_series = [r.q_door for r in refs]
        steps = np.diff(q_series)
        self.assertLessEqual(float(np.max(steps)), D2R(0.6) * TK.DT + 1e-9)
        self.assertAlmostEqual(pol.q_ref, D2R(48.0), places=12)

    def test_e2_finger_loss(self):
        def inj(sim, m):
            if sim.pol.phase == 'OPEN' and sim.pol.q_ref is not None and sim.pol.q_ref > D2R(3.0):
                m.finger_bar_min = (0.2, 2.0)
        self.assertEqual(EdgeSim(inj).run(t_max=260.0).abort, 'finger_contact_lost')

    def test_e3_rotational_drift(self):
        def inj(sim, m):
            if sim.pol.phase == 'OPEN':
                R = np.eye(4)
                c, s_ = math.cos(D2R(2.5)), math.sin(D2R(2.5))
                R[:3, :3] = [[c, -s_, 0], [s_, c, 0], [0, 0, 1]]
                m.T_tcp = m.T_tcp @ R
        self.assertEqual(EdgeSim(inj).run(t_max=260.0).abort, 'grasp_drift')

    def test_e4_door_rate(self):
        def inj(sim, m):
            if sim.pol.phase == 'OPEN':
                m.qdot_door = 0.06
        self.assertEqual(EdgeSim(inj).run(t_max=260.0).abort, 'door_rate')

    def test_e5_front_default_unchanged(self):
        pol = P.K3APolicy(P.K3AConfig(), TK.T_HG())
        np.testing.assert_allclose(pol.a_H, [0.0, -1.0, 0.0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
