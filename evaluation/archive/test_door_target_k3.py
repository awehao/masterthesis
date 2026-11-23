#!/usr/bin/env python3
"""door_target_k3 的離線反例。    python3 evaluation/test_door_target_k3.py"""
import math
import os
import sys
import unittest

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_geom as G                                 # noqa: E402
import door_target_k3 as T3                           # noqa: E402

D = yaml.safe_load(open(os.path.join(os.path.dirname(HERE), 'src/my_omnibot_description/config/door_local_v2.yaml')))
CFG = T3.K3TargetConfig(hinge_point_world=(0, 0, 0), hinge_axis_world=(0, 0, -1), theta_goal_rad=math.radians(8.0))
D2R = math.radians


def tgt():
    H0 = G.handle_pose(D, 0.0, 0.85)
    T_HG = np.eye(4)
    T_HG[:3, :3] = [[-1, 0, 0], [0, 0, 1], [0, 1, 0]]
    T_HG[:3, 3] = [0, 0.0068, 0]
    return T3.DoorTargetK3(H0 @ T_HG, H0, CFG), T_HG


class K3(unittest.TestCase):
    def test_codex_reproduction_rejects_overshoot(self):
        """前一步參考 8°、實測緩增至 8.51°、目標 8° ⇒ 舊 DoorTarget 輸出 8.01°；本版必須衝突（不得 > 8°）。"""
        t, _ = tgt()
        t.q_ref = D2R(8.0)
        with self.assertRaises(T3.DoorTargetK3Conflict):
            t.step_ref(D2R(8.51), 0.05)

    def test_ref_never_exceeds_goal(self):
        t, _ = tgt()
        q, rng = 0.0, np.random.default_rng(0)
        for k in range(2000):
            try:
                r = t.step_ref(q, 0.05)
            except T3.DoorTargetK3Conflict:
                break
            self.assertLessEqual(r, D2R(8.0) + 1e-12)
            q = min(r + rng.normal(0, D2R(0.05)), D2R(8.4))           # 實測追參考（含雜訊、不過頭太多）

    def test_three_constraints_every_step(self):
        t, _ = tgt()
        q, prev = 0.0, None
        for k in range(1500):
            r = t.step_ref(q, 0.05)
            self.assertLessEqual(abs(r - q), D2R(0.5) + 1e-12)                       # 超前
            if prev is not None:
                self.assertLessEqual(abs(r - prev), CFG.rate(prev) * 0.05 + 1e-12)   # 限速（依前一步 q_ref）
            self.assertLessEqual(r, D2R(8.0) + 1e-12)                                # 範圍
            prev = r
            q = r                                                                     # 理想跟隨

    def test_decel_law(self):
        self.assertAlmostEqual(CFG.rate(0.0), D2R(1.0))
        self.assertAlmostEqual(CFG.rate(D2R(7.0)), D2R(0.25) + (D2R(1.0) - D2R(0.25)) * 0.5)
        self.assertAlmostEqual(CFG.rate(D2R(8.0)), D2R(0.25))

    def test_reaches_goal_without_jump_then_holds(self):
        t, _ = tgt()
        q, prev = 0.0, 0.0
        for k in range(3000):
            r = t.step_ref(q, 0.05)
            self.assertLessEqual(r - prev, D2R(1.0) * 0.05 + 1e-12)
            prev, q = r, r
            if t.reached:
                break
        self.assertTrue(t.reached)
        for _ in range(20):
            self.assertAlmostEqual(t.step_ref(q, 0.05), D2R(8.0), places=12)          # 保持

    def test_lagging_door_limits_ref(self):
        t, _ = tgt()
        for _ in range(200):
            t.step_ref(0.0, 0.05)                                                      # 門卡住不動
        self.assertLessEqual(t.q_ref, D2R(0.5) + 1e-12)

    def test_door_pushed_back_conflicts(self):
        t, _ = tgt()
        t.q_ref = D2R(5.0)
        with self.assertRaises(T3.DoorTargetK3Conflict):
            t.step_ref(D2R(4.0), 0.05)                                                 # 門被推回 1°

    def test_target_keeps_grasp_relation(self):
        t, T_HG = tgt()
        q = 0.0
        for _ in range(100):
            H = G.handle_pose(D, q, 0.85)
            Tg, d = t.step(H, q, 0.05)
            np.testing.assert_allclose(Tg, G.handle_pose(D, d['q_ref'], 0.85) @ T_HG, atol=1e-12)
            q = d['q_ref']

    # ---- Codex 20261021_181050 第 3 項：OPEN 銜接
    def test_first_output_equals_last_published_and_no_advance(self):
        H0 = G.handle_pose(D, 0.0, 0.85)
        T_HG = np.eye(4)
        T_HG[:3, :3] = [[-1, 0, 0], [0, 0, 1], [0, 1, 0]]
        T_HG[:3, 3] = [0, 0.0068, 0]
        T_last = H0 @ T_HG
        T_last[:3, 3] += (0.0031, -0.0012, 0.0024)                  # 上一筆 HOLD 目標與實測抓取差數 mm
        t = T3.DoorTargetK3(T_last, H0, CFG, G_T_H_grasp=np.eye(4))
        Tg, d = t.step(H0, 0.0, 0.05)                                 # dt>0 也不推進
        self.assertEqual(d['q_ref'], 0.0)
        np.testing.assert_allclose(Tg, T_last, atol=1e-12)
        np.testing.assert_allclose(t.G_T_H_grasp, np.eye(4))
        Tg2, d2 = t.step(H0, 0.0, 0.05)                               # 第二步才推進，且相鄰目標位移受限速約束
        self.assertGreater(d2['q_ref'], 0.0)
        r = np.linalg.norm(T_last[:3, 3] - np.array([0, 0, T_last[2, 3]]))
        self.assertLessEqual(np.linalg.norm(Tg2[:3, 3] - Tg[:3, 3]), CFG.r_max_rps * 0.05 * r + 1e-9)

    def test_start_outside_range_conflicts(self):
        t, _ = tgt()
        with self.assertRaises(T3.DoorTargetK3Conflict):
            t.step_ref(D2R(-1.2), 0.05)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            T3.K3TargetConfig((0, 0, 0), (0, 0, -1), D2R(8), r_min_rps=0.0).validate()
        t, _ = tgt()
        with self.assertRaises(ValueError):
            t.step_ref(float('nan'), 0.05)


if __name__ == '__main__':
    unittest.main(verbosity=2)
