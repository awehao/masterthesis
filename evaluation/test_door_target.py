#!/usr/bin/env python3
"""door_geom／door_target 的離線單元測試與反例（不依賴模擬器）。

    python3 evaluation/test_door_target.py
"""
import math
import os
import sys
import unittest

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import door_geom as G                                  # noqa: E402
import door_target as DT                               # noqa: E402
import object_target_geometry as OT                    # noqa: E402

D = yaml.safe_load(open(os.path.join(WS, 'src/my_omnibot_description/config/door_local_v0.yaml')))
A_OPEN = (0.0, 0.0, -1.0)          # door_local 約定：門扇繞 +z 轉 −q ⇒ 開門軸 = −z
C0 = (0.0, 0.0, 0.0)


def T_HG_baseline():
    """實錄基準（dl2_obs_to_target_v2.baseline_grasp 同式）：R_grasp 用設計 R_DES0 近似、grasp_depth 0.0068。"""
    T = np.eye(4)
    T[:3, :3] = np.array([[-1.0, 0, 0], [0, 0, 1.0], [0, 1.0, 0]])
    T[:3, 3] = [0.0, 0.0068, 0.0]
    return T


class HingedPose(unittest.TestCase):
    def setUp(self):
        self.T0 = G.handle_pose(D, 0.0, 0.85)

    def pose(self, q):
        r = DT.hinged_handle_pose(self.T0, C0, A_OPEN, q, 0.0, math.radians(110))
        self.assertTrue(r['ok'], r)
        return r['T_WH']

    def test_zero_is_identity(self):
        np.testing.assert_allclose(self.pose(0.0), self.T0, atol=1e-12)

    def test_matches_door_geom_at_all_angles(self):
        for qd in (5, 30, 60, 80, 100):
            np.testing.assert_allclose(self.pose(math.radians(qd)), G.handle_pose(D, math.radians(qd), 0.85), atol=1e-12)

    def test_radius_height_preserved(self):
        r0 = np.linalg.norm(self.T0[:2, 3])
        for qd in np.linspace(0, 110, 23):
            T = self.pose(math.radians(qd))
            self.assertAlmostEqual(np.linalg.norm(T[:2, 3]), r0, places=12)
            self.assertAlmostEqual(T[2, 3], 0.85, places=12)

    def test_orientation_follows_leaf(self):
        for qd in (20, 90):
            q = math.radians(qd)
            T = self.pose(q)
            np.testing.assert_allclose(T[:3, 0], [math.cos(q), -math.sin(q), 0], atol=1e-12)   # 沿門扇
            np.testing.assert_allclose(T[:3, 1], [math.sin(q), math.cos(q), 0], atol=1e-12)    # 指向門扇內側

    def test_opening_direction_toward_robot(self):
        self.assertLess(self.pose(math.radians(10))[1, 3], self.T0[1, 3])   # 拉向機器人（−y）

    def test_composition(self):
        q1, q2 = 0.3, 0.45
        T12 = DT.rotate_about_hinge(self.pose(q1), np.zeros(3), np.array(A_OPEN), q2)
        np.testing.assert_allclose(T12, self.pose(q1 + q2), atol=1e-12)

    def test_rejects(self):
        self.assertFalse(DT.hinged_handle_pose(self.T0, C0, A_OPEN, math.radians(120), 0.0, math.radians(110))['ok'])
        self.assertFalse(DT.hinged_handle_pose(self.T0, C0, (0, 0, 0), 0.1, 0.0, 1.0)['ok'])
        self.assertFalse(DT.hinged_handle_pose(self.T0, C0, A_OPEN, 0.1, 1.0, 0.0)['ok'])
        bad = self.T0.copy()
        bad[:3, :3] *= 1.01
        self.assertFalse(DT.hinged_handle_pose(bad, C0, A_OPEN, 0.1, 0.0, 1.0)['ok'])
        self.assertFalse(DT.hinged_handle_pose(self.T0, (0, float('nan'), 0), A_OPEN, 0.1, 0.0, 1.0)['ok'])

    def test_object_target_consistency(self):
        T_HG = T_HG_baseline()
        for qd in (0, 10, 45):
            T_WH = self.pose(math.radians(qd))
            r = OT.object_target(T_WH, T_HG, np.array([0.0, -1.0, 0.0]), 0.0)
            self.assertTrue(r['ok'])
            np.testing.assert_allclose(r['T_WE'], T_WH @ T_HG, atol=1e-12)
            # TCP 在桿心朝門扇內側 6.8 mm（不是 14.7 mm 舊值）
            self.assertAlmostEqual(float((r['T_WE'][:3, 3] - T_WH[:3, 3]) @ T_WH[:3, 1]), 0.0068, places=12)


class DoorTargetTest(unittest.TestCase):
    def setUp(self):
        self.cfg = DT.DoorTargetConfig(hinge_point_world=C0, hinge_axis_world=A_OPEN, q_min_rad=0.0, q_max_rad=math.radians(110),
                                       lead_max_rad=0.0125, rate_max_rps=0.044)
        self.T_HG = T_HG_baseline()
        self.H0 = G.handle_pose(D, 0.0, 0.85)
        self.tg = DT.DoorTarget(self.H0 @ self.T_HG, self.H0, self.cfg)

    def test_first_step_from_measurement(self):
        Hm = G.handle_pose(D, 0.2, 0.85)
        tg = DT.DoorTarget(self.H0 @ self.T_HG, self.H0, self.cfg)
        _, d = tg.step(Hm, 0.2, 0.2, 0.05)
        self.assertAlmostEqual(d['q_ref'], 0.2)

    def test_target_keeps_grasp_relation(self):
        for k in range(20):
            q = 0.01 * k
            Tg, d = self.tg.step(G.handle_pose(D, q, 0.85), q, math.radians(90), 0.05)
            np.testing.assert_allclose(Tg, G.handle_pose(D, d['q_ref'], 0.85) @ self.T_HG, atol=1e-12)

    def test_bounded_lead_when_stuck(self):
        for _ in range(200):
            _, d = self.tg.step(self.H0, 0.0, math.radians(90), 0.05)       # 門卡住：實測不動
        self.assertLessEqual(d['q_ref'], 0.0125 + 1e-12)
        self.assertTrue(d['lead_clamped'])

    def test_bounded_lead_both_directions(self):
        q = 0.5
        tg = DT.DoorTarget(self.H0 @ self.T_HG, self.H0, self.cfg)
        tg.step(G.handle_pose(D, q, 0.85), q, q, 0.05)
        _, d = tg.step(G.handle_pose(D, q, 0.85), q, 0.0, 10.0)            # 大 dt：限速放寬，仍受超前限制
        self.assertGreaterEqual(d['q_ref'], q - 0.0125 - 1e-12)

    def test_rate_limit(self):
        _, d = self.tg.step(self.H0, 0.0, 1.0, 0.05)
        _, d = self.tg.step(G.handle_pose(D, 0.002, 0.85), 0.002, 1.0, 0.05)
        self.assertLessEqual(d['q_ref'], 2 * 0.044 * 0.05 + 1e-12)
        self.assertTrue(d['rate_limited'])

    def test_range_clamp(self):
        qm = math.radians(109.9)
        tg = DT.DoorTarget(self.H0 @ self.T_HG, self.H0, self.cfg)
        tg.step(G.handle_pose(D, qm, 0.85), qm, math.radians(200), 0.05)
        _, d = tg.step(G.handle_pose(D, qm, 0.85), qm, math.radians(200), 1.0)
        self.assertLessEqual(d['q_ref'], math.radians(110) + 1e-12)

    def test_drift(self):
        Hm = G.handle_pose(D, 0.3, 0.85)
        self.assertAlmostEqual(self.tg.drift_vs(Hm @ self.T_HG, Hm), 0.0, places=12)
        slipped = Hm @ self.T_HG
        slipped[:3, 3] += [0.0, 0.0, 0.01]
        self.assertAlmostEqual(self.tg.drift_vs(slipped, Hm), 0.01, places=9)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            self.tg.step(self.H0, float('nan'), 0.1, 0.05)
        with self.assertRaises(ValueError):
            self.tg.step(self.H0, 0.0, 0.1, 0.0)
        with self.assertRaises(ValueError):
            DT.DoorTargetConfig(C0, A_OPEN, 0.0, 1.0, 0.0, 0.1).validate()
        with self.assertRaises(ValueError):
            DT.DoorTargetConfig(C0, (0, 0, 0), 0.0, 1.0, 0.01, 0.1).validate()


class DistanceCounterexamples(unittest.TestCase):
    """有向方盒／橫桿距離：與暴力表面取樣對照、內外號、旋轉後的門扇。"""

    def brute_box(self, P, c, R, hs, n=60):
        g = np.linspace(-1, 1, n)
        S = []
        for ax in range(3):
            for sgn in (-1, 1):
                a, b = [k for k in range(3) if k != ax]
                U, V = np.meshgrid(g, g)
                pts = np.zeros((U.size, 3))
                pts[:, ax] = sgn
                pts[:, a] = U.ravel()
                pts[:, b] = V.ravel()
                S.append(pts * hs)
        S = np.vstack(S) @ R.T + c
        return np.array([np.linalg.norm(S - p, axis=1).min() for p in P])

    def test_rotated_leaf_vs_bruteforce(self):
        rng = np.random.default_rng(0)
        for qd in (0, 37, 80):
            name, c, R, hs = G.shapes(D, math.radians(qd), 0.85)[0]
            P = c + rng.uniform(-1.2, 1.2, (40, 3)) * np.array([1.0, 1.0, 0.3])
            d = G.box_dist(P, c, R, hs)
            outside = d > 0
            bf = self.brute_box(P[outside], c, R, hs)
            np.testing.assert_allclose(d[outside], bf, atol=0.02 * max(hs))   # 取樣解析度內
            self.assertTrue((G.box_dist(c[None, :], c, R, hs) < 0).all())  # 中心在內為負

    def test_face_zero_and_sign(self):
        name, c, R, hs = G.shapes(D, 0.6, 0.85)[0]
        face = c + R[:, 1] * hs[1]
        self.assertAlmostEqual(float(G.box_dist(face[None, :], c, R, hs)[0]), 0.0, places=12)
        self.assertGreater(float(G.box_dist((face + R[:, 1] * 0.01)[None, :], c, R, hs)[0]), 0.0)
        self.assertLess(float(G.box_dist((face - R[:, 1] * 0.01)[None, :], c, R, hs)[0]), 0.0)

    def test_axis_aligned_wrong_would_fail(self):
        """反例：若誤把旋轉 80° 的門扇當軸對齊方盒，距離錯很大——確認本實作有轉座標。"""
        name, c, R, hs = G.shapes(D, math.radians(80), 0.85)[0]
        p = c + R[:, 0] * (hs[0] - 0.05)                    # 門扇內部（沿 u 接近端點）
        self.assertLess(float(G.box_dist(p[None, :], c, R, hs)[0]), 0.0)
        self.assertGreater(float(G.box_dist(p[None, :], c, np.eye(3), hs)[0]), 0.05)

    def test_bar_distance(self):
        T = G.handle_pose(D, 0.4, 0.85)
        c, u, y = T[:3, 3], T[:3, 0], T[:3, 1]
        r = D['handle']['bar']['radius']
        self.assertAlmostEqual(float(G.bar_dist(D, (c + y * r)[None, :], 0.4, 0.85)[0]), 0.0, places=12)
        self.assertAlmostEqual(float(G.bar_dist(D, (c + y * 0.05)[None, :], 0.4, 0.85)[0]), 0.05 - r, places=12)
        end = c + u * (D['handle']['bar']['length'] / 2 + 0.03)
        self.assertAlmostEqual(float(G.bar_dist(D, end[None, :], 0.4, 0.85)[0]), 0.03 - r, places=12)

    def test_disc_clear(self):
        name, c, R, hs = G.shapes(D, 0.0, 0.85)[0]
        y_face = c[1] - hs[1]                               # 門扇機器人側面（q = 0：y = 0）
        self.assertAlmostEqual(G.disc_clear(0.5, y_face - 0.40, c[2], 0.30, c, R, hs), 0.10, places=12)

    def test_pass_angle_projection(self):
        qd, need = G.pass_angle(D, 0.341, 0.05)
        q = math.radians(qd)
        self.assertLessEqual(0.95 * math.cos(q) + 0.04 * math.sin(q), need + 1e-12)
        q2 = math.radians(qd - 0.5)
        self.assertGreater(0.95 * math.cos(q2) + 0.04 * math.sin(q2), need)


if __name__ == '__main__':
    unittest.main(verbosity=2)
