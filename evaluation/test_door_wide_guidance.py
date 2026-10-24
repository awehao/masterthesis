#!/usr/bin/env python3
"""door_wide_guidance（W1b 底盤路徑引導）離線反例：
  G1 CorridorRef 的參考位姿＝走廊監看 Corridor.ref_pose（同一表、同一函式）；
  G2 世界→本體座標轉換（偏航 90° 時世界 +x 誤差 ⇒ 本體 −vy）、各軸限幅；
  G3 相位開關：OPEN／SETTLE 且 free ⇒ w_vref＝1；OPEN 但尚未 free、PAUSE、ABORTED、DONE ⇒ 關閉（w 0、參考 NaN）；
  G4 非有限輸入 ⇒ 關閉並回報錯誤，不拋例外；
  G5 輸出能被求解器的 parse_coord（版面 v1）解析：開啟時帶本體 v_ref，關閉時 base_vref 為 None；
  G6 簡化積分（門角固定、底盤照 v_ref 運動）：初始偏差 (15 mm, −10 mm, 0.7°) 在 5 s 內收斂到 < 1 mm／0.02°。

    python3 evaluation/test_door_wide_guidance.py      # 需 ROS 環境（G5 匯入求解節點）
"""
import importlib.util
import json
import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_corridor as COR                        # noqa: E402
import door_wide_guidance as G                         # noqa: E402

TABLE = os.path.join(HERE, 'results', 'door', 'door_wide_corridor_table_90.json')


def corr():
    return COR.Corridor.from_json(json.load(open(TABLE))['table'])


class T(unittest.TestCase):
    def test_g1_same_reference(self):
        c = corr()
        r = G.CorridorRef(c)
        for qd in (0.0, 8.0, 17.3, 30.0, 45.0):
            for u, v in zip(r.pose(math.radians(qd)), c.ref_pose(qd)):     # 度↔弧度互換的捨入 ≤ 1e-12
                self.assertAlmostEqual(u, v, places=12)

    def test_g2_transform_and_clip(self):
        class Fixed:
            def pose(self, q):
                return (1.0, 0.0, math.pi / 2)

            def dpose_dq(self, q):
                return (0.0, 0.0, 0.0)
        cfg = G.GuidanceCfg(k_lin=1.0, k_yaw=1.0, v_cap=0.02, w_cap=0.05)
        coord, info = G.guidance(Fixed(), cfg, (1.0 - 0.01, 0.0, math.pi / 2), 0.1, 0.0)   # 世界 +x 誤差 10 mm
        self.assertAlmostEqual(coord[2], 0.0, places=9)
        self.assertAlmostEqual(coord[3], -0.01, places=9)                               # 本體 −vy
        coord, _ = G.guidance(Fixed(), cfg, (0.0, 0.0, math.pi / 2 - 0.5), 0.1, 0.0)    # 大誤差 ⇒ 限幅
        self.assertLessEqual(abs(coord[2]), 0.02 + 1e-12)
        self.assertLessEqual(abs(coord[3]), 0.02 + 1e-12)
        self.assertAlmostEqual(coord[4], 0.05, places=12)
        self.assertEqual(coord[1], cfg.w_vref)

    def test_g3_phase_switch(self):
        r, cfg = G.CorridorRef(corr()), G.GuidanceCfg()
        pose = r.pose(math.radians(10.0))
        for ph, ms, on in (('OPEN', 'free', True), ('SETTLE', 'free', True), ('OPEN', 'waiting', False), ('OPEN', 'hold', False),
                           ('PAUSE', 'free', False), ('ABORTED', 'free', False), ('DONE', 'free', False), ('HOLD', 'hold', False)):
            coord, info, err = G.phase_coord(r, cfg, ph, ms, pose, math.radians(10.0), 0.017)
            self.assertEqual(coord[1] > 0, on, ph)
            self.assertEqual(info is not None, on, ph)
            self.assertIsNone(err)
            if not on:
                self.assertTrue(all(math.isnan(v) for v in coord[2:5]))

    def test_g4_nonfinite(self):
        r, cfg = G.CorridorRef(corr()), G.GuidanceCfg()
        coord, info, err = G.phase_coord(r, cfg, 'OPEN', 'free', (float('nan'), 0.0, 0.0), 0.1, 0.0)
        self.assertEqual(coord[1], 0.0)
        self.assertIsNone(info)
        self.assertIsNotNone(err)

    def test_g5_parse_coord(self):
        spec = importlib.util.spec_from_file_location('adm8', os.path.join(HERE, 'dl3_wgmpc_wg2_adm8_node.py'))
        m = importlib.util.module_from_spec(spec)
        sys.modules['adm8'] = m                                   # dataclass 需要模組已登錄
        try:
            spec.loader.exec_module(m)
        except ImportError as e:
            self.skipTest(f'無 ROS 環境：{e}')
        r, cfg = G.CorridorRef(corr()), G.GuidanceCfg()
        on, _, _ = G.phase_coord(r, cfg, 'OPEN', 'free', r.pose(math.radians(5.0)), math.radians(5.0), 0.017)
        c, why = m.parse_coord(on)
        self.assertIsNone(why)
        self.assertEqual(c['w_vref'], 1.0)
        self.assertEqual(len(c['base_vref']), 3)
        self.assertEqual(c['w_qn'], 0.0)
        c, why = m.parse_coord(G.off_coord())
        self.assertIsNone(why)
        self.assertEqual(c['w_vref'], 0.0)
        self.assertIsNone(c['base_vref'])

    def test_g6_converges(self):
        c = corr()
        r, cfg = G.CorridorRef(c), G.GuidanceCfg()
        q = math.radians(12.0)
        ref = r.pose(q)
        p = COR.se2_compose(ref, (0.015, -0.010, math.radians(0.7)))
        x, y, th = p
        for _ in range(100):                                   # 5 s，20 Hz
            coord, _, _ = G.phase_coord(r, cfg, 'OPEN', 'free', (x, y, th), q, 0.0)
            vx, vy, w = coord[2:5]
            cth, sth = math.cos(th), math.sin(th)
            x, y, th = x + 0.05 * (cth * vx - sth * vy), y + 0.05 * (sth * vx + cth * vy), th + 0.05 * w
        e = c.deviation((x, y, th), math.degrees(q))
        self.assertLess(max(abs(e[0]), abs(e[1])), 1e-3)
        self.assertLess(abs(math.degrees(e[2])), 0.02)


if __name__ == '__main__':
    unittest.main(verbosity=2)
