#!/usr/bin/env python3
"""door_coopen_closure_guidance（C2）純邏輯測試。
  Z1 啟用權重：36° 以下 0、40° 以上 1、中間 smoothstep 單調；36° 以下 c 不累積（維持 0）。
  Z2 一階＋變化率：固定誤差 5 mm、dt 0.05 ⇒ 每軸每輪變化 ≤ 2 mm/s·dt；長時間收斂到誤差值；不超過 8 mm（誤差 20 mm 時截到 8 mm、計數）。
  Z3 失效處置：非有限／維度錯／dt≤0 ⇒ 不更新 c、沿用上一個有效 s，n_invalid＋1，不拋例外。
  Z4 ShiftedTable：pose() 平移加 offset、偏航不變；dpose_dq 等其他屬性原樣轉交 ⇒ door_wide_guidance.guidance 的前饋不變；
     offset 0 時 guidance 輸出與原表逐位元相同；offset 非 0 時世界速度只差 k_lin·offset（未觸限幅時），且仍受 v_cap 限幅。
  Z5 方向：TCP 在目標 +x 方向 3 mm（e＝−3 mm）⇒ 收斂後參考位姿往 −x 移（底盤被引向使 TCP 回到目標）。

    python3 evaluation/test_door_coopen_closure_guidance.py
"""
import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_closure_guidance as CG              # noqa: E402
import door_wide_guidance as G                         # noqa: E402


def table():
    rows = [{'q_door_deg': float(q), 'fwd_m': 0.001 * q, 'lat_m': -0.002 * q, 'dyaw_deg': 0.1 * q, 'q': [0.0] * 9} for q in range(0, 50)]
    return G.RefTable(rows, (1.0, -0.5, 3.0))


class Z(unittest.TestCase):
    def test_z1_enable(self):
        vals = [CG.smoothstep(q, 36.0, 40.0) for q in np.linspace(30, 45, 61)]
        self.assertEqual(vals[0], 0.0)
        self.assertEqual(vals[-1], 1.0)
        self.assertTrue(all(b >= a for a, b in zip(vals, vals[1:])))
        c = CG.ClosureCorrector()
        for _ in range(100):
            s, cc, _ = c.update(0.05, 30.0, [0.0, 0.0], [0.005, 0.0])
        self.assertEqual((s, float(np.linalg.norm(cc))), (0.0, 0.0))

    def test_z2_filter_rate_mag(self):
        c = CG.ClosureCorrector()
        prev = np.zeros(2)
        for _ in range(2000):
            s, cc, _ = c.update(0.05, 45.0, [0.0, 0.0], [0.005, 0.0])
            self.assertTrue(np.all(np.abs(cc - prev) <= 0.002 * 0.05 + 1e-12))
            prev = cc
        np.testing.assert_allclose(cc, [0.005, 0.0], atol=1e-6)
        c = CG.ClosureCorrector()
        for _ in range(4000):
            s, cc, _ = c.update(0.05, 45.0, [0.0, 0.0], [0.02, 0.0])
        self.assertAlmostEqual(float(np.linalg.norm(cc)), 0.008, places=9)
        self.assertGreater(c.n_clip_mag, 0)

    def test_z3_invalid(self):
        c = CG.ClosureCorrector()
        for _ in range(50):
            c.update(0.05, 45.0, [0.0, 0.0], [0.005, 0.0])
        c_before, s_before = c.c.copy(), c.s
        for bad in ((0.05, float('nan'), [0, 0], [0, 0]), (0.05, 45.0, [0, float('inf')], [0, 0]), (0.05, 45.0, [0, 0, 0], [0, 0]),
                    (0.0, 45.0, [0, 0], [0, 0]), (0.05, 45.0, None, [0, 0])):
            s, cc, info = c.update(*bad)
            self.assertEqual(info['why'], 'input_invalid')
            np.testing.assert_array_equal(cc, c_before)
            self.assertEqual(s, s_before)
        self.assertEqual(c.n_invalid, 5)

    def test_z4_shifted_table(self):
        t = table()
        cfg = G.GuidanceCfg(k_lin=1.0, k_yaw=1.0, v_cap=0.035, w_cap=0.10)
        pose, q, rate = (1.0, -0.6, 3.0), math.radians(20.0), 0.01
        a, _ = G.guidance(t, cfg, pose, q, rate)
        b, _ = G.guidance(CG.ShiftedTable(t, (0.0, 0.0)), cfg, pose, q, rate)
        np.testing.assert_array_equal(np.array(a, float), np.array(b, float))   # NaN 位置相同視為相等
        sh = CG.ShiftedTable(t, (0.003, -0.002))
        self.assertEqual(sh.pose(q)[2], t.pose(q)[2])
        self.assertEqual(sh.dpose_dq(q), t.dpose_dq(q))
        p_on = t.pose(q)                                   # 位於表位姿上 ⇒ 回授未觸限幅，速度差＝k_lin·offset
        _, ia = G.guidance(t, cfg, p_on, q, rate)
        _, ib = G.guidance(sh, cfg, p_on, q, rate)
        np.testing.assert_allclose(np.array(ib['v_world'][:2]) - np.array(ia['v_world'][:2]), [0.003, -0.002], atol=1e-12)
        self.assertEqual(ib['v_world'][2], ia['v_world'][2])
        _, ic = G.guidance(CG.ShiftedTable(t, (1.0, 0.0)), cfg, pose, q, rate)
        self.assertLessEqual(abs(ic['v_world'][0]), 0.035 + 1e-12)

    def test_z5_direction(self):
        c = CG.ClosureCorrector()
        for _ in range(2000):
            c.update(0.05, 45.0, [0.003, 0.0], [0.0, 0.0])
        self.assertLess(c.ref_offset()[0], -0.0029)


if __name__ == '__main__':
    unittest.main(verbosity=2)
