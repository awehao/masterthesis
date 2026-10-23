#!/usr/bin/env python3
"""door_k3m_corridor 的反例：SE(2) 合成／反元素、偏差以參考座標表示（旋轉參考）、帶寬判定、無效輸入、表驗證、端點夾取、
JSON 往返；以及真實走廊表（door_k3m_corridor_table.json）上候選軌跡本身的偏差 ≈ 0。

    python3 evaluation/test_door_k3m_corridor.py
"""
import json
import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_corridor as CO                         # noqa: E402

D2R = math.radians


def table(yaw0=0.0):
    rows = [{'q_door_deg': 0.0, 'fwd_m': 0.0, 'lat_m': 0.0, 'dyaw_deg': 0.0},
            {'q_door_deg': 8.0, 'fwd_m': -0.08, 'lat_m': 0.06, 'dyaw_deg': -3.4}]
    return CO.Corridor(rows, (0.6, -0.53, yaw0))


class Corridor(unittest.TestCase):
    def test_se2_inverse(self):
        a = (0.3, -0.2, 0.7)
        e = CO.se2_compose(CO.se2_inv(a), a)
        for v in e:
            self.assertAlmostEqual(v, 0.0, places=12)

    def test_deviation_in_reference_frame(self):
        C = table(yaw0=math.pi / 2)                                       # 錨點朝 +y
        ref = C.ref_pose(0.0)
        b = (ref[0] + 0.010, ref[1], ref[2])                              # 世界 +x 10 mm ＝ 參考座標的右方（−y）
        e = C.deviation(b, 0.0)
        self.assertAlmostEqual(e[0], 0.0, places=12)
        self.assertAlmostEqual(e[1], -0.010, places=12)

    def test_interpolation_and_clamp(self):
        C = table()
        f, l, y = C.ref_local(4.0)
        self.assertAlmostEqual(f, -0.04)
        self.assertAlmostEqual(l, 0.03)
        self.assertAlmostEqual(y, -1.7)
        self.assertEqual(C.ref_local(-1.0), C.ref_local(0.0))
        self.assertEqual(C.ref_local(9.0), C.ref_local(8.0))

    def test_band(self):
        C = table()
        ref = C.ref_pose(4.0)
        ok, e, why = C.check(ref, 4.0, 0.025, D2R(1.5))
        self.assertTrue(ok)
        off = CO.se2_compose(ref, (0.026, 0.0, 0.0))
        self.assertEqual(C.check(off, 4.0, 0.025, D2R(1.5))[2], 'corridor_position')
        off = CO.se2_compose(ref, (0.0, 0.0, D2R(1.6)))
        self.assertEqual(C.check(off, 4.0, 0.025, D2R(1.5))[2], 'corridor_yaw')

    def test_moving_reference(self):
        """門角改變而底盤不動 ⇒ 相對走廊的偏差隨參考移動。"""
        C = table()
        b = C.ref_pose(4.0)
        e = C.deviation(b, 4.5)
        self.assertGreater(abs(e[0]) + abs(e[1]), 1e-3)
        self.assertAlmostEqual(e[2], -D2R(-3.4 * 0.5 / 8.0), places=12)

    def test_invalid_inputs(self):
        C = table()
        for b, q in ((None, 1.0), ((0.6, float('nan'), 0.0), 1.0), ((0.6, -0.5, 0.0), float('nan')), ((0.6, -0.5), 1.0), ((0.6, -0.5, 0.0), None)):
            self.assertEqual(C.check(b, q, 0.025, 0.03)[2], 'corridor_input_invalid')

    def test_table_validation(self):
        with self.assertRaises(ValueError):
            CO.Corridor([{'q_door_deg': 0.0, 'fwd_m': 0.0, 'lat_m': 0.0, 'dyaw_deg': 0.0}], (0, 0, 0))
        with self.assertRaises(ValueError):
            CO.Corridor([{'q_door_deg': 0.0, 'fwd_m': float('nan'), 'lat_m': 0.0, 'dyaw_deg': 0.0},
                         {'q_door_deg': 1.0, 'fwd_m': 0.0, 'lat_m': 0.0, 'dyaw_deg': 0.0}], (0, 0, 0))
        with self.assertRaises(ValueError):
            CO.Corridor([{'q_door_deg': 0.0, 'fwd_m': 0.0, 'lat_m': 0.0, 'dyaw_deg': 0.0},
                         {'q_door_deg': 1.0, 'fwd_m': 0.0, 'lat_m': 0.0, 'dyaw_deg': 0.0}], (0, float('inf'), 0))

    def test_json_roundtrip(self):
        C = table()
        C2 = CO.Corridor.from_json(json.loads(json.dumps(C.to_json())))
        self.assertEqual(C.ref_pose(3.3), C2.ref_pose(3.3))

    def test_real_table_candidate_on_reference(self):
        p = os.path.join(HERE, 'results', 'door', 'door_k3m_corridor_table.json')
        rp = os.path.join(HERE, 'results', 'door', 'door_k3m_offline_replay_r2.json')
        if not os.path.exists(p):
            self.skipTest('走廊表尚未產生')
        C = CO.Corridor.from_json(json.load(open(p))['table'])
        rows = next(c for c in json.load(open(rp))['configs'] if c['name'] == 'free')['rows']
        worst = 0.0
        for r in rows:
            b = CO.se2_compose(C.anchor, (r['base_fwd_m'], r['base_lat_m'], D2R(r['base_dyaw_deg'])))
            e = C.deviation(b, r['q_door_deg'])
            worst = max(worst, abs(e[0]), abs(e[1]))
        self.assertLess(worst, 0.003)                                     # 候選本身在走廊中心附近（門角相同時取平均造成的小差）


if __name__ == '__main__':
    unittest.main(verbosity=2)
