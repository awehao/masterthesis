#!/usr/bin/env python3
"""door_k3b_pair_geom 反例。
  G1 HELD 實測組態（door_wide_w2c_30_1）⇒ 兩配對 −0.19／−0.36 mm（與離線 v4c 一致，±0.01 mm）。
  G2 全開（8.9 mm）同組態 ⇒ 兩配對皆 > 0。
  G3 預測：兩輪間距離減少 1 mm／0.05 s ⇒ 接近率 0.02 m/s，d_pred＝d − 0.02·T；未接近時用 v_floor 0.002。
  G4 驗證：qf 非 9 維、開度 NaN、時間不遞增、年齡 0.3 s、後端拋例外 ⇒ invalid。

    python3 evaluation/test_door_k3b_pair_geom.py
"""
import json
import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3b_pair_geom as PG                        # noqa: E402


def held():
    v4c = json.load(open(os.path.join(HERE, 'results', 'door', 'door_k3b_offline_v4c.json')))
    v4 = json.load(open(os.path.join(HERE, 'results', 'door', 'door_k3b_offline_v4.json')))
    hs = v4['held_state']
    qf = np.array(list(hs['base']) + list(hs['q_arm']))
    return qf, hs['q_door'], v4c['held_pair_distance_m']['q_finger_measured_m'], v4c['held_pair_distance_m']['per_finger']


class TG(unittest.TestCase):
    def test_g1_g2_held(self):
        qf, qd, fq, ref = held()
        st, d, dp, det = PG.PairGeom().step(1.0, qf, qd, fq, 0.01)
        self.assertEqual(st, 'ok', det)
        self.assertAlmostEqual(d[0], ref['uflite_finger1'], delta=1e-5)
        self.assertAlmostEqual(d[1], ref['uflite_finger2'], delta=1e-5)
        self.assertLess(max(d), 0.0)
        st, d, dp, det = PG.PairGeom().step(1.0, qf, qd, [0.0089, 0.0089], 0.01)
        self.assertGreater(min(d), 0.0)

    def test_g3_pred(self):
        seq = iter([[0.010, 0.012], [0.009, 0.012]])
        g = PG.PairGeom(fn=lambda q, qd, f: next(seq))
        qf = np.zeros(9)
        st, d, dp, det = g.step(1.0, qf, 0.5, [0.008, 0.008], 0.0)
        self.assertAlmostEqual(dp[0], 0.010 - 0.002 * det['T_pred_s'], places=9)
        st, d, dp, det = g.step(1.05, qf, 0.5, [0.008, 0.008], 0.0)
        self.assertAlmostEqual(det['closing_rate'][0], 0.02, places=9)
        self.assertAlmostEqual(dp[0], 0.009 - 0.02 * det['T_pred_s'], places=9)
        self.assertAlmostEqual(dp[1], 0.012 - 0.002 * det['T_pred_s'], places=9)

    def test_g4_invalid(self):
        ok_fn = lambda q, qd, f: [0.01, 0.01]                                # noqa: E731
        self.assertEqual(PG.PairGeom(fn=ok_fn).step(1.0, np.zeros(8), 0.5, [0.008, 0.008], 0.0)[0], 'invalid')
        self.assertEqual(PG.PairGeom(fn=ok_fn).step(1.0, np.zeros(9), 0.5, [math.nan, 0.008], 0.0)[0], 'invalid')
        g = PG.PairGeom(fn=ok_fn)
        g.step(1.0, np.zeros(9), 0.5, [0.008, 0.008], 0.0)
        self.assertEqual(g.step(1.0, np.zeros(9), 0.5, [0.008, 0.008], 0.0)[0], 'invalid')
        self.assertEqual(PG.PairGeom(fn=ok_fn).step(1.0, np.zeros(9), 0.5, [0.008, 0.008], 0.3)[0], 'invalid')

        def boom(q, qd, f):
            raise RuntimeError('x')
        self.assertEqual(PG.PairGeom(fn=boom).step(1.0, np.zeros(9), 0.5, [0.008, 0.008], 0.0)[3]['why'], 'pair_compute_failed')


if __name__ == '__main__':
    unittest.main(verbosity=1)
