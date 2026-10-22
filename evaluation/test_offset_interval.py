#!/usr/bin/env python3
"""offset_interval.IntervalOffsetObserver 的離線反例。合成手臂依同一模型逐物理步生成（區間步數 4／5／6 混合）。

    python3 evaluation/test_offset_interval.py
"""
import json
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import offset_interval as OI                          # noqa: E402

ID = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
ALPHA = np.asarray(ID['alpha'], float)
DTP = float(ID['phys_dt_measured_s'])
D0 = np.array([0.0, 0.017, -0.008, -0.001, -0.001, 0.0])


def run(obs, n, u_of_k, d_of_k=lambda k: D0, m_of_k=lambda k: 5, drop=None):
    q = np.array([-0.7, 0.45, 2.1, -1.64, -0.69, 0.09])
    s = q - D0
    recs = []
    for k in range(n):
        m = m_of_k(k)
        d = d_of_k(k)
        applied = [np.asarray(u_of_k(k, j), float) for j in range(m)]
        q0, s0 = q.copy(), s.copy()
        for u in applied:
            q = q + ALPHA * (s - q) + ALPHA * d
            s = s + DTP * u
        if drop is not None and drop(k):
            recs.append(obs.update(q0, s0, applied, [float('nan')] * 6))
        else:
            recs.append(obs.update(q0, s0, applied, q))
    return recs


def mk(**kw):
    return OI.IntervalOffsetObserver(ALPHA, DTP, **kw)


class Interval(unittest.TestCase):
    def test_static_converges(self):
        o = mk()
        run(o, 200, lambda k, j: np.zeros(6))
        np.testing.assert_allclose(o.d, D0, atol=1e-9)

    def test_constant_velocity_converges(self):
        v = np.array([0.0, 0.08, -0.05, 0.0, 0.03, 0.0])
        o = mk()
        run(o, 200, lambda k, j: v)
        np.testing.assert_allclose(o.d, D0, atol=1e-9)

    def test_accel_decel_converges(self):
        o = mk()
        run(o, 200, lambda k, j: np.array([0.0, 0.1, -0.05, 0.02, 0.0, 0.0]) * np.sin(0.3 * (5 * k + j)))
        np.testing.assert_allclose(o.d, D0, atol=1e-9)

    def test_obs_exact_under_model(self):
        """模型成立時，每區間的 d_obs 恰為真值（含 4／6 步區間），與先前 d̂ 無關。"""
        o = mk()
        recs = run(o, 30, lambda k, j: np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.0]) * (k % 3), m_of_k=lambda k: (4, 5, 6)[k % 3])
        for r in recs:
            np.testing.assert_allclose(r['d_obs'], D0, atol=1e-10)
        self.assertEqual({r['m'] for r in recs}, {4, 5, 6})

    def test_out_of_sample_residual_uses_prior_estimate(self):
        o = mk()
        recs = run(o, 3, lambda k, j: np.zeros(6))
        # 首區間 d̂＝0 ⇒ 樣本外殘差＝D0·[1−(1−α)^5]，不是 0
        np.testing.assert_allclose(recs[0]['pred_residual'], D0 * (1 - (1 - ALPHA) ** 5), atol=1e-12)
        np.testing.assert_allclose(recs[0]['d_hat_before'], 0.0)

    def test_known_offset_step_tracked_with_tau(self):
        d1 = D0 + np.array([0.0, 0.004, 0.0, 0.0, 0.0, 0.0])
        o = mk()
        run(o, 200, lambda k, j: np.zeros(6))
        recs = run(o, 10, lambda k, j: np.zeros(6), d_of_k=lambda k: d1)
        # 一階濾波：每區間 m·dt_p/τ＝0.1 ⇒ 10 區間後剩 0.9^10 ≈ 0.349 的落差
        np.testing.assert_allclose(o.d[1] - d1[1], (D0[1] - d1[1]) * 0.9 ** 10, rtol=1e-6)
        self.assertTrue(all(abs(r['d_obs'][1] - d1[1]) < 1e-10 for r in recs))

    def test_missing_data_held(self):
        o = mk()
        run(o, 100, lambda k, j: np.zeros(6))
        d_before = o.d.copy()
        recs = run(o, 5, lambda k, j: np.zeros(6), drop=lambda k: True)
        self.assertTrue(all(r['used'] == 'held' for r in recs))
        np.testing.assert_array_equal(o.d, d_before)
        r = o.update(np.zeros(6), np.zeros(6), [], np.zeros(6))
        self.assertEqual(r['used'], 'held')
        r = o.update(np.zeros(6), None, [np.zeros(6)], np.zeros(6))
        self.assertEqual(r['used'], 'held')
        r = o.update(np.zeros(6), np.zeros(6), [[float('inf')] * 6], np.zeros(6))
        self.assertEqual(r['used'], 'held')

    def test_clip(self):
        o = mk()
        run(o, 300, lambda k, j: np.zeros(6), d_of_k=lambda k: np.array([0.0, 0.2, 0, 0, 0, 0]))
        self.assertAlmostEqual(o.d[1], 0.05)

    def test_invalid_config(self):
        with self.assertRaises(ValueError):
            OI.IntervalOffsetObserver(np.full(6, 1.2), DTP)
        with self.assertRaises(ValueError):
            OI.IntervalOffsetObserver(ALPHA, DTP, tau_s=0.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
