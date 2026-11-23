#!/usr/bin/env python3
"""door_k3a_rot_locate.predict_next 的合成反例：依同一模型生成的資料，相鄰快照殘差必須近零；注入關節擾動則必須被看見。

    python3 evaluation/test_door_k3a_rot_locate.py
"""
import json
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3a_rot_locate as RL                      # noqa: E402

ID = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
ALPHA = np.asarray(ID['alpha'], float)
DTP = float(ID['phys_dt_measured_s'])


def generate(n_cycles=40, steps=5, d=np.array([0.0, 0.017, -0.008, -0.001, -0.001, 0.0]), perturb=None, seed=0):
    """以模型逐物理步生成 (q, s, applied) 快照序列；perturb(k) → 第 k 週期末加到 q 的關節擾動。"""
    rng = np.random.default_rng(seed)
    q = np.array([-0.7, 0.45, 2.1, -1.64, -0.69, 0.09])
    s = q - d
    snaps = []
    for k in range(n_cycles):
        u_cyc = rng.normal(0, 0.03, 6)
        applied = [u_cyc.copy() for _ in range(steps)]
        snaps.append((q.copy(), s.copy(), applied))
        for u in applied:
            q = q + ALPHA * (s - q) + ALPHA * d
            s = s + DTP * u
        if perturb is not None:
            q = q + perturb(k)
    snaps.append((q.copy(), s.copy(), None))
    return snaps, d


class Residual(unittest.TestCase):
    def test_same_model_gives_zero_residual(self):
        snaps, d = generate()
        worst = 0.0
        for (q0, s0, ap), (q1, s1, _) in zip(snaps, snaps[1:]):
            qp, sp = RL.predict_next(q0, s0, d, ap, ALPHA, DTP)
            worst = max(worst, float(np.abs(q1 - qp).max()))
            self.assertLess(float(np.abs(s1 - sp).max()), 1e-12)
        self.assertLess(worst, 1e-12)

    def test_injected_perturbation_is_seen(self):
        bump = np.array([0.0, 0.0, 0.0, 0.002, 0.0, 0.0])
        snaps, d = generate(perturb=lambda k: bump if k == 10 else np.zeros(6))
        res = [snaps[k + 1][0] - RL.predict_next(snaps[k][0], snaps[k][1], d, snaps[k][2], ALPHA, DTP)[0] for k in range(len(snaps) - 1)]
        np.testing.assert_allclose(res[10], bump, atol=1e-12)
        self.assertLess(max(float(np.abs(r).max()) for i, r in enumerate(res) if i != 10), 1e-12)   # 每區間重新初始化 ⇒ 不累積

    def test_wrong_offset_shows_constant_bias(self):
        snaps, d = generate()
        res = [snaps[k + 1][0] - RL.predict_next(snaps[k][0], snaps[k][1], d * 0.0, snaps[k][2], ALPHA, DTP)[0] for k in range(len(snaps) - 1)]
        self.assertTrue(all(r[1] > 0 for r in res))                  # 漏掉 d̂ ⇒ j2 同號偏差（區間內一致）

    def test_dist_helper(self):
        dd = RL.dist([[1.0, -1.0], [3.0, -3.0]])
        self.assertEqual(dd['n'], 2)
        self.assertEqual(dd['frac_negative'], [0.0, 1.0])
        self.assertIsNone(RL.dist([]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
