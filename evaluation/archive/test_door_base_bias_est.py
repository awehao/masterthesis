#!/usr/bin/env python3
"""door_base_bias_est 測試（Codex 20261027_010258 §3 反例）。合成受控體＝wgmpc_core_sp_db.plant_phys_step 同一式（逐物理步 u＋b）。
  E1 已知 b、變動命令與偏航：區間反解與窗平均還原 b（1e-9）；零偏差 ⇒ 0。
  E2 樣本不足（< N_MIN）⇒ None；過期（> MAX_AGE_S）⇒ None。
  E3 缺步（漏一筆命令）、命令 exec_mode≠0、api 未套用、命令值非有限 ⇒ 該區間不用。
  E4 同步號命令再到：值同 ⇒ dup 不重複；值不同 ⇒ 衝突、含它的區間不用。位姿同時刻再到值同 ⇒ dup、不同 ⇒ conflict；倒退 ⇒ 不重開。
  E5 超界：區間觀測超 CAP ⇒ out_of_bounds 不採用。
  E6 啟用：停用 ⇒ None 且清窗；再啟用從零累積（N_MIN 前 None）；停用期間不接受區間。
  E7 錯誤估計不放大：受控體偏差在中途改變 ⇒ 窗只保留 0.5 s，估計在一窗後轉為新值。
  E8 與核心一致：用估計值的 plant_phys_step 推進＝受控體真值（1e-12）。

    python3 evaluation/test_door_base_bias_est.py
"""
import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import door_base_bias_est as E                               # noqa: E402

DTP = 0.01
B = np.array([-0.00024, 0.0009, 0.0024])


def plant(p, u, b):
    th = p[2]
    c, s = math.cos(th), math.sin(th)
    w = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]) @ (np.asarray(u) + b)
    return p + DTP * w


class Sim:
    """100 Hz 命令、50 Hz 位姿；命令於 t_k 套用，位姿 stamp＝步後時刻。"""
    def __init__(self, est, b=B, seed=0):
        self.e, self.b = est, np.asarray(b, float)
        self.rng = np.random.default_rng(seed)
        self.p = np.array([1.02, -0.5, 3.03])
        self.k = 0
        self.e.on_pose(0.0, self.p)

    def run(self, n_steps, drop=(), mode=0, api=1, b=None):
        if b is not None:
            self.b = np.asarray(b, float)
        out = []
        for _ in range(n_steps):
            u = np.array([0.003, -0.002, 0.01]) + self.rng.normal(0, 0.002, 3)
            t = round(self.k * DTP, 9)
            if self.k not in drop:
                self.e.on_cmd(self.k, t, u, mode, api)
            self.p = plant(self.p, u, self.b)
            self.k += 1
            if self.k % 2 == 0:
                out.append(self.e.on_pose(round(self.k * DTP, 9), self.p))
        return out

    @property
    def t(self):
        return round(self.k * DTP, 9)


def mk(enabled=True):
    e = E.BaseBiasEstimator(DTP)
    e.set_enabled(enabled)
    return e


class T(unittest.TestCase):
    def test_e1_recover(self):
        e = mk()
        s = Sim(e)
        recs = s.run(100)
        self.assertTrue(all(r['used'] for r in recs[1:]) and recs[0]['used'])
        for r in recs:
            np.testing.assert_allclose(r['b_obs'], B, atol=1e-9)
        b, why, info = e.estimate(s.t)
        self.assertIsNone(why)
        np.testing.assert_allclose(b, B, atol=1e-9)
        e0 = mk()
        s0 = Sim(e0, b=np.zeros(3))
        s0.run(60)
        np.testing.assert_allclose(e0.estimate(s0.t)[0], 0.0, atol=1e-10)

    def test_e2_few_and_stale(self):
        e = mk()
        s = Sim(e)
        s.run(2 * (E.N_MIN - 1))
        self.assertEqual(e.estimate(s.t)[1], 'too_few')
        s.run(2)
        self.assertIsNone(e.estimate(s.t)[1])
        self.assertIsNone(e.estimate(s.t + E.MAX_AGE_S - 1e-3)[1])
        self.assertEqual(e.estimate(s.t + E.MAX_AGE_S + 1e-3)[1], 'stale')

    def test_e3_bad_steps(self):
        for kw, key in ((dict(drop={5}), 'cmd_count'), (dict(mode=4), 'exec_mode'), (dict(api=0), 'exec_mode')):
            e = mk()
            s = Sim(e)
            recs = s.run(10, **kw)
            self.assertTrue(any(not r['used'] for r in recs), kw)
            self.assertIn(key, e.rejects, kw)
            if 'drop' in kw:
                self.assertEqual(sum(r['used'] for r in recs), 4)
        e = mk()
        e.on_pose(0.0, [0, 0, 0])
        e.on_cmd(0, 0.0, [float('nan'), 0, 0], 0, 1)
        e.on_cmd(1, 0.01, [0, 0, 0], 0, 1)
        r = e.on_pose(0.02, [0, 0, 0])
        self.assertEqual(r['why'], 'cmd_nonfinite')

    def test_e4_dup_conflict_monotone(self):
        e = mk()
        e.on_pose(0.0, [0, 0, 0])
        e.on_cmd(0, 0.0, [0, 0, 0], 0, 1)
        self.assertFalse(e.on_cmd(0, 0.0, [0, 0, 0], 0, 1))
        self.assertEqual(e.n_dup, 1)
        e.on_cmd(1, 0.01, [0, 0, 0], 0, 1)
        e.on_cmd(1, 0.01, [0.001, 0, 0], 0, 1)
        self.assertEqual(e.n_conflict, 1)
        r = e.on_pose(0.02, [0, 0, 0])
        self.assertFalse(r['used'])
        self.assertIn('cmd_count', r['why'])
        self.assertEqual(e.on_pose(0.02, [0, 0, 0])['why'], 'dup')
        self.assertEqual(e.on_pose(0.02, [1e-6, 0, 0])['why'], 'pose_conflict')
        self.assertEqual(e.on_pose(0.01, [0, 0, 0])['why'], 'pose_not_advancing')
        e2 = mk()
        s = Sim(e2)
        s.run(20)
        n = e2.n_used
        for t, p in [(0.02, [1.0, 0, 0]), (0.04, [1.0, 0, 0])]:
            e2.on_pose(t, p)
        self.assertEqual(e2.n_used, n)

    def test_e5_out_of_bounds(self):
        e = mk()
        s = Sim(e, b=np.array([0.0, 0.006, 0.0]))
        recs = s.run(30)
        self.assertTrue(all(not r['used'] for r in recs))
        self.assertEqual(e.rejects.get('out_of_bounds'), len(recs) - 0)
        self.assertEqual(e.estimate(s.t)[1], 'too_few')

    def test_e6_enable(self):
        e = mk(False)
        s = Sim(e)
        recs = s.run(40)
        self.assertTrue(all(r['why'] == 'disabled' for r in recs))
        self.assertEqual(e.estimate(s.t)[1], 'disabled')
        e.set_enabled(True)
        s.run(2 * E.N_MIN)
        self.assertEqual(e.estimate(s.t)[1], 'too_few')       # 第一筆位姿只設錨點
        s.run(2)
        self.assertIsNone(e.estimate(s.t)[1])
        e.set_enabled(False)
        self.assertEqual(e.estimate(s.t)[1], 'disabled')
        self.assertEqual(len(e.win), 0)
        e.set_enabled(True)
        s.run(2 * (E.N_MIN - 1))
        self.assertEqual(e.estimate(s.t)[1], 'too_few')

    def test_e7_change(self):
        e = mk()
        s = Sim(e)
        s.run(100)
        b2 = np.array([0.001, -0.001, -0.003])
        s.run(20, b=b2)
        b, _, _ = e.estimate(s.t)
        self.assertGreater(np.abs(b - b2).max(), 1e-4)        # 窗內仍混有舊值
        s.run(int(E.WIN_S / DTP) + 2)
        np.testing.assert_allclose(e.estimate(s.t)[0], b2, atol=1e-9)

    def test_e8_core_consistency(self):
        from ammr_wholebody_mpc import wgmpc_core_sp_db as S1
        import json
        with open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')) as f:
            idn = json.load(f)
        am = S1.ArmSetpointModel(alpha=idn['alpha'], bias=idn['bias_rad'], phys_dt=DTP)
        cfg = S1.WGMPCConfigSP(N=5, dt=0.05, tcp='link_tcp', arm_model=am)
        e = mk()
        s = Sim(e)
        s.run(40)
        b, _, _ = e.estimate(s.t)
        q = np.concatenate([s.p, np.zeros(6)])
        u = np.concatenate([[0.003, -0.002, 0.01], np.zeros(6)])
        qq, _ = S1.plant_phys_step(q, np.zeros(6), u, cfg, 3, base_bias=b)
        p = s.p.copy()
        for _ in range(3):
            p = plant(p, u[:3], B)
        np.testing.assert_allclose(qq[:3], p, atol=1e-12)


if __name__ == '__main__':
    unittest.main(verbosity=2)
