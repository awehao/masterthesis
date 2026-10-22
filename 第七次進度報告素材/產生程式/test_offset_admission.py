#!/usr/bin/env python3
"""offset_admission 的離線反例。合成手臂依模型逐物理步：q⁺ = q + α(s − q) + α·d、s⁺ = s + dt_p·ṡ（每求解週期 5 步）。
另以 door_k3a_1 求解紀錄重放：legacy 必須逐週期重現實錄 d̂（舊模式預設不變）。

    python3 evaluation/test_offset_admission.py
"""
import json
import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import offset_admission as OA                         # noqa: E402

ID = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
ALPHA = np.asarray(ID['alpha'], float)
DTP = float(ID['phys_dt_measured_s'])
DT = 0.05
D_TRUE = np.array([0.0, 0.017, -0.008, -0.001, -0.001, 0.0])


def est(adm, **kw):
    return OA.OffsetEstimator(ALPHA, DTP, DT, admission=adm, **kw)


def simulate(e, sdot_of_t, T, d=D_TRUE, q0=None, s0=None):
    """回傳每週期紀錄。sdot_of_t(t) → 6 軸設定點速度。"""
    s = np.zeros(6) if s0 is None else np.array(s0, float)
    q = s + d if q0 is None else np.array(q0, float)
    t, out = 0.0, []
    while t < T - 1e-9:
        out.append((t, e.update(t, s.copy(), q.copy()), s.copy(), q.copy()))
        for _ in range(int(round(DT / DTP))):
            q = q + ALPHA * (s - q) + ALPHA * d
            s = s + DTP * np.asarray(sdot_of_t(t), float)
        t = round(t + DT, 9)
    return out


class Admission(unittest.TestCase):
    def test_default_is_legacy(self):
        e = OA.OffsetEstimator(ALPHA, DTP, DT)
        self.assertEqual(e.admission, 'legacy')

    def test_consistent_requires_observer(self):
        with self.assertRaises(ValueError):
            OA.OffsetEstimator(ALPHA, DTP, DT, moving=False, admission='consistent')

    def test_static_start_converges_both(self):
        for adm in ('legacy', 'consistent'):
            e = est(adm)
            simulate(e, lambda t: np.zeros(6), 6.0)
            np.testing.assert_allclose(e.d, D_TRUE, atol=2e-5, err_msg=adm)

    def test_consistent_static_start_waits_for_window(self):
        e = est('consistent')
        out = simulate(e, lambda t: np.zeros(6), 0.25)
        self.assertTrue(all(r['used'] == 'held' for _, r, _, _ in out))   # 窗口未滿 ⇒ 保持
        np.testing.assert_allclose(e.d, 0.0)

    def test_constant_velocity_converges_consistent(self):
        v = np.array([0.0, 0.08, -0.05, 0.0, 0.03, 0.0])                    # 每週期 4 mrad > gate ⇒ legacy 也只走 moving
        e = est('consistent')
        simulate(e, lambda t: v, 8.0, q0=np.zeros(6) + D_TRUE, s0=np.zeros(6))
        np.testing.assert_allclose(e.d, D_TRUE, atol=2e-4)

    def test_accel_transient_bypass_counterexample(self):
        """慢速起步加速：設定點逐週期變化 < gate（static 准入），但非等速（觀測器拒絕）。
        legacy 經 static 分支以含遲滯的 q − s 更新（偏離真值）；consistent 保持。"""
        acc = np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.0])                      # rad/s²：0.4 s 內 ṡ 由 0 到 0.04 rad/s（每週期 < 2 mrad）
        # 0.30 s 窗內 ṡ 跨度 0.03 ⇒ 對均值偏差 0.015 > accel_tol 0.01 ⇒ 觀測器拒絕；逐週期變化 < gate ⇒ static 准入

        def sd(t):
            return acc * max(0.0, t - 3.0) if t < 3.4 else acc * 0.4
        res = {}
        for adm in ('legacy', 'consistent'):
            e = est(adm)
            out = simulate(e, sd, 3.4)
            w = [(t, r) for t, r, _, _ in out if 3.05 <= t < 3.4]
            res[adm] = (e.d.copy(), w)
        d_leg, w_leg = res['legacy']
        d_con, w_con = res['consistent']
        n_bypass = sum(1 for _, r in w_leg if r['used'] == 'static' and r['om'] is not None and not r['om']['ok'])
        self.assertGreater(n_bypass, 3)                                        # 重現 Codex 指出的繞過
        self.assertTrue(all(r['used'] == 'held' for _, r in w_con if r['om'] is not None and not r['om']['ok']))
        self.assertGreater(abs(d_leg[1] - D_TRUE[1]), abs(d_con[1] - D_TRUE[1]) + 2e-4)

    def test_missing_data_held(self):
        e = est('consistent')
        simulate(e, lambda t: np.zeros(6), 3.0)
        d0 = e.d.copy()
        for bad in ([float('nan')] * 6, None):
            r = e.update(3.05, bad, np.zeros(6))
            self.assertEqual(r['used'], 'held')
            np.testing.assert_allclose(e.d, d0)
        r = e.update(float('nan'), np.zeros(6), np.zeros(6))
        self.assertEqual(r['used'], 'held')

    def test_time_not_increasing_held(self):
        e = est('consistent')
        out = simulate(e, lambda t: np.zeros(6), 3.0)
        d0 = e.d.copy()
        t_last, _, s_last, q_last = out[-1]
        r = e.update(t_last, s_last, q_last)                                  # 同一時刻再送
        self.assertEqual(r['used'], 'held')
        self.assertIn('time_not_increasing', r['why'])
        np.testing.assert_allclose(e.d, d0)

    def test_clip(self):
        e = est('consistent')
        simulate(e, lambda t: np.zeros(6), 6.0, d=np.array([0.0, 0.2, 0, 0, 0, 0]))
        self.assertAlmostEqual(e.d[1], 0.05)

    def test_init_static_same_both(self):
        for adm in ('legacy', 'consistent'):
            e = est(adm, init_static=True)
            e.update(0.0, np.zeros(6), D_TRUE)
            np.testing.assert_allclose(e.init, D_TRUE)

    def test_legacy_reproduces_door_k3a_1_log(self):
        """舊模式預設不變：依 door_k3a_1 求解紀錄的同步輸入重放，d̂ 與更新類別逐週期與實錄一致。"""
        p = os.path.join(HERE, 'runs', 'door_k3a_1', 'align_solver.json')
        a = json.load(open(p))
        A = a['args']
        e = OA.OffsetEstimator(ALPHA, DTP, 1.0 / A['rate'], A['offset_tau_s'], A['offset_gate_rad'], A['offset_max_rad'],
                               bool(A.get('offset_init_static')), bool(A.get('offset_moving')), 'legacy')
        n = 0
        for x in a['log']:
            si = x.get('solve_in')
            if not si:
                continue
            r = e.update(si['snap_sim_t_exact'], si['s_meas'], si['q_meas'][3:])
            self.assertLess(float(np.abs(np.array(r['d_hat_after']) - np.array(si['offset_d_hat'])).max()), 1e-12)
            om = x.get('offset_moving') or {}
            self.assertEqual(om.get('used'), r['used'])
            n += 1
        self.assertEqual(n, a['stats']['n_solve_rows_logged'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
