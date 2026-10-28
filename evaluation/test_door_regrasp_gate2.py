#!/usr/bin/env python3
"""停止准入 gate 2（door_regrasp_exec_core.Gate2StopObs）與 door_regrasp_mode2.RegraspAuthority2 驗證（Codex r66 §3）。
  Q1 靜止、ZERO_PUB 於 1.0 s 宣告、套用命令零 ⇒ t_cmd＝ZERO_PUB 起第 N_CONFIRM 步；t ≥ t_cmd＋OBS_S＋0.2 s 才 CONFIRMED（之前 WAIT）；
  Q2 檢查時刻之間的超標：5.02 s 底盤 x 瞬增 0.2 mm、下一步回原位 ⇒ 每 5 步取窗端點（20 Hz 線上取樣）的同一判準全數低速（漏判），
     逐物理步的 gate 2 FAILED stop_obs_moved；
  Q3 觀察資料缺步（步號跳 1）⇒ FAILED stop_obs_data_invalid（不跳過）；
  Q4 ZERO_PUB 未宣告 ⇒ 20 s 仍 WAIT、不 fresh；
  Q5 CONFIRMED 後（張爪提交前）手臂開始動 ⇒ FAILED、confirmed_fresh False；
  Q6 線上窗涵蓋判定器窗：同一筆資料以判定器 P3 定義（[ZERO_PUB, t_cmd＋OBS_S]）重算皆低速，且 gate 2 確認時刻 ≥ 判定器窗末；
  A1 RegraspAuthority2 × 真 E2 鏈：gate 2 不成立 ⇒ mode 2 請求 nack gate2:<why>、停在 mode 1；請求時成立、提交當下不成立 ⇒ gate2_at_commit；
     其餘 gate 皆真 ⇒ 1→6 正常（同 mode1 R1）；缺 gate 2 ⇒ gate2:gate_missing；
  A2 執行端核心 gate 2 FAILED 不觸發核心中止（只使 mode 2 不准入）。

    python3 evaluation/test_door_regrasp_gate2.py
"""
import json
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_check_p as CP                          # noqa: E402
import door_regrasp_exec_core as XC                    # noqa: E402
import door_regrasp_mode2 as RG2                       # noqa: E402
import test_door_regrasp_exec_core as TX               # noqa: E402
import test_door_regrasp_mode1_chain as TR             # noqa: E402

DT = 0.01


def run(g, n, zero_at=1.0, mutate=None, skip=None):
    base, arm, door = np.array([0.95, -0.92, 3.07]), np.zeros(6), 0.84
    rows, k = [], 0
    for i in range(n):
        k += 1
        if skip is not None and k == skip:
            k += 1
        t = round(k * DT, 6)
        b, a = base.copy(), arm.copy()
        if mutate:
            b, a = mutate(k, b, a)
        if zero_at is not None and t >= zero_at - 1e-9:
            g.set_zero(zero_at)
        st = g.step(k, t, b, door, a, [0.0, 0.0, 0.0], [0.0] * 6)
        rows.append([k, t, list(b), list(a), door])
        if st == 'FAILED':
            break
    return rows, k


class TQ(unittest.TestCase):
    def test_q1_confirm(self):
        g = XC.Gate2StopObs()
        rows, k = run(g, 1200)
        self.assertAlmostEqual(g.t_cmd, 1.0 + (CP.N_CONFIRM - 1) * DT, places=6)
        self.assertEqual(g.state, 'CONFIRMED')
        self.assertGreaterEqual(g.t['CONFIRMED'], g.t_cmd + CP.OBS_S + XC.MARGIN_S - 1e-9)
        g2 = XC.Gate2StopObs()
        run(g2, int(round((g.t_cmd + CP.OBS_S + XC.MARGIN_S) / DT)) - 2)
        self.assertEqual(g2.state, 'WAIT')

    def test_q2_between_checks(self):
        def m(k, b, a):
            if k == 502:
                b[0] += 0.0002
            return b, a
        g = XC.Gate2StopObs()
        rows, _ = run(g, 1300, mutate=m)
        self.assertEqual((g.state, g.why), ('FAILED', 'stop_obs_moved'))
        g_sparse = XC.Gate2StopObs()
        rows_all, _ = run(g_sparse, 1300, mutate=m)
        S = rows_all
        I = {'step': 0, 'sim_t': 1, 'base_xyth': 2, 'q_arm_meas': 3, 'door_q_rad': 4}
        full = [i for i in range(len(S)) if S[i][1] >= 1.0]
        sparse = [i for i in full if S[i][0] % 5 == 0]
        self.assertTrue(all(CP.low(CP.window_speeds(S, I, i)) for i in sparse))          # 20 Hz 端點取樣全數低速（漏判）
        self.assertFalse(all(CP.low(CP.window_speeds(S, I, i)) for i in full))           # 逐步窗口有超標

    def test_q3_gap(self):
        g = XC.Gate2StopObs()
        run(g, 1100, skip=400)
        self.assertEqual((g.state, g.why), ('FAILED', 'stop_obs_data_invalid'))

    def test_q4_no_zero(self):
        g = XC.Gate2StopObs()
        run(g, 2000, zero_at=None)
        self.assertEqual(g.state, 'WAIT')
        self.assertFalse(g.confirmed_fresh(2000))

    def test_q5_motion_after_confirm(self):
        def m(k, b, a):
            if k >= 1150:
                a[0] = 0.002 * (k - 1149) * DT
            return b, a
        g = XC.Gate2StopObs()
        run(g, 1200, mutate=m)
        self.assertEqual((g.state, g.why), ('FAILED', 'stop_obs_moved'))
        self.assertFalse(g.confirmed_fresh(1200))

    def test_q6_covers_judge(self):
        g = XC.Gate2StopObs()
        rows, _ = run(g, 1200)
        I = {'step': 0, 'sim_t': 1, 'base_xyth': 2, 'q_arm_meas': 3, 'door_q_rad': 4}
        win = [i for i in range(len(rows)) if 1.0 - 1e-9 <= rows[i][1] <= g.t_cmd + CP.OBS_S + 1e-9]
        self.assertTrue(all(CP.low(CP.window_speeds(rows, I, i)) for i in win))
        self.assertGreaterEqual(g.t['CONFIRMED'], g.t_cmd + CP.OBS_S)


class TA(unittest.TestCase):
    def _w(self, g):
        w = TR.WorldR(gates=g)
        w.auth = RG2.RegraspAuthority2('r', run_id_num=TR.RUNN, dt=TR.DT, gates=g)
        w.wx = TR.WX3.W2Exec3(w.auth, w.intake)
        return w

    def test_a1_authority(self):
        gt = {j: (lambda s: True) for j in RG2.GATED}
        w = self._w(dict(gt))
        TR.full(w)
        self.assertEqual([c['mode_seq'] for c in w.auth.commits], [1, 2, 3, 4, 5, 6])
        g = dict(gt)
        g[2] = lambda s: (False, 'stop_obs_moved')
        w = self._w(g)
        TR.full(w)
        self.assertEqual((w.auth.mode_seq, w.auth.rejections[0]['why']), (1, 'gate2:stop_obs_moved'))
        st = {'n': 0}

        def gk(s, st=st):
            st['n'] += 1
            return st['n'] <= 1
        g = dict(gt)
        g[2] = gk
        w = self._w(g)
        TR.full(w)
        self.assertEqual(w.auth.mode_seq, 1)
        self.assertTrue(str(w.auth.rejections[0]['why']).startswith('gate2_at_commit'), w.auth.rejections[0])
        g = {j: f for j, f in gt.items() if j != 2}
        w = self._w(g)
        TR.full(w)
        self.assertEqual((w.auth.mode_seq, w.auth.rejections[0]['why']), (1, 'gate2:gate_missing'))

    def test_a2_core_no_abort_on_gate2(self):
        auth = RG2.RegraspAuthority2('r', run_id_num=TR.RUNN, dt=DT)
        c = XC.RegraspExecCore(auth, json.load(open(os.path.join(HERE, 'results', 'door', 'door_regrasp_ref_v4.json'))), TX.H, dt=DT)
        c.g[2].set_zero(0.0)
        T = np.eye(4)
        for k in range(1, 30):
            b = [0.95 + (0.001 if k == 15 else 0.0), -0.92, 3.07]
            why = c.step(k, round(k * DT, 6), 0, TX.rec(k), T, b, [0.0] * 6, (0.0, 0.0), TX.Q, [0.0] * 3, [0.0] * 6)
            self.assertIsNone(why)
        self.assertEqual(c.g[2].state, 'FAILED')
        self.assertEqual(auth.gates[2](30)[0], False)


if __name__ == '__main__':
    unittest.main(verbosity=1)
