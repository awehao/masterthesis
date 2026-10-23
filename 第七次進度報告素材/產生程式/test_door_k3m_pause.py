#!/usr/bin/env python3
"""door_k3m_pause 的反例：凍結＝最後已發布目標（逐元素連續、凍結後不再更新）、減速命令以 applied u_prev 承接且在共同可行集合內
（含 Codex 例：applied 0.035255、量測 0.010）、減速→保持切換可行、完成序列（零速→確認套用→停止→觀察）與失敗路徑。

    python3 evaluation/test_door_k3m_pause.py
"""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_pause as PZ                            # noqa: E402

LIM = PZ.BaseLimits()


class Reference(unittest.TestCase):
    def test_freeze_last_published(self):
        R = PZ.PauseReference()
        with self.assertRaises(ValueError):
            R.freeze()
        T1, T2 = np.eye(4), np.eye(4)
        T2[0, 3] = 0.01
        R.published(1, 20.0, T1)
        R.published(2, 20.05, T2)
        info = R.freeze()
        self.assertEqual(info, {'seq': 2, 't': 20.05})
        np.testing.assert_array_equal(R.target(), T2)
        T3 = np.eye(4)
        T3[0, 3] = 0.02
        R.published(3, 20.10, T3)                                       # 凍結後不更新
        np.testing.assert_array_equal(R.target(), T2)
        out = R.target()
        out[0, 3] = 9.0
        np.testing.assert_array_equal(R.target(), T2)                   # 回傳複本


class Decel(unittest.TestCase):
    def test_codex_applied_vs_measured(self):
        """Codex 例：applied 0.035255、量測 0.010 ⇒ 舊律要求 0（變化 0.035 > 0.025 框）；新律由 applied 承接，變化 ≤ 0.8 框。"""
        up = [0.035255, 0.0, 0.0]
        self.assertFalse(PZ.feasible(up, [0.0, 0.0, 0.0], LIM))
        u = PZ.decel_step(up, [0.0, 0.0, 0.0], LIM)
        self.assertTrue(PZ.feasible(up, u, LIM))
        self.assertAlmostEqual(u[0], 0.035255 - 0.8 * 0.5 * 0.05)

    def test_sequence_reaches_zero_and_hold_switch_feasible(self):
        rng = np.random.default_rng(1)
        for _ in range(200):
            up = rng.uniform(-1, 1, 3) * np.array([LIM.v_lin, LIM.v_lin, LIM.v_ang])
            u = up.copy()
            for k in range(30):
                goal = [0.0, 0.0, 0.0] if k < 10 else rng.uniform(-1, 1, 3) * np.array([0.005, 0.005, 0.02])   # 之後換伺服目標
                un = PZ.decel_step(u, goal, LIM)
                self.assertTrue(PZ.feasible(u, un, LIM), (u, un))
                u = un
        z = PZ.decel_step([0.001, -0.001, 0.003], [0.0, 0.0, 0.0], LIM)
        np.testing.assert_allclose(z, 0.0)

    def test_wheel_limit_and_invalid(self):
        lim = PZ.BaseLimits(v_lin=1.0, v_ang=5.0, a_lin=100.0, a_ang=100.0)
        u = PZ.decel_step([0.0, 0.0, 0.0], [0.3, 0.3, 1.0], lim)          # 輪加速度約束生效
        self.assertTrue(PZ.feasible([0.0, 0.0, 0.0], u, lim))
        self.assertFalse(PZ.feasible([0.0, 0.0, 0.0], [0.3, 0.3, 1.0], lim))
        u2 = u
        for _ in range(50):
            u2n = PZ.decel_step(u2, [0.3, 0.3, 1.0], lim)
            self.assertTrue(PZ.feasible(u2, u2n, lim))
            u2 = u2n
        W = PZ.WL.wheel_matrix(lim.wheel.wheel_base_L)
        self.assertLessEqual(np.abs(W @ u2).max(), lim.wheel.w_lim + 1e-9)  # 輪速上限生效
        with self.assertRaises(ValueError):
            PZ.decel_step([np.nan, 0, 0], [0, 0, 0], LIM)


def M(v=(0.0, 0.0, 0.0), dr=0.0, ar=0.0):
    return {'v_body': list(v), 'door_rate': dr, 'arm_rate_applied': ar}


class Sequencer(unittest.TestCase):
    def run_nominal(self, S):
        t, seq = 0.0, 0
        while S.state in ('LOWSPEED', 'ZERO_PUB'):
            r = S.cycle(t, M(), True, base_cmd_zero=True)
            if r['publish']:
                seq += 1
                S.note_published(seq, r['zero'])
            t += 0.05
        for k in range(10):
            S.applied(t + 0.01 * k, [0.0] * 3, [0.0] * 6, seq)
        t2 = S.t_stop
        while S.state == 'STOPPED':
            t2 += 0.05
            S.cycle(t2, M(), True, base_cmd_zero=True)
        return S

    def test_nominal_done(self):
        S = self.run_nominal(PZ.PauseSequencer())
        self.assertEqual(S.state, 'DONE', S.log)
        self.assertEqual([x[1] for x in S.log], ['ZERO_PUB', 'CONFIRM', 'STOPPED', 'DONE'])

    def test_solve_failure_is_fail(self):
        S = PZ.PauseSequencer()
        S.cycle(0.0, M(), True, True)
        S.cycle(0.05, M(), False, True)
        self.assertEqual((S.state, S.why), ('FAIL', 'solve_failed'))

    def test_confirm_needs_zero_from_zero_seq(self):
        S = PZ.PauseSequencer()
        t, seq = 0.0, 0
        while S.state in ('LOWSPEED', 'ZERO_PUB'):
            r = S.cycle(t, M(), True, True)
            if r['publish']:
                seq += 1
                S.note_published(seq, r['zero'])
            t += 0.05
        for k in range(10):
            S.applied(t, [0.0] * 3, [0.0] * 6, S.zero_seq0 - 1)          # 舊序號的套用不算
        self.assertEqual(S.state, 'CONFIRM')
        for k in range(10):
            S.applied(t, [0.0, 0.0, 0.0], [2e-6] + [0.0] * 5, seq)          # 手臂未歸零
        self.assertEqual(S.state, 'CONFIRM')

    def test_moved_after_stop_and_timeout_and_invalid(self):
        S = PZ.PauseSequencer()
        self.run_nominal(S) if False else None
        S2 = PZ.PauseSequencer(PZ.PauseCfg(timeout_s=0.5))
        for k in range(20):
            S2.cycle(0.05 * k, M(v=(0.01, 0, 0)), True, False)
        self.assertEqual(S2.why, 'timeout')
        S3 = PZ.PauseSequencer()
        S3.cycle(0.0, {'v_body': [0, 0], 'door_rate': 0.0, 'arm_rate_applied': 0.0}, True, False)
        self.assertEqual(S3.why, 'measurement_invalid')
        S4 = PZ.PauseSequencer()
        t, seq = 0.0, 0
        while S4.state in ('LOWSPEED', 'ZERO_PUB'):
            r = S4.cycle(t, M(), True, True)
            if r['publish']:
                seq += 1
                S4.note_published(seq, r['zero'])
            t += 0.05
        for k in range(10):
            S4.applied(t, [0.0] * 3, [0.0] * 6, seq)
        S4.cycle(S4.t_stop + 0.05, M(dr=0.01), True, True)
        self.assertEqual(S4.why, 'moved_after_stop')


if __name__ == '__main__':
    unittest.main(verbosity=2)
