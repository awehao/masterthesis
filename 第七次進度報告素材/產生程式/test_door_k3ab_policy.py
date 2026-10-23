#!/usr/bin/env python3
"""door_k3ab_policy 的離線反例（沿用 test_door_k3a_policy 的合成世界；每輪 5 個物理步，配對流逐步送入，可超前本輪狀態步號）。

    python3 evaluation/test_door_k3ab_policy.py
"""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3ab_policy as AB                          # noqa: E402
import test_door_k3a_policy as TP                      # noqa: E402
from door_target import rotate_about_hinge            # noqa: E402

E12 = np.array([0.0, -1.0, 0.0])


class ABSim(TP.Sim):
    """f(sim, step) → (f1, f2)；ahead＝送出超過本輪狀態步號的配對數（未來資料）；hook(sim, m) 可破壞量測。"""

    def __init__(self, f=None, ahead=2, hook=None, inject=None):
        super().__init__(inject)
        self.cfg = AB.K3ABConfig()
        self.pol = AB.K3ABPolicy(self.cfg, TP.T_HG())
        self.f = f or (lambda sim, s: (2.0, 2.0))
        self.ahead = ahead
        self.hook = hook
        self.sent = -1
        self.rows = []
        self.wall_age = 0.02

    def meas(self):
        m = super().meas()
        s = int(round(self.t * 100))
        m.state_step = s
        new = []
        for k in range(self.sent + 1, s + self.ahead + 1):
            f1, f2 = self.f(self, k)
            new.append([k, round(k * 0.01, 4), f1, f2, True])
        msg = {'pairs': new}
        if self.hook:
            msg = self.hook(self, m, msg)
        if msg is not None:
            self.pol.pairs.push(msg, 0.0)
            if msg.get('pairs'):
                self.sent = max(self.sent, max(int(e[0]) for e in msg['pairs'] if isinstance(e, list) and e and isinstance(e[0], int)))
        m.pairs_age_wall = self.wall_age
        return m

    def run(self, t_max=120.0):
        orig = self.pol.step

        def wrapped(m, dt):
            a = orig(m, dt)
            self.rows.append((m.t, m.state_step, dict(self.pol.row or {}), a['target'], self.pol.q_ref, a))
            return a
        self.pol.step = wrapped
        return super().run(t_max)


def imbalance(sim, s):
    return (1.0, 4.0) if sim.pol.phase in ('OPEN', 'SETTLE', 'END_HOLD') else (2.0, 2.0)


def rows_of(s, ph0):
    return [r for r in s.rows if r[2].get('ph0') == ph0]


class Policy(unittest.TestCase):
    def test_balanced_done_and_published_equals_base(self):
        s = ABSim()
        pol = s.run()
        self.assertEqual(pol.phase, 'DONE', pol.events[-3:])
        for t, st, row, T, _, _ in s.rows:
            if T is not None and row.get('base') is not None:
                np.testing.assert_allclose(T, row['base'], atol=0)
        self.assertEqual(pol.bal.delta, 0.0)

    def test_imbalance_moves_along_e12_and_bounded(self):
        s = ABSim(f=imbalance)
        pol = s.run()
        self.assertEqual(pol.phase, 'DONE', pol.events[-3:])
        prev = None
        for t, st, row, T, _, _ in rows_of(s, 'OPEN') + rows_of(s, 'SETTLE'):
            base, d, e12 = np.asarray(row['base']), row['delta'], np.asarray(row['e12'])
            np.testing.assert_allclose(e12, base[:3, :3] @ E12, atol=1e-15)
            np.testing.assert_allclose(T[:3, :3], base[:3, :3], atol=0)
            np.testing.assert_allclose(T[:3, 3] - base[:3, 3], d * e12, atol=1e-15)
            self.assertLessEqual(abs(d), 1e-3 + 1e-15)
            if prev is not None:
                self.assertLessEqual(abs(d - prev[1]), 1e-3 * (t - prev[0]) + 1e-12)
            prev = (t, d)
        self.assertGreater(pol.bal.delta, 0.0)                               # F2 > F1 ⇒ 往 finger2 側
        self.assertLessEqual(pol.bal.delta, 1e-3)

    def test_first_open_continuous_and_zero_before(self):
        s = ABSim(f=lambda sim, k: (0.5, 9.0))                              # 進 OPEN 前就有承載差
        s.run()
        hold = rows_of(s, 'HOLD')
        first = rows_of(s, 'OPEN')[0]
        self.assertTrue(all(r[2]['delta'] == 0.0 for r in hold))
        self.assertEqual(first[2]['delta'], 0.0)
        np.testing.assert_allclose(first[3], hold[-1][3], atol=0)          # 實際發布目標連續
        ev = next(e['balance_start'] for e in s.pol.events if 'balance_start' in e)
        self.assertEqual(ev['s0'], first[1])
        self.assertEqual(s.pol.bal.records[0]['step'], ev['s0'] + 1)       # ≤ s0 不積分

    def test_base_is_pure_hinge_rotation(self):
        """基礎目標＝首筆 OPEN 基礎目標繞鉸鏈轉 (q_ref − q_ref₀)，與 Δ 無關（Δ 不併入基礎序列）。"""
        s = ABSim(f=imbalance)
        s.run()
        op = [r for r in s.rows if r[2].get('ph0') in ('OPEN', 'SETTLE')]
        T0, q0 = np.asarray(op[0][2]['base']), op[0][4]
        for r in op:
            Te = rotate_about_hinge(T0, np.zeros(3), TP.AX, r[4] - q0)
            np.testing.assert_allclose(np.asarray(r[2]['base']), Te, atol=1e-12)
        self.assertGreater(max(r[2]['delta'] for r in op), 1e-4)

    def test_end_hold_frozen_and_done_stops(self):
        s = ABSim(f=imbalance)
        pol = s.run()
        st_rows = rows_of(s, 'SETTLE')
        eh = [r for r in rows_of(s, 'END_HOLD') if r[3] is not None]          # 轉入 DONE 那輪不發布
        self.assertTrue(eh)
        for r in eh:
            np.testing.assert_allclose(r[3], st_rows[-1][3], atol=0)
            self.assertEqual(r[2]['delta'], st_rows[-1][2]['delta'])
        self.assertLessEqual(pol.bal.records[-1]['step'], st_rows[-1][1])  # END_HOLD 不再消費
        self.assertTrue(all(r[2]['consumed'] is None for r in eh))
        last = s.rows[-1]
        self.assertEqual(pol.phase, 'DONE')
        self.assertIsNone(last[3])                                          # DONE：停止發布、停止求解器
        self.assertTrue(last[5]['stop_solver'] and last[5]['finish'])

    def test_no_future_data_consumed(self):
        s = ABSim(f=imbalance, ahead=4)
        s.run(t_max=25.0)
        for t, st, row, T, _, _ in s.rows:
            c = row.get('consumed')
            if c:
                self.assertLessEqual(c[1], st)
        cons = [r[2]['consumed'] for r in s.rows if r[2].get('consumed')]
        for a, b in zip(cons, cons[1:]):
            self.assertEqual(b[0], a[1] + 1)                                # 每步只消費一次、連續
        self.assertTrue(any(k > s.rows[-1][1] for k in s.pol.pairs.pending))

    def test_stale_wall_aborts(self):
        def hook(sim, m, msg):
            if sim.pol.phase == 'OPEN' and sim.pol.dtk is not None:
                sim.wall_age = 0.25
            return msg
        s = ABSim(hook=hook)
        pol = s.run()
        self.assertEqual(pol.abort, 'balance_meas_invalid')
        self.assertEqual(pol.bal.stopped['why'], 'stale_wall')
        self.assertTrue(all(a['target'] is None for t, p, a, _ in s.acts if t > pol.t_abort))

    def test_stale_sim_aborts(self):
        def hook(sim, m, msg):
            if sim.pol.phase == 'OPEN' and sim.pol.dtk is not None and sim.t > sim.pol.open_entry['t'] + 0.3:
                return {'pairs': []}                                        # 配對停更（牆鐘年齡維持正常）
            return msg
        s = ABSim(hook=hook)
        pol = s.run()
        self.assertEqual(pol.abort, 'balance_meas_invalid')
        self.assertEqual(pol.bal.stopped['why'], 'stale_sim')

    def test_gap_conflict_malformed_abort(self):
        def gap(sim, m, msg):
            if sim.pol.phase == 'OPEN' and sim.pol.dtk is not None and sim.t > sim.pol.open_entry['t'] + 0.2:
                msg = {'pairs': [e for e in msg['pairs'] if e[0] % 7 != 0]}
            return msg

        def conflict(sim, m, msg):
            if sim.pol.phase == 'OPEN' and sim.pol.dtk is not None and sim.t > sim.pol.open_entry['t'] + 0.2:
                msg = {'pairs': msg['pairs'] + [[sim.sent - 1, round((sim.sent - 1) * 0.01, 4), 9.0, 9.0, True]]}
            return msg

        def malformed(sim, m, msg):
            if sim.pol.phase == 'OPEN' and sim.pol.dtk is not None and sim.t > sim.pol.open_entry['t'] + 0.2:
                return {'pairs': [[1, 2, 3]]}
            return msg
        for hook, why in ((gap, 'gap'), (conflict, 'buffer_conflict'), (malformed, 'buffer_malformed')):
            s = ABSim(hook=hook)
            pol = s.run()
            self.assertEqual(pol.abort, 'balance_meas_invalid', why)
            self.assertEqual(pol.bal.stopped['why'], why)

    def test_invalid_pair_aborts(self):
        def f(sim, k):
            return (None, 2.0) if sim.pol.phase == 'OPEN' and sim.pol.dtk is not None and sim.t > sim.pol.open_entry['t'] + 0.2 else (2.0, 2.0)
        s = ABSim(f=f)
        pol = s.run()
        self.assertEqual(pol.abort, 'balance_meas_invalid')
        self.assertEqual(pol.bal.stopped['why'], 'invalid')

    def test_parent_safety_still_first(self):
        def inj(sim, m):
            if sim.pol.phase == 'OPEN' and sim.pol.dtk is not None and sim.t > sim.pol.open_entry['t'] + 0.3:
                m.wrist_max = 31.0
        s = ABSim(f=imbalance, inject=inj)
        pol = s.run()
        self.assertEqual(pol.abort, 'wrist_force')
        self.assertIsNone(pol.bal.stopped)

    def test_drift_against_original_grasp(self):
        s = ABSim(f=lambda sim, k: (0.5, 9.5))
        pol = s.run()
        G0 = pol.G_T_H.copy()
        self.assertEqual(pol.phase, 'DONE')
        np.testing.assert_allclose(pol.G_T_H, G0, atol=0)
        self.assertGreaterEqual(pol.max_drift[0], 0.9e-3)                  # TCP 跟目標 ⇒ 漂移含 Δ（不被重新初始化吸收）

    def test_rotated_handle_direction(self):
        with __import__('test_door_k3ax_policy').rotated_handle(30.0):
            s = ABSim(f=imbalance)
            pol = s.run()
            self.assertEqual(pol.phase, 'DONE', pol.events[-3:])
            r = rows_of(s, 'SETTLE')[-1][2]
            np.testing.assert_allclose(r['e12'], np.asarray(r['base'])[:3, :3] @ E12, atol=1e-15)

    def test_e12_follows_tool_frame_not_world_z(self):
        from scipy.spatial.transform import Rotation
        T = np.eye(4)
        T[:3, :3] = Rotation.from_rotvec([0.3, -0.5, 0.9]).as_matrix()
        e = AB.K3ABPolicy.e12_world(T)
        np.testing.assert_allclose(e, -T[:3, 1], atol=1e-15)
        self.assertGreater(np.abs(e - np.array([0.0, 0.0, -1.0])).max(), 0.1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
