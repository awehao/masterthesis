#!/usr/bin/env python3
"""長窗隔離診斷的反例（Codex 20261023_230237 第 3 節）：
觀察器保護（door_k3ab_lw_guard）：短脈衝接觸不漏、量測停更提前結束、非有限／倒退／欄位缺失、未 arm 不觸發、正常結束後不觸發、首次觸發鎖定；
分析（door_k3ab_lw_analysis）：缺步不能判停住、低速後再運動被標出、無閂鎖（DONE／其他）、觀察窗不足、保護提前終止、首次低速與確認分列。

    python3 evaluation/test_door_k3ab_lw.py
"""
import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3ab_lw_guard as G                       # noqa: E402
import door_k3ab_lw_analysis as A                    # noqa: E402


def st(step, t, q=0.01, qdot=0.0):
    return [t, step, q] + [0.0] * 16 + [0.6, -0.53, 1.57] + [0.0] * 6 + [qdot]


def ct(step, t, unexp=0.0, wall=0.0, wrist=10.0, inv=0):
    return {'sim_t': t, 'step': step, 'n_steps': 5, 'n_invalid': inv, 'finger_bar_max_n': [0.0, 10.0], 'finger_bar_min_n': [0.0, 9.0],
            'unexpected_max_n': unexp, 'unexpected_who': 'uflite_gripper_link' if unexp else None, 'wall_max_n': wall, 'wrist_max_n': wrist}


class Guard(unittest.TestCase):
    def armed(self):
        g = G.LWGuard()
        g.on_state(st(1, 0.01), 0.0)
        g.on_contacts(ct(1, 0.01), 0.0)
        g.arm(0.05)
        return g

    def test_not_armed_never_trips(self):
        g = G.LWGuard()
        g.on_state(st(1, 0.01, q=math.radians(9)), 0.0)
        g.on_contacts(ct(1, 0.01, unexp=5.0), 0.0)
        self.assertIsNone(g.check(10.0))
        self.assertIsNone(g.trip)

    def test_short_pulse_contact(self):
        g = self.armed()
        g.on_contacts(ct(2, 0.02, unexp=0.11), 0.06)                       # 單一窗口的峰值即觸發，無持續時間要求
        self.assertEqual(g.trip['why'], 'unexpected_contact')
        self.assertEqual(g.trip['src_step'], 2)
        g.on_contacts(ct(3, 0.03, wall=9.0), 0.07)
        self.assertEqual(g.trip['why'], 'unexpected_contact')              # 首次觸發鎖定

    def test_wall_wrist_door(self):
        for kw, why in ((dict(wall=0.2), 'wall_contact'), (dict(wrist=30.0), 'wrist'), (dict(inv=1), 'contacts_invalid')):
            g = self.armed()
            g.on_contacts(ct(2, 0.02, **kw), 0.06)
            self.assertEqual(g.trip['why'], why)
        g = self.armed()
        g.on_state(st(2, 0.02, q=math.radians(5.0)), 0.06)
        self.assertEqual(g.trip['why'], 'door_range')
        g = self.armed()
        g.on_state(st(2, 0.02, qdot=-0.051), 0.06)
        self.assertEqual(g.trip['why'], 'door_rate')
        g = self.armed()
        g.on_state(st(2, 0.02, qdot=0.05), 0.06)
        self.assertIsNone(g.trip)

    def test_stale_and_invalid(self):
        g = self.armed()
        g.on_state(st(2, 0.02), 0.1)
        g.on_contacts(ct(2, 0.02), 0.1)
        self.assertIsNone(g.check(0.29))
        g.on_contacts(ct(3, 0.03), 0.25)
        self.assertEqual(g.check(0.31)['why'], 'state_stale')             # 狀態停更（接觸仍到）
        g = self.armed()
        g.on_state(st(2, 0.02), 0.1)
        g.on_state(st(2, 0.02), 0.25)                                      # 重複步號不更新年齡
        g.on_contacts(ct(3, 0.03), 0.25)
        self.assertEqual(g.check(0.31)['why'], 'state_stale')
        g = self.armed()
        g.on_state(st(2, 0.02), 0.1)
        g2 = G.LWGuard()
        g2.arm(1.0)                                                        # arm 後從未收到 ⇒ 由 arm 起算
        self.assertEqual(g2.check(1.21)['why'], 'state_stale')
        g = self.armed()
        bad = st(2, 0.02)
        bad[5] = float('nan')
        g.on_state(bad, 0.06)
        self.assertEqual(g.trip['why'], 'state_invalid')
        g = self.armed()
        g.on_state(st(5, 0.05), 0.06)
        g.on_state(st(4, 0.04), 0.07)
        self.assertEqual(g.trip['why'], 'state_invalid')
        g = self.armed()
        g.on_contacts({'sim_t': 0.1}, 0.06)
        self.assertEqual(g.trip['why'], 'contacts_invalid')

    def test_contacts_stale(self):
        g = self.armed()
        for k in range(2, 30):
            g.on_state(st(k, 0.01 * k), 0.01 * k)
        self.assertEqual(g.check(0.29)['why'], 'contacts_stale')

    def test_closed_after_normal_end(self):
        g = self.armed()
        g.close(0.1, 'door_k3a 結束：ABORTED')
        self.assertIsNone(g.check(5.0))
        g.on_contacts(ct(2, 0.02, unexp=3.0), 5.0)
        self.assertIsNone(g.trip)


COLS = ['step', 'sim_t', 'base_xyth', 'q_arm_meas', 'door_q_rad', 'door_qdot_rad_s', 'q_finger', 'leaf_contacts', 'wall_contacts', 'wrist_force_n']


def room(n, v_of, latch_step=100, drop=(), dup=(), nan=()):
    """v_of(t) → (底盤 x 速度 m/s, 門速 rad/s)；t 為相對閂鎖的時間。"""
    steps, x, qd = [], 0.6, 0.01
    for k in range(n):
        t = round(10.0 + 0.01 * k, 4)
        vb, vd = v_of(round((k - (latch_step or 0)) * 0.01, 6))
        x += vb * 0.01
        qd += vd * 0.01
        row = [k, t, [x, -0.53, 1.57], [0.0] * 6, qd, vd, [0.4, 0.4], {}, {}, 10.0]
        if k in nan:
            row[2] = [float('nan'), -0.53, 1.57]
        if k in drop:
            continue
        steps.append(row)
        if k in dup:
            steps.append(list(row))
    ev = [] if latch_step is None else [{'k2_abort_latched': {'step': latch_step}}]
    return {'steps_cols': COLS, 'steps': steps, 'events': ev}


def task(**kw):
    t = {'policy_events': [{'t': 10.97, 'abort': 'finger_contact_lost'}], 'final_phase': 'ABORTED'}
    t.update(kw)
    return t


def decay(t):
    return (0.004 if t < 2.0 else 0.0, 0.01 if t < 2.0 else 0.0)


class Analysis(unittest.TestCase):
    def test_stop_confirmed_split_times(self):
        R = A.analyse(task(), room(100 + 1101, decay))
        self.assertEqual(R['branch'], 'latched')
        self.assertEqual(R['stop_result'], 'stopped_confirmed')
        a = R['detect']['all']
        # 最後一次位移在 1.99；窗口 [t−0.1, t] 內只剩 1 步位移（底盤 0.4 mm/s、門 0.001 rad/s）時已低於門檻 ⇒ 2.08（窗口平均的性質）
        self.assertAlmostEqual(a['first_low_t'], 2.08, places=6)
        self.assertAlmostEqual(a['confirm']['run_start_t'], 2.08, places=6)
        self.assertAlmostEqual(a['confirm']['confirm_t'], 3.08, places=6)
        self.assertFalse(R['window_short'])

    def test_missing_step_blocks_stop(self):
        R = A.analyse(task(), room(100 + 1101, decay, drop=range(400, 1300, 80)))   # 每 0.8 s 缺一步 ⇒ 永遠湊不滿 1 s
        self.assertTrue(R['stop_result'].startswith('not_stopped_within_covered_'))
        R2 = A.analyse(task(), room(100 + 1101, decay, dup=range(400, 1300, 80)))
        self.assertTrue(R2['stop_result'].startswith('not_stopped_within_covered_'))
        R3 = A.analyse(task(), room(100 + 1101, decay, nan=range(400, 1300, 80)))
        self.assertTrue(R3['stop_result'].startswith('not_stopped_within_covered_'))

    def test_re_motion_flagged(self):
        def v(t):
            return (0.004 if t < 2.0 or 5.0 <= t < 5.5 else 0.0, 0.0)
        R = A.analyse(task(), room(100 + 1101, v))
        self.assertEqual(R['stop_result'], 'stopped_confirmed_then_re_motion')
        self.assertTrue(R['detect']['all']['re_motion'][0]['t'] >= 5.0 - 1e-9)

    def test_window_short_wording(self):
        R = A.analyse(task(), room(100 + 300, lambda t: (0.004, 0.0)))
        self.assertTrue(R['window_short'])
        self.assertEqual(R['stop_result'], 'not_stopped_within_covered_2.99_s')

    def test_no_latch_branches(self):
        R = A.analyse(task(final_phase='DONE', policy_events=[]), room(500, decay, latch_step=None))
        self.assertEqual((R['branch'], R['stop_result']), ('done_branch_observation_insufficient', 'observation_insufficient'))
        R = A.analyse(task(), room(500, decay, latch_step=None))
        self.assertEqual((R['branch'], R['stop_result']), ('no_latch_other', 'not_applicable'))

    def test_guard_terminated_is_unverified(self):
        R = A.analyse(task(lw_guard={'msg': '{"experiment_guard": {"why": "wrist"}}', 'wall_recv': 1.0}), room(100 + 1101, decay))
        self.assertEqual(R['branch'], 'observation_terminated_early')
        self.assertEqual(R['stop_result'], 'unverified_observation_terminated_early')

    def test_geometry_segments(self):
        def fn(qf, qd, qfg):
            return {'nonfinger': (0.004, 'x'), 'finger': (0.03, 'y'), 'base': (0.2, 'z'), 'self': (0.01, 's')}
        calls = {'k': 0}

        def parts(qf, qd, qfg):
            calls['k'] += 1
            return (0.004 if 50 <= calls['k'] < 60 else 0.006), -0.0003
        R = A.analyse(task(), room(100 + 200, decay), fn, parts)
        segs = R['geometry']['shell_bar_below_5mm_segments']
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0]['n_steps'], 10)
        self.assertAlmostEqual(R['geometry']['sampled_min_m']['shell_bar'], 0.004)


if __name__ == '__main__':
    unittest.main(verbosity=2)
