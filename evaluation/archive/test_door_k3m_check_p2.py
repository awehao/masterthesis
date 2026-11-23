#!/usr/bin/env python3
"""door_k3m_check_p2（SETTLE 完成訊號判準）反例（合成紀錄；K3-M 與暫停 P1–P4 沿用既有測試）：
  一致 ⇒ 無理由；PAUSE 前 1 s 差分 > 0.002 ⇒ settle_p2_condition_not_held_1s；窗口內缺步 ⇒ settle_p2_rate_window_missing；
  紀錄未覆蓋 1 s ⇒ settle_p2_window_not_covered；trace 的訊號與 room 不符 ⇒ settle_p2_trace_signal_mismatch；
  轉換事件的訊號 invalid ⇒ settle_p2_transition_signal_invalid；轉換前 1 s 有 invalid 列 ⇒ settle_p2_trace_not_held_1s；
  未進 PAUSE ⇒ settle_p2_unverifiable；judge 只移除父判定器的 settle_condition_not_held_1s，原始門速等其他理由保留。

    python3 evaluation/test_door_k3m_check_p2.py
"""
import copy
import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_check_p2 as C                          # noqa: E402
import door_k3m_settle_sig as SS                       # noqa: E402

COLS = ['step', 'sim_t', 'base_xyth', 'q_arm_meas', 'door_q_rad', 'leaf_contacts']
B = [0.6, -0.533, 1.5708]
QA = [0.0, 0.3, 0.6, 0.0, -1.0, 0.0]
Q8 = math.radians(8.0)
LC = {'uflite_finger1': {'bar': 3.0}, 'uflite_finger2': {'bar': 3.0}}
T_P = 20.0


def make(rate=0.0005, gap=None, start_step=1850, sig_bad=None, bump=None):
    """q(t)＝8°＋rate·(t − 18)；PAUSE 於 20.0 s；bump(t) 另加偏移。"""
    GTH = np.linalg.inv(C.KC3.KC._tcp(B, QA)) @ C.KC3.KC._H_of(Q8)      # 8° 時漂移 0
    steps, S = [], {}
    for k in range(start_step, 2010):
        if gap and gap[0] <= k < gap[1]:
            continue
        t = round(k * 0.01, 4)
        q = Q8 + rate * (t - 18.0) + (bump(t) if bump else 0.0)
        steps.append([k, t, B, QA, q, LC])
        S[k] = (t, q)
    win = SS.SettleRateWindow()
    trace = []
    for k in range(start_step, 2000, 2):
        if k not in S:
            continue
        win.on_sync(S[k][0], k, S[k][1], S[k][0])
        if k % 8 == 0 and S[k][0] >= 19.0 - 1e-9:
            trace.append({'t': S[k][0], 'step': k, 'phase': 'SETTLE', 'settle_sig': win.value(k, S[k][0])})
    if sig_bad:
        sig_bad(trace)
    sig_t = copy.deepcopy(trace[-1]['settle_sig'])
    task = {'schema': 'door_k3m_task_p2/1', 'G_T_H': GTH.tolist(), 'trace': trace,
            'policy_events': [{'t': 19.0, 'from': 'OPEN', 'to': 'SETTLE'},
                              {'t': T_P, 'from': 'SETTLE', 'to': 'END_HOLD', 'settle_sig': sig_t},
                              {'t': T_P, 'from': 'END_HOLD', 'to': 'PAUSE'}]}
    return task, {'steps_cols': COLS, 'steps': steps}


def reasons(task, room):
    why, st = [], {}
    C.check_settle_p2(task, room, why, st)
    return why


class T(unittest.TestCase):
    def test_consistent(self):
        self.assertEqual(reasons(*make()), [])

    def test_rate_exceeded(self):
        self.assertIn('settle_p2_condition_not_held_1s', reasons(*make(rate=0.003)))

    def test_short_motion_inside_window(self):
        bump = (lambda t: 0.0 if t < 19.5 else min(0.0004, 0.004 * (t - 19.5)))   # 0.1 s 內動 0.0004 rad（0.004 rad/s）
        self.assertIn('settle_p2_condition_not_held_1s', reasons(*make(bump=bump)))

    def test_gap_and_cover(self):
        self.assertIn('settle_p2_rate_window_missing', reasons(*make(gap=(1950, 1952))))
        self.assertIn('settle_p2_window_not_covered', reasons(*make(start_step=1910)))

    def test_trace_mismatch(self):
        def bad(tr):
            tr[3]['settle_sig']['rate'] += 1e-4
        self.assertIn('settle_p2_trace_signal_mismatch', reasons(*make(sig_bad=bad)))

        def nosig(tr):
            del tr[2]['settle_sig']
        self.assertIn('settle_p2_trace_signal_mismatch', reasons(*make(sig_bad=nosig)))

    def test_transition_and_trace_invalid(self):
        t, r = make()
        t['policy_events'][1]['settle_sig']['valid'] = False
        self.assertIn('settle_p2_transition_signal_invalid', reasons(t, r))

        def inval(tr):
            tr[-3]['settle_sig'] = dict(tr[-3]['settle_sig'], valid=False, why='gap')
        self.assertIn('settle_p2_trace_not_held_1s', reasons(*make(sig_bad=inval)))

    def test_no_pause(self):
        t, r = make()
        t['policy_events'] = t['policy_events'][:1]
        self.assertEqual(reasons(t, r), ['settle_p2_unverifiable'])

    def test_judge_replaces_only_parent_settle_reason(self):
        t, r = make()
        orig_km, orig_cp = C.CP.KM.judge, C.CP.check_pause
        try:
            C.CP.KM.judge = lambda *a, **k: {'why': ['door_rate', C.PARENT_SETTLE_REASON], 'stats': {}}
            C.CP.check_pause = lambda task, room, solver, why, st: (True, True)
            res = C.judge(t, r, None, None)
        finally:
            C.CP.KM.judge, C.CP.check_pause = orig_km, orig_cp
        self.assertEqual(res['why'], ['door_rate'])                   # 原始門速等理由保留
        self.assertTrue(res['stats']['parent_settle_reason_replaced'])
        self.assertTrue(res['settle_p2_ok'])
        self.assertFalse(res['pass'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
