#!/usr/bin/env python3
"""door_k3m_check_p4b2（只刪零長度 END_HOLD 事件）反例：
  同時刻 SETTLE→END_HOLD＋END_HOLD→PAUSE ⇒ 刪 1 筆；END_HOLD 與 PAUSE 間隔 > 1e-9 ⇒ 不刪；END_HOLD 後非 PAUSE ⇒ 不刪；
  既有各趟（pause_1、p2c_2、p3_1、p3b_1、p4_1）以 p4b2 重判：除 p4b 已有的理由外不新增理由、也不移除 phase_sequence 以外的理由。

    python3 evaluation/test_door_k3m_check_p4b2.py
"""
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_check_p4b as C4B                      # noqa: E402
import door_k3m_check_p4b2 as C                       # noqa: E402


def ev(*xs):
    return {'policy_events': [{'t': t, 'from': f, 'to': to} for t, f, to in xs]}


class T(unittest.TestCase):
    def test_drop_rules(self):
        t, n = C.drop_zero_length_end_hold(ev((1.0, 'OPEN', 'SETTLE'), (2.0, 'SETTLE', 'END_HOLD'), (2.0, 'END_HOLD', 'PAUSE'), (9.0, 'PAUSE', 'DONE')))
        self.assertEqual(n, 1)
        self.assertEqual([e['to'] for e in t['policy_events']], ['SETTLE', 'PAUSE', 'DONE'])
        _, n = C.drop_zero_length_end_hold(ev((2.0, 'SETTLE', 'END_HOLD'), (2.1, 'END_HOLD', 'PAUSE')))
        self.assertEqual(n, 0)
        _, n = C.drop_zero_length_end_hold(ev((2.0, 'SETTLE', 'END_HOLD'), (2.0, 'END_HOLD', 'DONE')))
        self.assertEqual(n, 0)

    def test_earlier_runs_unchanged(self):
        for r in ('door_k3m_pause_p2c_2', 'door_k3m_pause_p3_1', 'door_k3m_pause_p3b_1', 'door_k3m_pause_p4_1'):
            d = os.path.join(HERE, 'runs', r)
            if not os.path.exists(os.path.join(d, 'task.json')):
                continue
            a = [json.load(open(os.path.join(d, f))) for f in ('task.json', 'room_run.json', 'park_gate.json', 'align_solver.json')]
            norm = (lambda ws: {('phase_sequence' if w.startswith('phase_sequence') else w) for w in ws})   # 理由字串內含序列內容
            w4b = norm(C4B.judge(a[0], a[1], a[2], a[3])['why'])
            w2 = norm(C.judge(a[0], a[1], a[2], a[3])['why'])
            self.assertTrue(w2 <= w4b, (r, w2 - w4b))
            self.assertTrue(all(x.startswith('phase_sequence') or x == 'end_hold_too_short_or_missing' for x in w4b - w2), (r, w4b - w2))


if __name__ == '__main__':
    unittest.main(verbosity=2)
