#!/usr/bin/env python3
"""door_k3m_finger_effort（假控制器）＋door_k3m_check_w1e.check_finger_effort（合成 room）離線測試（W1e 規格 §5）。

    python3 evaluation/test_door_k3m_finger_effort.py
"""
import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import door_k3m_finger_effort as F                     # noqa: E402
import door_k3m_check_w1e as E                         # noqa: E402

J = ['finger_joint1', 'finger_joint2']
IDX = [107, 108]


class FakeCtrl:
    def __init__(self, n=109, v0=5.0, cap=None, fail_read=False):
        self.v = [100.0] * n
        for i in IDX:
            self.v[i] = v0
        self.cap, self.fail_read, self.calls = cap, fail_read, []

    def get_max_efforts(self):
        return None if self.fail_read else list(self.v)

    def set_max_efforts(self, values, joint_indices=None):
        self.calls.append((list(values), list(joint_indices)))
        for v, i in zip(values, joint_indices):
            self.v[i] = v if self.cap is None else min(v, self.cap)


class TModule(unittest.TestCase):
    def test_disabled(self):
        p = F.PauseFingerEffort(0.0, IDX, J)
        c = FakeCtrl()
        self.assertIsNone(p.on_pause_switch(c, 10, 1.0))
        self.assertEqual(c.calls, [])

    def test_switch_and_readback(self):
        c = FakeCtrl()
        ini = F.read_init(c, IDX, J)
        self.assertEqual(ini['readback'], [5.0, 5.0])
        p = F.PauseFingerEffort(20.0, IDX, J)
        r = p.on_pause_switch(c, 5419, 54.21)
        self.assertEqual(c.calls, [([20.0, 20.0], IDX)])                          # 只改兩指
        self.assertEqual(r['old'], [5.0, 5.0])
        self.assertEqual(r['readback'], [20.0, 20.0])
        self.assertTrue(r['ok'])
        self.assertEqual([c.v[i] for i in range(109) if i not in IDX], [100.0] * 107)
        self.assertIs(p.on_pause_switch(c, 6000, 60.0), r)                          # 只設一次
        self.assertEqual(len(c.calls), 1)

    def test_readback_mismatch_not_ok(self):
        c = FakeCtrl(cap=8.0)
        r = F.PauseFingerEffort(20.0, IDX, J).on_pause_switch(c, 1, 0.1)
        self.assertFalse(r['ok'])
        c = FakeCtrl(fail_read=True)
        r = F.PauseFingerEffort(20.0, IDX, J).on_pause_switch(c, 1, 0.1)
        self.assertFalse(r['ok'])
        self.assertIsNone(r['readback'])

    def test_periodic(self):
        c = FakeCtrl()
        p = F.PauseFingerEffort(20.0, IDX, J)
        self.assertIsNone(p.periodic(c, 0, 0.0))                                    # 切換前不讀
        p.on_pause_switch(c, 100, 1.0)
        self.assertIsNone(p.periodic(c, 150, 1.5))
        self.assertTrue(p.periodic(c, 200, 2.0)['ok'])
        c.v[IDX[1]] = 5.0                                                           # 中途被改回 ⇒ 該筆 ok False
        self.assertFalse(p.periodic(c, 300, 3.0)['ok'])
        self.assertEqual(len(p.record['periodic']), 2)


def room_ok(V=20.0, t_sw=54.21, t_end=62.0):
    per = [{'step': int((t_sw + k) * 100), 't': t_sw + k, 'readback': [V, V], 'ok': True} for k in range(1, int(t_end - t_sw) + 1)]
    return {'finger_effort': {'init': {'joints': J, 'indices': IDX, 'readback': [5.0, 5.0], 'ok': True}, 'arg_pause_max_effort': V,
                              'pause': {'step': 5421, 't': t_sw, 'joints': J, 'indices': IDX, 'old': [5.0, 5.0], 'set': V,
                                        'readback': [V, V], 'ok': True, 'periodic': per}, 'tol': 1e-3},
            'k3m': {'pause_switch': {'switch_step': 5421, 'switch_t': t_sw}}, 'steps': [[6200, t_end]]}


EC = {'pause_finger_max_effort': 20.0, 'original_effort': 5.0, 'joints': J}


def chk(room, ec=EC):
    w, st = [], {}
    return E.check_finger_effort(room, ec, w, st), w


class TJudge(unittest.TestCase):
    def test_ok(self):
        ok, w = chk(room_ok())
        self.assertTrue(ok, w)

    def test_counterexamples(self):
        cases = [
            (lambda r: r.pop('finger_effort'), 'finger_effort_missing'),
            (lambda r: r['finger_effort'].__setitem__('arg_pause_max_effort', 10.0), 'finger_effort_arg_mismatch'),
            (lambda r: r['finger_effort']['init'].__setitem__('readback', [5.0, 4.0]), 'finger_effort_init_invalid'),
            (lambda r: r['finger_effort']['init'].__setitem__('ok', False), 'finger_effort_init_invalid'),
            (lambda r: r['finger_effort'].__setitem__('pause', None), 'finger_effort_pause_missing'),
            (lambda r: r['finger_effort']['pause'].__setitem__('step', 5422), 'finger_effort_not_at_pause_switch'),
            (lambda r: r['finger_effort']['pause'].__setitem__('readback', [20.0, 8.0]), 'finger_effort_not_effective'),
            (lambda r: r['finger_effort']['pause'].__setitem__('ok', False), 'finger_effort_not_effective'),
            (lambda r: r['finger_effort']['pause'].__setitem__('readback', None), 'finger_effort_not_effective'),
            (lambda r: r['finger_effort']['pause'].__setitem__('old', [20.0, 20.0]), 'finger_effort_old_mismatch'),
            (lambda r: r['finger_effort']['pause'].__setitem__('periodic', []), 'finger_effort_periodic_short_0'),
            (lambda r: r['finger_effort']['pause']['periodic'][2].__setitem__('readback', [20.0, 5.0]), 'finger_effort_periodic_mismatch'),
        ]
        for f, tag in cases:
            r = copy.deepcopy(room_ok())
            f(r)
            ok, w = chk(r)
            self.assertFalse(ok, tag)
            self.assertIn(tag, w)

    def test_registration(self):
        import door_k3m_check_w1c as W1C
        orig = W1C.judge_with
        W1C.judge_with = lambda *a, **k: {'pass': True, 'why': [], 'stats': {}}
        try:
            self.assertIn('w1e_executor_change_not_registered', E.judge_with({}, room_ok(), None, None, None, {})['why'])
            self.assertTrue(E.judge_with({}, room_ok(), None, None, None, {'executor_change': EC})['pass'])
        finally:
            W1C.judge_with = orig


if __name__ == '__main__':
    unittest.main(verbosity=1)
