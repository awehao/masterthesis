#!/usr/bin/env python3
"""door_k3m_finger_effort（假控制器）＋door_k3m_check_w1e.check_finger_effort（合成 room）離線測試（W1e 規格 §5；r2＝Codex 20261025_024339）。

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
N = 109
NAMES = [f'dof{i}' for i in range(N - 2)] + J
IDX = [N - 2, N - 1]


class FakeCtrl:
    def __init__(self, v0=5.0, cap=None, fail_read=False, set_raises=False, touch_other=False):
        self.v = [100.0] * N
        for i in IDX:
            self.v[i] = v0
        self.cap, self.fail_read, self.set_raises, self.touch_other, self.calls = cap, fail_read, set_raises, touch_other, []

    def get_max_efforts(self):
        return None if self.fail_read else list(self.v)

    def set_max_efforts(self, values, joint_indices=None):
        if self.set_raises:
            raise RuntimeError('boom')
        self.calls.append((list(values), list(joint_indices)))
        for v, i in zip(values, joint_indices):
            self.v[i] = v if self.cap is None else min(v, self.cap)
        if self.touch_other:
            self.v[3] = 1.0


class TModule(unittest.TestCase):
    def test_init(self):
        r = F.read_init(FakeCtrl(), IDX, J, NAMES, 5.0)
        self.assertIs(r['ok'], True)
        self.assertEqual(r['readback'], [5.0, 5.0])
        for kw, args, why in ((dict(fail_read=True), (IDX, J, NAMES, 5.0), 'get_max_efforts 回傳 None'),
                              (dict(v0=4.0), (IDX, J, NAMES, 5.0), 'finger_init_readback_not_original'),
                              (dict(v0=float('nan')), (IDX, J, NAMES, 5.0), 'finger_init_readback_not_original'),
                              ({}, ([0, 1], J, NAMES, 5.0), 'finger_index_or_name_mapping_invalid'),
                              ({}, ([IDX[0], IDX[0]], J, NAMES, 5.0), 'finger_index_or_name_mapping_invalid'),
                              ({}, (IDX, J, NAMES[:-1], 5.0), 'finger_index_or_name_mapping_invalid')):
            r = F.read_init(FakeCtrl(**kw), *args)
            self.assertIs(r['ok'], False, why)
            self.assertEqual(r['why'], why)

    def test_disabled(self):
        c = FakeCtrl()
        self.assertIsNone(F.PauseFingerEffort(0.0, IDX, J).on_pause_switch(c, 10, 1.0))
        self.assertEqual(c.calls, [])

    def test_switch_ok_only_fingers(self):
        c = FakeCtrl()
        p = F.PauseFingerEffort(20.0, IDX, J)
        r = p.on_pause_switch(c, 5419, 54.21)
        self.assertEqual(c.calls, [([20.0, 20.0], IDX)])
        self.assertIs(r['ok'], True)
        self.assertEqual(r['old'], [5.0, 5.0])
        self.assertEqual(r['readback'], [20.0, 20.0])
        self.assertEqual(r['full_before'][:N - 2], r['full_after'][:N - 2])
        self.assertIs(p.on_pause_switch(c, 6000, 60.0), r)
        self.assertEqual(len(c.calls), 1)                                           # 不重試

    def test_switch_failures(self):
        for kw, why in ((dict(cap=8.0), 'finger_readback_not_set_value'),
                        (dict(set_raises=True), 'set_max_efforts 例外：RuntimeError'),
                        (dict(touch_other=True), 'non_finger_max_effort_changed'),
                        (dict(fail_read=True), 'get_max_efforts 回傳 None')):
            c = FakeCtrl(**kw)
            r = F.PauseFingerEffort(20.0, IDX, J).on_pause_switch(c, 1, 0.1)
            self.assertIs(r['ok'], False, why)
            self.assertEqual(r['why'], why)
            self.assertLessEqual(len(c.calls), 1)                                   # 不重試、不加力

    def test_periodic(self):
        c = FakeCtrl()
        p = F.PauseFingerEffort(20.0, IDX, J)
        self.assertIsNone(p.periodic(c, 0, 0.0))
        p.on_pause_switch(c, 100, 1.0)
        self.assertIsNone(p.periodic(c, 150, 1.5))
        e = p.periodic(c, 200, 2.0)
        self.assertIs(e['ok'], True)
        self.assertIs(e['others_same'], True)
        c.v[IDX[1]] = 5.0
        e = p.periodic(c, 300, 3.0)
        self.assertIs(e['ok'], False)
        self.assertEqual(e['why'], 'finger_readback_reverted')
        c.v[IDX[1]] = 20.0
        c.v[7] = 0.5
        e = p.periodic(c, 400, 4.0)
        self.assertEqual(e['why'], 'non_finger_max_effort_changed')


def room_ok(V=20.0, s_sw=5421, t_end=62.0):
    t_sw = s_sw / 100.0
    full0 = [100.0] * (N - 2) + [5.0, 5.0]
    full1 = [100.0] * (N - 2) + [V, V]
    per = []
    for k in range(1, int(t_end - t_sw) + 1):
        s = s_sw + 100 * k
        per.append({'step': s, 't': s / 100.0, 'readback': [V, V], 'others_same': True, 'ok': True, 'why': None})
    steps = [[s, s / 100.0] for s in range(5000, int(t_end * 100) + 1)]
    return {'finger_effort': {'init': {'joints': J, 'indices': IDX, 'names': NAMES, 'original': 5.0, 'full': full0,
                                       'readback': [5.0, 5.0], 'ok': True, 'why': None},
                              'arg_pause_max_effort': V, 'arg_original_effort': 5.0,
                              'pause': {'step': s_sw, 't': t_sw, 'joints': J, 'indices': IDX, 'old': [5.0, 5.0], 'set': V,
                                        'readback': [V, V], 'full_before': full0, 'full_after': full1, 'ok': True, 'why': None,
                                        'periodic': per}, 'pause_switch_failed': None, 'tol': 1e-3},
            'k3m': {'pause_switch': {'switch_step': s_sw, 'switch_t': t_sw}}, 'steps': steps}


EC = {'pause_finger_max_effort': 20.0, 'original_effort': 5.0, 'joints': J}


def chk(room, ec=EC):
    w, st = [], {}
    return E.check_finger_effort(room, ec, w, st), w


def P(r):
    return r['finger_effort']['pause']


class TJudge(unittest.TestCase):
    def test_ok(self):
        ok, w = chk(room_ok())
        self.assertTrue(ok, w)

    def test_codex_counterexamples(self):
        r = room_ok()                                                               # 重複同一筆 periodic 七次
        P(r)['periodic'] = [copy.deepcopy(P(r)['periodic'][0]) for _ in range(7)]
        ok, w = chk(r)
        self.assertFalse(ok)
        self.assertTrue(any(x.startswith('finger_effort_periodic_invalid_') for x in w), w)
        self.assertTrue(any(x.startswith('finger_effort_periodic_short_') for x in w), w)
        r = room_ok()                                                               # 切換時間 99999
        P(r)['t'] = 99999.0
        ok, w = chk(r)
        self.assertIn('finger_effort_not_at_pause_switch', w)
        r = room_ok()                                                               # 索引 [0,1]
        r['finger_effort']['init']['indices'] = [0, 1]
        P(r)['indices'] = [0, 1]
        ok, w = chk(r)
        self.assertIn('finger_effort_init_invalid', w)

    def test_counterexamples(self):
        cases = [
            (lambda r: r.pop('finger_effort'), 'finger_effort_missing'),
            (lambda r: r['finger_effort'].__setitem__('arg_pause_max_effort', 10.0), 'finger_effort_arg_mismatch'),
            (lambda r: r['finger_effort'].__setitem__('arg_original_effort', 6.0), 'finger_effort_arg_mismatch'),
            (lambda r: r['finger_effort']['init'].__setitem__('readback', [5.0, 4.0]), 'finger_effort_init_invalid'),
            (lambda r: r['finger_effort']['init'].__setitem__('ok', 1), 'finger_effort_init_invalid'),
            (lambda r: r['finger_effort']['init'].__setitem__('names', NAMES[:-2] + J[::-1]), 'finger_effort_init_invalid'),
            (lambda r: r['finger_effort'].__setitem__('pause', None), 'finger_effort_pause_missing'),
            (lambda r: P(r).__setitem__('step', 5422), 'finger_effort_not_at_pause_switch'),
            (lambda r: r['k3m']['pause_switch'].__setitem__('switch_t', 55.0), 'finger_effort_not_at_pause_switch'),
            (lambda r: P(r).__setitem__('readback', [20.0, 8.0]), 'finger_effort_not_effective'),
            (lambda r: P(r).__setitem__('ok', 'True'), 'finger_effort_not_effective'),
            (lambda r: P(r).__setitem__('readback', [20.0, float('nan')]), 'finger_effort_not_effective'),
            (lambda r: P(r).__setitem__('old', [20.0, 20.0]), 'finger_effort_old_mismatch'),
            (lambda r: P(r)['full_after'].__setitem__(5, 1.0), 'finger_effort_non_finger_changed_or_missing'),
            (lambda r: P(r).__setitem__('full_before', None), 'finger_effort_non_finger_changed_or_missing'),
            (lambda r: P(r).__setitem__('indices', [IDX[1], IDX[0]]), 'finger_effort_index_mismatch'),
            (lambda r: P(r).__setitem__('periodic', []), 'finger_effort_periodic_short_0'),
            (lambda r: P(r)['periodic'][2].__setitem__('readback', [20.0, 5.0]), 'finger_effort_periodic_invalid_1'),
            (lambda r: P(r)['periodic'][2].__setitem__('ok', 1), 'finger_effort_periodic_invalid_1'),
            (lambda r: P(r)['periodic'][2].__setitem__('others_same', False), 'finger_effort_periodic_invalid_1'),
            (lambda r: P(r)['periodic'][2].__setitem__('t', P(r)['periodic'][2]['t'] + 0.005), 'finger_effort_periodic_invalid_1'),
            (lambda r: r['finger_effort'].__setitem__('pause_switch_failed', {'why': 'x'}), 'finger_effort_switch_failed'),
        ]
        for f, tag in cases:
            r = room_ok()
            f(r)
            ok, w = chk(r)
            self.assertFalse(ok, tag)
            self.assertIn(tag, w, tag)

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
