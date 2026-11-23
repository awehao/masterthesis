#!/usr/bin/env python3
"""door_k3b_finger_restore 反例（假控制器）。
  F1 正常：暫停切換 5→20 N 後恢復 20→5 N ⇒ ok；之後 periodic 以 5 N 核對通過（不誤判 reverted）。
  F2 set_max_efforts 拋例外 ⇒ ok False、參考仍 20 N（periodic 讀到 20 仍通過）。
  F3 控制器只改到一指 ⇒ finger_readback_not_restored；之後 periodic 以 20 N 核 ⇒ finger_readback_reverted（走既有中止）。
  F4 非手指上限被改 ⇒ non_finger_max_effort_changed；F5 沒有暫停切換紀錄 ⇒ no_pause_switch_record；F6 恢復兩次 ⇒ restore_twice。

    python3 evaluation/test_door_k3b_finger_restore.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import door_k3b_finger_restore as FR                   # noqa: E402
import door_k3m_finger_effort as FEFF                  # noqa: E402

IDX = [7, 8]


class Ctrl:
    def __init__(self, mode='ok'):
        self.v = [50.0] * 7 + [5.0, 5.0]
        self.mode = mode
        self.n_set = 0

    def get_max_efforts(self):
        return list(self.v)

    def set_max_efforts(self, vals, joint_indices):
        self.n_set += 1
        if self.mode == 'raise' and self.n_set > 1:
            raise RuntimeError('x')
        for i, x in zip(joint_indices, vals):
            self.v[i] = x
        if self.mode == 'one' and self.n_set > 1:
            self.v[8] = 20.0
        if self.mode == 'other' and self.n_set > 1:
            self.v[0] = 1.0


def switched(mode='ok'):
    c = Ctrl(mode)
    fe = FEFF.PauseFingerEffort(20.0, IDX, ['finger_joint1', 'finger_joint2'])
    r = fe.on_pause_switch(c, 100, 1.0)
    assert r['ok'], r
    return c, fe


class TF(unittest.TestCase):
    def test_f1_normal(self):
        c, fe = switched()
        r = FR.restore(fe, c, 5.0, 500, 5.0)
        self.assertTrue(r['ok'], r)
        self.assertEqual(c.v[7:], [5.0, 5.0])
        p = fe.periodic(c, 600, 6.0)
        self.assertTrue(p['ok'], p)

    def test_f2_raise(self):
        c, fe = switched('raise')
        r = FR.restore(fe, c, 5.0, 500, 5.0)
        self.assertFalse(r['ok'])
        self.assertIn('例外', r['why'])
        self.assertTrue(fe.periodic(c, 600, 6.0)['ok'])

    def test_f3_partial(self):
        c, fe = switched('one')
        r = FR.restore(fe, c, 5.0, 500, 5.0)
        self.assertEqual(r['why'], 'finger_readback_not_restored')
        self.assertEqual(fe.periodic(c, 600, 6.0)['why'], 'finger_readback_reverted')

    def test_f4_f5_f6(self):
        c, fe = switched('other')
        self.assertEqual(FR.restore(fe, c, 5.0, 500, 5.0)['why'], 'non_finger_max_effort_changed')
        fe2 = FEFF.PauseFingerEffort(20.0, IDX, ['finger_joint1', 'finger_joint2'])
        self.assertEqual(FR.restore(fe2, Ctrl(), 5.0, 1, 0.1)['why'], 'no_pause_switch_record')
        c, fe = switched()
        FR.restore(fe, c, 5.0, 500, 5.0)
        self.assertEqual(FR.restore(fe, c, 5.0, 501, 5.01)['why'], 'restore_twice')


if __name__ == '__main__':
    unittest.main(verbosity=1)
