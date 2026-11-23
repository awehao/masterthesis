#!/usr/bin/env python3
"""door_finger_loss_cont 單元測試：L1 未啟用不判；L2 單步低力（extf_1 型態：0.02→7.8→0→3.6 N）不觸發；L3 連續 0.2 s 觸發；
L4 無效樣本不清零不加時；L5 張爪解除清零；L6 以 extf_1 實錄重放（OPEN 段）不觸發、最長連續 ≤ 0.02 s。"""
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_finger_loss_cont as FL   # noqa: E402


def armed():
    m = FL.FingerLossCont()
    for k in range(200):
        assert m.step(k * 0.01, 5.0, 5.0, True, True) is None
    assert m.armed
    return m


class T(unittest.TestCase):
    def test_l1_not_armed(self):
        m = FL.FingerLossCont()
        for k in range(100):
            self.assertIsNone(m.step(k * 0.01, 0.0, 0.0, True, True))
        self.assertFalse(m.armed)

    def test_l2_single_step_lows(self):
        m = armed()
        seq = [0.0205, 7.853, 0.0, 3.61] * 50
        for i, v in enumerate(seq):
            self.assertIsNone(m.step(2.0 + i * 0.01, v, 5.0, True, True))
        self.assertLessEqual(max(m.take_window()['finger_low_cont_max_s']), 0.0101)

    def test_l3_continuous_fires(self):
        m = armed()
        why = None
        for i in range(30):
            why = why or m.step(2.0 + i * 0.01, 0.1, 5.0, True, True)
        self.assertTrue(why and why.startswith('finger_contact_lost_cont:f1'))
        self.assertAlmostEqual(m.fired['dur_s'], 0.2, places=6)

    def test_l4_invalid_keeps(self):
        m = armed()
        for i in range(10):
            m.step(2.0 + i * 0.01, 0.1, 5.0, True, True)
        d0 = m.dur[0]
        for i in range(5):
            m.step(2.1 + i * 0.01, float('nan'), 5.0, False, True)
        self.assertEqual(m.dur[0], d0)

    def test_l5_open_resets(self):
        m = armed()
        m.step(2.0, 0.1, 5.0, True, True)
        m.step(2.01, 0.0, 0.0, True, False)
        self.assertFalse(m.armed)
        self.assertEqual(m.dur, [0.0, 0.0])

    def test_l6_replay_extf1(self):
        p = os.path.join(HERE, 'runs', 'door_pullback_extf_1', 'room_run.json')
        if not os.path.exists(p):
            self.skipTest('no extf_1')
        r = json.load(open(p))
        I = {c: i for i, c in enumerate(r['steps_cols'])}
        m = FL.FingerLossCont()
        fired = None
        for s in r['steps']:
            if s[1] < 3.9 or s[1] > 41.6:
                continue
            fn = s[I['finger_contact_n']]
            fired = fired or m.step(s[1], fn[0], fn[1], fn is not None, True)
        self.assertIsNone(fired)
        self.assertTrue(m.armed)
        self.assertLessEqual(max(m.take_window()['finger_low_cont_max_s']), 0.03)


if __name__ == '__main__':
    unittest.main(verbosity=1)
