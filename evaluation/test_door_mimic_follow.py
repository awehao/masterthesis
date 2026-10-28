#!/usr/bin/env python3
"""door_mimic_follow：M1 跟隨 ⇒ pass；M2 follower 不動 ⇒ fail not_following；M3 行程未足即接觸 ⇒ fail；M4 資料無效 ⇒ fail；M5 以 extf_grip3_1 實錄（雙驅動）閉爪段重播 ⇒ pass。"""
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_mimic_follow as MF   # noqa: E402


def ramp(m, q2fn, contact_at=None):
    r = None
    for k in range(200):
        q1 = max(0.0, 0.0089 - k * 0.0089 / 100)
        f = 1.0 if (contact_at is not None and k >= contact_at) else 0.0
        r = r or m.step(k * 0.01, q1, q2fn(q1), f, f, True, True)
    return r


class T(unittest.TestCase):
    def test_m1(self):
        self.assertEqual(ramp(MF.MimicFollowCheck(), lambda q: q + 3e-6), 'pass')

    def test_m2(self):
        self.assertTrue(ramp(MF.MimicFollowCheck(), lambda q: 0.0089).startswith('fail:not_following'))

    def test_m3(self):
        self.assertTrue(ramp(MF.MimicFollowCheck(), lambda q: q, contact_at=5).startswith('fail:contact_before_travel'))

    def test_m4(self):
        m = MF.MimicFollowCheck()
        m.step(0.0, 0.0089, 0.0089, 0, 0, True, True)
        self.assertEqual(m.step(0.01, 0.0088, 0.0088, None, 0, False, True), 'fail:data_invalid')

    def test_m5_replay(self):
        p = os.path.join(HERE, 'runs', 'door_pullback_extf_grip3_1', 'room_run.json')
        if not os.path.exists(p):
            self.skipTest('no run')
        r = json.load(open(p))
        I = {c: i for i, c in enumerate(r['steps_cols'])}
        m = MF.MimicFollowCheck()
        res = None
        for s in r['steps']:
            if s[1] < 0.9:
                continue
            q, f = s[I['q_finger']], s[I['finger_contact_n']]
            res = res or m.step(s[1], q[0], q[1], f[0] if f else None, f[1] if f else None, f is not None, True)
            if res:
                break
        self.assertEqual(res, 'pass', m.report())


if __name__ == '__main__':
    unittest.main(verbosity=1)
