#!/usr/bin/env python3
"""door_k2_io 的離線反例。    python3 evaluation/test_door_k2_io.py"""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import door_k2_io as KIO                               # noqa: E402

W = {'finger_bar_max_n': [0.0, 0.0], 'finger_bar_min_n': [0.0, 0.0], 'unexpected_max_n': 0.0, 'wall_max_n': 0.0, 'wrist_max_n': 1.0,
     'n_invalid': 0, 'sim_t': 1.0, 'unexpected_who': None}


class Sync(unittest.TestCase):
    def test_age_grows_when_stalled(self):
        s = KIO.SyncTracker()
        self.assertIsNone(s.age(0.0))
        self.assertTrue(s.on_msg(10, 0.1, 100.0))
        self.assertFalse(s.on_msg(10, 0.1, 100.5))                 # 重複不刷新
        self.assertAlmostEqual(s.age(101.0), 1.0)
        self.assertEqual(s.n_dup, 1)

    def test_backward_and_nonfinite(self):
        s = KIO.SyncTracker()
        s.on_msg(10, 0.1, 0.0)
        self.assertFalse(s.on_msg(8, 0.08, 0.1))
        self.assertFalse(s.ok())
        s2 = KIO.SyncTracker()
        s2.on_msg(float('nan'), 0.1, 0.0)
        self.assertFalse(s2.ok())


class Agg(unittest.TestCase):
    def test_short_pulse_survives_next_window(self):
        a = KIO.ContactsAggregator()
        a.on_msg(dict(W, wrist_max_n=35.0), 0.0)
        a.on_msg(dict(W, wrist_max_n=1.0), 0.02)
        c = a.consume()
        self.assertEqual(c['wrist_max_n'], 35.0)
        self.assertEqual(c['n_windows'], 2)
        self.assertIsNone(a.consume())                             # 已清空

    def test_min_of_finger_bar_and_invalid_sum(self):
        a = KIO.ContactsAggregator()
        a.on_msg(dict(W, finger_bar_min_n=[2.0, 2.0]), 0.0)
        a.on_msg(dict(W, finger_bar_min_n=[2.0, 0.1], n_invalid=1), 0.02)
        c = a.consume()
        self.assertEqual(c['finger_bar_min_n'], [2.0, 0.1])
        self.assertEqual(c['n_invalid'], 1)

    def test_nan_and_malformed_propagate(self):
        a = KIO.ContactsAggregator()
        a.on_msg(dict(W, wall_max_n=float('nan')), 0.0)
        a.on_msg(W, 0.02)
        self.assertTrue(math.isnan(a.consume()['wall_max_n']))
        a.on_msg({'garbage': 1}, 0.04)
        self.assertGreaterEqual(a.consume()['n_invalid'], 1)


class Latch(unittest.TestCase):
    def test_filter_and_freeze(self):
        L = KIO.K2AbortLatch()
        self.assertEqual(L.filter_wb([1.0] * 9), [1.0] * 9)
        L.on_msg('door_moving', 100, 2.0, 0.004)
        self.assertIsNone(L.filter_wb([1.0] * 9))
        self.assertEqual(L.grip(0.0), 0.004)                       # 不續閉
        self.assertEqual(L.grip(0.0089), 0.004)                    # 不放爪
        L.on_msg('other', 200, 3.0, 0.0)
        self.assertEqual(L.latched['why'], 'door_moving')          # 閂鎖不覆寫


class Classify(unittest.TestCase):
    def setUp(self):
        self.T = np.eye(4)
        self.T[:3, 3] = [0.8, -0.04, 0.85]
        self.args = dict(T_WH=self.T, bar_r=0.013, bar_half=0.10, post_u=[-0.085, 0.085], post_half_u=0.008, tol=0.006)

    def C(self, P):
        return KIO.classify_contacts(np.array(P), np.ones(len(P)), **self.args)

    def test_bar_center(self):
        self.assertEqual(self.C([[0.8, -0.04 - 0.013, 0.85]])['bar'], 1.0)

    def test_post_edge_is_ambiguous(self):
        r = self.C([[0.8 + 0.085 - 0.008, -0.04 + 0.012, 0.85]])        # 支柱邊緣、靠近橫桿軸
        self.assertEqual(r['bar'], 0.0)
        self.assertEqual(r['ambiguous'], 1.0)

    def test_beyond_bar_end_is_ambiguous(self):
        r = self.C([[0.8 + 0.105, -0.04, 0.85 + 0.013]])
        self.assertEqual(r['ambiguous'], 1.0)

    def test_panel(self):
        self.assertEqual(self.C([[0.8, 0.0, 1.2]])['panel'], 1.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
