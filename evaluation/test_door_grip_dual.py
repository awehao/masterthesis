#!/usr/bin/env python3
"""door_grip_dual：D1 登錄前純量；D2 確認窗滿 2 s 登錄基準且固定；D3 登錄後閉合 ⇒ 雙指 basis−Δ；D4 張爪 ⇒ 清除且不再登錄；D5 閂鎖凍結最後雙指；D6 窗中斷重來；r2：E1 無觸發不登錄；E2 窗滿後觸發即登錄；E3 觸發先於窗滿 ⇒ 窗滿時登錄。"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import door_grip_dual as GD   # noqa: E402


def reg(g, t0=0.0, q=(0.00166, 0.00168)):
    for k in range(200):
        g.observe(t0 + k * 0.01, q[0], q[1], 5.0, 5.0, True, 0.0)


class T(unittest.TestCase):
    def test_d1_d2_d3(self):
        g = GD.GripDual(0.00035, 0.0)
        self.assertEqual(g.targets(0.0), (0.0, 0.0))
        reg(g)
        self.assertIsNotNone(g.registered)
        g.observe(3.0, 0.002, 0.002, 5.0, 5.0, True, 0.0)
        a, b = g.targets(0.0)
        self.assertAlmostEqual(a, 0.00166 - 0.00035, places=9)
        self.assertAlmostEqual(b, 0.00168 - 0.00035, places=9)

    def test_d4_open_clears(self):
        g = GD.GripDual(0.00035, 0.0)
        reg(g)
        self.assertEqual(g.targets(0.0089), (0.0089, 0.0089))
        self.assertEqual(g.targets(0.0), (0.0, 0.0))
        reg(g, 10.0)
        self.assertEqual(g.targets(0.0), (0.0, 0.0))

    def test_d5_latch_freezes_dual(self):
        g = GD.GripDual(0.00035, 0.0)
        reg(g)
        d = g.targets(0.0)
        self.assertEqual(g.targets(0.0, latched=True), d)
        self.assertEqual(g.targets(0.0089, latched=True), d)

    def test_d6_window_break(self):
        g = GD.GripDual(0.00035, 0.0)
        for k in range(150):
            g.observe(k * 0.01, 0.00166, 0.00168, 5.0, 5.0, True, 0.0)
        g.observe(1.5, 0.00166, 0.00168, 0.2, 5.0, True, 0.0)
        for k in range(150):
            g.observe(1.51 + k * 0.01, 0.00166, 0.00168, 5.0, 5.0, True, 0.0)
        self.assertIsNone(g.registered)


class T2(unittest.TestCase):
    def test_e1_no_trigger_no_register(self):
        g = GD.GripDual2(0.00035, 0.0)
        reg(g)
        self.assertIsNone(g.registered)
        self.assertEqual(g.targets(0.0), (0.0, 0.0))

    def test_e2_trigger_after_full_window(self):
        g = GD.GripDual2(0.00035, 0.0)
        reg(g)
        g.trigger(2.5)
        g.observe(2.5, 0.00166, 0.00168, 5.0, 5.0, True, 0.0)
        self.assertIsNotNone(g.registered)
        self.assertEqual(g.registered['trigger']['t'], 2.5)

    def test_e3_trigger_before_full_window(self):
        g = GD.GripDual2(0.00035, 0.0)
        g.trigger(0.0)
        for k in range(150):
            g.observe(k * 0.01, 0.00166, 0.00168, 5.0, 5.0, True, 0.0)
        self.assertIsNone(g.registered)
        for k in range(150, 200):
            g.observe(k * 0.01, 0.00166, 0.00168, 5.0, 5.0, True, 0.0)
        self.assertIsNotNone(g.registered)


if __name__ == '__main__':
    unittest.main(verbosity=1)
