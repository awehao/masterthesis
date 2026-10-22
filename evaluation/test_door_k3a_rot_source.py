#!/usr/bin/env python3
"""door_k3a_rot_source.decompose 的反例：共同座標的非交換旋轉、閉合、缺 d̂ ⇒ NA。

    python3 evaluation/test_door_k3a_rot_source.py
"""
import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3a_rot_source as RS                      # noqa: E402


def Rz(d):
    c, s = math.cos(math.radians(d)), math.sin(math.radians(d))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def Rx(d):
    c, s = math.cos(math.radians(d)), math.sin(math.radians(d))
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


class Decompose(unittest.TestCase):
    def test_codex_noncommutative_counterexample(self):
        """命令 yaw 90°、實測在空間座標額外 roll 10° ⇒ 共同座標殘差應為 x＝+10°（舊式 R_cᵀR_m 會得 y＝−10°）。"""
        Rc = Rz(90.0)
        Rm = Rx(10.0) @ Rc
        d = RS.decompose(Rc, None, Rm)
        np.testing.assert_allclose(d['D_MC_deg'], [10.0, 0.0, 0.0], atol=1e-9)
        old = np.degrees(RS.rotvec(Rc.T @ Rm))
        np.testing.assert_allclose(old, [0.0, -10.0, 0.0], atol=1e-9)                 # 證明舊式在命令局部座標

    def test_closure(self):
        Rc, Rs, Rm = Rz(30.0), Rx(3.0) @ Rz(31.0), Rx(7.0) @ Rz(5.0) @ Rz(31.0)
        d = RS.decompose(Rc, Rs, Rm)
        self.assertLess(d['closure_err'], 1e-12)
        self.assertIsNotNone(d['D_SC_deg'])
        self.assertIsNotNone(d['D_MS_deg'])

    def test_missing_dhat_is_na(self):
        d = RS.decompose(Rz(1.0), None, Rz(2.0))
        self.assertIsNone(d['D_SC_deg'])
        self.assertIsNone(d['D_MS_deg'])
        self.assertIsNone(d['closure_err'])
        np.testing.assert_allclose(d['D_MC_deg'], [0.0, 0.0, 1.0], atol=1e-9)


if __name__ == '__main__':
    unittest.main(verbosity=2)
