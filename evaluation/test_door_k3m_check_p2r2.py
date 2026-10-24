#!/usr/bin/env python3
"""door_k3m_check_p2r2（凍結目標旋轉差改用穩定算法）反例：
  同一矩陣 ⇒ 原 rot_angle ≈ 3.65e-8（偽拒的成因）、穩定算法 0；真實 1e-6 rad 旋轉、1e-8 m 位移、序號不符、PAUSE 前無已發布 ⇒ 仍拒；
  穩定算法與 acos 在中等角度一致。

    python3 evaluation/test_door_k3m_check_p2r2.py
"""
import json
import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_check_p2r2 as C                        # noqa: E402
import door_k3m_pause_mode as PM                       # noqa: E402

RUN = os.path.join(HERE, 'runs', 'door_k3m_pause_p2c_2', 'task.json')


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def T_of(R, p):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


def task_with(T_last, T_req, seq_last=10, seq_req=10):
    return {'k3m_pause': {'request': {'frozen_target_T': [float(v) for v in T_req.reshape(-1)], 'frozen_target_seq': seq_req}},
            'trace': [{'phase': 'SETTLE', 'pub_seq': seq_last, 'target': T_last.tolist()},
                      {'phase': 'PAUSE', 'pub_seq': seq_last + 1, 'target': T_req.tolist()}]}


class T(unittest.TestCase):
    def test_real_run_matrix(self):
        if not os.path.exists(RUN):
            self.skipTest('no run record')
        TR = np.asarray(json.load(open(RUN))['k3m_pause']['request']['frozen_target_T'])
        self.assertGreater(PM.rot_angle(TR, TR), 1e-9)                  # 原算法：自己對自己就超過 1e-9
        self.assertEqual(C.rot_angle_stable(TR, TR), 0.0)

    def test_ok_and_rejections(self):
        R0 = Rz(0.3) @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]])
        T0 = T_of(R0, [0.78, -0.14, 0.85])
        self.assertTrue(C.frozen_target_ok(task_with(T0, T0.copy()))[0])
        self.assertFalse(C.frozen_target_ok(task_with(T0, T_of(R0 @ Rz(1e-6), T0[:3, 3])))[0])
        self.assertFalse(C.frozen_target_ok(task_with(T0, T_of(R0, T0[:3, 3] + [1e-8, 0, 0])))[0])
        self.assertFalse(C.frozen_target_ok(task_with(T0, T0, seq_last=10, seq_req=11))[0])
        t = task_with(T0, T0)
        t['trace'] = t['trace'][1:]
        self.assertFalse(C.frozen_target_ok(t)[0])

    def test_agrees_with_acos_moderate(self):
        for a in (1e-3, 0.1, 1.0, 2.5):
            self.assertAlmostEqual(C.rot_angle_stable(T_of(np.eye(3), [0, 0, 0]), T_of(Rz(a), [0, 0, 0])), a, places=9)


if __name__ == '__main__':
    unittest.main(verbosity=2)
