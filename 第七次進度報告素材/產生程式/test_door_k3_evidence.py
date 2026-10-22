#!/usr/bin/env python3
"""door_k3_evidence 的離線反例（合成接觸串流；不啟動 Isaac）。

    python3 evaluation/test_door_k3_evidence.py
"""
import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3_evidence as EV                         # noqa: E402

NAMES = ['link5', 'link6', 'uflite_finger1', 'uflite_finger2', 'link_base']


def stream():
    # pair 順序同 NAMES：link5 0 點、link6 1 點、finger1 2 點、finger2 1 點、link_base 1 點（不在 keep）
    cnt = np.array([[0, 1, 2, 1, 1]])
    start = np.array([[0, 0, 1, 3, 4]])
    pts = np.arange(15, dtype=float).reshape(5, 3)
    nrm = np.tile([0.0, 0.0, 1.0], (5, 1))
    fn = np.array([[0.5], [1.0], [2.0], [3.0], [9.0]])
    sep = np.array([-1e-4, 0.0, -2e-4, 1e-4, 0.0])
    return fn, pts, nrm, sep, cnt, start


class Evidence(unittest.TestCase):
    def test_extract_contacts_slices_and_keep(self):
        c = EV.extract_contacts(*stream(), NAMES)
        self.assertEqual(set(c), {'link6', 'uflite_finger1', 'uflite_finger2'})       # link_base 不在 keep、link5 無接觸
        self.assertEqual(len(c['uflite_finger1']), 2)
        self.assertEqual(c['uflite_finger1'][0]['fn'], 1.0)
        self.assertEqual(c['uflite_finger1'][1]['p'], [6.0, 7.0, 8.0])
        self.assertEqual(c['uflite_finger2'][0]['fn'], 3.0)
        self.assertEqual(c['link6'][0]['fn'], 0.5)

    def test_pair_count_mismatch_raises_and_safe_wraps(self):
        fn, pts, nrm, sep, cnt, start = stream()
        with self.assertRaises(ValueError):
            EV.extract_contacts(fn, pts, nrm, sep, cnt, start, NAMES[:-1])
        v, why = EV.safe(EV.extract_contacts, fn, pts, nrm, sep, cnt, start, NAMES[:-1])
        self.assertIsNone(v)
        self.assertIn('ValueError', why)

    def test_nonfinite_values_recorded_as_none(self):
        fn, pts, nrm, sep, cnt, start = stream()
        fn = fn.copy()
        fn[1, 0] = float('nan')
        c = EV.extract_contacts(fn, pts, nrm, sep, cnt, start, NAMES)
        self.assertIsNone(c['uflite_finger1'][0]['fn'])                                # 不寫成 0

    def test_friction(self):
        ff = np.array([[0.1, 0, 0], [0, 0.2, 0], [0, 0, 0.3]])
        fp = np.ones((3, 3))
        cnt = np.array([[0, 0, 2, 1, 0]])
        start = np.array([[0, 0, 0, 2, 3]])
        f = EV.extract_friction(ff, fp, cnt, start, NAMES)
        self.assertEqual(f['uflite_finger1'][1]['ft'], [0.0, 0.2, 0.0])
        self.assertEqual(f['uflite_finger2'][0]['ft'], [0.0, 0.0, 0.3])

    def test_wrench_and_efforts(self):
        F = np.arange(60, dtype=float).reshape(10, 6)
        self.assertEqual(EV.wrench_row(F, 7), [42.0, 43.0, 44.0, 45.0, 46.0, 47.0])
        self.assertEqual(EV.wrench_row(F[None], 7), [42.0, 43.0, 44.0, 45.0, 46.0, 47.0])     # (1, J, 6)
        with self.assertRaises(ValueError):
            EV.wrench_row(np.zeros((3, 3)), 1)
        names = ['base', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'finger_joint1', 'finger_joint2']
        e = EV.efforts_by_name(np.arange(9.0), names, EV.ARM_DOF + EV.FINGER_DOF)
        self.assertEqual(e['joint1'], 1.0)
        self.assertEqual(e['finger_joint2'], 8.0)
        with self.assertRaises(ValueError):
            EV.efforts_by_name(np.arange(8.0), names, EV.ARM_DOF)
        with self.assertRaises(ValueError):
            EV.efforts_by_name(np.arange(9.0), names, ('joint7',))

    def test_summarize_in_handle_frame(self):
        T = np.eye(4)
        T[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]                                 # 把手 x＝世界 y
        T[:3, 3] = [1.0, 2.0, 3.0]
        con = {'uflite_finger1': [{'p': [1.0, 2.5, 3.0], 'n': [0.0, 1.0, 0.0], 'fn': 2.0, 'sep': 0.0},
                                  {'p': [1.0, 2.5, 3.2], 'n': [0.0, 1.0, 0.0], 'fn': 2.0, 'sep': 0.0}]}
        s = EV.summarize_contacts(con, T)['uflite_finger1']
        np.testing.assert_allclose(s['p_handle_mean'], [0.5, 0.0, 0.1], atol=1e-12)
        np.testing.assert_allclose(s['n_handle_mean'], [1.0, 0.0, 0.0], atol=1e-12)
        self.assertEqual(s['fn_sum'], 4.0)

    def test_torch_like_input(self):
        class T:
            def __init__(self, a):
                self.a = a

            def cpu(self):
                return self

            def numpy(self):
                return self.a
        fn, pts, nrm, sep, cnt, start = stream()
        c = EV.extract_contacts(T(fn), T(pts), T(nrm), T(sep), cnt, start, NAMES)
        self.assertIn('uflite_finger1', c)


if __name__ == '__main__':
    unittest.main(verbosity=2)
