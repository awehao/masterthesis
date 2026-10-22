#!/usr/bin/env python3
"""door_k3a_preload_locate.finger_sums 的反例：缺單指資料 ⇒ 合量 NA；讀取成功但無接觸 ⇒ 0。

    python3 evaluation/test_door_k3a_preload_locate.py
"""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3a_preload_locate as PL                  # noqa: E402

R = np.eye(3)
C1 = [{'p': [0, 0, 0], 'n': [0, 0, 1.0], 'fn': -2.0, 'sep': 0.0}]
C2 = [{'p': [0, 0, 0], 'n': [0, 0, -1.0], 'fn': -3.0, 'sep': 0.0}]
F1 = [{'p': [0, 0, 0], 'ft': [0.0, 1.0, 0.0]}]


class Sums(unittest.TestCase):
    def test_both_valid(self):
        x = {'contacts': {'uflite_finger1': C1, 'uflite_finger2': C2}, 'friction': {'uflite_finger1': F1, 'uflite_finger2': F1}}
        d = PL.finger_sums(x, R)
        np.testing.assert_allclose(d['sum_fn_n_handle'], [0, 0, 1.0])
        np.testing.assert_allclose(d['sum_ft_handle'], [0, 2.0, 0])
        self.assertTrue(d['sums_valid'])

    def test_missing_one_finger_value_is_na(self):
        bad = [{'p': [0, 0, 0], 'n': [0, 0, -1.0], 'fn': None, 'sep': 0.0}]
        x = {'contacts': {'uflite_finger1': C1, 'uflite_finger2': bad}, 'friction': {'uflite_finger1': F1, 'uflite_finger2': F1}}
        d = PL.finger_sums(x, R)
        self.assertEqual(d['uflite_finger2']['contacts_validity'], 'missing')
        self.assertIsNone(d['sum_fn_n_handle'])
        self.assertFalse(d['sums_valid'])

    def test_read_error_is_na_not_zero(self):
        x = {'contacts': None, 'contacts_err': 'RuntimeError: x', 'friction': {}}
        d = PL.finger_sums(x, R)
        self.assertIsNone(d['sum_fn_n_handle'])
        self.assertEqual(d['uflite_finger1']['contacts_validity'], 'missing')

    def test_no_contact_is_zero(self):
        x = {'contacts': {}, 'friction': {}}
        d = PL.finger_sums(x, R)
        np.testing.assert_allclose(d['sum_fn_n_handle'], [0, 0, 0])
        self.assertEqual(d['uflite_finger1']['contacts_validity'], 'no_contact')
        self.assertTrue(d['sums_valid'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
