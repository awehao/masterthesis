#!/usr/bin/env python3
"""door_k3a_evidence_report.per_step 的反例（不讀實錄）。

    python3 evaluation/test_door_k3a_evidence_report.py
"""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3a_evidence_report as RP                 # noqa: E402

T = np.eye(4).reshape(-1).tolist()


class PerStep(unittest.TestCase):
    def test_invalid_friction_is_na_not_zero(self):
        row = {'T_WH': T, 'contacts': {'uflite_finger1': [{'p': [0, 0, 0], 'n': [0, 0, 1.0], 'fn': -2.0, 'sep': 0.0}]},
               'friction': {'uflite_finger1': [{'p': [0, 0, 0], 'ft': [None, None, None]}, {'p': [0, 0, 0], 'ft': [None, 0.0, 0.0]}]}}
        d = RP.per_step(row)['uflite_finger1']
        self.assertIsNone(d['ft_handle_sum'])
        self.assertIsNone(d['proxy_ratio'])
        self.assertEqual(d['n_friction'], 2)
        self.assertEqual(d['n_friction_valid'], 0)

    def test_points_without_force_counted_separately(self):
        row = {'T_WH': T, 'contacts': {'uflite_finger1': [{'p': [0, 0, 0], 'n': [0, 0, 1.0], 'fn': 0.0, 'sep': 0.001},
                                                          {'p': [1, 0, 0], 'n': [0, 0, 1.0], 'fn': -0.0, 'sep': 0.001}]},
               'friction': {}}
        d = RP.per_step(row)['uflite_finger1']
        self.assertEqual(d['n_points_reported'], 2)
        self.assertEqual(d['n_points_nonzero_fn'], 0)
        self.assertIsNone(d['p_handle_absw'])                                        # 全零 ⇒ 未定

    def test_partial_valid_friction(self):
        row = {'T_WH': T, 'contacts': {'uflite_finger2': [{'p': [0, 0, 0], 'n': [0, 0, -1.0], 'fn': -4.0, 'sep': 0.0}]},
               'friction': {'uflite_finger2': [{'p': [0, 0, 0], 'ft': [0.0, 2.0, 0.0]}, {'p': [0, 0, 0], 'ft': [None, None, None]}]}}
        d = RP.per_step(row)['uflite_finger2']
        self.assertEqual(d['n_friction_valid'], 1)
        np.testing.assert_allclose(d['ft_handle_sum'], [0.0, 2.0, 0.0])
        self.assertAlmostEqual(d['proxy_ratio'], 0.5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
