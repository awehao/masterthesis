#!/usr/bin/env python3
"""door_regrasp_ref_registry 錯配拒絕測試（Codex r60 §1）。
  K1 參考包 v3 通過；K2 舊 v2（含模糊 T_HG、無 inner_grasp）拒絕；K3 phi 改 0（與 T_HG 不符）拒絕；K4 grasp_T 平移 3.5 mm 拒絕（原錯配量級）；
  K5 B1 改動拒絕；K6 外側基準 sha 不符拒絕；K7 來源 sha 不符拒絕；K8 T_HG 與 phi 同時改為舊抓法（自洽但與參考端點不符）拒絕。

    python3 evaluation/test_door_regrasp_ref_registry.py
"""
import copy
import json
import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_reach_v1 as RV                      # noqa: E402
import door_regrasp_ref_registry as REG                # noqa: E402

V3 = json.load(open(os.path.join(HERE, 'results', 'door', 'door_regrasp_ref_v3.json')))
V2 = json.load(open(os.path.join(HERE, 'results', 'door', 'door_regrasp_ref_v2.json')))


class TK(unittest.TestCase):
    def test_k1(self):
        self.assertEqual(REG.validate(V3), (True, None))

    def test_k2(self):
        self.assertFalse(REG.validate(V2)[0])

    def test_k3(self):
        p = copy.deepcopy(V3)
        p['inner_grasp']['phi_deg'] = 0.0
        self.assertEqual(REG.validate(p)[1], 'inner_T_HG_ne_phi')

    def test_k4(self):
        p = copy.deepcopy(V3)
        T = np.asarray(p['grasp_T'], float).reshape(4, 4)
        T[0, 3] += 0.0035
        p['grasp_T'] = T.reshape(-1).tolist()
        self.assertTrue(REG.validate(p)[1].startswith('grasp_ne_registered_grasp'))

    def test_k5(self):
        p = copy.deepcopy(V3)
        p['B1'] = [p['B1'][0] + 0.05, p['B1'][1], p['B1'][2]]
        self.assertEqual(REG.validate(p)[1], 'B1_ne_mode5_start')

    def test_k6_k7(self):
        p = copy.deepcopy(V3)
        p['outer_grasp_basis']['sha256'] = '0' * 64
        self.assertEqual(REG.validate(p)[1], 'outer_grasp_basis_missing_or_changed')
        p = copy.deepcopy(V3)
        p['source_file']['sha256'] = '0' * 64
        self.assertEqual(REG.validate(p)[1], 'source_file_missing_or_changed')

    def test_k8(self):
        p = copy.deepcopy(V3)
        p['inner_grasp'] = {'phi_deg': 0.0, 'T_HG': np.asarray(RV.T_HG(), float).reshape(-1).tolist()}
        self.assertTrue(REG.validate(p)[1].startswith('grasp_ne_registered_grasp'), REG.validate(p))
        self.assertGreater(abs(math.radians(V3['inner_grasp']['phi_deg'])), 0.5)


if __name__ == '__main__':
    unittest.main(verbosity=1)
