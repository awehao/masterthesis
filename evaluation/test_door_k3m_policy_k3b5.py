#!/usr/bin/env python3
"""door_k3m_policy_k3b5 反例（**合成時間軸**）：
  P0 r5 物件使用 door_k3b_release5 監看；k3b_report 版本 r5。
  R  回歸：test_door_k3m_policy_k3b4 全部反例改以 K3MReleasePolicy5＋release5 監看執行。

    python3 evaluation/test_door_k3m_policy_k3b5.py
"""
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import door_k3b_release5 as KR5                        # noqa: E402
import door_k3m_policy_k3b5 as P5                      # noqa: E402
import test_door_k3m_policy_k3b3 as T3                 # noqa: E402
import test_door_k3m_policy_k3b4 as T4                 # noqa: E402


class TP(unittest.TestCase):
    def test_p0_and_regression(self):
        orig_pk, orig_kr = T4.PK, T3.KR
        T4.PK = types.SimpleNamespace(K3MReleasePolicy4=P5.K3MReleasePolicy5, VERSION=P5.VERSION)
        T3.KR = KR5
        try:
            p, _ = T4.run(T4.Sim4())
            self.assertIsInstance(p, P5.K3MReleasePolicy5)
            self.assertIsInstance(p.k3b, KR5.TaskReleaseMonitor)
            self.assertEqual(p.k3b_report()['version'], P5.VERSION)
            self.assertEqual(p.phase, 'DONE', p.abort)
            res = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w')).run(unittest.defaultTestLoader.loadTestsFromModule(T4))
        finally:
            T4.PK, T3.KR = orig_pk, orig_kr
        self.assertTrue(res.wasSuccessful(), [str(x[1])[-300:] for x in res.failures + res.errors])
        self.assertEqual(res.testsRun, 5)


if __name__ == '__main__':
    unittest.main(verbosity=1)
