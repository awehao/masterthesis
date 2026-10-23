#!/usr/bin/env python3
"""door_k3ab_task_node 的 ROS 接線測試（進程內、無 Isaac）：/door/finger_pair 原始訊息進到策略的 PairBuffer；
解析失敗 ⇒ malformed；只接線，不跑任務流程。使用獨立 ROS_DOMAIN_ID（不干擾其他程序）。

    python3 evaluation/test_door_k3ab_task_node_ros.py
"""
import json
import os
import sys
import time
import unittest

os.environ['ROS_DOMAIN_ID'] = os.environ.get('K3AB_TEST_DOMAIN', '141')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                     # noqa: E402
import rclpy                                           # noqa: E402
from std_msgs.msg import String                        # noqa: E402

import door_k3ab_task_node as TN                       # noqa: E402
import test_door_k3a_policy as TP                      # noqa: E402


class Wiring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def spin(self, nodes, cond, timeout=5.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout and not cond():
            for n in nodes:
                rclpy.spin_once(n, timeout_sec=0.02)
        return cond()

    def test_pairs_reach_buffer_and_malformed(self):
        nd = TN.Node2()
        pol = TN.P.K3ABPolicy(TN.P.K3ABConfig(), TP.T_HG())
        nd.pol = pol
        pub_node = rclpy.create_node('k3ab_pair_test_pub')
        pub = pub_node.create_publisher(String, '/door/finger_pair', 50)
        try:
            ok = self.spin([nd, pub_node], lambda: nd.count_publishers('/door/finger_pair') > 0 and pub.get_subscription_count() > 0)
            self.assertTrue(ok)
            msg = {'sim_t': 1.02, 'step': 102, 'pairs': [[101, 1.01, 2.0, 2.5, True], [102, 1.02, 2.0, 2.6, True]]}
            pub.publish(String(data=json.dumps(msg)))
            self.assertTrue(self.spin([nd, pub_node], lambda: len(pol.pairs.pending) == 2))
            self.assertEqual(sorted(pol.pairs.pending), [101, 102])
            self.assertIsNone(pol.pairs.fault)
            self.assertTrue(np.isfinite(pol.pairs.age(time.monotonic())))
            pub.publish(String(data='{not json'))
            self.assertTrue(self.spin([nd, pub_node], lambda: pol.pairs.fault is not None))
            self.assertEqual(pol.pairs.fault, 'malformed')
        finally:
            pub_node.destroy_node()
            nd.destroy_node()


if __name__ == '__main__':
    unittest.main(verbosity=2)
