#!/usr/bin/env python3
"""door_k3m_task_node 的 ROS 接線測試（進程內、無 Isaac）：執行端／求解器 ack 進到策略的 TaskModeWaiter（t_recv＝最新狀態模擬時間）、
/park/violation 存入 exec_violation、策略 outbox 的請求由 /door/base_mode_req 發出可被收到。使用獨立 ROS_DOMAIN_ID。

    python3 evaluation/test_door_k3m_task_node_ros.py
"""
import json
import os
import sys
import time
import unittest

os.environ['ROS_DOMAIN_ID'] = os.environ.get('K3M_TEST_DOMAIN', '142')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                     # noqa: E402
import rclpy                                           # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy   # noqa: E402
from std_msgs.msg import String                        # noqa: E402

import door_k3m_task_node as TN                        # noqa: E402
import door_k3m_corridor as CO                         # noqa: E402
import door_k3m_geom_monitor as GM                     # noqa: E402
import test_door_k3a_policy as TP                      # noqa: E402

RUN = 'k3m_ros_test'


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

    def test_routing(self):
        nd = TN.Node2()
        cor = CO.Corridor([{'q_door_deg': 0.0, 'fwd_m': 0.0, 'lat_m': 0.0, 'dyaw_deg': 0.0},
                           {'q_door_deg': 8.0, 'fwd_m': 0.0, 'lat_m': 0.0, 'dyaw_deg': 0.0}], (0, 0, 0))
        pol = TN.P.K3MPolicy(TN.P.K3MConfig(), TP.T_HG(), RUN, cor, GM.GeometryMonitor(GM.GeomMonitorConfig(), None))
        nd.pol = pol
        nd.sync = (20.37, 2037, 0.0, np.eye(4), [0, 0, 0], np.zeros(6), 0.0)
        lat = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        peer = rclpy.create_node('k3m_peer')
        pe, ps = peer.create_publisher(String, '/door/base_mode_ack', lat), peer.create_publisher(String, '/wgmpc/base_mode_ack', lat)
        pv = peer.create_publisher(String, '/park/violation', 10)
        got = []
        peer.create_subscription(String, '/door/base_mode_req', lambda m: got.append(json.loads(m.data)), lat)
        try:
            self.assertTrue(self.spin([nd, peer], lambda: pe.get_subscription_count() > 0 and ps.get_subscription_count() > 0
                                      and pv.get_subscription_count() > 0 and nd.mode_req_pub.get_subscription_count() > 0))
            req = pol.waiter.make_request(2030, 20.30, time.monotonic())
            nd.mode_req_pub.publish(String(data=json.dumps(req)))
            self.assertTrue(self.spin([nd, peer], lambda: len(got) == 1))
            self.assertEqual(got[0], req)
            ea = {'run_id': RUN, 'seq': 1, 'mode': 'free', 'switch_step': 2031, 'switch_t': 20.31}
            pe.publish(String(data=json.dumps(ea)))
            self.assertTrue(self.spin([nd, peer], lambda: pol.waiter.exec_ack is not None))
            self.assertEqual(pol.waiter.t_exec, 20.37)                                         # t_recv＝最新狀態模擬時間
            sa = dict(ea, first_free_slot=5, first_free_sim_t=20.35, publish_sim_t=20.36, published=True)
            ps.publish(String(data=json.dumps(sa)))
            self.assertTrue(self.spin([nd, peer], lambda: pol.waiter.solver_ack is not None))
            self.assertEqual(pol.waiter.status(20.38, time.monotonic()), 'confirmed')
            pv.publish(String(data=json.dumps({'why': 'corridor'})))
            self.assertTrue(self.spin([nd, peer], lambda: nd.exec_violation is not None))
            self.assertEqual(nd.exec_violation['why'], 'corridor')
            self.assertEqual([m['src'] for m in nd.mode_msgs], ['executor', 'solver'])
        finally:
            peer.destroy_node()
            nd.destroy_node()


if __name__ == '__main__':
    unittest.main(verbosity=2)
