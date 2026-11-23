#!/usr/bin/env python3
"""door_k3m_task_node_p 的 ROS 接線測試（進程內、無 Isaac）：原有 free 握手路由照舊；暫停請求由 /door/pause_req 發出可被收到；
/door/pause_ack_exec、/door/pause_ack_solver 進到策略的 TaskPauseWaiter（t_recv＝最新狀態模擬時間、牆鐘為回呼內實測）並可確認；
/wgmpc/pause_status 進到策略。使用獨立 ROS_DOMAIN_ID。

    python3 evaluation/test_door_k3m_task_node_p_ros.py
"""
import json
import os
import sys
import time
import unittest

os.environ['ROS_DOMAIN_ID'] = os.environ.get('K3MP_TEST_DOMAIN', '152')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                     # noqa: E402
import rclpy                                           # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy   # noqa: E402
from std_msgs.msg import String                        # noqa: E402
import door_k3m_task_node_p as TN                      # noqa: E402
import door_k3m_corridor as CO                         # noqa: E402
import door_k3m_geom_monitor as GM                     # noqa: E402
import test_door_k3a_policy as TP                      # noqa: E402

RUN = 'k3mp_ros_test'
XYZ = [0.8, -0.05, 0.85]


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

    def test_pause_routing(self):
        nd = TN.Node2()
        cor = CO.Corridor([{'q_door_deg': 0.0, 'fwd_m': 0.0, 'lat_m': 0.0, 'dyaw_deg': 0.0},
                           {'q_door_deg': 8.0, 'fwd_m': 0.0, 'lat_m': 0.0, 'dyaw_deg': 0.0}], (0, 0, 0))
        pol = TN.PP.K3MPPolicy(TN.PP.K3MPConfig(), TP.T_HG(), RUN, cor, GM.GeometryMonitor(GM.GeomMonitorConfig(), None))
        nd.pol = pol
        nd.sync = (30.37, 3037, 0.14, np.eye(4), [0, 0, 0], np.zeros(6), 0.0)
        lat = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        peer = rclpy.create_node('k3mp_peer')
        pe, ps = peer.create_publisher(String, '/door/pause_ack_exec', lat), peer.create_publisher(String, '/door/pause_ack_solver', lat)
        pst = peer.create_publisher(String, '/wgmpc/pause_status', lat)
        fe = peer.create_publisher(String, '/door/base_mode_ack', lat)
        got, gotf = [], []
        peer.create_subscription(String, '/door/pause_req', lambda m: got.append(json.loads(m.data)), lat)
        peer.create_subscription(String, '/door/base_mode_req', lambda m: gotf.append(json.loads(m.data)), lat)
        try:
            self.assertTrue(self.spin([nd, peer], lambda: pe.get_subscription_count() > 0 and ps.get_subscription_count() > 0
                                      and pst.get_subscription_count() > 0 and nd.pause_req_pub.get_subscription_count() > 0
                                      and fe.get_subscription_count() > 0))
            freq = pol.waiter.make_request(2030, 20.30, time.monotonic())             # 原 free 路由照舊
            nd.mode_req_pub.publish(String(data=json.dumps(freq)))
            self.assertTrue(self.spin([nd, peer], lambda: len(gotf) == 1))
            TF = np.eye(4)
            TF[:3, 3] = XYZ
            req = pol.pwaiter.make_request(3030, 30.30, time.monotonic(), 400, TF)
            nd.pause_req_pub.publish(String(data=json.dumps(req)))
            self.assertTrue(self.spin([nd, peer], lambda: len(got) == 1))
            self.assertEqual(got[0], req)
            ea = {'run_id': RUN, 'seq': 2, 'mode': 'pause', 'switch_step': 3031, 'switch_t': 30.31}
            pe.publish(String(data=json.dumps(ea)))
            self.assertTrue(self.spin([nd, peer], lambda: pol.pwaiter.exec_ack is not None))
            self.assertEqual(pol.pwaiter.t_exec, 30.37)
            sa = dict(ea, first_pause_slot=9, first_pause_sim_t=30.33, publish_sim_t=30.35, published=True, frozen_target_xyz=XYZ,
                      frozen_target_T=[float(v) for v in TF.reshape(-1)])
            ps.publish(String(data=json.dumps(sa)))
            self.assertTrue(self.spin([nd, peer], lambda: pol.pwaiter.solver_ack is not None))
            self.assertEqual(pol.pwaiter.status(30.38, time.monotonic()), 'confirmed')
            pst.publish(String(data=json.dumps({'outcome': {'state': 'CONFIRM'}})))
            self.assertTrue(self.spin([nd, peer], lambda: pol.pause_status is not None))
            self.assertEqual(pol.pause_status['outcome']['state'], 'CONFIRM')
            self.assertEqual([m['src'] for m in nd.pause_msgs], ['executor', 'solver'])
        finally:
            peer.destroy_node()
            nd.destroy_node()


if __name__ == '__main__':
    unittest.main(verbosity=2)
