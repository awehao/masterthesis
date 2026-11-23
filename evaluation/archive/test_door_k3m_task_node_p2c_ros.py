#!/usr/bin/env python3
"""door_k3m_task_node_p2c 的 ROS 接線測試（test_door_k3m_task_node_p2b_ros.py 的複本＋交棒前置條件）：
H1：/wb_vel_cmd 未達 20 筆 ⇒ wb_cmd_ready False；第 20 筆 ⇒ True；停送超過 0.2 s 牆鐘 ⇒ False；長度不足的訊息不計。
H2：p2c 只訂閱 /wb_vel_cmd 計數，不發布該話題（不轉送、不改命令）。
原說明（p2b）：door_k3m_task_node_p2b 的 ROS 接線測試（test_door_k3m_task_node_p2_ros.py 的複本＋送達反例）：
B1（door_k3m_pause_p2_1 的缺陷情境）：兩輪之間連發 8 筆、中間不 spin，再照主迴圈 64 次 spin_once ⇒ p2b 窗口收齊全部樣本、值 valid；
   同情境 p2 節點只留每輪最新一筆 ⇒ long_window（重現缺陷）。
S（p2b 語意）：重送 ⇒ 窗口拒收；倒退 ⇒ 窗口清空、需重新累積滿 0.1 s（p2 由 SyncTracker 先擋，p2b 較保守）；量測路徑 SyncTracker 照舊。
B2：排空後窗口最新樣本步號＝量測路徑 nd.sync 的步號（本輪 step_mismatch 不會因兩個訂閱而系統性發生）。
原說明（p2）：door_k3m_task_node_p2 的 ROS 接線測試（test_door_k3m_task_node_p_ros.py 的複本＋同步流→SETTLE 完成訊號窗口）：
S：/door/sync_state（29 欄）每筆被 SyncTracker 接受後才進 SettleRateWindow；重送（同步號）與倒退不進窗口；窗口值與 50 Hz 序列的 0.1 s 差分相同。
原說明（p 版）：door_k3m_task_node_p 的 ROS 接線測試（進程內、無 Isaac）：原有 free 握手路由照舊；暫停請求由 /door/pause_req 發出可被收到；
/door/pause_ack_exec、/door/pause_ack_solver 進到策略的 TaskPauseWaiter（t_recv＝最新狀態模擬時間、牆鐘為回呼內實測）並可確認；
/wgmpc/pause_status 進到策略。使用獨立 ROS_DOMAIN_ID。

    python3 evaluation/test_door_k3m_task_node_p_ros.py
"""
import json
import os
import sys
import time
import unittest

os.environ['ROS_DOMAIN_ID'] = os.environ.get('K3MP2C_TEST_DOMAIN', '156')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                     # noqa: E402
import rclpy                                           # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy   # noqa: E402
from std_msgs.msg import Float64MultiArray, String      # noqa: E402
import door_k3m_task_node_p2c as TN                    # noqa: E402
import door_k3m_task_node_p2 as TN_P2                  # noqa: E402  缺陷重現對照
import door_k3m_corridor as CO                         # noqa: E402
import door_k3m_geom_monitor as GM                     # noqa: E402
import test_door_k3a_policy as TP                      # noqa: E402

RUN = 'k3mp2c_ros_test'
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
        pol = TN.PP2.K3MP2Policy(TN.PP2.K3MP2Config(), TP.T_HG(), RUN, cor, GM.GeometryMonitor(GM.GeomMonitorConfig(), None))
        nd.pol = pol
        nd.sync = (30.37, 3037, 0.14, np.eye(4), [0, 0, 0], np.zeros(6), 0.0)
        lat = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        peer = rclpy.create_node('k3mp2b_peer')
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


    def test_sync_feeds_settle_window(self):
        nd = TN.Node2()
        peer = rclpy.create_node('k3mp2b_sync_peer')
        pub = peer.create_publisher(Float64MultiArray, '/door/sync_state', 10)

        def msg(t, step, q):
            return Float64MultiArray(data=[t, float(step), q] + [float(v) for v in np.eye(4).reshape(-1)] + [0.6, -0.53, 1.57] + [0.0] * 6 + [0.016])
        try:
            self.assertTrue(self.spin([nd, peer], lambda: pub.get_subscription_count() > 0))
            seq = [(round(30.0 + 0.02 * i, 4), 3000 + 2 * i, 0.14 + 0.001 * 0.02 * i) for i in range(12)]
            for k, (t, st, q) in enumerate(seq):
                pub.publish(msg(t, st, q))
                self.assertTrue(self.spin([nd, peer], lambda: nd.sync is not None and int(nd.sync[1]) == st))
                if k == 6:
                    pub.publish(msg(t, st, 0.5))                                 # 重送（同步號）：窗口拒收
                    pub.publish(msg(t - 0.04, st - 4, 0.5))                       # 倒退：窗口清空
                    for _ in range(20):
                        rclpy.spin_once(nd, timeout_sec=0.01)
            # p2b：窗口訂閱不經 SyncTracker——重送由窗口拒收（duplicate），倒退由窗口清空（之後須重新累積滿 0.1 s；比 p2 保守）
            for _ in range(20):
                rclpy.spin_once(nd, timeout_sec=0.01)
            self.assertEqual(nd.settle_win.anomalies['duplicate'], 1)
            self.assertEqual(nd.settle_win.anomalies['backward'], 1)
            self.assertEqual([b[1] for b in nd.settle_win.buf], [s[1] for s in seq[7:]])
            self.assertEqual(nd.settle_win.value(seq[-1][1], nd.settle_win.buf[-1][3])['why'], 'short_window')   # 清空後 0.08 s：未滿窗口（牆鐘取最新樣本，排除停更）
            for i in range(12, 14):                                          # 再兩筆 ⇒ 滿 0.1 s
                t, st, q = round(30.0 + 0.02 * i, 4), 3000 + 2 * i, 0.14 + 0.001 * 0.02 * i
                pub.publish(msg(t, st, q))
                self.assertTrue(self.spin([nd, peer], lambda: nd.settle_win.buf and nd.settle_win.buf[-1][1] == st))
            v = nd.settle_win.value(3026, time.monotonic())
            self.assertTrue(v['valid'], v)
            self.assertAlmostEqual(v['rate'], 0.001, places=6)
            self.assertEqual(nd.sync_tr.n_back, 1)                           # 量測路徑的 SyncTracker 照舊記到倒退
        finally:
            peer.destroy_node()
            nd.destroy_node()

    def burst(self, mod, n_cyc=4, per=8):
        nd = mod.Node2()
        peer = rclpy.create_node('k3mp2b_burst_peer')
        pub = peer.create_publisher(Float64MultiArray, '/door/sync_state', 64)

        def msg(t, step, q):
            return Float64MultiArray(data=[t, float(step), q] + [float(v) for v in np.eye(4).reshape(-1)] + [0.6, -0.53, 1.57] + [0.0] * 6 + [0.016])
        try:
            self.assertTrue(self.spin([nd, peer], lambda: pub.get_subscription_count() >= 1))
            k = 0
            for _ in range(n_cyc):
                for _ in range(per):                                       # 兩輪之間到達多筆、中間不 spin
                    pub.publish(msg(round(30.0 + 0.02 * k, 4), 3000 + 2 * k, 0.14 + 0.001 * 0.02 * k))
                    k += 1
                time.sleep(0.05)
                for _ in range(64):                                        # 與主迴圈相同：每輪 ≤ 64 次 spin_once
                    rclpy.spin_once(nd, timeout_sec=0.0)
            v = nd.settle_win.value(int(nd.sync[1]), time.monotonic())
            return k, [b[1] for b in nd.settle_win.buf], int(nd.sync[1]), v, getattr(nd, 'win_rx', None)
        finally:
            peer.destroy_node()
            nd.destroy_node()

    def test_b1_burst_delivery(self):
        k, steps, sync_step, v, rx = self.burst(TN)
        self.assertEqual(rx, k)
        self.assertEqual(steps[-1], 3000 + 2 * (k - 1))
        self.assertEqual(len(steps), len([s for s in range(3000, 3000 + 2 * k, 2) if s >= steps[0]]))   # 保留範圍內無缺
        self.assertTrue(v['valid'], v)
        self.assertAlmostEqual(v['rate'], 0.001, places=6)
        k2, steps2, _, v2, _ = self.burst(TN_P2)                            # p2：每輪只留最新一筆（缺陷重現）
        self.assertEqual(len(steps2), 4)
        self.assertEqual(v2['why'], 'long_window')

    def test_b2_latest_matches_measurement(self):
        k, steps, sync_step, v, rx = self.burst(TN, n_cyc=3, per=5)
        self.assertEqual(steps[-1], sync_step)
        self.assertEqual(v['step1'], sync_step)

    def test_h1_handover_precondition(self):
        nd = TN.Node2()
        peer = rclpy.create_node('k3mp2c_wb_peer')
        pub = peer.create_publisher(Float64MultiArray, '/wb_vel_cmd', 10)
        try:
            self.assertTrue(self.spin([nd, peer], lambda: pub.get_subscription_count() > 0))
            self.assertFalse(nd.wb_cmd_ready(time.monotonic()))
            pub.publish(Float64MultiArray(data=[0.0] * 5))                       # 長度不足：不計
            for i in range(19):
                pub.publish(Float64MultiArray(data=[0.0] * 9))
                self.assertTrue(self.spin([nd, peer], lambda: nd.wb_cmd_n == i + 1))
            self.assertFalse(nd.wb_cmd_ready(time.monotonic()))
            pub.publish(Float64MultiArray(data=[0.0] * 9))
            self.assertTrue(self.spin([nd, peer], lambda: nd.wb_cmd_n == 20))
            self.assertTrue(nd.wb_cmd_ready(time.monotonic()))
            self.assertFalse(nd.wb_cmd_ready(nd.wb_cmd_wall + 0.25))           # 停送 ⇒ 未就緒
        finally:
            peer.destroy_node()
            nd.destroy_node()

    def test_h2_no_wb_cmd_publisher(self):
        nd = TN.Node2()
        try:
            pubs = [n for n, _ in nd.get_publisher_names_and_types_by_node(nd.get_name(), nd.get_namespace())]
            self.assertNotIn('/wb_vel_cmd', pubs)
        finally:
            nd.destroy_node()

if __name__ == '__main__':
    unittest.main(verbosity=2)
