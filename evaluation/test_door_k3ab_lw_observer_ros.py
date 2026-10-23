#!/usr/bin/env python3
"""獨立觀察器的 ROS 假世界核對（子行程啟動 door_k3ab_lw_observer.py；隔離 ROS_DOMAIN_ID）：
  O1：中止後狀態停更 ⇒ /door/lw_guard（state_stale）與 /room/stop 都發出，報告記 trip；
  O2：中止後單一窗口短脈衝非設計接觸 0.2 N ⇒ unexpected_contact，來源步正確；
  O3：中止後他方先發 /room/stop（正常結束）再停更 ⇒ 不觸發，報告記 closed；
  O4：未收到中止 ⇒ 門角 9°、停更都不觸發。

    ROS_DOMAIN_ID=143 python3 evaluation/test_door_k3ab_lw_observer_ros.py
"""
import json
import os
import subprocess
import sys
import tempfile
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray, String

HERE = os.path.dirname(os.path.abspath(__file__))
LAT = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)


class W(Node):
    def __init__(self):
        super().__init__('lw_fake_world')
        self.sync = self.create_publisher(Float64MultiArray, '/door/sync_state', 100)
        self.ct = self.create_publisher(String, '/door/contacts', 100)
        self.ab = self.create_publisher(String, '/door/k2_abort', LAT)
        self.rs = self.create_publisher(String, '/room/stop', 10)
        self.guard, self.stops = [], []
        self.create_subscription(String, '/door/lw_guard', lambda m: self.guard.append(m.data), LAT)
        self.create_subscription(String, '/room/stop', lambda m: self.stops.append(m.data), 10)

    def tick(self, k, q=0.01, unexp=0.0, state=True):
        t = 20.0 + 0.02 * k
        if state:
            self.sync.publish(Float64MultiArray(data=[t, float(k), q] + [0.0] * 16 + [0.6, -0.53, 1.57] + [0.0] * 6 + [0.0]))
        self.ct.publish(String(data=json.dumps({'sim_t': t, 'step': k, 'n_steps': 2, 'n_invalid': 0, 'finger_bar_max_n': [0, 10],
                                                'finger_bar_min_n': [0, 9], 'unexpected_max_n': unexp, 'unexpected_who': None,
                                                'wall_max_n': 0.0, 'wrist_max_n': 10.0})))


def spin(nd, s):
    t0 = time.monotonic()
    while time.monotonic() - t0 < s:
        rclpy.spin_once(nd, timeout_sec=0.005)


def case(name, script):
    d = tempfile.mkdtemp(prefix='lwobs_')
    out = os.path.join(d, 'obs.json')
    p = subprocess.Popen([sys.executable, '-u', os.path.join(HERE, 'door_k3ab_lw_observer.py'), '--out', out],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    nd = W()
    try:
        spin(nd, 2.5)                                                   # 等發現
        script(nd)
        spin(nd, 0.5)
    finally:
        p.terminate()
        p.wait(10)
        nd.destroy_node()
    rep = json.load(open(out))
    return rep, nd


def main():
    rclpy.init()
    ok = True

    def k_run(nd, n, k0=0, **kw):
        for k in range(k0, k0 + n):
            nd.tick(k, **kw)
            spin(nd, 0.02)

    def o1(nd):
        k_run(nd, 10)
        nd.ab.publish(String(data='{"why": "finger_contact_lost"}'))
        k_run(nd, 10, 10)
        k_run(nd, 20, 20, state=False)                                  # 狀態停更，接觸仍到
    rep, nd = case('O1', o1)
    r1 = rep['trip'] and rep['trip']['why'] == 'state_stale' and nd.guard and any('experiment_guard' in s for s in nd.stops)
    print('O1', 'OK' if r1 else 'FAIL', rep['trip'], len(nd.guard), nd.stops[:1])
    ok &= bool(r1)

    def o2(nd):
        k_run(nd, 10)
        nd.ab.publish(String(data='{"why": "finger_contact_lost"}'))
        k_run(nd, 5, 10)
        nd.tick(15, unexp=0.2)
        k_run(nd, 5, 16)
    rep, nd = case('O2', o2)
    r2 = rep['trip'] and rep['trip']['why'] == 'unexpected_contact' and rep['trip']['src_step'] == 15
    print('O2', 'OK' if r2 else 'FAIL', rep['trip'])
    ok &= bool(r2)

    def o3(nd):
        k_run(nd, 10)
        nd.ab.publish(String(data='{"why": "finger_contact_lost"}'))
        k_run(nd, 10, 10)
        nd.rs.publish(String(data='door_k3a 結束：ABORTED finger_contact_lost'))
        spin(nd, 0.6)
    rep, nd = case('O3', o3)
    r3 = rep['trip'] is None and rep['closed'] is not None and not nd.guard
    print('O3', 'OK' if r3 else 'FAIL', rep['trip'], rep['closed'])
    ok &= bool(r3)

    def o4(nd):
        k_run(nd, 10, q=0.16)
        spin(nd, 0.6)
    rep, nd = case('O4', o4)
    r4 = rep['trip'] is None and rep['armed_wall'] is None and not nd.guard
    print('O4', 'OK' if r4 else 'FAIL', rep['trip'], rep['armed_wall'])
    ok &= bool(r4)
    rclpy.try_shutdown()
    print('全部通過' if ok else '有失敗')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
