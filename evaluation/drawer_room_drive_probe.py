#!/usr/bin/env python3
"""驅動探針：對房間模擬器發 /cmd_vel，核 lidar、位姿與套用回報。

**只是接線核對**，不是導航，也不產生任何結果判定。
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64MultiArray, String

AP_IDX = {'step': 0, 'sim_t': 1, 'bvx': 2, 'bvy': 3, 'wz': 4,
          'exec_mode': 11, 'n_recv': 13, 'api_applied': 15}


class Probe(Node):
    def __init__(self, a):
        super().__init__('drawer_room_drive_probe')
        self.set_parameters([rclpy.parameter.Parameter(
            'use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        self.a = a
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(LaserScan, '/scan_raw', self._scan, 10)
        self.create_subscription(Odometry, '/odom', self._odom, 10)
        self.create_subscription(Float64MultiArray, '/coman/applied_cmd',
                                 self._ap, 10)
        self.create_subscription(String, '/handover/state', self._ho, 10)
        self.scan = None
        self.n_scan = 0
        self.odom0 = None
        self.odom = None
        self.n_odom = 0
        self.ap = None
        self.n_ap = 0
        self.ho = None

    def _scan(self, m):
        self.scan = m
        self.n_scan += 1

    def _odom(self, m):
        p = (m.pose.pose.position.x, m.pose.pose.position.y)
        if self.odom0 is None:
            self.odom0 = p
        self.odom = p
        self.n_odom += 1

    def _ap(self, m):
        if len(m.data) == 18:
            self.ap = list(m.data)
            self.n_ap += 1

    def _ho(self, m):
        try:
            self.ho = json.loads(m.data)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vx', type=float, default=0.25)
    ap.add_argument('--drive-s', type=float, default=6.0)
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    rclpy.init()
    nd = Probe(a)
    import time
    t0 = time.monotonic()
    # 等感測器上線
    while rclpy.ok() and time.monotonic() - t0 < 30.0:
        rclpy.spin_once(nd, timeout_sec=0.05)
        if nd.n_scan > 0 and nd.n_odom > 0 and nd.n_ap > 0:
            break
    rep = {'scan_seen': nd.n_scan > 0, 'odom_seen': nd.n_odom > 0,
           'applied_seen': nd.n_ap > 0}
    if not all(rep.values()):
        print(f'[probe] **感測器未全部上線** {rep}', flush=True)
        return 2
    rng = [r for r in nd.scan.ranges if math.isfinite(r)]
    rep['scan_n'] = len(nd.scan.ranges)
    rep['scan_finite'] = len(rng)
    rep['scan_min_m'] = min(rng) if rng else None
    rep['scan_max_m'] = max(rng) if rng else None
    print(f'[probe] lidar {rep["scan_finite"]}/{rep["scan_n"]} 條有回波；'
          f'距離 {rep["scan_min_m"]:.3f} … {rep["scan_max_m"]:.3f} m',
          flush=True)
    p_start = nd.odom
    rep['pose_start'] = p_start

    # 驅動
    tw = Twist()
    tw.linear.x = a.vx
    t1 = time.monotonic()
    while rclpy.ok() and time.monotonic() - t1 < a.drive_s:
        nd.pub.publish(tw)
        rclpy.spin_once(nd, timeout_sec=0.02)
    tw.linear.x = 0.0
    for _ in range(20):
        nd.pub.publish(tw)
        rclpy.spin_once(nd, timeout_sec=0.02)

    p_end = nd.odom
    rep['pose_end'] = p_end
    rep['moved_m'] = math.hypot(p_end[0] - p_start[0], p_end[1] - p_start[1])
    rep['applied_last'] = {k: nd.ap[i] for k, i in AP_IDX.items()} \
        if nd.ap else None
    rep['handover_state'] = nd.ho
    print(f'[probe] 位移 {rep["moved_m"]:.4f} m '
          f'（{p_start} → {p_end}）', flush=True)
    print(f'[probe] 最後套用回報 {rep["applied_last"]}', flush=True)
    print(f'[probe] 交棒狀態 {rep["handover_state"]}', flush=True)
    ok = rep['moved_m'] > 0.3
    print(f'[probe] {"接線通過" if ok else "**機器人沒有動**"}', flush=True)
    if a.out:
        json.dump(rep, open(a.out, 'w'), ensure_ascii=False, indent=1)
    nd.destroy_node()
    rclpy.shutdown()
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
