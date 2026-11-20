#!/usr/bin/env python3
"""回程關門綁定共轉期間，給 Shield 的雷射遮掉「被操作的門扇」（Howard 11/20 選 (b)）。

/scan（原樣）→ 本節點 → /scan_shield（只給 scan_safety_shield；AMCL／costmap／追蹤器仍用 /scan）。
平常：原樣轉發（不轉發 ⇒ Shield 判掃描過期而限速，所以一定持續發）。
監督送 /shield/door_mask（JSON：active、seg＝底盤座標下門扇線段 [hx, hy, ex, ey]、t＝牆鐘）：
active 且訊息新鮮（≤0.5 s 牆鐘）⇒ 離線段 ≤HALF_W 的回波設 inf；其餘角度（牆、門框、行人）不動。
理由：Shield 把掃描點當靜止障礙，關門時機器人跟著遠離中的門扇前進＝「接近障礙」被擋（替身 s103d）；
門扇本身的碰撞由每拍全身停止軌跡淨空核對負責（env 含門扇、門框、牆）。不改 Shield 任何參數，只換它的 scan_topic。"""
import json
import math
import os
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

HALF_W = float(os.environ.get('LEAF_MASK_HALF_W', '0.12'))           # 門扇厚 0.04＋把手外凸＋追蹤誤差
FRESH_S = 0.5
QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1, durability=DurabilityPolicy.VOLATILE)


def seg_dist(px, py, hx, hy, ex, ey):
    dx, dy = ex - hx, ey - hy
    L2 = dx * dx + dy * dy
    u = np.clip(((px - hx) * dx + (py - hy) * dy) / max(L2, 1e-9), 0.0, 1.0)
    return np.hypot(px - (hx + u * dx), py - (hy + u * dy))


class LeafMask(Node):
    def __init__(self):
        super().__init__('door_leaf_scan_mask')
        self.mask, self.n_masked, self.n_msg = None, 0, 0
        self.pub = self.create_publisher(LaserScan, '/scan_shield', QOS)
        self.create_subscription(LaserScan, '/scan', self._scan, QOS)
        self.create_subscription(String, '/shield/door_mask', self._mask, 10)
        self.t_log = time.time()

    def _mask(self, m):
        try:
            self.mask = json.loads(m.data)
        except Exception:                                             # noqa: BLE001
            self.mask = None

    def _scan(self, m):
        mk = self.mask
        if mk and mk.get('active') and time.time() - float(mk.get('t', 0.0)) <= FRESH_S and mk.get('seg') is not None:
            r = np.asarray(m.ranges, float)
            a = m.angle_min + np.arange(len(r)) * m.angle_increment
            ok = np.isfinite(r)
            px, py = np.where(ok, r, 0.0) * np.cos(a), np.where(ok, r, 0.0) * np.sin(a)
            hx, hy, ex, ey = (float(x) for x in mk['seg'])
            hit = ok & (seg_dist(px, py, hx, hy, ex, ey) <= HALF_W)
            if hit.any():
                r = np.where(hit, np.inf, r)
                self.n_masked += int(hit.sum())
                m.ranges = [float(x) for x in r]
            self.n_msg += 1
        self.pub.publish(m)
        if time.time() - self.t_log > 5.0 and self.n_msg:
            self.get_logger().info(f'leaf mask active: {self.n_msg} scans, {self.n_masked} returns masked')
            self.t_log, self.n_msg, self.n_masked = time.time(), 0, 0


def main():
    rclpy.init()
    n = LeafMask()
    try:
        rclpy.spin(n)
    finally:
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
