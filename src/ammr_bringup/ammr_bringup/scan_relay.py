#!/usr/bin/env python3
"""
訂閱 /scan_raw，把 frame_id 改成 lidar_link 後重發為 /scan。
Gazebo Harmonic 強制把 scan frame_id 命名為 <model>/base_footprint/lidar，
這個 relay 把它換回 URDF 標準名稱 lidar_link。

可選：角度遮罩（self-occlusion filter）
  某些底盤的 LiDAR 周圍有結構柱，會在固定機體角度產生「假近距回波」，
  污染 AMCL 定位。用參數把這些扇區的 range 設成 inf（等同無回波），
  其餘角度不動。預設不濾 → 不影響沒有此問題的機器人（如 ammr_base）。

  blocked_centers_deg   : list[float]  機體座標下被擋扇區的中心角（度，0=+x 前方）
  blocked_halfwidth_deg : float        每個中心的半寬（度）
"""
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import LaserScan

BEST_EFFORT_QOS = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)


def _ang_diff_deg(a, b):
    """smallest absolute angular difference in degrees"""
    d = (a - b) % 360.0
    return min(d, 360.0 - d)


class ScanRelay(Node):
    def __init__(self):
        super().__init__('scan_relay')

        self.declare_parameter('blocked_centers_deg', [])
        self.declare_parameter('blocked_halfwidth_deg', 0.0)
        self.centers = [float(c) for c in
                        self.get_parameter('blocked_centers_deg').value]
        self.halfwidth = float(self.get_parameter('blocked_halfwidth_deg').value)

        self._mask = None  # built lazily once we know the scan geometry
        if self.centers and self.halfwidth > 0.0:
            self.get_logger().info(
                f'self-occlusion filter ON: centers={self.centers} '
                f'±{self.halfwidth}deg')

        self.pub = self.create_publisher(LaserScan, '/scan', BEST_EFFORT_QOS)
        self.create_subscription(LaserScan, '/scan_raw', self.cb, BEST_EFFORT_QOS)

    def _build_mask(self, msg: LaserScan):
        n = len(msg.ranges)
        mask = [False] * n
        if not (self.centers and self.halfwidth > 0.0):
            self._mask = mask
            return
        for i in range(n):
            bearing = math.degrees(msg.angle_min + i * msg.angle_increment) % 360.0
            if any(_ang_diff_deg(bearing, c) <= self.halfwidth for c in self.centers):
                mask[i] = True
        self._mask = mask
        self.get_logger().info(
            f'masked {sum(mask)}/{n} rays as self-occlusion')

    def cb(self, msg: LaserScan):
        msg.header.frame_id = 'lidar_link'
        if self._mask is None or len(self._mask) != len(msg.ranges):
            self._build_mask(msg)
        if any(self._mask):
            ranges = list(msg.ranges)
            for i, blocked in enumerate(self._mask):
                if blocked:
                    ranges[i] = float('inf')   # no return -> ignored by AMCL/costmap
            msg.ranges = ranges
        self.pub.publish(msg)


def main():
    rclpy.init()
    rclpy.spin(ScanRelay())
    rclpy.shutdown()
