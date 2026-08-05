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

        # Comma-separated string (e.g. "45,135,225,315") rather than a list
        # parameter: rclpy infers an empty-list default as BYTE_ARRAY and then
        # rejects float-array overrides. A string default '' has no such issue.
        self.declare_parameter('blocked_centers_deg', '')
        self.declare_parameter('blocked_halfwidth_deg', 0.0)
        # Per-sector half-widths, comma-separated, aligned with the centres.
        # One shared half-width was wrong in both directions: measured against
        # 194 frames of /scan_raw, the four +-15 deg sectors masked 120 rays
        # while only 68 are genuinely self-occluded -- 78 good rays thrown away
        # (22% of the lidar) -- and they missed a 34 deg block near -69 deg
        # entirely, so 26 rays of self-return kept feeding AMCL. The real
        # sectors are narrow near +-45 and +-135 (8-9 deg) and wide near -69.
        # Empty -> fall back to blocked_halfwidth_deg for every centre.
        self.declare_parameter('blocked_halfwidths_deg', '')
        centers_str = str(self.get_parameter('blocked_centers_deg').value)
        self.centers = [float(x) for x in centers_str.split(',') if x.strip()]
        self.halfwidth = float(self.get_parameter('blocked_halfwidth_deg').value)
        hw_str = str(self.get_parameter('blocked_halfwidths_deg').value)
        hws = [float(x) for x in hw_str.split(',') if x.strip()]
        if hws and len(hws) != len(self.centers):
            raise ValueError(
                f'blocked_halfwidths_deg has {len(hws)} entries for '
                f'{len(self.centers)} centres')
        self.halfwidths = hws or [self.halfwidth] * len(self.centers)

        self._mask = None  # built lazily once we know the scan geometry
        if self.centers and any(h > 0.0 for h in self.halfwidths):
            pairs = ', '.join(f'{c:+.1f}+-{h:.1f}'
                              for c, h in zip(self.centers, self.halfwidths))
            self.get_logger().info(f'self-occlusion filter ON: {pairs} deg')

        self.pub = self.create_publisher(LaserScan, '/scan', BEST_EFFORT_QOS)
        self.create_subscription(LaserScan, '/scan_raw', self.cb, BEST_EFFORT_QOS)

    def _build_mask(self, msg: LaserScan):
        n = len(msg.ranges)
        mask = [False] * n
        if not (self.centers and any(h > 0.0 for h in self.halfwidths)):
            self._mask = mask
            return
        for i in range(n):
            bearing = math.degrees(msg.angle_min + i * msg.angle_increment) % 360.0
            if any(_ang_diff_deg(bearing, c) <= h
                   for c, h in zip(self.centers, self.halfwidths)):
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
