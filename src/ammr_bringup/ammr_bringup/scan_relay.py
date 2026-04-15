#!/usr/bin/env python3
"""
訂閱 /scan，把 frame_id 改成 lidar_link 後重發。
Gazebo Harmonic 強制把 scan frame_id 命名為 ammr_base/base_footprint/lidar，
這個 relay 把它換回 URDF 標準名稱 lidar_link。
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import LaserScan

BEST_EFFORT_QOS = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)


class ScanRelay(Node):
    def __init__(self):
        super().__init__('scan_relay')
        self.pub = self.create_publisher(LaserScan, '/scan', BEST_EFFORT_QOS)
        self.create_subscription(LaserScan, '/scan_raw', self.cb, BEST_EFFORT_QOS)

    def cb(self, msg: LaserScan):
        msg.header.frame_id = 'lidar_link'
        self.pub.publish(msg)


def main():
    rclpy.init()
    rclpy.spin(ScanRelay())
    rclpy.shutdown()
