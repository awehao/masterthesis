#!/usr/bin/env python3
"""
訂閱 /odom，發布 odom -> ammr_base/base_footprint 的 TF。
確保時間戳與 odom 完全一致，避免 AMCL Message Filter 掉包。
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

BEST_EFFORT_QOS = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)


class OdomTfBroadcaster(Node):
    def __init__(self):
        super().__init__('odom_tf_broadcaster')
        self.br = TransformBroadcaster(self)
        self._last_stamp = None
        self.create_subscription(Odometry, '/odom', self.odom_cb, BEST_EFFORT_QOS)

    def odom_cb(self, msg: Odometry):
        stamp = msg.header.stamp
        # Skip backwards timestamps to prevent "jump back in time" TF errors
        if self._last_stamp is not None:
            last_ns = self._last_stamp.sec * 1_000_000_000 + self._last_stamp.nanosec
            curr_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
            if curr_ns <= last_ns:
                return
        self._last_stamp = stamp

        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_footprint'
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation = msg.pose.pose.orientation
        self.br.sendTransform(t)


def main():
    rclpy.init()
    rclpy.spin(OdomTfBroadcaster())
    rclpy.shutdown()
