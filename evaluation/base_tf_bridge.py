"""Live world -> base_footprint TF from the simulator's odometry.

The fixed-base tests published this transform as a STATIC one, which was correct
exactly because the base could not move. The moment the base is part of the
solution that is no longer true: the safety filter reads TF to build the base
part of q, and a static transform would tell it the robot is still where it
started while the arm's Jacobian is computed about a base that has driven away.
Every barrier row would then be built at the wrong place, with a residual that
still looks healthy.

The source is `/odom`, bridged from the simulator's OdometryPublisher, which is
derived from the true pose rather than dead-reckoned. That makes this a
GROUND-TRUTH pose feed: it is legitimate for testing the whole-body motion and
the safety layer, and it means this test says nothing about localisation.

    python3 evaluation/base_tf_bridge.py [--z 0.05]
"""
from __future__ import annotations

import argparse
import sys

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from tf2_ros import TransformBroadcaster


class BaseTF(Node):

    def __init__(self, a):
        super().__init__('base_tf_bridge')
        self.a = a
        self.br = TransformBroadcaster(self)
        self.n = 0
        self.create_subscription(Odometry, a.odom, self._on,
                                 qos_profile_sensor_data)
        self.create_timer(2.0, self._report)

    def _on(self, m: Odometry) -> None:
        t = TransformStamped()
        t.header.stamp = m.header.stamp
        t.header.frame_id = self.a.frame
        t.child_frame_id = self.a.child
        p = m.pose.pose.position
        t.transform.translation.x = float(p.x)
        t.transform.translation.y = float(p.y)
        # z from the spawn height, not from odometry: the odometry frame is
        # planar and reports z = 0, while the model root sits above the ground.
        t.transform.translation.z = float(self.a.z)
        t.transform.rotation = m.pose.pose.orientation
        self.br.sendTransform(t)
        self.n += 1

    def _report(self):
        if self.n == 0:
            self.get_logger().warn(f'{self.a.odom} 尚未收到任何訊息')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--odom', default='/odom')
    ap.add_argument('--frame', default='world')
    ap.add_argument('--child', default='base_footprint')
    ap.add_argument('--z', type=float, default=0.05)
    a = ap.parse_args()
    rclpy.init()
    n = BaseTF(a)
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    rclpy.try_shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
