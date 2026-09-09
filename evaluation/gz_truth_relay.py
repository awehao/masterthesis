#!/usr/bin/env python3
"""Publish the Gazebo robot's ground-truth pose on the same topic Isaac uses.

Gazebo never published a robot ground-truth pose: only the dynamic obstacles
carried a PosePublisher, so /odom -- which omni_drive_controller integrates
from /cmd_vel -- was the only robot position available, and there was nothing
to check it against. The scene broadcaster already emits every moving model's
true pose on /world/<world>/dynamic_pose/info, so nothing in the world, the
model or the launch has to change: bridge that topic and pull one model out
of it.

    ros2 run ros_gz_bridge parameter_bridge \
        /world/random_room/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V

Republished as PoseStamped on /model/<model>/pose at --rate Hz, matching what
isaac_bench_sim.py publishes, so one analysis reads both simulators.
"""
import argparse

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from tf2_msgs.msg import TFMessage

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='ammr_base')
ap.add_argument('--in-topic', default='/world/random_room/dynamic_pose/info')
ap.add_argument('--rate', type=float, default=20.0)


class Relay(Node):
    def __init__(self, a):
        super().__init__('gz_truth_relay')
        self.set_parameters([rclpy.parameter.Parameter(
            'use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        self.a = a
        self.period = 1.0 / a.rate
        self.last = None
        self.n_in = self.n_out = 0
        self.pub = self.create_publisher(
            PoseStamped, f'/model/{a.model}/pose', 10)
        self.create_subscription(TFMessage, a.in_topic, self.cb, 50)
        self.create_timer(5.0, self.report)

    def cb(self, msg):
        self.n_in += 1
        for tr in msg.transforms:
            if tr.child_frame_id != self.a.model:
                continue
            now = self.get_clock().now().nanoseconds * 1e-9
            if self.last is not None and (now - self.last) < self.period:
                return
            self.last = now
            p = PoseStamped()
            p.header.stamp = tr.header.stamp
            p.header.frame_id = 'world'
            p.pose.position.x = tr.transform.translation.x
            p.pose.position.y = tr.transform.translation.y
            p.pose.position.z = tr.transform.translation.z
            p.pose.orientation = tr.transform.rotation
            self.pub.publish(p)
            self.n_out += 1
            return

    def report(self):
        if self.n_out == 0:
            names = 'unknown'
            self.get_logger().warn(
                f'no "{self.a.model}" in {self.a.in_topic} yet '
                f'({self.n_in} messages seen) — check the model name')
        else:
            self.get_logger().info(
                f'truth relay: {self.n_in} in, {self.n_out} out')


def main():
    a = ap.parse_args()
    rclpy.init()
    n = Relay(a)
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
