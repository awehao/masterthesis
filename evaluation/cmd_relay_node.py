#!/usr/bin/env python3
"""Twist 純轉送（門→抽屜串接 S1 抽屜段；results/specs/door_drawer_chain_s1_r1.md）：/cmd_vel_pre_mux → /cmd_vel_pre_shield。
門段由 door_nav1_mission_node 的選擇器（CRUISE 轉送）接這兩個話題；抽屜段 nav1 已結束，改由本節點做同一件事（不整形、不限幅、不補零）。
    python3 evaluation/cmd_relay_node.py [--in-topic /cmd_vel_pre_mux] [--out-topic /cmd_vel_pre_shield]
"""
import argparse
import json

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


class Relay(Node):
    def __init__(self, a):
        super().__init__('cmd_relay')
        self.pub = self.create_publisher(Twist, a.out_topic, 10)
        self.create_subscription(Twist, a.in_topic, self._cb, 10)
        self.n = 0

    def _cb(self, m):
        self.pub.publish(m)
        self.n += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-topic', default='/cmd_vel_pre_mux')
    ap.add_argument('--out-topic', default='/cmd_vel_pre_shield')
    a, _ = ap.parse_known_args()
    rclpy.init()
    nd = Relay(a)
    print(f'[relay] {a.in_topic} → {a.out_topic}', flush=True)
    try:
        rclpy.spin(nd)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        print(f'[relay] 結束：轉送 {nd.n} 筆 {json.dumps(vars(a))}', flush=True)
        nd.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
