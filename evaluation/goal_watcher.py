#!/usr/bin/env python3
"""Exit 0 as soon as the robot reaches the goal; exit 1 on timeout.

Used by run_one_trial.sh so a successful trial doesn't have to wait out
the full DURATION budget. We mirror the controller's goal-reached logic
(distance in map frame) instead of guessing from /odom, since /odom drifts
relative to map after AMCL corrections.

Usage:
    goal_watcher.py --goal-x 17 --goal-y 17 --tol 0.25 --timeout 250
"""
from __future__ import annotations

import argparse
import math
import sys
import time

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener


class GoalWatcher(Node):
    def __init__(self, goal_xy, tol, debounce):
        super().__init__('goal_watcher')
        self.goal_xy = goal_xy
        self.tol = tol
        self.debounce = debounce        # consecutive hits required → debounces TF jitter
        self.below_count = 0
        self.reached = False
        self.last_dist = float('nan')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(0.2, self.check)

    def check(self):
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_footprint', rclpy.time.Time()
            )
        except Exception:
            return                       # TF not ready yet, just keep waiting

        x = float(t.transform.translation.x)
        y = float(t.transform.translation.y)
        dist = math.hypot(x - self.goal_xy[0], y - self.goal_xy[1])
        self.last_dist = dist

        if dist < self.tol:
            self.below_count += 1
            if self.below_count >= self.debounce:
                self.get_logger().info(
                    f'[goal_watcher] REACHED: dist={dist:.3f} m  at ({x:.2f}, {y:.2f})'
                )
                self.reached = True
        else:
            self.below_count = 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--goal-x', type=float, required=True)
    p.add_argument('--goal-y', type=float, required=True)
    p.add_argument('--tol',     type=float, default=0.25,
                   help='map-frame distance to declare success [m]')
    p.add_argument('--timeout', type=float, default=250.0,
                   help='give up after this many wall-clock seconds')
    p.add_argument('--debounce', type=int, default=3,
                   help='consecutive in-tol samples needed (≈ debounce × 0.2 s)')
    args = p.parse_args()

    rclpy.init()
    node = GoalWatcher((args.goal_x, args.goal_y), args.tol, args.debounce)

    start = time.monotonic()
    try:
        while rclpy.ok() and not node.reached:
            rclpy.spin_once(node, timeout_sec=0.1)
            elapsed = time.monotonic() - start
            if elapsed > args.timeout:
                node.get_logger().info(
                    f'[goal_watcher] TIMEOUT after {elapsed:.1f}s '
                    f'(last dist = {node.last_dist:.2f} m)'
                )
                break
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    sys.exit(0 if node.reached else 1)


if __name__ == '__main__':
    main()
