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
    def __init__(self, goal_xy, tol, debounce, stall_s=15.0, stall_pad=0.10,
                 stall_move=0.02):
        super().__init__('goal_watcher')
        self.goal_xy = goal_xy
        self.tol = tol
        self.debounce = debounce        # consecutive hits required → debounces TF jitter
        self.below_count = 0
        self.reached = False
        self.last_dist = float('nan')

        # Second, slower exit: the robot has stopped just outside tol and is not
        # going to get any closer.
        #
        # The controller and this watcher judge "arrived" from DIFFERENT pose
        # estimates -- gmpc_node uses its own pose_source (the EKF on
        # /odometry/filtered in the benchmark), this watcher uses TF
        # map->base_footprint (AMCL-corrected). When the two disagree by a
        # centimetre at the wrong moment, the controller halts at 0.293 m while
        # the watcher still measures more than 0.30, and since a halted robot
        # never gets closer the trial runs to its full duration cap. Measured on
        # the archive: 8% of GMPC trials, up to 146 s each, of a parked robot
        # sitting in traffic. MPPI and RPP never hit it, so the waste was
        # asymmetric across the methods being compared.
        #
        # Deliberately conservative: a robot that pauses to let an obstacle pass
        # must NOT be declared arrived, because that would cut the recording
        # while it is still navigating -- a far worse failure than wasting time.
        # Hence both conditions, and 15 s rather than 2-3.
        self.stall_s = stall_s          # s of no motion before conceding
        self.stall_pad = stall_pad      # extra distance beyond tol still eligible
        self.stall_move = stall_move    # m of motion that resets the stall timer
        self._stall_xy = None
        self._stall_t0 = None

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
            self._stall_xy = None
            self._stall_t0 = None
            return

        self.below_count = 0
        now = time.time()
        if dist < self.tol + self.stall_pad:
            if (self._stall_xy is None
                    or math.hypot(x - self._stall_xy[0], y - self._stall_xy[1])
                    > self.stall_move):
                self._stall_xy = (x, y)
                self._stall_t0 = now
            elif now - self._stall_t0 >= self.stall_s:
                # Distinct wording on purpose: this is not the same event as
                # crossing the tolerance, and the logs have to say which one
                # ended the trial.
                self.get_logger().info(
                    f'[goal_watcher] STALLED at dist={dist:.3f} m '
                    f'({x:.2f}, {y:.2f}) for {self.stall_s:.0f}s -> conceding'
                )
                self.reached = True
        else:
            self._stall_xy = None
            self._stall_t0 = None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--goal-x', type=float, required=True)
    p.add_argument('--goal-y', type=float, required=True)
    p.add_argument('--stall-s', type=float, default=15.0,
                   help='s of no motion just outside tol before conceding')
    p.add_argument('--stall-pad', type=float, default=0.10,
                   help='how far beyond tol a stalled robot still counts [m]')
    p.add_argument('--tol',     type=float, default=0.15,
                   help='map-frame distance to declare success [m]')
    p.add_argument('--timeout', type=float, default=250.0,
                   help='give up after this many wall-clock seconds')
    p.add_argument('--debounce', type=int, default=3,
                   help='consecutive in-tol samples needed (≈ debounce × 0.2 s)')
    args = p.parse_args()

    rclpy.init()
    node = GoalWatcher((args.goal_x, args.goal_y), args.tol, args.debounce,
                       stall_s=args.stall_s, stall_pad=args.stall_pad)

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
