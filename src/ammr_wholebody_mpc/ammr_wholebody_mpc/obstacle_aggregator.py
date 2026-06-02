"""Subscribe to each /model/dyn_obs_*/pose from the Gazebo bridge, estimate
each obstacle's velocity by finite-differencing its pose history, and republish
the whole set as a single std_msgs/Float32MultiArray on /gmpc/obstacles for
the GMPC + CBF controller to consume.

Wire format (Float32MultiArray.data) — flat array of 5 numbers per obstacle:
    [x1, y1, r1, vx1, vy1,
     x2, y2, r2, vx2, vy2,
     ...]
in the global frame. Length = 5 × n_obstacles.

Velocity estimate
-----------------
Each pose callback stores (t, x, y). The published velocity is the slope of
the last `window` samples computed by least-squares. window=5 at 20 Hz gives
~0.25 s averaging — enough to suppress noise but short enough to track 0.4 m/s
ping-pong motion.

This is the *ground-truth* obstacle channel (Gazebo provides perfect poses).
Future work: replace with a /scan-based clusterer + Kalman tracker so the
pipeline doesn't depend on simulation ground truth.
"""

from __future__ import annotations

import os
from collections import deque
from typing import List

import numpy as np
import yaml

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg      import Float32MultiArray


class ObstacleAggregator(Node):

    FIELDS_PER_OBS = 5    # x, y, r, vx, vy

    def __init__(self):
        super().__init__('obstacle_aggregator')

        self.declare_parameter('trajectories_file', '')
        self.declare_parameter('publish_rate',      20.0)
        self.declare_parameter('output_topic',      '/gmpc/obstacles')
        self.declare_parameter('vel_window',        5)      # samples for LSQ slope
        self.declare_parameter('vel_max_age_s',     0.50)   # drop samples older

        traj_file = str(self.get_parameter('trajectories_file').value)
        if not traj_file or not os.path.isfile(traj_file):
            raise RuntimeError(
                f'trajectories_file param empty or invalid: {traj_file!r}. '
                f'Pass e.g. dynamic_trajectories.yaml.'
            )

        with open(traj_file, 'r') as f:
            cfg = yaml.safe_load(f) or {}
        entries = cfg.get('dynamic_obstacles', [])
        self._radius = {e['name']: float(e.get('radius', 0.25)) for e in entries}

        win = int(self.get_parameter('vel_window').value)
        self._history = {name: deque(maxlen=win) for name in self._radius}

        # Ground-truth velocity from Gazebo's /model/<name>/cmd_vel topic.
        # The dynamic_obstacle_driver publishes Twist commands here, and because
        # the cylinders are <kinematic>true</kinematic> in the SDF, that command
        # IS their actual velocity (no physics lag). This bypasses the 250 ms
        # LSQ window so the CBF predictor reacts instantly when the obstacle
        # reverses direction at a ping-pong endpoint.
        self._vel_gt = {name: (0.0, 0.0) for name in self._radius}
        self._vel_gt_seen = set()

        self._max_age_ns = int(float(self.get_parameter('vel_max_age_s').value) * 1e9)

        for name in self._radius:
            self.create_subscription(
                PoseStamped, f'/model/{name}/pose',
                lambda msg, n=name: self._pose_cb(n, msg),
                10,
            )
            self.create_subscription(
                Twist, f'/model/{name}/cmd_vel',
                lambda msg, n=name: self._vel_cb(n, msg),
                10,
            )

        self.pub = self.create_publisher(
            Float32MultiArray,
            str(self.get_parameter('output_topic').value),
            10,
        )

        rate = float(self.get_parameter('publish_rate').value)
        self.create_timer(1.0 / rate, self._tick)

        self.get_logger().info(
            f'obstacle_aggregator up: tracking {len(self._radius)} obstacles '
            f'@ {rate:.1f} Hz with velocity window {win} samples '
            f'-> {self.get_parameter("output_topic").value}'
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _stamp_to_ns(msg: PoseStamped) -> int:
        return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)

    def _pose_cb(self, name: str, msg: PoseStamped):
        t_ns = self._stamp_to_ns(msg)
        if t_ns == 0:
            # Bridge sometimes emits zero stamps; fall back to wall-clock now()
            t_ns = self.get_clock().now().nanoseconds
        x = float(msg.pose.position.x)
        y = float(msg.pose.position.y)
        self._history[name].append((t_ns, x, y))

    def _vel_cb(self, name: str, msg: Twist):
        self._vel_gt[name] = (float(msg.linear.x), float(msg.linear.y))
        if name not in self._vel_gt_seen:
            self._vel_gt_seen.add(name)
            self.get_logger().info(
                f'Ground-truth velocity stream established for {name}'
            )

    def _estimate_velocity(self, name: str) -> tuple[float, float]:
        """Least-squares slope of recent samples; 0 if insufficient."""
        h = self._history[name]
        if len(h) < 2:
            return 0.0, 0.0
        # Drop samples older than max_age_s relative to newest
        t_newest = h[-1][0]
        pts = [(t, x, y) for (t, x, y) in h if (t_newest - t) <= self._max_age_ns]
        if len(pts) < 2:
            return 0.0, 0.0
        ts = np.array([p[0] for p in pts], dtype=float) * 1e-9
        xs = np.array([p[1] for p in pts])
        ys = np.array([p[2] for p in pts])
        t_mean = ts.mean()
        dt = ts - t_mean
        denom = float(np.sum(dt * dt))
        if denom < 1e-9:
            return 0.0, 0.0
        vx = float(np.sum(dt * (xs - xs.mean())) / denom)
        vy = float(np.sum(dt * (ys - ys.mean())) / denom)
        return vx, vy

    def _tick(self):
        out = Float32MultiArray()
        flat: List[float] = []
        for name, r in self._radius.items():
            h = self._history[name]
            if not h:
                continue
            x, y = h[-1][1], h[-1][2]
            # Prefer Gazebo ground-truth velocity (zero lag). Fall back to LSQ
            # only if no cmd_vel has been seen yet for this obstacle.
            if name in self._vel_gt_seen:
                vx, vy = self._vel_gt[name]
            else:
                vx, vy = self._estimate_velocity(name)
            flat.extend([x, y, r, vx, vy])
        out.data = flat
        self.pub.publish(out)


def main():
    rclpy.init()
    node = ObstacleAggregator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
