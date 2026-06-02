"""Aggregate dynamic obstacle state for the GMPC + horizon CBF controller.

For each obstacle named in the trajectories YAML:
  - Subscribe to  /model/<name>/pose      (the Gazebo bridge ground-truth pose)
  - Subscribe to  /model/<name>/cmd_vel   (logged for diagnostics only)
  - Run a 4-state Kalman Filter (CV model) over the pose stream to produce a
    smooth (px, py, vx, vy) estimate
  - Republish all current estimates as a single Float32MultiArray on
    /gmpc/obstacles

Wire format (Float32MultiArray.data) — 5 numbers per obstacle, flat:
    [x1, y1, r1, vx1, vy1,
     x2, y2, r2, vx2, vy2,
     ...]

Why a Kalman filter rather than the previous LSQ slope or cmd_vel ground truth?
  1. Realism: cmd_vel ground truth is a *cheat* — a real robot has no such
     channel. The KF only ingests pose measurements, so the same pipeline
     transfers directly to /scan-based detection (DBSCAN cluster centroids).
  2. Smoothness: LSQ slope over a fixed window has jagged transitions at
     ping-pong endpoints. The KF naturally interpolates with bounded noise.
  3. Calibrated uncertainty: P is available for future fusion with /scan
     or more sophisticated motion models.
"""

from __future__ import annotations

import os
from typing import List

import yaml

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg      import Float32MultiArray

from .kalman_tracker import KalmanTracker2D


class ObstacleAggregator(Node):

    FIELDS_PER_OBS = 5    # x, y, r, vx, vy

    def __init__(self):
        super().__init__('obstacle_aggregator')

        self.declare_parameter('trajectories_file', '')
        self.declare_parameter('publish_rate',      20.0)
        self.declare_parameter('output_topic',      '/gmpc/obstacles')

        # KF tuning — exposed so we can dial it from the launch file if needed.
        self.declare_parameter('kf_sigma_pos',   0.005)   # process σ pos [m/√s]
        self.declare_parameter('kf_sigma_vel',   1.5)     # process σ vel [m/s²]
        self.declare_parameter('kf_sigma_meas',  0.01)    # meas σ pos [m]
        self.declare_parameter('kf_init_vel_var', 1.0)    # initial P_vv

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

        sigma_pos    = float(self.get_parameter('kf_sigma_pos').value)
        sigma_vel    = float(self.get_parameter('kf_sigma_vel').value)
        sigma_meas   = float(self.get_parameter('kf_sigma_meas').value)
        init_vel_var = float(self.get_parameter('kf_init_vel_var').value)

        self._kf: dict[str, KalmanTracker2D | None] = {n: None for n in self._radius}
        self._kf_kwargs = dict(
            sigma_pos=sigma_pos, sigma_vel=sigma_vel,
            sigma_meas=sigma_meas, init_vel_var=init_vel_var,
        )

        # Logged for sanity-checking the KF estimate (kept off the wire).
        self._cmd_vel_gt = {n: (0.0, 0.0) for n in self._radius}
        self._cmd_vel_seen = set()

        for name in self._radius:
            self.create_subscription(
                PoseStamped, f'/model/{name}/pose',
                lambda msg, n=name: self._pose_cb(n, msg), 10,
            )
            self.create_subscription(
                Twist, f'/model/{name}/cmd_vel',
                lambda msg, n=name: self._vel_cb(n, msg), 10,
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
            f'with Kalman filter (σ_p={sigma_pos:.3f}, σ_v={sigma_vel:.2f}, '
            f'σ_meas={sigma_meas:.3f}) @ {rate:.1f} Hz -> '
            f'{self.get_parameter("output_topic").value}'
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _stamp_to_ns(msg: PoseStamped) -> int:
        return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)

    def _pose_cb(self, name: str, msg: PoseStamped):
        t_ns = self._stamp_to_ns(msg)
        if t_ns == 0:
            t_ns = self.get_clock().now().nanoseconds
        x = float(msg.pose.position.x)
        y = float(msg.pose.position.y)
        kf = self._kf[name]
        if kf is None:
            self._kf[name] = KalmanTracker2D(init_xy=(x, y), **self._kf_kwargs)
            self.get_logger().info(f'KF initialised for {name} at ({x:.2f}, {y:.2f})')
        else:
            kf.step(t_ns=t_ns, y_xy=(x, y))

    def _vel_cb(self, name: str, msg: Twist):
        self._cmd_vel_gt[name] = (float(msg.linear.x), float(msg.linear.y))
        if name not in self._cmd_vel_seen:
            self._cmd_vel_seen.add(name)
            self.get_logger().info(
                f'cmd_vel ground-truth stream seen for {name} '
                f'(KF is using pose measurements only)'
            )

    # ------------------------------------------------------------------
    def _tick(self):
        out = Float32MultiArray()
        flat: List[float] = []
        for name, r in self._radius.items():
            kf = self._kf[name]
            if kf is None:
                continue
            px, py = kf.position
            vx, vy = kf.velocity
            flat.extend([px, py, r, vx, vy])
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
