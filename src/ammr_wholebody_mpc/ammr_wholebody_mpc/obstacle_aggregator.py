"""Bridge: subscribe to each dynamic-obstacle pose topic that the Gazebo bridge
publishes, look up the obstacle radius from a YAML, and republish all current
positions as a single  std_msgs/Float32MultiArray  on  /gmpc/obstacles  for the
GMPC controller to consume.

Format on the wire:
    Float32MultiArray.data = [x1, y1, r1,  x2, y2, r2,  ...]
i.e. length = 3 × n_obstacles, all values in the global frame.

This is the *ground-truth* obstacle channel — useful for the first stage of
CBF integration. Future work: replace this aggregator with a /scan-based
detector + Kalman tracker so the pipeline doesn't depend on simulation
ground truth.
"""

from __future__ import annotations

import os
from typing import List

import yaml

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from std_msgs.msg      import Float32MultiArray


class ObstacleAggregator(Node):

    def __init__(self):
        super().__init__('obstacle_aggregator')

        self.declare_parameter('trajectories_file', '')
        self.declare_parameter('publish_rate',      20.0)
        self.declare_parameter('output_topic',      '/gmpc/obstacles')

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
        self._latest = {}                # name -> (x, y) most-recent pose

        for name in self._radius:
            topic = f'/model/{name}/pose'
            self.create_subscription(
                PoseStamped, topic,
                lambda msg, n=name: self._pose_cb(n, msg),
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
            f'@ {rate:.1f} Hz -> {self.get_parameter("output_topic").value}'
        )

    def _pose_cb(self, name: str, msg: PoseStamped):
        self._latest[name] = (float(msg.pose.position.x),
                              float(msg.pose.position.y))

    def _tick(self):
        out = Float32MultiArray()
        flat: List[float] = []
        for name, r in self._radius.items():
            pos = self._latest.get(name)
            if pos is None:
                continue
            flat.extend([pos[0], pos[1], r])
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
