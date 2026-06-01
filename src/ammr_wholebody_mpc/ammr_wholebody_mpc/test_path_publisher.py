"""Publish a one-shot test plan on /plan so the GMPC node can be exercised
without a full Nav2 planner stack.

Currently supports a single shape via the `shape` parameter:
  * 'line'   : straight line from (0,0) to (`length`, 0)
  * 'square' : closed-loop square of side `length`
  * 'arc'    : quarter circle of radius `length` starting at origin going +x then curving +y

Parameters
----------
shape       : 'line' | 'square' | 'arc'
length      : trajectory characteristic length [m]
spacing     : path-point spacing [m]
frame_id    : header frame (default 'map')
publish_period : seconds between re-publishes (latched-style for stragglers)
"""

from __future__ import annotations

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg      import Path


def _line(length: float, ds: float) -> np.ndarray:
    n = max(2, int(round(length / ds)) + 1)
    return np.stack([np.linspace(0.0, length, n), np.zeros(n)], axis=1)


def _square(side: float, ds: float) -> np.ndarray:
    n = max(2, int(round(side / ds)) + 1)
    seg = np.linspace(0.0, side, n)
    pts = [np.stack([seg,           np.zeros(n)],       axis=1),   # +x
           np.stack([seg[-1] + np.zeros(n), seg],       axis=1),   # +y
           np.stack([seg[::-1],     seg[-1] + np.zeros(n)], axis=1),  # -x
           np.stack([np.zeros(n),   seg[::-1]],         axis=1)]   # -y
    return np.concatenate(pts, axis=0)


def _arc(radius: float, ds: float) -> np.ndarray:
    arclen = 0.5 * np.pi * radius                # quarter circle
    n = max(4, int(round(arclen / ds)) + 1)
    t = np.linspace(0.0, np.pi / 2.0, n)
    return np.stack([radius * np.sin(t), radius * (1.0 - np.cos(t))], axis=1)


def _build_path_msg(xy: np.ndarray, frame_id: str, stamp) -> Path:
    msg = Path()
    msg.header.frame_id = frame_id
    msg.header.stamp    = stamp
    for i in range(xy.shape[0]):
        ps = PoseStamped()
        ps.header = msg.header
        ps.pose.position.x = float(xy[i, 0])
        ps.pose.position.y = float(xy[i, 1])
        ps.pose.orientation.w = 1.0       # path_processor uses tangent for yaw
        msg.poses.append(ps)
    return msg


class TestPathPublisher(Node):

    def __init__(self):
        super().__init__('gmpc_test_path_publisher')

        self.declare_parameter('shape',          'line')
        self.declare_parameter('length',         3.0)
        self.declare_parameter('spacing',        0.05)
        self.declare_parameter('frame_id',       'map')
        self.declare_parameter('topic',          '/plan')
        self.declare_parameter('publish_period', 1.0)

        shape   = str(self.get_parameter('shape').value)
        length  = float(self.get_parameter('length').value)
        ds      = float(self.get_parameter('spacing').value)
        frame   = str(self.get_parameter('frame_id').value)
        topic   = str(self.get_parameter('topic').value)
        period  = float(self.get_parameter('publish_period').value)

        if   shape == 'line'  : xy = _line(length, ds)
        elif shape == 'square': xy = _square(length, ds)
        elif shape == 'arc'   : xy = _arc(length, ds)
        else: raise ValueError(f'unknown shape: {shape!r}')

        self._frame = frame
        self._xy    = xy

        # Transient-local QoS so a late subscriber still gets the path
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history    =QoSHistoryPolicy.KEEP_LAST,
            depth      =1,
            durability =QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub = self.create_publisher(Path, topic, qos)
        self.create_timer(period, self._tick)

        self.get_logger().info(
            f'Publishing {shape!r} plan ({xy.shape[0]} points, '
            f'~{ds:.3f}m spacing) on {topic} in frame {frame!r}'
        )

    def _tick(self):
        msg = _build_path_msg(self._xy, self._frame, self.get_clock().now().to_msg())
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = TestPathPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
