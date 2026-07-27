"""Live CBF diagnostic: why does the robot hit a wall after the CBF kicks in?

Prints one line per second with everything needed to tell the two failure modes
apart:

  * static-CBF blind : /gmpc/static_obstacles is EMPTY while the robot is close
    to a wall -> the CBF cannot see the wall, so dodging a dynamic obstacle
    pushes it straight into one.
  * static-CBF active but overruled : wall points present, but min_h goes
    negative anyway -> margin/alpha too weak, or the QP slacks the constraint.

Columns
  t        seconds since start
  x,y      robot position (ground-truth /odom)
  wall     true distance to the nearest wall surface (from /map, EDT)
  clr      wall distance minus robot radius (0.30) -> NEGATIVE means overlap
  nStat    number of wall points the CBF currently has
  nDyn     number of dynamic obstacles the tracker publishes
  min_h    CBF barrier value (<0 = inside the keep-out of something)
  cmd      the velocity command being applied

Run alongside the simulation:
    python3 evaluation/watch_cbf.py
"""
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry, OccupancyGrid
from std_msgs.msg import Float32MultiArray, Float32
from geometry_msgs.msg import Twist

ROBOT_R = 0.30


class Watch(Node):
    def __init__(self):
        super().__init__('watch_cbf')
        self.odom = self.stat = self.dyn = self.mh = self.cmd = None
        self.wall_dist = None
        self.info = None
        self.create_subscription(Odometry, '/odom', self._odom, 10)
        self.create_subscription(Float32MultiArray, '/gmpc/static_obstacles',
                                 lambda m: setattr(self, 'stat', list(m.data)), 10)
        self.create_subscription(Float32MultiArray, '/gmpc/obstacles',
                                 lambda m: setattr(self, 'dyn', list(m.data)), 10)
        self.create_subscription(Float32, '/gmpc/min_h',
                                 lambda m: setattr(self, 'mh', m.data), 10)
        self.create_subscription(Twist, '/cmd_vel',
                                 lambda m: setattr(self, 'cmd', m), 10)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, '/map', self._map, qos)
        self.t0 = time.time()
        self.worst = 1e9
        self.create_timer(1.0, self._tick)
        print(f"{'t':>5} {'x':>6} {'y':>6} {'wall':>6} {'clr':>7} "
              f"{'nStat':>5} {'nDyn':>4} {'min_h':>7}  cmd")

    def _odom(self, m):
        self.odom = m

    def _map(self, m):
        from scipy.ndimage import distance_transform_edt
        g = np.array(m.data, dtype=np.int16).reshape(m.info.height, m.info.width)
        self.wall_dist = distance_transform_edt(~(g >= 50)) * m.info.resolution
        self.info = m.info
        print(f"[map] {m.info.width}x{m.info.height} @ {m.info.resolution} m")

    def _tick(self):
        if self.odom is None:
            return
        p = self.odom.pose.pose.position
        wall = float('nan')
        if self.wall_dist is not None:
            c = int((p.x - self.info.origin.position.x) / self.info.resolution)
            r = int((p.y - self.info.origin.position.y) / self.info.resolution)
            if 0 <= r < self.wall_dist.shape[0] and 0 <= c < self.wall_dist.shape[1]:
                wall = float(self.wall_dist[r, c])
        clr = wall - ROBOT_R
        ns = len(self.stat) // 5 if self.stat is not None else -1
        nd = len(self.dyn) // 5 if self.dyn is not None else -1
        mh = self.mh if self.mh is not None else float('nan')
        c = self.cmd
        cmds = (f"vx={c.linear.x:+.2f} vy={c.linear.y:+.2f} wz={c.angular.z:+.2f}"
                if c else "-")
        flag = ''
        if math.isfinite(clr):
            self.worst = min(self.worst, clr)
            if clr < 0:
                flag = '  <== 撞牆(重疊)'
            elif clr < 0.10 and ns == 0:
                flag = '  <== 貼牆但 CBF 沒有牆點!'
        print(f"{time.time()-self.t0:5.0f} {p.x:6.2f} {p.y:6.2f} {wall:6.2f} "
              f"{clr:+7.2f} {ns:5d} {nd:4d} {mh:7.3f}  {cmds}{flag}")


def main():
    rclpy.init()
    n = Watch()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        print(f"\n最小淨距 = {n.worst:+.3f} m  (<0 表示真的撞進牆裡)")
    finally:
        n.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
