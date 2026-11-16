#!/usr/bin/env python3
"""探測：原導航（goal_to_plan_relay→Smac2D）從當下位置到幾個門邊候選目標的 /plan 形狀（只錄路徑，不評估控制）。
    python3 probe_smac_paths.py OUT.json "x,y,yaw;x,y,yaw;..." """
import json, math, sys, time
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node


class P(Node):
    def __init__(self):
        super().__init__('smac_probe')
        self.plan, self.t_plan = None, 0.0
        self.create_subscription(Path, '/plan', self._p, 10)
        self.pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

    def _p(self, m):
        self.plan = [[round(p.pose.position.x, 4), round(p.pose.position.y, 4), round(2 * math.atan2(p.pose.orientation.z, p.pose.orientation.w), 4)] for p in m.poses]
        self.t_plan = time.time()


def main():
    out, goals = sys.argv[1], [[float(v) for v in g.split(',')] for g in sys.argv[2].split(';')]
    rclpy.init(); n = P(); res = []
    for g in goals:
        m = PoseStamped(); m.header.frame_id = 'map'
        m.pose.position.x, m.pose.position.y = g[0], g[1]; m.pose.orientation.z, m.pose.orientation.w = math.sin(g[2] / 2), math.cos(g[2] / 2)
        t0 = time.time(); n.plan = None
        while time.time() - t0 < 8.0 and n.plan is None:
            m.header.stamp = n.get_clock().now().to_msg(); n.pub.publish(m)
            end = time.time() + 1.0
            while time.time() < end and n.plan is None:
                rclpy.spin_once(n, timeout_sec=0.05)
        res.append(dict(goal=g, plan=n.plan, n=None if n.plan is None else len(n.plan)))
        print('goal', g, 'plan pts', res[-1]['n'], 'end', None if n.plan is None else n.plan[-5:], flush=True)
    json.dump(res, open(out, 'w'), indent=1)
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
