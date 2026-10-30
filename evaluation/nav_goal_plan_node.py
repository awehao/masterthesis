#!/usr/bin/env python3
"""導航分段節點（返程；規格 door_return_phase3_r1.md）：依 /odom 實際位姿發密集折線計畫（點距 ≤ 5 cm）給 GMPC＋CBF，到達半徑內或期限（模擬秒）結束；訂閱者增加時補發。不改控制與保護。
    python3 evaluation/nav_goal_plan_node.py --wp "x,y;x,y" --tol 0.25 --timeout-s 120 --out ...
"""
import argparse
import json
import math

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy


class Nd(Node):
    def __init__(self):
        super().__init__('nav_goal_plan')
        self.pub = self.create_publisher(Path, '/plan', QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(Odometry, '/odom', self._o, 10)
        self.xy, self.t = None, None

    def _o(self, m):
        self.xy = (m.pose.pose.position.x, m.pose.pose.position.y)
        self.t = m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wp', required=True)
    ap.add_argument('--tol', type=float, default=0.25)
    ap.add_argument('--timeout-s', type=float, default=120.0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rclpy.init()
    nd = Nd()
    while nd.xy is None:
        rclpy.spin_once(nd, timeout_sec=0.1)
    wp = [nd.xy] + [tuple(float(v) for v in p.split(',')) for p in a.wp.split(';')]
    p = Path()
    p.header.frame_id = 'odom'
    for (x0, y0), (x1, y1) in zip(wp[:-1], wp[1:]):
        n = max(1, int(math.ceil(math.hypot(x1 - x0, y1 - y0) / 0.05)))
        h = math.atan2(y1 - y0, x1 - x0)
        for k in range(n + 1):
            ps = PoseStamped()
            ps.header.frame_id = 'odom'
            ps.pose.position.x, ps.pose.position.y = x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n
            ps.pose.orientation.z, ps.pose.orientation.w = math.sin(h / 2), math.cos(h / 2)
            p.poses.append(ps)
    rep = {'wp': wp, 'n': len(p.poses), 't0': nd.t, 'ok': False}
    ns0, t0 = -1, nd.t
    print(f'[nav] 計畫 {len(p.poses)} 點 → {wp[-1]}', flush=True)
    while rclpy.ok():
        rclpy.spin_once(nd, timeout_sec=0.05)
        ns = nd.pub.get_subscription_count()
        if ns > ns0:
            nd.pub.publish(p)
            ns0 = ns
        d = math.hypot(nd.xy[0] - wp[-1][0], nd.xy[1] - wp[-1][1])
        if d <= a.tol:
            rep.update(ok=True, t1=nd.t, d=d, xy=nd.xy)
            print(f'[nav] 到達 d={d:.3f} m（{nd.t - t0:.1f} s）', flush=True)
            break
        if nd.t - t0 > a.timeout_s:
            rep.update(why='timeout', t1=nd.t, d=d, xy=nd.xy)
            print(f'[nav] **逾時** d={d:.3f} m', flush=True)
            break
    json.dump(rep, open(a.out, 'w'), ensure_ascii=False, indent=1, default=float)
    return 0 if rep['ok'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
