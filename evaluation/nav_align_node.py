#!/usr/bin/env python3
"""末端對位（返程回起點；語義同 door_nav1 PARK_ALIGN）：GMPC 到達後以 P 伺服收斂位置與朝向，輸出接護盾入口，不繞過掃描保護。
起跑前呼叫端須停掉 /cmd_vel_pre_shield 的其他來源。
    python3 evaluation/nav_align_node.py --goal="-0.6,-2.8,1.5708" --out x.json
"""
import argparse
import json
import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock


class Nd(Node):
    def __init__(self):
        super().__init__('nav_align')
        self.pub = self.create_publisher(Twist, '/cmd_vel_pre_shield', 10)
        self.create_subscription(Odometry, '/odom', self._o, 10)
        self.create_subscription(Clock, '/clock', self._c, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.p, self.t = None, None

    def _o(self, m):
        q = m.pose.pose.orientation
        self.p = (m.pose.pose.position.x, m.pose.pose.position.y, math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2)))

    def _c(self, m):
        self.t = float(m.clock.sec) + 1e-9 * float(m.clock.nanosec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--goal', required=True)
    ap.add_argument('--v-max', type=float, default=0.10)
    ap.add_argument('--w-max', type=float, default=0.30)
    ap.add_argument('--k-v', type=float, default=1.0)
    ap.add_argument('--k-w', type=float, default=1.0)
    ap.add_argument('--a-lin', type=float, default=0.25)
    ap.add_argument('--a-ang', type=float, default=0.6)
    ap.add_argument('--tol-xy', type=float, default=0.02)
    ap.add_argument('--tol-yaw', type=float, default=0.03)
    ap.add_argument('--hold-s', type=float, default=0.5)
    ap.add_argument('--timeout-s', type=float, default=40.0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    gx, gy, gth = (float(v) for v in a.goal.split(','))
    rclpy.init()
    nd = Nd()
    while nd.p is None or nd.t is None:
        rclpy.spin_once(nd, timeout_sec=0.1)
    t0, tl, ok_t, u = nd.t, nd.t, None, [0.0, 0.0, 0.0]
    rep = {'goal': [gx, gy, gth], 'start': list(nd.p), 't0': t0, 'ok': False}
    print(f'[nav] 末端對位 起 {nd.p[0]:.3f},{nd.p[1]:.3f},{nd.p[2]:.3f} → {gx},{gy},{gth}', flush=True)
    while rclpy.ok():
        rclpy.spin_once(nd, timeout_sec=0.02)
        if nd.t <= tl:
            continue
        dt, tl = nd.t - tl, nd.t
        x, y, th = nd.p
        ex, ey = gx - x, gy - y
        ey_ = math.atan2(math.sin(gth - th), math.cos(gth - th))
        d = math.hypot(ex, ey)
        s = min(1.0, a.v_max / max(a.k_v * d, 1e-9))
        c, sn = math.cos(th), math.sin(th)
        tg = [a.k_v * s * (ex * c + ey * sn), a.k_v * s * (-ex * sn + ey * c), max(-a.w_max, min(a.w_max, a.k_w * ey_))]
        for i, lim in ((0, a.a_lin), (1, a.a_lin), (2, a.a_ang)):
            u[i] += max(-lim * dt, min(lim * dt, tg[i] - u[i]))
        m = Twist()
        m.linear.x, m.linear.y, m.angular.z = u
        nd.pub.publish(m)
        if d <= a.tol_xy and abs(ey_) <= a.tol_yaw:
            ok_t = nd.t if ok_t is None else ok_t
            if nd.t - ok_t >= a.hold_s:
                rep.update(ok=True)
                break
        else:
            ok_t = None
        if nd.t - t0 > a.timeout_s:
            rep['why'] = 'timeout'
            break
    nd.pub.publish(Twist())
    rep.update(end=list(nd.p), t1=nd.t, d=math.hypot(gx - nd.p[0], gy - nd.p[1]), yaw_e=math.atan2(math.sin(gth - nd.p[2]), math.cos(gth - nd.p[2])))
    json.dump(rep, open(a.out, 'w'), ensure_ascii=False, indent=1, default=float)
    print(f"[nav] 末端對位 {'完成' if rep['ok'] else '**未完成**'} d={rep['d']:.3f} m、朝向差 {rep['yaw_e']:.3f} rad（{nd.t - t0:.1f} s）", flush=True)
    return 0 if rep['ok'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
