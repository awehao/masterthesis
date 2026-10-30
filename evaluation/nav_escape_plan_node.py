#!/usr/bin/env python3
"""離門短段（門→抽屜串接 S1；Codex m5c 核可 A＋B）：交接後依 /odom 實際位姿，發一條沿世界 +y 的密集直線計畫給 GMPC（GMPC＋CBF、護盾、限制照舊）。
mission_dd_1：抽屜段起點距門扇 0.514 m（< 動態 CBF 邊距 0.69），整段導航命令為零；本段先離開門扇再接抽屜計畫。
離線核對（mission_room.pgm＋門扇 1.66° 線段）：起點→+0.4 m 路徑上最近佔用格 0.649→0.951 m、門扇 0.514→0.914 m。
成功＝沿 +y 推進 ≥ --min-progress-m（或 GMPC 到達）；期限 --timeout-s（模擬時間，自計畫發布起）內未達 ⇒ 失敗（rc 2）。結束時不再發布（計畫由下一段覆蓋）。
    python3 evaluation/nav_escape_plan_node.py --dy 0.4 --min-progress-m 0.25 --timeout-s 40 --out runs/<id>/escape.json
"""
import argparse
import json
import math
import time

import rclpy
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy


class Escape(Node):
    def __init__(self, a):
        super().__init__('nav_escape_plan')
        self.a = a
        self.pub = self.create_publisher(Path, '/plan', QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                                                   durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(Odometry, '/odom', self._odom, 10)
        self.pose = None
        self.t = None

    def _odom(self, m):
        q = m.pose.pose.orientation
        self.pose = (m.pose.pose.position.x, m.pose.pose.position.y, math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))
        self.t = m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec

    def plan(self, x0, y0):
        p = Path()
        p.header.frame_id = 'odom'
        n = max(2, int(math.ceil(self.a.dy / 0.05)) + 1)
        yaw = math.pi / 2
        for i in range(n):
            ps = PoseStamped()
            ps.header.frame_id = 'odom'
            ps.pose.position.x = x0
            ps.pose.position.y = y0 + self.a.dy * i / (n - 1)
            ps.pose.orientation.z, ps.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
            p.poses.append(ps)
        return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dy', type=float, default=0.4)
    ap.add_argument('--min-progress-m', type=float, default=0.25)
    ap.add_argument('--timeout-s', type=float, default=40.0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rclpy.init()
    nd = Escape(a)
    rep = {'args': vars(a), 'ok': False}
    w0 = time.monotonic()
    while nd.pose is None and time.monotonic() - w0 < 15.0:
        rclpy.spin_once(nd, timeout_sec=0.1)
    if nd.pose is None:
        rep['why'] = 'no_odom'
        json.dump(rep, open(a.out, 'w'), ensure_ascii=False, indent=1)
        print('[escape] **讀不到 /odom** ⇒ 失敗', flush=True)
        return 2
    x0, y0, th0 = nd.pose
    p = nd.plan(x0, y0)
    rep.update(start=[x0, y0, th0], t_start=nd.t, n_points=len(p.poses), goal=[x0, y0 + a.dy])
    nsub_prev, t_pub = -1, nd.t
    print(f'[escape] 起點 ({x0:.3f}, {y0:.3f}, {math.degrees(th0):.1f}°) → +y {a.dy} m（{len(p.poses)} 點）；期限 {a.timeout_s} s', flush=True)
    while rclpy.ok():
        rclpy.spin_once(nd, timeout_sec=0.05)
        ns = nd.pub.get_subscription_count()
        if ns > nsub_prev:                                   # 訂閱者增加時（含第一次）發布；GMPC 訂閱為 VOLATILE
            nd.pub.publish(p)
            rep.setdefault('published', []).append({'t': nd.t, 'n_sub': ns})
            nsub_prev = ns
        prog = nd.pose[1] - y0
        if prog >= a.min_progress_m:
            rep.update(ok=True, t_end=nd.t, end=list(nd.pose), progress_m=prog)
            print(f'[escape] **離門完成**：+y 推進 {prog:.3f} m（{nd.t - rep["t_start"]:.1f} s）；位姿 ({nd.pose[0]:.3f}, {nd.pose[1]:.3f}, {math.degrees(nd.pose[2]):.1f}°)', flush=True)
            break
        if nd.t is not None and t_pub is not None and nd.t - t_pub > a.timeout_s:
            rep.update(why='timeout', t_end=nd.t, end=list(nd.pose), progress_m=prog)
            print(f'[escape] **離門逾時**（{a.timeout_s} s）：推進 {prog:.3f} m ⇒ 失敗', flush=True)
            break
    json.dump(rep, open(a.out, 'w'), ensure_ascii=False, indent=1, default=float)
    nd.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    return 0 if rep['ok'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
