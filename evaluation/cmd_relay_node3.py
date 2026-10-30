#!/usr/bin/env python3
"""門→抽屜待命接手（銜接包）：cmd_relay_node2 的版本化擴充。起跑即啟動、預先載入，待命期**不建立** /plan 與護盾入口發布者，不影響門段唯一輸出。
收到 /mission/subtask_done（門段 DONE）後：確認舊程序群組已退出、指定話題發布者全為 0 → 由實測位置發本段計畫 → 收到 GMPC 採用回報 → 才建立輸出並以 cmd_relay_node2 規則承接。
隔離未過、計畫未採用皆逾時退出（rc 2），不輸出。
    python3 evaluation/cmd_relay_node3.py --plan-wp=x,y --zero /a /b --dead-pgids 1 2
"""
import argparse
import math
import os
import sys
import time

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cmd_relay_node2 import Relay as Relay2                    # noqa: E402


def plan_msg(xy, wps, stamp):
    w = [tuple(float(v) for v in xy)] + [tuple(float(v) for v in p.split(',')) for p in wps.split(';')]
    p = Path()
    p.header.frame_id = 'odom'
    p.header.stamp.sec, p.header.stamp.nanosec = stamp
    for (x0, y0), (x1, y1) in zip(w[:-1], w[1:]):
        n = max(1, int(math.ceil(math.hypot(x1 - x0, y1 - y0) / 0.05)))
        h = math.atan2(y1 - y0, x1 - x0)
        for k in range(n + 1):
            ps = PoseStamped()
            ps.header.frame_id = 'odom'
            ps.pose.position.x, ps.pose.position.y = x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n
            ps.pose.orientation.z, ps.pose.orientation.w = math.sin(h / 2), math.cos(h / 2)
            p.poses.append(ps)
    return p


class Standby(Relay2):
    """Relay2 的輸出延到接手才建立。"""

    def __init__(self, a):
        self.out_ready = False
        super().__init__(a)

    def create_publisher(self, typ, topic, *x, **k):
        if not self.out_ready and topic == getattr(getattr(self, 'a', None), 'out_topic', None):
            return None                                                         # 待命：護盾入口不建立發布者
        return super().create_publisher(typ, topic, *x, **k)

    def _create_out(self):
        self.out_ready = True
        self.pub = self.create_publisher(Twist, self.a.out_topic, 10)

    def _cb(self, m):
        if self.out_ready:
            super()._cb(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-topic', default='/cmd_vel_pre_mux')
    ap.add_argument('--out-topic', default='/cmd_vel_pre_shield')
    ap.add_argument('--a-lin', type=float, default=0.5)
    ap.add_argument('--a-ang', type=float, default=1.2)
    ap.add_argument('--blend-max-s', type=float, default=6.0)
    ap.add_argument('--plan-wp', required=True)
    ap.add_argument('--zero', nargs='+', required=True)
    ap.add_argument('--dead-pgids', nargs='*', type=int, default=[])
    ap.add_argument('--hold-s', type=float, default=0.3)
    ap.add_argument('--iso-timeout-s', type=float, default=20.0)
    ap.add_argument('--ack-timeout-s', type=float, default=5.0)
    a, _ = ap.parse_known_args()
    rclpy.init()

    class Nd(Standby):
        def __init__(self, a_):
            self.done, self.od, self.ack = None, None, None
            super().__init__(a_)
            lat = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.create_subscription(String, '/mission/subtask_done', lambda m: setattr(self, 'done', m.data), lat)
            self.od_wall = None
            self.create_subscription(Odometry, '/odom', self._od, 10)
            self.create_subscription(String, '/gmpc/plan_ack', lambda m: setattr(self, 'ack', m.data), 10)

        def _od(self, m):
            self.od, self.od_wall = m, time.monotonic()

    nd = Nd(a)
    print(f'[relay] 待命：{a.zero} 隔離後接手；計畫 → {a.plan_wp}', flush=True)

    def alive(pg):
        try:
            os.killpg(pg, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def spin_until(fn, timeout):
        w0 = time.monotonic()
        while rclpy.ok() and not fn():
            rclpy.spin_once(nd, timeout_sec=0.01)
            if timeout is not None and time.monotonic() - w0 > timeout:
                return False
        return True
    rc = 0
    try:
        spin_until(lambda: nd.done is not None, None)
        t_d = time.monotonic()
        print(f'[relay] 收到 subtask_done，隔離核對 @ sim {nd.t}', flush=True)
        t_ok = [None]

        def iso():
            if any(alive(p) for p in a.dead_pgids) or any(nd.count_publishers(t) > 0 for t in a.zero):
                t_ok[0] = None
                return False
            t_ok[0] = t_ok[0] if t_ok[0] is not None else time.monotonic()
            return time.monotonic() - t_ok[0] >= a.hold_s
        if not spin_until(iso, a.iso_timeout_s):
            print(f'[relay] **隔離未通過**：{[p for p in a.dead_pgids if alive(p)]} {[t for t in a.zero if nd.count_publishers(t) > 0]}', flush=True)
            return 2
        def od_fresh():                                                         # 計畫起點：時間戳新鮮、有限、晚於隔離完成
            if nd.od is None or nd.t is None or nd.od_wall is None or nd.od_wall < t_iso:
                return False
            ps, h = nd.od.pose.pose.position, nd.od.header.stamp
            ts = float(h.sec) + 1e-9 * float(h.nanosec)
            return all(math.isfinite(v) for v in (ps.x, ps.y)) and abs(float(nd.t) - ts) <= 0.2
        t_iso = time.monotonic()
        if not spin_until(od_fresh, 3.0):
            print('[relay] **無新鮮里程** ⇒ 不發計畫、不輸出', flush=True)
            return 2
        p = nd.od.pose.pose.position
        t = float(nd.t)
        st = (int(t), int((t % 1) * 1e9))
        pid = f'{st[0]}.{st[1]:09d}'
        pub = nd.create_publisher(Path, '/plan', QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        msg = plan_msg((p.x, p.y), a.plan_wp, st)
        pub.publish(msg)
        if not spin_until(lambda: nd.ack == pid, a.ack_timeout_s):
            print(f'[relay] **計畫未獲採用** id {pid} 收到 {nd.ack}', flush=True)
            return 2
        nd.destroy_publisher(pub)                                              # 採用後不留舊的 transient-local 計畫
        nd._create_out()
        print(f'[relay] ODOM {p.x:.6f},{p.y:.6f}；隔離＋計畫 {time.monotonic() - t_d:.2f} s（牆鐘）；開放輸出 @ sim {nd.t}', flush=True)
        rclpy.spin(nd)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        print(f"[relay] 結束：{nd.rep['n']} 筆、丟棄 {nd.rep['n_drop']}、接手 {nd.rep['takeover']}、失效 {nd.fail}", flush=True)
        nd.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
