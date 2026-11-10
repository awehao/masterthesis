#!/usr/bin/env python3
"""第八次：主動保持的功能核對探針（Codex 20261110_200117 核對順序 1–3；假世界）。
時程（模擬時間，自 CRUISE 起）：0–T_GO 沿車頭方向的直線計畫前進；T_GO 發 /gmpc/hold active＝true（主動保持）；保持 T_HOLD 秒；
解除保持並重發原計畫，再跑 T_AFTER 秒結束。選擇器同導航節點（轉送平滑器輸出到 /cmd_vel_pre_shield）。發 /m8nav/state 供假世界起步同步。
QP 是否連續求解、CBF 是否改變命令、進出是否跳變，由 bag（/gmpc/diag_v2、/cmd_vel*、/shield/diag）事後分析（m8_hold_check.py）。
"""
import argparse
import json
import math
import os
import signal

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

T_GO, T_HOLD, T_AFTER = float(os.environ.get('M8_T_GO', 3.0)), float(os.environ.get('M8_T_HOLD', 14.0)), float(os.environ.get('M8_T_AFTER', 4.0))


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class HoldProbe(Node):
    def __init__(self, a):
        super().__init__('m8_hold_probe')
        self.a = a
        lat = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.plan_pub = self.create_publisher(Path, '/plan', lat)
        self.hold_pub = self.create_publisher(String, '/gmpc/hold', 10)
        self.out_pub = self.create_publisher(Twist, '/cmd_vel_pre_shield', 10)
        self.state_pub = self.create_publisher(String, '/m8nav/state', 10)
        self.create_subscription(Odometry, '/odom', self._odom, 10)
        self.create_subscription(Twist, '/cmd_vel_pre_mux', self._cin, 10)
        self.pose = self.t = None
        self.cin, self.cin_t = (0.0, 0.0, 0.0), None
        self.seg, self.t0, self.events = 'WAIT', None, []
        self.rep = dict(version='m8_hold_probe/r1', T_GO=T_GO, T_HOLD=T_HOLD, T_AFTER=T_AFTER, result=None, events=self.events)
        self.create_timer(0.05, self._tick)

    def _odom(self, m):
        self.t = m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec
        p = m.pose.pose
        self.pose = (p.position.x, p.position.y, yaw_of(p.orientation))

    def _cin(self, m):
        self.cin, self.cin_t = (m.linear.x, m.linear.y, m.angular.z), self.t

    def _hold(self, act):
        self.hold_pub.publish(String(data=json.dumps(dict(active=act, id=len(self.events)))))
        self.events.append(dict(t=self.t, hold=act, pose=list(self.pose)))

    def _tick(self):
        if self.pose is None:
            return
        x, y, th = self.pose
        if self.seg == 'WAIT':
            p = Path()
            p.header.frame_id = 'odom'
            p.header.stamp = self.get_clock().now().to_msg()
            for k in range(41):
                ps = PoseStamped()
                ps.header = p.header
                ps.pose.position.x, ps.pose.position.y = x + k * 0.05 * math.cos(th), y + k * 0.05 * math.sin(th)
                ps.pose.orientation.z, ps.pose.orientation.w = math.sin(th / 2), math.cos(th / 2)
                p.poses.append(ps)
            self.plan = p
            self.plan_pub.publish(p)
            self.seg, self.t0 = 'CRUISE', self.t
            self.events.append(dict(t=self.t, seg='CRUISE', pose=list(self.pose)))
        el = self.t - self.t0
        if self.seg == 'CRUISE' and el >= T_GO:
            self._hold(True)
            self.seg = 'HOLD'
        elif self.seg == 'HOLD':
            if int(el * 20) % 20 == 0:
                self.hold_pub.publish(String(data=json.dumps(dict(active=True, id=1))))
            if el >= T_GO + T_HOLD:
                self._hold(False)
                self.plan_pub.publish(self.plan)
                self.seg = 'AFTER'
        elif self.seg == 'AFTER' and el >= T_GO + T_HOLD + T_AFTER:
            self.seg = 'ZERO'
            self.rep['result'] = 'done'
        u = self.cin if (self.seg in ('CRUISE', 'HOLD', 'AFTER') and self.cin_t is not None and self.t - self.cin_t <= 0.5) else (0.0, 0.0, 0.0)
        tw = Twist()
        tw.linear.x, tw.linear.y, tw.angular.z = u
        self.out_pub.publish(tw)
        self.state_pub.publish(String(data=json.dumps(dict(seg='CRUISE' if self.seg != 'WAIT' else 'WAIT', probe=self.seg, t=self.t))))
        json.dump(self.rep, open(self.a.out, 'w'), ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a, _ = ap.parse_known_args()
    rclpy.init()
    n = HoldProbe(a)
    stop = {'v': False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(v=True))
    try:
        while rclpy.ok() and not stop['v']:
            rclpy.spin_once(n, timeout_sec=0.02)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
