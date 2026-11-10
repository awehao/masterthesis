#!/usr/bin/env python3
"""第八次第一包：計畫／航向執行契約核對（Codex 20261110_190030 第 1 項；無行人）。
從當下位姿發一條 1.0 m 的直線計畫（各點朝向＝當下車頭；方向由 M8_PROBE_DIR：fwd 車頭前方、lat 車頭左方、back 車頭後方），
選擇器同導航節點（轉送 /cmd_vel_pre_mux 到 /cmd_vel_pre_shield），到終點 0.10 m 內或逾時即結束。
介面同導航節點（--goal-file、--out、--nav-timeout-s 可給但不用目標檔），讓 run_m8_fake.sh 可直接替換 NAV_NODE。
記錄：實測軌跡、最終偏航改變量、到達時間；GMPC 的航向參考、解與輪級動作另由 bag（/gmpc/heading、/gmpc/diag_v2、/cmd_vel*）分析。
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


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class Probe(Node):
    def __init__(self, a):
        super().__init__('m8_contract_probe')
        self.a = a
        self.dir = os.environ.get('M8_PROBE_DIR', 'fwd')
        lat = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.plan_pub = self.create_publisher(Path, '/plan', lat)
        self.out_pub = self.create_publisher(Twist, '/cmd_vel_pre_shield', 10)
        self.state_pub = self.create_publisher(String, '/m8nav/state', 10)
        self.create_subscription(Odometry, '/odom', self._odom, 10)
        self.create_subscription(Twist, '/cmd_vel_pre_mux', self._cin, 10)
        self.pose = self.t = None
        self.cin, self.cin_t = (0.0, 0.0, 0.0), None
        self.seg, self.t0, self.goal, self.traj = 'WAIT', None, None, []
        self.rep = dict(version='m8_contract_probe/r1', dir=self.dir, result=None, segments=[])
        self.create_timer(0.05, self._tick)

    def _odom(self, m):
        self.t = m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec
        p = m.pose.pose
        self.pose = (p.position.x, p.position.y, yaw_of(p.orientation))

    def _cin(self, m):
        self.cin, self.cin_t = (m.linear.x, m.linear.y, m.angular.z), self.t

    def _tick(self):
        if self.pose is None:
            return
        x, y, th = self.pose
        if self.seg == 'WAIT':
            ang = th + dict(fwd=0.0, lat=math.pi / 2, back=math.pi)[self.dir]
            self.goal = (x + math.cos(ang), y + math.sin(ang))
            p = Path()
            p.header.frame_id = 'odom'
            p.header.stamp = self.get_clock().now().to_msg()
            for k in range(21):
                ps = PoseStamped()
                ps.header = p.header
                ps.pose.position.x, ps.pose.position.y = x + k * 0.05 * math.cos(ang), y + k * 0.05 * math.sin(ang)
                ps.pose.orientation.z, ps.pose.orientation.w = math.sin(th / 2), math.cos(th / 2)
                p.poses.append(ps)
            self.plan_msg = p
            self.plan_pub.publish(p)
            self.seg, self.t0 = 'CRUISE', self.t
            self.rep.update(start=[x, y, th], goal=list(self.goal))
            self.rep['segments'].append(dict(seg='CRUISE', t=self.t))
        if self.seg == 'CRUISE':
            if self.t - self.t0 < 3.0:
                self.plan_pub.publish(self.plan_msg)
            d = math.hypot(self.goal[0] - x, self.goal[1] - y)
            self.traj.append([self.t, x, y, th])
            if d <= 0.10 or self.t - self.t0 > 15.0:
                self.rep.update(result='done' if d <= 0.10 else 'timeout', t_used=self.t - self.t0, dist=d, yaw_change=math.atan2(math.sin(th - self.rep['start'][2]), math.cos(th - self.rep['start'][2])))
                self.seg = 'ZERO'
                self.rep['segments'].append(dict(seg='ZERO', t=self.t))
        u = self.cin if (self.seg == 'CRUISE' and self.cin_t is not None and self.t - self.cin_t <= 0.5) else (0.0, 0.0, 0.0)
        tw = Twist()
        tw.linear.x, tw.linear.y, tw.angular.z = u
        self.out_pub.publish(tw)
        self.state_pub.publish(String(data=json.dumps(dict(seg=self.seg, t=self.t))))
        self.rep['traj'] = self.traj[::5]
        json.dump(self.rep, open(self.a.out, 'w'), ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a, _ = ap.parse_known_args()
    rclpy.init()
    n = Probe(a)
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
