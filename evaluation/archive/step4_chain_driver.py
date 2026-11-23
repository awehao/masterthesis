#!/usr/bin/env python3
"""命令鏈整合驗收的驅動與記錄節點（無障礙開闊區域）。

鏈路：本節點 -> /cmd_vel_nav -> smoother -> /cmd_vel_smoothed -> guard -> /cmd_vel -> Isaac

本節點只發固定前進命令並記錄四條序列；故障注入由外部腳本以終止程序的方式進行，
注入時刻由該腳本另外寫檔，事後合併。

停止判定（**事前固定**）：真值平移速度 <= 0.01 m/s 且持續 >= 1.0 s 模擬時間，
真值取樣為模擬器的 20 Hz 位姿發布。
"""
import argparse, json, math, os, time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--seconds', type=float, default=40.0, help='牆鐘上限')
ap.add_argument('--vx', type=float, default=0.20)
ap.add_argument('--wz', type=float, default=0.50)
ap.add_argument('--rate', type=float, default=20.0)
a = ap.parse_args()

rclpy.init()
n = Node('step4_chain_driver')
be = QoSProfile(depth=20, reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST)
pub = n.create_publisher(Twist, '/cmd_vel_nav', 10)

nav, smo, out, sta, pose = [], [], [], [], []
mk = lambda L: (lambda m: L.append((time.monotonic(), m.linear.x, m.linear.y,  # noqa
                                    m.angular.z)))
n.create_subscription(Twist, '/cmd_vel_smoothed', mk(smo), 10)
n.create_subscription(Twist, '/cmd_vel', mk(out), 10)
n.create_subscription(String, '/wheel_guard/status',
                      lambda m: sta.append((time.monotonic(), m.data)), 50)

def _pose(m):
    s = m.header.stamp
    pose.append((time.monotonic(), s.sec + s.nanosec * 1e-9,
                 m.pose.position.x, m.pose.position.y,
                 m.pose.orientation.z, m.pose.orientation.w))
n.create_subscription(PoseStamped, '/model/omni_bot/pose', _pose, be)

t0 = time.monotonic()
while time.monotonic() - t0 < 20.0 and len(pose) < 3:
    rclpy.spin_once(n, timeout_sec=0.1)
if len(pose) < 3:
    print('  !! 沒有收到 /model/omni_bot/pose，模擬器未就緒'); raise SystemExit(2)
print(f'  模擬器已就緒（{len(pose)} 筆位姿）')

period = 1.0 / a.rate
nxt = time.monotonic()
t_start = time.monotonic()
while time.monotonic() - t_start < a.seconds:
    m = Twist()
    m.linear.x, m.angular.z = float(a.vx), float(a.wz)
    pub.publish(m)
    nav.append((time.monotonic(), m.linear.x, m.linear.y, m.angular.z))
    nxt += period
    while time.monotonic() < nxt:
        rclpy.spin_once(n, timeout_sec=0.002)
n.destroy_node(); rclpy.shutdown()

rec = dict(schema='step4_chain/1', vx=a.vx, wz=a.wz, rate=a.rate,
           stop_speed_thresh=0.01, stop_hold_s=1.0,
           nav=nav, smoothed=smo, out=out, pose=pose,
           status=[[t, s] for t, s in sta])
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump(rec, open(a.out, 'w'), ensure_ascii=False)
print(f'  /cmd_vel_nav {len(nav)}  /cmd_vel_smoothed {len(smo)}  '
      f'/cmd_vel {len(out)}  status {len(sta)}  pose {len(pose)}')
print(f'  -> {a.out}')
