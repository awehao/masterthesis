#!/usr/bin/env python3
"""比對 /clock 與真值位姿：/clock 是否與實際運動同步。

rosbag2 在被中止時可能不寫出 metadata（實測留下 0 位元組的 mcap），所以這裡
直接訂閱並自己寫 CSV，收尾不依賴錄製器的生命週期。
"""
import argparse, csv, math, sys, time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rosgraph_msgs.msg import Clock
from geometry_msgs.msg import PoseStamped

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--seconds', type=float, default=90.0, help='牆鐘上限')
ap.add_argument('--cmd', type=float, default=0.20, help='預期速度 m/s')
a = ap.parse_args()

rclpy.init()
n = Node('clock_probe')
be = QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST)
rows = []          # (wall, clock_sim, pose_stamp, x, y)
state = {'clk': None}
n.create_subscription(Clock, '/clock',
                      lambda m: state.__setitem__(
                          'clk', m.clock.sec + m.clock.nanosec * 1e-9), 10)

def pose_cb(m):
    s = m.header.stamp
    rows.append((time.monotonic(), state['clk'],
                 s.sec + s.nanosec * 1e-9,
                 m.pose.position.x, m.pose.position.y))

n.create_subscription(PoseStamped, '/model/omni_bot/pose', pose_cb, be)

t0 = time.monotonic()
last = 0
while time.monotonic() - t0 < a.seconds:
    rclpy.spin_once(n, timeout_sec=0.2)
    if len(rows) > 30 and len(rows) == last:
        break                      # 發布停了，模擬已結束
    last = len(rows)
n.destroy_node(); rclpy.shutdown()

with open(a.out, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['wall', 'clock_sim', 'pose_stamp', 'x', 'y'])
    w.writerows(rows)

good = [r for r in rows if r[1] is not None]
print(f'樣本 {len(rows)}，其中有 /clock 的 {len(good)}')
if len(good) < 20:
    print('  !! 樣本不足'); sys.exit(1)
d_clock = good[-1][1] - good[0][1]
d_stamp = good[-1][2] - good[0][2]
dist = math.hypot(good[-1][3] - good[0][3], good[-1][4] - good[0][4])
print(f'  /clock 前進 {d_clock:.4f} s；位姿時戳前進 {d_stamp:.4f} s；'
      f'兩者差 {d_clock - d_stamp:+.6f} s')
print(f'  位移 {dist:.4f} m -> 依 /clock 的速度 {dist/d_clock:.5f} m/s'
      f'（命令 {a.cmd}），比值 {dist/d_clock/a.cmd:.5f}')
mx = max(abs(r[1] - r[2]) for r in good)
print(f'  同一則位姿上 |/clock − 位姿時戳| 最大 {mx*1000:.3f} ms')
