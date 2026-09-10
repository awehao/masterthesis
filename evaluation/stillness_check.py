#!/usr/bin/env python3
"""Confirm nothing is drifting before the goal is published.

Not publishing a mover command is not the same claim as the mover standing
still: a kinematic body can still be carrying a pose the integrator left it
with, and the robot itself can hold residual velocity from spawn settling.
Both are measured here from what is actually on the wire, over a window, and
the worst case is reported so the caller can gate on it.
"""
import argparse, math, sys
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

ap = argparse.ArgumentParser()
ap.add_argument('--movers', type=int, default=10)
ap.add_argument('--window', type=float, default=3.0, help='wall seconds to watch')
ap.add_argument('--max-speed', type=float, default=0.02, help='m/s allowed')
ap.add_argument('--traffic', action='store_true',
                help='移動障礙預期會動：只要求機器人靜止，並反過來要求至少一個'
                     '移動體確實在動（traffic driver 靜默失效也要被抓到）')
a = ap.parse_args()

rclpy.init()
n = Node('stillness_check', parameter_overrides=[])
n.set_parameters([rclpy.parameter.Parameter('use_sim_time', value=True)])
be = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST)

hist = {}          # topic -> list of (sim_t, x, y)
def mk(topic):
    def cb(m, t=topic):
        st = m.header.stamp
        hist.setdefault(t, []).append(
            (st.sec + st.nanosec * 1e-9, m.pose.position.x, m.pose.position.y))
    n.create_subscription(PoseStamped, topic, cb, be)

for i in range(a.movers):
    mk(f'/model/dyn_obs_{i}/pose')
mk('/model/omni_bot/pose')

twist = []
def odom_cb(m):
    twist.append((abs(m.twist.twist.linear.x), abs(m.twist.twist.linear.y),
                  abs(m.twist.twist.angular.z)))
n.create_subscription(Odometry, '/odom', odom_cb, be)

import time
t0 = time.monotonic()
while time.monotonic() - t0 < a.window:
    rclpy.spin_once(n, timeout_sec=0.1)

bad = 0
mover_moving = 0
mode = '移動障礙運行中（只要求機器人靜止）' if a.traffic else '全部應靜止'
print(f'--- 靜止檢查（觀察 {a.window:.1f} s 牆鐘；{mode}）---')
for t in sorted(hist):
    h = hist[t]
    is_robot = t.endswith('/omni_bot/pose')
    if len(h) < 2:
        print(f'  {t:28s} 樣本 {len(h)} 不足，無法判定'); bad += 1; continue
    span = h[-1][0] - h[0][0]
    if span <= 1e-6:
        print(f'  {t:28s} 模擬時間未前進，無法判定'); bad += 1; continue
    # worst instantaneous speed, not just endpoint-to-endpoint: a body that
    # moved out and back would look stationary from the endpoints alone
    v = 0.0
    for (ta, xa, ya), (tb, xb, yb) in zip(h, h[1:]):
        dt = tb - ta
        if dt > 1e-6:
            v = max(v, math.hypot(xb - xa, yb - ya) / dt)
    # With traffic running the movers are SUPPOSED to move; judging them by the
    # stationary criterion would fail a correct run. They are still measured and
    # printed, and their motion is required below -- a traffic driver that never
    # started must not pass as "everything is still".
    if a.traffic and not is_robot:
        if v > a.max_speed:
            mover_moving += 1
        print(f'  {t:28s} n={len(h):4d} span={span:5.2f}s 最大瞬時速度={v:.4f} m/s'
              f'  （預期移動{"，確認在動" if v > a.max_speed else "，但未偵測到移動"}）')
        continue
    flag = '' if v <= a.max_speed else '  << 超出'
    if v > a.max_speed: bad += 1
    print(f'  {t:28s} n={len(h):4d} span={span:5.2f}s 最大瞬時速度={v:.4f} m/s{flag}')
if a.traffic:
    n_mov = sum(1 for t in hist if not t.endswith('/omni_bot/pose'))
    print(f'  移動體 {mover_moving}/{n_mov} 個確認在動')
    if mover_moving == 0:
        print('  !! 沒有任何移動體在動：traffic driver 可能未啟動'); bad += 1

if twist:
    mx = max(t[0] for t in twist); my = max(t[1] for t in twist)
    mw = max(t[2] for t in twist)
    print(f'  /odom twist 最大 |vx|={mx:.4f} |vy|={my:.4f} m/s |wz|={mw:.4f} rad/s')
    if max(mx, my) > a.max_speed: bad += 1
else:
    print('  /odom 無資料'); bad += 1

n.destroy_node(); rclpy.shutdown()
print(f'--- 結果：{"通過" if bad == 0 else f"{bad} 項未通過"} ---')
sys.exit(0 if bad == 0 else 1)
