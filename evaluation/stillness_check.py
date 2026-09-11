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
ap.add_argument('--discover', type=float, default=15.0,
                help='開始量測前，等訂閱配對上的最長秒數')
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
# 建立訂閱不等於已經配對上。v2_on_095657 那趟這裡印出「/odom 無資料」而中止，
# 但同一趟的 bag 錄到 /odom 3596 筆、/clock 10391 筆 —— 固定長度的觀察視窗
# 在配對完成前就結束了。所以先等配對（每個主題至少一則），再清空重新計時；
# 這樣「無資料」才真的代表那個主題沒有發布。
want = [f'/model/dyn_obs_{i}/pose' for i in range(a.movers)] \
       + ['/model/omni_bot/pose']
t_disc = time.monotonic()
while time.monotonic() - t_disc < a.discover:
    if twist and all(hist.get(t) for t in want):
        break
    rclpy.spin_once(n, timeout_sec=0.05)
disc_s = time.monotonic() - t_disc
missing = ([t for t in want if not hist.get(t)] + ([] if twist else ['/odom']))
if missing:
    print(f'  !! 等待配對 {disc_s:.1f} s 後仍有 {len(missing)} 個主題沒有樣本：'
          f'{", ".join(missing)}；量測照常進行並據實判定')
else:
    print(f'  訂閱配對完成，用時 {disc_s:.1f} s；'
          f'開始 {a.window:.1f} s 觀察視窗')
hist.clear(); twist.clear()

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
