#!/usr/bin/env python3
"""velocity_smoother 黑箱測試：最終命令是否仍在輪級限制內。

**隔離**：使用獨立的 ROS_DOMAIN_ID 與測試專用 topic
（/t_cmd_in -> /t_cmd_out），不接任何機器人命令入口，不啟動模擬器。

smoother 的每軸限制（max_accel [6.25, 6.25, 25.51]）看不到輪級耦合，
所以「各軸都在限制內」不蘊含「輪速／輪加速度在限制內」。本檔量測的是
**輸出訊息序列**：輪速可直接由單筆輸出驗算；輪加速度只能以**輸出訊息之間的
單調牆鐘時間差**計算，那**不是** smoother 內部計算限制時使用的 dt
（本機無 .cpp，無法核對）。

第一輪固定節奏，不做時間抖動。
"""
import argparse, json, math, os, subprocess, sys, time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--out', default='evaluation/results/smoother_wheel_test.json')
ap.add_argument('--rate', type=float, default=20.0, help='輸入發布頻率（設定值）')
a = ap.parse_args()

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

R, L = 0.05, 0.245
W_MAX, A_MAX = 5.55, 125.0          # rad/s, rad/s²  (gmpc.py 的 wheel_w_max/a_max)
W = np.array([[0.0, 1.0, L], [-1.0, 0.0, L], [0.0, -1.0, L], [1.0, 0.0, L]])


def omega(u):
    return (W @ np.asarray(u, float)) / R


# ---- 輸入排程：四段，皆以「兩端各自輪級可行」構造 -------------------------
def schedule():
    seq = []
    def hold(u, n):
        seq.extend([tuple(u)] * n)
    hold((0.0, 0.0, 0.0), 10)
    # 1. 平移＋旋轉：兩端各自可行，但各軸過渡進度不同
    hold((0.20, 0.0, 0.0), 30)          # |ω|max = 4.0
    hold((0.0, 0.0, 1.00), 30)          # |ω|max = 4.9
    # 2. 反向
    hold((0.0, 0.0, -1.00), 30)
    # 3. 停止
    hold((0.0, 0.0, 0.0), 20)
    # 4. 對角平移＋反向旋轉，兩端仍各自可行
    hold((0.15, -0.15, 0.0), 30)
    hold((-0.15, 0.15, 0.40), 30)
    hold((0.0, 0.0, 0.0), 20)
    return seq


SEQ = schedule()
for u in set(SEQ):
    assert np.max(np.abs(omega(u))) <= W_MAX + 1e-9, f'輸入端點本身不可行: {u}'

rclpy.init()
n = Node('smoother_wheel_test')
pub = n.create_publisher(Twist, '/t_cmd_in', 10)
rx = []
n.create_subscription(
    Twist, '/t_cmd_out',
    lambda m: rx.append((time.monotonic(), m.linear.x, m.linear.y, m.angular.z)), 10)

t0 = time.monotonic()
while time.monotonic() - t0 < 10.0 and pub.get_subscription_count() < 1:
    rclpy.spin_once(n, timeout_sec=0.1)
subs = pub.get_subscription_count()
print(f'  /t_cmd_in 訂閱者 = {subs}')
if subs < 1:
    print('  !! smoother 未訂閱，請先啟動測試節點'); n.destroy_node(); rclpy.shutdown()
    sys.exit(2)

period = 1.0 / a.rate
sent = []
nxt = time.monotonic()
for u in SEQ:
    m = Twist()
    m.linear.x, m.linear.y, m.angular.z = float(u[0]), float(u[1]), float(u[2])
    pub.publish(m)
    sent.append((time.monotonic(),) + u)
    nxt += period
    while time.monotonic() < nxt:
        rclpy.spin_once(n, timeout_sec=0.002)
t_end = time.monotonic()
while time.monotonic() - t_end < 1.0:
    rclpy.spin_once(n, timeout_sec=0.01)
n.destroy_node(); rclpy.shutdown()

O = np.array(rx)
print(f'  送出 {len(sent)} 則，收到 {len(O)} 則')
if len(O) < 10:
    print('  !! 輸出樣本不足'); sys.exit(3)

w = np.array([omega(r[1:]) for r in O])
wmax = np.max(np.abs(w), axis=1)
dt = np.diff(O[:, 0])
dw = np.max(np.abs(np.diff(w, axis=0)), axis=1)
acc = dw / np.maximum(dt, 1e-9)

# 每軸限制（smoother 自己保證的）
ax = np.abs(np.diff(O[:, 1])) / np.maximum(dt, 1e-9)
ay = np.abs(np.diff(O[:, 2])) / np.maximum(dt, 1e-9)
aw = np.abs(np.diff(O[:, 3])) / np.maximum(dt, 1e-9)

TOL_W, TOL_A = 1e-6, 1e-6           # 事前固定的通過容差
res = dict(
    n_sent=len(sent), n_recv=len(O), rate_setpoint=a.rate,
    recv_dt_median=float(np.median(dt)), recv_dt_min=float(dt.min()),
    recv_dt_max=float(dt.max()),
    wheel_speed_max=float(wmax.max()), wheel_speed_limit=W_MAX,
    wheel_speed_over=int((wmax > W_MAX + TOL_W).sum()),
    wheel_speed_max_excess=float(max(wmax.max() - W_MAX, 0.0)),
    wheel_acc_max=float(acc.max()), wheel_acc_limit=A_MAX,
    wheel_acc_over=int((acc > A_MAX + TOL_A).sum()),
    wheel_acc_max_excess=float(max(acc.max() - A_MAX, 0.0)),
    axis_acc_max=[float(ax.max()), float(ay.max()), float(aw.max())],
    axis_acc_limit=[6.25, 6.25, 25.51])
print(f'\n  接收間隔：中位 {res["recv_dt_median"]:.4f} s  '
      f'範圍 {res["recv_dt_min"]:.4f}–{res["recv_dt_max"]:.4f}'
      f'（設定 {a.rate} Hz = {period:.4f} s；**不宣稱內部 dt 相同**）')
print(f'  輪速 max {res["wheel_speed_max"]:.4f} / 上限 {W_MAX}  '
      f'超出 {res["wheel_speed_over"]}/{len(wmax)} 筆，最大超出 {res["wheel_speed_max_excess"]:.6f}')
print(f'  輪加速 max {res["wheel_acc_max"]:.2f} / 上限 {A_MAX}  '
      f'超出 {res["wheel_acc_over"]}/{len(acc)} 筆，最大超出 {res["wheel_acc_max_excess"]:.4f}')
print(f'    （以**輸出訊息序列**的接收時間差計算，非內部 dt）')
print(f'  每軸加速 max {[round(v,3) for v in res["axis_acc_max"]]} / '
      f'上限 {res["axis_acc_limit"]}')
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(dict(result=res, sent=sent, recv=O.tolist()), open(a.out, 'w'), indent=1)
print(f'\n  -> {a.out}')
print(f'  判定：輪速 {"PASS" if res["wheel_speed_over"]==0 else "FAIL"}  '
      f'輪加速 {"PASS" if res["wheel_acc_over"]==0 else "FAIL"}')
