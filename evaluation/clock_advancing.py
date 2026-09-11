#!/usr/bin/env python3
"""/clock 是否在前進 —— 配對只付一次的版本。

原本的 shell 版本每次嘗試都跑兩次 `ros2 topic echo --once`，每次都是新行程、
都要重跑一次 DDS 配對，而各自的 timeout 只有 3 s。導航鏈有十幾個節點時，
配對經常超過 3 s，於是回報「拿不到 /clock」並重試，把 60 s 的預算耗光：
v2_off_093455 在這個閘用掉 56 s（上限 60）勉強通過，v2_on_095657 直接逾時
中止，但同一趟的 bag 錄到 /clock 10391 筆 —— 那是誤判，不是時鐘停住。

這裡改成單一行程：先等到第一則 /clock（配對），再從那一刻起量測是否前進。
「等不到第一則」與「收到了但沒前進」是兩種不同的失敗，分開回報。
"""
import argparse
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock

ap = argparse.ArgumentParser()
ap.add_argument('--discover', type=float, default=30.0, help='等第一則 /clock 的秒數')
ap.add_argument('--window', type=float, default=10.0, help='量測前進的秒數')
ap.add_argument('--min-advance', type=float, default=0.5, help='至少前進幾秒模擬時間')
a = ap.parse_args()

rclpy.init()
n = Node('clock_advancing')
seen = {'t': None, 'first': None, 'n': 0}


def cb(m):
    t = m.clock.sec + m.clock.nanosec * 1e-9
    if seen['first'] is None:
        seen['first'] = t
    seen['t'] = t
    seen['n'] += 1


n.create_subscription(
    Clock, '/clock', cb,
    QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
               history=HistoryPolicy.KEEP_LAST))

t0 = time.monotonic()
while time.monotonic() - t0 < a.discover and seen['first'] is None:
    rclpy.spin_once(n, timeout_sec=0.05)
disc = time.monotonic() - t0
if seen['first'] is None:
    print(f'  !! 等 {disc:.1f} s 仍收不到 /clock（配對未完成或沒有發布者）')
    sys.exit(1)

base = seen['first']
t1 = time.monotonic()
while time.monotonic() - t1 < a.window:
    rclpy.spin_once(n, timeout_sec=0.05)
    if seen['t'] - base >= a.min_advance:
        break
adv = seen['t'] - base
print(f'  配對用時 {disc:.1f} s；{time.monotonic()-t1:.1f} s 內模擬時間前進 '
      f'{adv:.2f} s（{seen["n"]} 則，要求 >= {a.min_advance}）')
sys.exit(0 if adv >= a.min_advance else 1)
