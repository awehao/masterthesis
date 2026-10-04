#!/usr/bin/env python3
"""導航命令逾時的驗證：**送一筆非零命令後停發**，看是否進入停止處置。

先前的缺陷：接收時間在主迴圈每個物理步都被刷新，只要曾收到過命令就永遠
算新鮮 —— 導航停止發布後舊命令會一直被沿用，逾時處置永遠不會觸發。
"""
from __future__ import annotations

import json
import math
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node


class P(Node):
    def __init__(s):
        super().__init__('nav_timeout_probe')
        s.set_parameters([rclpy.parameter.Parameter(
            'use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        s.pub = s.create_publisher(Twist, '/cmd_vel', 10)
        s.create_subscription(Odometry, '/odom', s.cb, 10)
        s.p = None
        s.n = 0

    def cb(s, m):
        s.p = (float(m.pose.pose.position.x), float(m.pose.pose.position.y),
               float(m.header.stamp.sec) + m.header.stamp.nanosec * 1e-9)
        s.n += 1


def main():
    rclpy.init()
    nd = P()
    t0 = time.monotonic()
    while rclpy.ok() and time.monotonic() - t0 < 30 and nd.n == 0:
        rclpy.spin_once(nd, timeout_sec=0.05)
    if nd.p is None:
        print('[timeout] **收不到 /odom**')
        return 2
    a = nd.p
    tw = Twist()
    tw.linear.x = 0.25
    # **只送一段非零命令，然後停發**
    t1 = time.monotonic()
    while rclpy.ok() and time.monotonic() - t1 < 2.0:
        nd.pub.publish(tw)
        rclpy.spin_once(nd, timeout_sec=0.02)
    b = nd.p
    moved_while_cmd = math.hypot(b[0] - a[0], b[1] - a[1])
    print(f'[timeout] 送命令 2 s：移動 {moved_while_cmd:.4f} m', flush=True)
    # 停發，只收不送
    t2 = time.monotonic()
    marks = []
    while rclpy.ok() and time.monotonic() - t2 < 8.0:
        rclpy.spin_once(nd, timeout_sec=0.02)
        marks.append((nd.p[2], nd.p[0], nd.p[1]))
    c = nd.p
    moved_after = math.hypot(c[0] - b[0], c[1] - b[1])
    # 停發後**最後 3 秒**的位移：逾時若生效，這段應該近乎為零
    t_end = marks[-1][0]
    late = [m for m in marks if m[0] >= t_end - 3.0]
    late_mv = math.hypot(late[-1][1] - late[0][1], late[-1][2] - late[0][2])
    print(f'[timeout] 停發後總移動 {moved_after:.4f} m；'
          f'**最後 3 秒（模擬）移動 {late_mv*1e3:.2f} mm**', flush=True)
    ok = late_mv < 0.005
    print(f'[timeout] {"**逾時處置生效**（停發後已停住）" if ok else
                      "**逾時處置未生效**（停發後仍在移動）"}', flush=True)
    json.dump({'moved_while_cmd_m': moved_while_cmd,
               'moved_after_stop_m': moved_after,
               'late_3s_move_m': late_mv, 'ok': ok},
              open(sys.argv[1], 'w') if len(sys.argv) > 1 else sys.stdout,
              ensure_ascii=False, indent=1)
    nd.destroy_node()
    try:
        rclpy.shutdown()
    except Exception:
        pass   # 已經關過就不是錯
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
