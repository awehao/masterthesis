#!/usr/bin/env python3
"""隔離閘門（一次性圖查詢）：確認指定話題發布者全為 0；取代逐項 ros2 topic info。先確認舊程序群組已退出（--dead-pgids），再以已知仍在發布的話題（--known）確認本節點已探索到圖，才判 0；查詢失敗不當 0。
    python3 evaluation/ros_graph_gate.py --zero /a /b --dead-pgids 123 456 --print-odom --timeout-s 10
rc 0＝全為 0（連續 0.3 s）；rc 2＝逾時（印出仍有發布者的話題）；rc 3＝探索未完成；rc 4＝舊程序未退出；rc 5＝查詢失敗。
"""
import argparse
import math
import os
import time

import rclpy
from nav_msgs.msg import Odometry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zero', nargs='+', required=True)
    ap.add_argument('--timeout-s', type=float, default=10.0)
    ap.add_argument('--hold-s', type=float, default=0.3)
    ap.add_argument('--known', nargs='+', default=['/clock', '/odom', '/scan'])
    ap.add_argument('--dead-pgids', nargs='*', type=int, default=[])
    ap.add_argument('--print-odom', action='store_true')
    a = ap.parse_args()
    rclpy.init()
    nd = rclpy.create_node('ros_graph_gate')
    w0 = time.monotonic()
    od = []
    if a.print_odom:
        nd.create_subscription(Odometry, '/odom', od.append, 1)

    def alive(pg):
        try:
            os.killpg(pg, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    while any(alive(p) for p in a.dead_pgids):
        if time.monotonic() - w0 > a.timeout_s:
            print(f'[gate] **舊程序未退出**：{[p for p in a.dead_pgids if alive(p)]}', flush=True)
            return 4
        time.sleep(0.02)
    try:
        while not all(nd.count_publishers(t) > 0 for t in a.known):
            rclpy.spin_once(nd, timeout_sec=0.05)
            if time.monotonic() - w0 > a.timeout_s:
                print(f'[gate] 探索未完成：{[t for t in a.known if nd.count_publishers(t) == 0]}', flush=True)
                return 3
    except Exception as e:                                                      # noqa: BLE001  查詢失敗不當零
        print(f'[gate] 圖查詢失敗：{e!r}', flush=True)
        return 5
    t_ok, bad = None, a.zero
    while time.monotonic() - w0 <= a.timeout_s:
        rclpy.spin_once(nd, timeout_sec=0.02)
        try:
            bad = [t for t in a.zero if nd.count_publishers(t) > 0]
        except Exception as e:                                                  # noqa: BLE001
            print(f'[gate] 圖查詢失敗：{e!r}', flush=True)
            return 5
        if bad:
            t_ok = None
            continue
        t_ok = t_ok if t_ok is not None else time.monotonic()
        if time.monotonic() - t_ok >= a.hold_s and (od or not a.print_odom):
            print(f'[gate] 隔離通過（{time.monotonic() - w0:.2f} s）', flush=True)
            if a.print_odom:
                p, q = od[-1].pose.pose.position, od[-1].pose.pose.orientation
                print(f'ODOM {p.x:.6f},{p.y:.6f},{math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2)):.6f}', flush=True)
            return 0
    print(f'[gate] **隔離未通過**：{bad}', flush=True)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
