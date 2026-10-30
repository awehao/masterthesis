#!/usr/bin/env python3
"""隔離閘門（一次性圖查詢）：確認指定話題發布者全為 0；取代逐項 ros2 topic info。先以 /clock 有發布者確認探索完成，再判 0，避免探索未完成的假 0。
    python3 evaluation/ros_graph_gate.py --zero /a /b --timeout-s 10
rc 0＝全為 0（連續 0.3 s）；rc 2＝逾時（印出仍有發布者的話題）；rc 3＝探索未完成。
"""
import argparse
import time

import rclpy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zero', nargs='+', required=True)
    ap.add_argument('--timeout-s', type=float, default=10.0)
    ap.add_argument('--hold-s', type=float, default=0.3)
    ap.add_argument('--sentinel', default='/clock')
    a = ap.parse_args()
    rclpy.init()
    nd = rclpy.create_node('ros_graph_gate')
    w0 = time.monotonic()
    while nd.count_publishers(a.sentinel) == 0:
        rclpy.spin_once(nd, timeout_sec=0.05)
        if time.monotonic() - w0 > a.timeout_s:
            print(f'[gate] 探索未完成：{a.sentinel} 無發布者', flush=True)
            return 3
    t_ok, bad = None, a.zero
    while time.monotonic() - w0 <= a.timeout_s:
        rclpy.spin_once(nd, timeout_sec=0.02)
        bad = [t for t in a.zero if nd.count_publishers(t) > 0]
        if bad:
            t_ok = None
            continue
        t_ok = t_ok if t_ok is not None else time.monotonic()
        if time.monotonic() - t_ok >= a.hold_s:
            print(f'[gate] 隔離通過（{time.monotonic() - w0:.2f} s）', flush=True)
            return 0
    print(f'[gate] **隔離未通過**：{bad}', flush=True)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
