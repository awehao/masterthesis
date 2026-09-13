"""等模擬時間到達指定值；達不到就回報，不假裝已到。"""
from __future__ import annotations
import argparse, sys, time

ap = argparse.ArgumentParser()
ap.add_argument('--until', type=float, required=True)
ap.add_argument('--plus', type=float, default=0.0)
ap.add_argument('--wall-timeout-s', type=float, default=90.0)
a = ap.parse_args()

import rclpy                                                   # noqa: E402
from rclpy.node import Node                                    # noqa: E402
from rosgraph_msgs.msg import Clock                            # noqa: E402

TARGET = a.until + a.plus
rclpy.init()
n = Node('wb_wait_sim_t')
v = []
n.create_subscription(Clock, '/clock',
                      lambda m: v.append(m.clock.sec + m.clock.nanosec * 1e-9), 10)
t0 = time.monotonic()
while time.monotonic() - t0 < a.wall_timeout_s and (not v or v[-1] < TARGET):
    rclpy.spin_once(n, timeout_sec=0.05)
if not v:
    print('[obs] **未收到 /clock**')
    rclpy.try_shutdown(); sys.exit(2)
ok = v[-1] >= TARGET
print(f'[obs] 觀察窗{"結束" if ok else "**未達成**"}：sim {v[-1]:.3f}（需要 >= {TARGET:.3f}）')
rclpy.try_shutdown()
sys.exit(0 if ok else 3)
