"""獨立觀測者：記錄 `/wb_vel_cmd` 的每一則訊息與其模擬時間。

**用途**：斷訊趟要能說「`/wb_vel_cmd` 實際停止更新」，
不能只靠執行端自己的 `cmd_age` 推論 —— 那是同一個程序的內部狀態。
本檔是**另一個程序**，只訂閱、不發布，對受測鏈路沒有影響。

收到 SIGTERM 會先落盤再結束；另外每 2 s 週期落盤，避免被清理時遺失。
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--topic', default='/wb_vel_cmd')
ap.add_argument('--msg-type', default='f64', choices=['f64', 'f32'],
                help='Float64MultiArray 或 Float32MultiArray')
ap.add_argument('--name', default='wb_topic_recorder')
ap.add_argument('--outfile', default='wb_vel_cmd_record.json')
a = ap.parse_args()

import rclpy                                                   # noqa: E402
from rclpy.node import Node                                    # noqa: E402
from std_msgs.msg import (Float32MultiArray,                   # noqa: E402
                          Float64MultiArray)
from rosgraph_msgs.msg import Clock                            # noqa: E402


class Rec(Node):
    def __init__(self):
        super().__init__(a.name)
        self.sim_t = None
        self.msgs = []
        self.create_subscription(Clock, '/clock', self._clk, 10)
        MT = Float64MultiArray if a.msg_type == 'f64' else Float32MultiArray
        self.create_subscription(MT, a.topic, self._msg, 50)
        self.t_wall0 = time.monotonic()
        self.last_dump = 0.0

    def _clk(self, m):
        self.sim_t = m.clock.sec + m.clock.nanosec * 1e-9

    def _msg(self, m):
        self.msgs.append([round(self.sim_t, 4) if self.sim_t is not None else None,
                          round(time.monotonic() - self.t_wall0, 4),
                          [round(float(x), 6) for x in m.data]])

    def dump(self):
        json.dump({'schema': 'wb_topic_recorder/1', 'topic': a.topic,
                   'n': len(self.msgs),
                   'cols': ['sim_t', 'wall_rel_s', 'data9'],
                   'first_sim_t': self.msgs[0][0] if self.msgs else None,
                   'last_sim_t': self.msgs[-1][0] if self.msgs else None,
                   'note': ('獨立觀測者，只訂閱不發布；'
                            '用於證明話題實際停止更新，而非執行端內部狀態推論'),
                   'msgs': self.msgs},
                  open(os.path.join(a.out, a.outfile), 'w'),
                  ensure_ascii=False)


def main():
    os.makedirs(a.out, exist_ok=True)
    rclpy.init()
    n = Rec()
    stop = {'v': False}

    def on_term(_s, _f):
        stop['v'] = True
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    print(f'[rec] 記錄 {a.topic} → {a.out}', flush=True)
    while rclpy.ok() and not stop['v']:
        rclpy.spin_once(n, timeout_sec=0.05)
        w = time.monotonic() - n.t_wall0
        if w - n.last_dump > 2.0:
            n.dump()
            n.last_dump = w
    n.dump()
    print(f'[rec] 共 {len(n.msgs)} 則；最後 sim '
          f'{n.msgs[-1][0] if n.msgs else "無"}', flush=True)
    rclpy.try_shutdown()
    return 0


sys.exit(main())
