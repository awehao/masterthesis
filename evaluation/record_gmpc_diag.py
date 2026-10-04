#!/usr/bin/env python3
"""把 /gmpc/diag_v2 原樣落成 JSONL。

為什麼要獨立一支：`ros2 bag` 會把整趟都錄下來，這裡只要求解器每輪的輸入，
而且要**逐筆不漏**。訂閱深度刻意開大（這是每輪一筆的低頻流，不是狀態流），
收到就寫、立刻 flush —— 趟次被中止時已寫的部分仍然有效。
"""
import argparse
import json
import sys

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--topic', default='/gmpc/diag_v2')
a = ap.parse_args()


class Rec(Node):
    def __init__(self):
        super().__init__('gmpc_diag_recorder')
        # 不要宣告 use_sim_time —— rclpy 已經替每個節點宣告過。
        # 這支只是落檔，時刻由 diag 自己帶（solver_in.t 取自 gmpc 的節點時鐘）。
        self.f = open(a.out, 'w')
        self.n = 0
        self.create_subscription(String, a.topic, self._cb, 2000)

    def _cb(self, m):
        try:
            json.loads(m.data)
        except Exception:
            self.f.write(json.dumps({'raw': m.data}) + '\n')
        else:
            self.f.write(m.data + '\n')
        self.n += 1
        self.f.flush()


def main():
    rclpy.init()
    nd = Rec()
    try:
        rclpy.spin(nd)
    except KeyboardInterrupt:
        pass
    finally:
        print(f'[diag] 收到 {nd.n} 筆 → {a.out}', flush=True)
        nd.f.close()
        nd.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    sys.exit(main() or 0)
