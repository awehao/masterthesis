"""WG2 介面測試用的**最小消費端樁**（不是模擬器）。

存在的唯一理由：`arm_vel_adapter` 會向消費端節點以 `get_parameters`
查詢 `joints` 並核對順序（F4 的守衛）。沒有消費端時 adapter **拒絕轉發**
—— 那是正確行為，但也讓介面測試量不到「實際套用」那一段。

本樁**只**做兩件事：
  1. 宣告 `joints` 參數（與安全鏈的順序一致）
  2. 訂閱 /wb_vel_cmd 並計數

**它不積分、不模擬物理、不回授狀態**，所以不得用它取代 Isaac 的物理驗證。
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'src/ammr_wholebody_mpc'))
from ammr_wholebody_mpc.arm_pregrasp import ARM_JOINTS          # noqa: E402


class Sink(Node):
    def __init__(self, name):
        super().__init__(name)
        self.declare_parameter('joints', list(ARM_JOINTS))
        self.n = 0
        self.last = None
        self.create_subscription(Float64MultiArray, '/wb_vel_cmd',
                                 self._on, 10)

    def _on(self, m):
        if len(m.data) >= 9:
            self.n += 1
            self.last = [float(x) for x in m.data[:9]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--node-name', default='wgmpc_wg2_sink')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    rclpy.init()
    nd = Sink(a.node_name)
    ex = SingleThreadedExecutor()      # **單一 executor 所有權**
    ex.add_node(nd)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if a.out:
            json.dump({'received': nd.n, 'last': nd.last},
                      open(a.out, 'w'), ensure_ascii=False)
        ex.shutdown()
        nd.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
