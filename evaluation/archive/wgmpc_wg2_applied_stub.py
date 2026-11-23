"""WG2 介面測試用的**套用回報樁**（不是模擬器）。

只做三件事：宣告 `joints`（adapter 的守衛需要）、訂閱 /wb_vel_cmd，
並依指定模式發布 /coman/applied_cmd（必要時加 /coman/applied_fail）。

**它不積分、不模擬物理、不回授狀態** ⇒ 不得用它取代 Isaac 的物理驗證。
用途只有一個：讓「健康零命令」與「失效閂鎖」兩條路徑都能在無 Isaac 下測到。

    --mode echo        把收到的命令原樣當作已套用（exec_mode=0）
    --mode healthy-zero 一律回報**零**但 exec_mode=0（**正常模式的零命令**）
    --mode fail-at N   前 N 筆 echo，之後 exec_mode=3 並發 applied_fail
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Float64MultiArray, String

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'src/ammr_wholebody_mpc'))
from ammr_wholebody_mpc.arm_pregrasp import ARM_JOINTS          # noqa: E402


class Stub(Node):
    def __init__(self, a):
        super().__init__(a.node_name)
        self.a = a
        self.declare_parameter('joints', list(ARM_JOINTS))
        self.n = 0
        self.step = 0
        self.fail_sent = False
        _lat = QoSProfile(depth=1)
        _lat.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.ap = self.create_publisher(Float64MultiArray,
                                        '/coman/applied_cmd', 10)
        self.fp = self.create_publisher(String, '/coman/applied_fail', _lat)
        self.create_subscription(Float64MultiArray, '/wb_vel_cmd', self._on, 10)

    def _on(self, m):
        if len(m.data) < 9:
            return
        self.n += 1
        self.step += 1
        t = self.get_clock().now().nanoseconds * 1e-9
        v = [float(x) for x in m.data[:9]]
        mode = 0
        if self.a.mode == 'healthy-zero':
            v = [0.0] * 9                      # **正常模式的零命令**
        elif self.a.mode == 'fail-at' and self.n > self.a.fail_at:
            v = [0.0] * 9
            mode = 3
            if not self.fail_sent:
                self.fail_sent = True
                self.fp.publish(String(data=json.dumps({
                    'fail': '注入測試：關節 3 積分結果超出限位',
                    'sim_t': t, 'physics_step_id': self.step,
                    'n_recv': self.n, 'n_rejected': 0,
                    'last_reject': 'None', 'n_frozen': 0,
                }, ensure_ascii=False)))
        out = Float64MultiArray()
        out.data = ([float(self.step), float(t)] + v
                    + [float(mode), 0.01, float(self.n), 0.0, 1.0,
                       float(self.n), float(t)])
        self.ap.publish(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--node-name', default='wgmpc_wg2_applied_stub')
    ap.add_argument('--mode', default='echo',
                    choices=['echo', 'healthy-zero', 'fail-at'])
    ap.add_argument('--fail-at', type=int, default=20)
    a = ap.parse_args()
    rclpy.init()
    nd = Stub(a)
    ex = SingleThreadedExecutor()          # **單一 executor 所有權**
    ex.add_node(nd)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        ex.shutdown()
        nd.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
