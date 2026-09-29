"""把各節點的診斷**分開落盤**（O4 要求的計時記錄）。

各節點處理時間、求解器耗時與端到端命令年齡**分開記錄**；
本檔只負責錄製，不做任何合併或平均相加的判定。
"""
from __future__ import annotations
import argparse, json, os, sys

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float32MultiArray, Float64MultiArray

# 欄位名單**由上游節點以 latched String 發布**（`~/diag_fields`）。
# 本檔不自帶硬編碼欄位名 —— 上游加欄位時硬編碼名單會錯位，
# 把別的量當成新欄位（曾把 tf_age 當成 node_ms）。
FIELD_TOPICS = {'dist': '/arm_link_distance/diag_fields',
                'safety': '/wholebody_safety/diag_fields'}
TOPICS = [
    ('/arm_link_distance/diag', Float32MultiArray, 'dist', None),
    ('/wholebody_safety/diag', Float32MultiArray, 'safety', None),
    ('/coman/cmd_meta', Float64MultiArray, 'solver_meta',
     ['seq', 'src_sim_t', 'solve_ms'] + [f'v{i}' for i in range(9)]),
    ('/wholebody_safety/cmd_meta', Float64MultiArray, 'safety_meta',
     ['src_seq', 'src_sim_t', 'solve_ms', 'safety_node_ms']
     + [f'out{i}' for i in range(9)]),
]


class Rec(Node):
    def __init__(self, out):
        super().__init__('coman_diag_record')
        from rclpy.parameter import Parameter
        self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL,
                                       True)])
        self.out = out
        self.data = {tag: [] for _, _, tag, _ in TOPICS}
        self.cols = {tag: (['sim_t'] + cols) if cols else None
                     for _, _, tag, cols in TOPICS}
        for topic, typ, tag, _ in TOPICS:
            self.create_subscription(
                typ, topic, self._make(tag), qos_profile_sensor_data)
        from rclpy.qos import QoSProfile, DurabilityPolicy
        from std_msgs.msg import String
        lat = QoSProfile(depth=1)
        lat.durability = DurabilityPolicy.TRANSIENT_LOCAL
        for tag, ftopic in FIELD_TOPICS.items():
            self.create_subscription(String, ftopic, self._fields(tag), lat)

    def _fields(self, tag):
        def cb(m):
            self.cols[tag] = ['sim_t'] + list(json.loads(m.data))
        return cb
        self.create_timer(2.0, self._flush)

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _make(self, tag):
        def cb(m):
            self.data[tag].append([round(self._now(), 5)]
                                  + [float(x) for x in m.data])
        return cb

    def _flush(self):
        os.makedirs(os.path.dirname(self.out) or '.', exist_ok=True)
        json.dump({'cols': self.cols,
                   'cols_source': '上游 ~/diag_fields（latched）；'
                                  'null 表示尚未收到名單，不得猜位置',
                   'counts': {k: len(v) for k, v in self.data.items()},
                   'data': self.data},
                  open(self.out, 'w'), ensure_ascii=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rclpy.init()
    n = Rec(a.out)
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n._flush()
        print(f'診斷錄製：{ {k: len(v) for k, v in n.data.items()} }', flush=True)
        n.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
