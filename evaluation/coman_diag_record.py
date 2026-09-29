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

TOPICS = [
    ('/arm_link_distance/diag', Float32MultiArray, 'dist',
     ['n_rows', 'n_ok', 'n_unk', 'n_stale', 'n_nodata', 'worst_age', 'min_d',
      'dropped', 'node_cycle_ms', 'dup_dropped', 'tight_lb', 'tight_ub',
      'tight_elapsed_s', 'tight_tol_met']),
    ('/wholebody_safety/diag', Float32MultiArray, 'safety',
     ['cycle', 'reason', 'n_rows', 'n_active', 'resid_before', 'resid_after',
      'iters', 'fallback', 'unresolved', 'filter_ms', 'speed_cap', 'min_d',
      'n_stale', 'n_nodata', 'n_occluded', 'safety_override', 'dt_prev_ms',
      'have_v_prev2', 'tf_age', 'node_ms', 'src_paired']),
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
        self.cols = {tag: ['sim_t'] + cols for _, _, tag, cols in TOPICS}
        for topic, typ, tag, _ in TOPICS:
            self.create_subscription(
                typ, topic, self._make(tag), qos_profile_sensor_data)
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
