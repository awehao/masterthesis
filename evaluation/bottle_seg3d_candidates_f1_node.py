#!/usr/bin/env python3
"""任務端候選節點的 F1 版（bottle_seg3d_candidates_node 的版本化子類；Codex F1 裁定）：候選、快照與檢查規則不變，只多輪次重設。
收到 /f1/round（[round_id, xlo, xhi, ylo, yhi, t_min]）：換全新 CandidateTracker 與 SnapshotChecker（序號從 1 重來），舊檢查完成通知丟棄；來源擷取戳早於 t_min 的列不進新追蹤器。不留前一輪的候選歷史。
    .venv-dl0/bin/python evaluation/bottle_seg3d_candidates_f1_node.py --log RUN_DIR/candidates.jsonl --ros-args -p use_sim_time:=true
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_seg3d_candidates as CT                    # noqa: E402
import bottle_seg3d_candidates_node as CN               # noqa: E402

import rclpy                                            # noqa: E402
from rclpy.executors import ExternalShutdownException  # noqa: E402
from std_msgs.msg import Float64MultiArray              # noqa: E402


class CandF1Node(CN.CandNode):
    def __init__(self, q_seed, log_path):
        super().__init__(q_seed, log_path)
        self.q_seed, self.gen, self.t_min, self.n_old = q_seed, 0, -float('inf'), 0
        self.create_subscription(Float64MultiArray, '/f1/round', self._round, 10)

    def _round(self, m):
        d = [float(v) for v in m.data]
        self.gen += 1
        g, old = self.gen, self.sc
        self.t_min = d[5]
        self.tr = CT.CandidateTracker()
        self.sc = CT.SnapshotChecker(self.q_seed, on_done=lambda s, r, g_=g: self._done(s, r) if g_ == self.gen else None)
        old.stop()
        self._write(dict(rec='round_reset', round=int(d[0]), t_min=d[5], gen=g))

    def _in(self, m):
        fs = (m.layout.dim[0].label.split('fields=')[1].split(',') if m.layout.dim and 'fields=' in m.layout.dim[0].label else None)
        if fs is not None and 'stamp' in fs and float(m.data[fs.index('stamp')]) < self.t_min:
            self.n_old += 1
            return
        super()._in(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', default=None)
    ap.add_argument('--design', default=os.path.join(HERE, 'results', 'vision', 'seg3d_collect_design.json'))
    a, ros_args = ap.parse_known_args()
    q_seed = np.array(json.load(open(a.design))['q_ready'])
    rclpy.init(args=ros_args)
    node = CandF1Node(q_seed, a.log)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
