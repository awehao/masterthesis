#!/usr/bin/env python3
"""手腕視覺節點的 F1 版（bottle_seg3d_node 的版本化子類；Codex 225852）：同一模型、同一三維定位、同一輸出欄位與話題；只把「尺寸閘後信心最高」換成 bottle_f1_select（給定子區域＋時序穩定＋鎖定）。
另把每幀全部實例寫進 --inst-log（JSON lines），供事後比對真值；真值不進本節點。下游候選節點不改。
    .venv-dl0/bin/python evaluation/bottle_seg3d_f1_node.py --weights ...best.pth --ns w1 --region 0,0.35,0.36,0.62 --inst-log RUN_DIR/instances.jsonl --ros-args -p use_sim_time:=true
"""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bottle_f1_select as FS                           # noqa: E402
import bottle_seg3d_node as SN                          # noqa: E402

import rclpy                                            # noqa: E402


class F1Node(SN.Seg3dNode):
    def __init__(self, weights, ns, region, log_path):
        self.sel = FS.F1Selector(region)
        self.log = open(log_path, 'a') if log_path else None
        super().__init__(weights, ns)

    def _write(self, st, rec):
        if self.log is not None:
            self.log.write(json.dumps(dict(stamp=float(st), **rec), ensure_ascii=False, default=float) + '\n')
            self.log.flush()

    def _no_instance(self, st):
        self.sel.hist = []
        self._write(st, dict(state='locked' if self.sel.lock is not None else 'search', why='no_instance', inst=[], lock=self.sel.lock))

    def _select(self, st, ests, confs):
        r, k, n_rej, rec = self.sel.step(st, ests, confs)
        self._write(st, rec)
        return r, k, n_rej


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--ns', default='w1', choices=['v1', 'w1'])
    ap.add_argument('--region', required=True, help='xlo,xhi,ylo,yhi（世界座標；有界子區域）')
    ap.add_argument('--inst-log', default='')
    a, ros_args = ap.parse_known_args()
    torch.set_num_threads(4)
    rclpy.init(args=ros_args)
    SN.run_node(F1Node(a.weights, a.ns, [float(v) for v in a.region.split(',')], a.inst_log))


if __name__ == '__main__':
    main()
