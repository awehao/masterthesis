#!/usr/bin/env python3
"""PARK_FIXED 整合反例（Codex 20261005_115350 §1）：展開期間／求解器未上線時收到 /park/violation，
任務節點之後**不得**再發目標、不得啟動求解器、不得請求換手，且須走既有中止收尾（/wgmpc/stop）。

不開 Isaac：真的啟動 drawer_task_node.py --park-fixed，由本測試用 /drawer/sync_state 餵狀態。
  0–2 s：底盤在停位、手臂收攏（求解器未上線前的分支，每輪會發目標）
  2 s  ：發 /park/violation；**同時**把手臂改為夾持姿態（滿足 ALIGN 條件——未修正時會發 solver_start）
判定：任務節點在 5 s 內結束、task.json abort 含 park_fixed_violation、收到 /wgmpc/stop、違規 0.3 s 後
不再有 /drawer/tcp_target、從未收到 solver_start=True、從未收到 /handover/request。

    source /opt/ros/jazzy/setup.bash; python3 evaluation/test_park_abort_integration.py
    PARK_TEST_FLAG=--park-hold python3 evaluation/test_park_abort_integration.py   （v2）
"""
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ['ROS_DOMAIN_ID'] = os.environ.get('PARK_TEST_DOMAIN', '121')

import rclpy                                                            # noqa: E402
from rclpy.node import Node                                             # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile                      # noqa: E402
from std_msgs.msg import Bool, Float64MultiArray, String                # noqa: E402

PARK = (-0.136412, 0.560, 1.297349)
STOW = [0.0, 0.0, 0.0, 0.0, -1.5707963, 0.0]
QG = [-0.0321, 0.9177, 1.4853, -0.3580, -1.0325, -1.3813]


class Probe(Node):
    def __init__(self):
        super().__init__('park_abort_probe')
        self.sync = self.create_publisher(Float64MultiArray, '/drawer/sync_state', 10)
        self.vio = self.create_publisher(String, '/park/violation', 10)
        lat = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.tgt, self.stop, self.start, self.ho = [], [], [], []
        self.create_subscription(Float64MultiArray, '/drawer/tcp_target',
                                 lambda m: self.tgt.append(time.monotonic()), 10)
        self.create_subscription(String, '/wgmpc/stop', lambda m: self.stop.append((time.monotonic(), m.data)), 10)
        self.create_subscription(Bool, '/drawer/solver_start',
                                 lambda m: self.start.append((time.monotonic(), m.data)), lat)
        self.create_subscription(String, '/handover/request', lambda m: self.ho.append(m.data), 10)


def main():
    out = os.path.join(tempfile.mkdtemp(prefix='park_abort_'), 'task.json')
    # 模式旗標：預設 v1 --park-fixed；PARK_TEST_FLAG=--park-hold 測 v2（違規停止語意相同）
    cmd = [sys.executable, '-u', os.path.join(HERE, 'drawer_task_node.py'),
           os.environ.get('PARK_TEST_FLAG', '--park-fixed'),
           '--open-m', '0.200', '--out', out]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    rclpy.init()
    nd = Probe()
    H = np.eye(4)
    H[:3, 3] = [0.0, 1.165, 0.55]
    t_sim, t0, t_vio = 10.0, time.monotonic(), None
    exited = None
    while time.monotonic() - t0 < 15.0:
        el = time.monotonic() - t0
        if t_vio is None and el >= 2.0 and len(nd.tgt) > 0:
            nd.vio.publish(String(data=json.dumps({'why': 'test_injected_violation_during_unfold'})))
            t_vio = time.monotonic()
        q = QG if t_vio is not None else STOW
        m = Float64MultiArray()
        m.data = [t_sim, 0.0, 0.0] + H.reshape(-1).tolist() + list(PARK) + q
        nd.sync.publish(m)
        t_sim += 0.02
        rclpy.spin_once(nd, timeout_sec=0.02)
        if proc.poll() is not None:
            exited = time.monotonic()
            break
    for _ in range(20):
        rclpy.spin_once(nd, timeout_sec=0.02)
    if proc.poll() is None:
        proc.kill()
    log = proc.stdout.read()
    rep = json.load(open(out)) if os.path.exists(out) else {}
    fails = []

    def check(name, cond, detail=''):
        print(('PASS ' if cond else 'FAIL ') + name + ('' if cond else f'  {detail}'))
        if not cond:
            fails.append(name)
    check('targets_were_flowing_before_violation', t_vio is not None and any(x < t_vio for x in nd.tgt),
          f'n_tgt={len(nd.tgt)}')
    check('task_exited_within_5s', exited is not None and t_vio is not None and exited - t_vio <= 5.0,
          log[-800:])
    check('abort_reason_recorded', 'park_fixed_violation' in str(rep.get('abort', '')), rep.get('abort'))
    check('solver_stop_sent', any(t >= (t_vio or 0) for t, _ in nd.stop), nd.stop)
    check('no_target_after_violation', t_vio is not None and not any(x > t_vio + 0.3 for x in nd.tgt),
          f'late={[round(x - t_vio, 3) for x in nd.tgt if x > t_vio + 0.3][:5]}')
    check('no_solver_start', not any(d for _, d in nd.start), nd.start)
    check('no_handover_request', not nd.ho, nd.ho)
    nd.destroy_node()
    rclpy.shutdown()
    print(f'{7 - len(fails)} 通過、{len(fails)} 失敗')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
