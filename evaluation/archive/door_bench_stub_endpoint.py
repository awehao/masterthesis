#!/usr/bin/env python3
"""台架初態啟動的**替身執行端**（無 Isaac、無物理；wgmpc_sp_stub_endpoint.py 的版本化延伸，只為測 ev9 的凍結／釋放協定與求解端 ADM17 接線）。
凍結期（同 ev9 B4）：模擬時間停在 t0＝0；每 10 ms 重發 /clock(t0)、/joint_states(q_nominal)、/odom、設定點回報 [0, t0, 1, s_init, 0, 1]、
  套用回報 [0, t0, 0×9, 0, nan, n_recv, 0, api 0, −1, nan]、/park/gate（passed False、bench True、bench_anchor）。
釋放（同 ev9 / door_bench_start.BenchRelease）：/bench/first_solution 宣告的 source_seq 與最新求解端封裝（/wgmpc/cmd_env）相同。
釋放後：每 10 ms 推進 PHYS_DT；設定點積分命令（封裝的手臂六軸）；手臂 x⁺＝x＋α(s−x)＋α⊙d_true（--d-true 未給 ⇒ d_init）。
輸出（--out）：凍結輪數、釋放紀錄、凍結期收到的封裝數、釋放後 1 s 內 |q − q_nominal| 最大、首筆套用 seq。

    （由 test_door_bench_start_ros.py 啟動）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_bench_start as BENCH                       # noqa: E402
import wgmpc_cmd_envelope as ENV                       # noqa: E402

ARM = [f'joint{i}' for i in range(1, 7)]
PHYS_DT = 0.01


class Stub(Node):
    def __init__(self, a, reg, alpha):
        super().__init__('door_bench_stub')
        self.a, self.reg = a, reg
        self.alpha = np.asarray(alpha, float) * np.ones(6)
        self.qn = np.asarray(reg['q_nominal'], float)
        self.q = self.qn.copy()
        self.sp = np.asarray(reg['s_init'], float)
        self.d_true = np.asarray(reg['d_init'] if a.d_true is None else [float(v) for v in a.d_true.split(',')], float)
        self.t, self.step = 0.0, 0
        self.u_arm = np.zeros(6)
        self.env_latest, self.first_announce = None, None
        self.rel = BENCH.BenchRelease(a.run_id, 0.0, a.timeout_wall_s, time.monotonic())
        lat = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.clock = self.create_publisher(Clock, '/clock', 10)
        self.js = self.create_publisher(JointState, '/joint_states', 10)
        self.od = self.create_publisher(Odometry, '/odom', 10)
        self.spp = self.create_publisher(Float64MultiArray, '/coman/arm_setpoint', 10)
        self.spm = self.create_publisher(String, '/coman/arm_setpoint_meta', lat)
        self.ap = self.create_publisher(Float64MultiArray, '/coman/applied_cmd', 10)
        self.gate = self.create_publisher(String, '/park/gate', 1)
        self.create_subscription(Float64MultiArray, ENV.TOPIC[ENV.ST_SOLVER], self._env, 50)
        self.create_subscription(String, BENCH.RELEASE_TOPIC, self._ann, lat)
        self.n_env_frozen, self.first_applied_seq, self.dev_max_1s = 0, None, 0.0
        self.t_release = None
        self.spm.publish(String(data=json.dumps({'cols': ['physics_step_id', 'sim_t', 'ready'] + [f'sp_{j}' for j in ARM]
                                                 + ['exec_mode_code', 'api_applied'], 'joint_order': ARM,
                                                 'sampling_instant': '**本步寫入後**（apply_action 之後）', 'physics_dt_s': PHYS_DT})))
        self.create_timer(PHYS_DT, self._tick)

    def _env(self, m):
        try:
            e = ENV.decode(m.data)
        except ValueError:
            return
        self.env_latest = e
        if self.rel.state == 'frozen':
            self.n_env_frozen += 1
        else:
            self.u_arm = np.asarray(list(e['u'])[3:9], float)

    def _ann(self, m):
        try:
            self.rel.on_announce(json.loads(m.data))
        except ValueError:
            pass

    def _pub(self, step, t, api_applied):
        c = Clock()
        c.clock.sec, c.clock.nanosec = int(t), min(int(round((t - int(t)) * 1e9)), 999999999)
        self.clock.publish(c)
        js = JointState()
        js.header.stamp = c.clock
        js.name, js.position, js.velocity = list(ARM), [float(x) for x in self.q], [0.0] * 6
        self.js.publish(js)
        o = Odometry()
        o.header.stamp = c.clock
        o.header.frame_id, o.child_frame_id = 'odom', 'base_footprint'
        o.pose.pose.orientation.w = 1.0
        self.od.publish(o)
        self.spp.publish(Float64MultiArray(data=[float(step), float(t), 1.0] + [float(x) for x in self.sp] + [0.0, 1.0]))
        self.ap.publish(Float64MultiArray(data=[float(step), float(t), 0.0, 0.0, 0.0] + [float(x) for x in (self.u_arm if api_applied else np.zeros(6))]
                                          + [0.0, float('nan'), 0.0, 0.0, 1.0 if api_applied else 0.0, -1.0, float('nan')]))
        self.gate.publish(String(data=json.dumps({'step': step, 'sim_t': t, 'passed': False, 'ok_now': False, 'anchor': None, 'mode': 'hold',
                                                  'violation': False, 'bench': True, 'bench_anchor': [0.0, 0.0, 0.0]})))

    def _tick(self):
        if self.rel.state == 'frozen':
            seq = None if self.env_latest is None else self.env_latest['source_seq']
            if self.rel.try_release(seq, time.monotonic()) == 'frozen':
                self._pub(0, 0.0, False)
                return
            if self.rel.state == 'released':
                self.t_release = time.monotonic()
                self.first_applied_seq = int(seq)
                self.u_arm = np.asarray(list(self.env_latest['u'])[3:9], float)
            else:
                return
        if self.rel.state != 'released':
            return
        self.sp = self.sp + self.u_arm * PHYS_DT
        self.q = self.q + self.alpha * (self.sp - self.q) + self.alpha * self.d_true
        self.t = round(self.t + PHYS_DT, 9)
        self.step += 1
        if self.t <= 1.0 + 1e-9:
            self.dev_max_1s = max(self.dev_max_1s, float(np.max(np.abs(self.q - self.qn))))
        self._pub(self.step, self.t, True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--params', required=True)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--d-true', default=None)
    ap.add_argument('--timeout-wall-s', type=float, default=60.0)
    ap.add_argument('--run-s', type=float, default=8.0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    reg = json.load(open(a.params))['bench']
    idn = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
    rclpy.init()
    nd = Stub(a, reg, idn['alpha'])
    w0 = time.monotonic()
    try:
        while rclpy.ok() and time.monotonic() - w0 < a.run_s + a.timeout_wall_s:
            rclpy.spin_once(nd, timeout_sec=0.005)
            if nd.rel.state == 'aborted' or (nd.t_release is not None and time.monotonic() - nd.t_release > a.run_s):
                break
    finally:
        json.dump({'release': nd.rel.record(), 'n_env_frozen': nd.n_env_frozen, 'first_applied_seq': nd.first_applied_seq,
                   'dev_max_1s_rad': nd.dev_max_1s, 'sim_t_end': nd.t, 'd_true': nd.d_true.tolist()}, open(a.out, 'w'), default=float)
        nd.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
