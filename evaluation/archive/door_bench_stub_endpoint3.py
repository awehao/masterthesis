#!/usr/bin/env python3
"""台架初態啟動的**替身執行端** r3（Codex r24：真 ADM＋真轉接器＋替身執行端；無 Isaac、無物理）。r1／r2 保留。
節點名 door_k2_sim、宣告參數 joints（真轉接器 arm_vel_adapter_k3m2 的順序核對對象）。凍結期同 ev9r2：t0＝0 重發 /clock、/joint_states、/odom（依排程）、
設定點回報、套用回報、/park/gate（bench_anchor）。訂閱**轉接器段**封裝（ENV.ST_ADAPTER；不直接接求解端）與 /bench/first_solution。
釋放與首步核對用**真模組**：door_bench_start2.BenchRelease（宣告 kind＝solution）、release_intake（真 EnvIntake ＋ 真 CmdChainE2，種子 s_init）、
DualSourceExecutor(initial＝wholebody).step、applied_source、verify_first_step。通過即記錄並結束（不跑物理）。
--odom-off "a:b,c:d"：本節點啟動後的牆鐘區間 [a, b)、[c, d) 不發 /odom（測延後 odom、yaw 過期／恢復）。
輸出（--out）：release 紀錄、first_intake、first_step、凍結輪數、收到的轉接器封裝數、釋放牆鐘時刻（相對啟動）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import rclpy
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import door_bench_start2 as B                          # noqa: E402
import door_k3m_applied_src as SRC                     # noqa: E402
import wgmpc_cmd_envelope as ENV                       # noqa: E402
from control_authority import AUTH_WHOLEBODY           # noqa: E402
from dual_source_executor import DualSourceExecutor    # noqa: E402
import test_door_bench_start as T1                     # noqa: E402

ARM = [f'joint{i}' for i in range(1, 7)]
PHYS_DT = 0.01


def parse_off(s):
    out = []
    for part in (s or '').split(','):
        if part.strip():
            a, b = part.split(':')
            out.append((float(a), float(b)))
    return out


class Stub(Node):
    def __init__(self, a, reg):
        super().__init__('door_k2_sim')
        self.declare_parameter('joints', ARM, ParameterDescriptor())
        self.a, self.reg, self.w0 = a, reg, time.monotonic()
        self.off = parse_off(a.odom_off)
        self.qn, self.s_init = list(reg['q_nominal']), list(reg['s_init'])
        self.rel = B.BenchRelease(a.run_id, 0.0, a.timeout_wall_s, time.monotonic())
        self.chain = T1.mk_chain()
        self.chain.seed_from_handover([0.0] * 9, list(self.s_init), 0.0, -1, source='bench_start')
        self.ex = DualSourceExecutor(self.chain, initial=AUTH_WHOLEBODY, stow_setpoint=tuple(self.qn), v_box_lin=0.035, v_box_ang=0.2)
        self.ex.wb_seeded = True
        self.intake = SRC.EnvIntake(run_id_num=ENV.run_id_num(a.run_id), stage_id=ENV.ST_ADAPTER, kinds=(ENV.K_NORMAL,))
        self.env_latest, self.n_env, self.done, self.out = None, 0, False, {}
        lat = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.clock = self.create_publisher(Clock, '/clock', 10)
        self.js = self.create_publisher(JointState, '/joint_states', 10)
        self.od = self.create_publisher(Odometry, '/odom', 10)
        self.spp = self.create_publisher(Float64MultiArray, '/coman/arm_setpoint', 10)
        self.spm = self.create_publisher(String, '/coman/arm_setpoint_meta', lat)
        self.ap = self.create_publisher(Float64MultiArray, '/coman/applied_cmd', 10)
        self.gate = self.create_publisher(String, '/park/gate', 1)
        self.create_subscription(Float64MultiArray, ENV.TOPIC[ENV.ST_ADAPTER], self._env, 50)
        self.create_subscription(String, B.RELEASE_TOPIC, self._ann, lat)
        self.spm.publish(String(data=json.dumps({'cols': ['physics_step_id', 'sim_t', 'ready'] + [f'sp_{j}' for j in ARM] + ['exec_mode_code', 'api_applied'],
                                                 'joint_order': ARM, 'sampling_instant': '**本步寫入後**（apply_action 之後）', 'physics_dt_s': PHYS_DT})))
        self.create_timer(PHYS_DT, self._tick)

    def _env(self, m):
        try:
            self.env_latest = ENV.decode(m.data)
            self.n_env += 1
        except ValueError:
            pass

    def _ann(self, m):
        try:
            self.rel.on_announce(json.loads(m.data))
        except ValueError:
            pass

    def _tick(self):
        if self.done:
            return
        w = time.monotonic() - self.w0
        c = Clock()
        self.clock.publish(c)
        js = JointState()
        js.name, js.position, js.velocity = list(ARM), [float(x) for x in self.qn], [0.0] * 6
        self.js.publish(js)
        if not any(a <= w < b for a, b in self.off):
            o = Odometry()
            o.header.frame_id, o.child_frame_id = 'odom', 'base_footprint'
            o.pose.pose.orientation.w = 1.0
            self.od.publish(o)
        self.spp.publish(Float64MultiArray(data=[0.0, 0.0, 1.0] + [float(x) for x in self.s_init] + [0.0, 1.0]))
        self.ap.publish(Float64MultiArray(data=[0.0, 0.0] + [0.0] * 9 + [0.0, float('nan'), 0.0, 0.0, 0.0, -1.0, float('nan')]))
        self.gate.publish(String(data=json.dumps({'step': 0, 'sim_t': 0.0, 'passed': False, 'ok_now': False, 'anchor': None, 'mode': 'hold',
                                                  'violation': False, 'bench': True, 'bench_anchor': [0.0, 0.0, 0.0]})))
        seq = None if self.env_latest is None else self.env_latest.get('source_seq')
        st = self.rel.try_release(seq, time.monotonic())
        if st == 'frozen':
            return
        self.out['release_wall_s'] = w
        if st == 'released':
            ok, why, rec = B.release_intake(self.intake, self.chain, self.env_latest, self.rel.announced['source_seq'], 0.0, lambda u: u, False)
            self.out['first_intake'] = {'ok': ok, 'why': why, **(rec or {})}
            if ok:
                res = self.ex.step(1, 0.0, PHYS_DT, q_arm_measured=list(self.qn), meas_vb=(0.0, 0.0, 0.0))
                src, mc = SRC.applied_source(self.chain, self.intake.map, res.owner == AUTH_WHOLEBODY)
                okf, whyf, frec = B.verify_first_step(res.owner, res.kind, src, mc, res.u_applied, res.arm_setpoint, self.s_init, PHYS_DT,
                                                      self.rel.announced['source_seq'], owner_wb=AUTH_WHOLEBODY)
                self.out['first_step'] = {'ok': okf, 'why': whyf, **frec}
        self.done = True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--params', required=True)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--odom-off', default='')
    ap.add_argument('--timeout-wall-s', type=float, default=40.0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    reg = json.load(open(a.params))['bench']
    rclpy.init()
    nd = Stub(a, reg)
    try:
        while rclpy.ok() and not nd.done and time.monotonic() - nd.w0 < a.timeout_wall_s + 5:
            rclpy.spin_once(nd, timeout_sec=0.005)
    finally:
        json.dump(dict(nd.out, release=nd.rel.record(), n_adapter_env=nd.n_env), open(a.out, 'w'), default=float)
        nd.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
