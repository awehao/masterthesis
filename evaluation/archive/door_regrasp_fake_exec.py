#!/usr/bin/env python3
"""換抓聯測的**替身執行端**（Codex r65／r66：三情境共用；無 Isaac、無物理引擎）。節點名 door_k2_sim；話題與訊息格式照 isaac_door_k2_sim_ev9r8。
控制邏輯用**真模組**（與 ev9r8 同一份）：CmdChainE2＋DualSourceExecutor（台架初態 wholebody、種子 s_init）、BenchRelease／verify_first_step、
  EnvIntake、W2Exec3＋door_regrasp_mode2.RegraspAuthority2＋door_regrasp_exec_core.RegraspExecCore（含 gate 2 停止准入、bench_release_w2）、
  GripGate、K2AbortLatch、K3-M ExecutorModeAuthority／ExecutorPauseAuthority、HoldMonitor（台架錨點）、走廊監看（交接前）、PhysicsContactHold。
受控體（名目運動學，**不是物理**）：
  底盤：套用的本體速度 × 增益 (0.9, 0.9, 0.82) 積分；手臂：設定點模型 q⁺＝q＋α(s−q)＋α⊙d_true（α＝free4 辨識，d_true＝台架登錄 d_init）；
  手指：一階追命令（τ 0.05 s），閉合時若 TCP 在登錄抓取位姿 5 mm／3° 內則停在接觸寬度並產生**合成**接觸力；
  門：外側夾持成立後門角由 TCP·G_T_H 的方位推得（同 door_coopen_inner_wgmpc），否則靜止。
合成接觸規則（明確、每筆紀錄 synthetic True；不讀任何「是否在等確認」的狀態）：
  外側：手指命令 < 接觸寬度 且 TCP 對 H_out(q)·T_HG_out ≤ 5 mm／3° ⇒ 兩指各 F_GRASP 於 bar_out（舊分類 bar）；否則 0；
  內側：手指命令 < 接觸寬度 且 TCP 對 H_in(q)·T_HG_in（參考包登錄）≤ 5 mm／3° ⇒ 兩指各 F_GRASP 於 bar_in；否則 0；其他類別恆 0。
故障注入（--fault，情境 ii）：none｜lose_force_hold（短保持 E0 成立後 0.5 s 起一指降至 0.3 N）｜drop_c3_mode4（mode 4 提交後 2 s 起 v3 紀錄 query_failed 0.3 s）｜
  abort_at_commit6（mode 6 提交當步送 K2 中止）｜stop_spike（ZERO_PUB 後 3 s 底盤 x 瞬增 0.2 mm、下一步回原位）｜stop_gap（ZERO_PUB 後 3 s 跳過一個物理步號）｜
  newer_before_step1（情境 i：首解宣告後、第 1 步前另注入一筆較新 v3 封裝）。
輸出（--out 目錄）：room_run.json（steps 摘要、bench、k3m、w2、regrasp、faults、synthetic 說明）、contact_v3.jsonl（合成 v3 紀錄）。

    python3 evaluation/door_regrasp_fake_exec.py --params … --run-id … --regrasp-ref … --out DIR [--scenario start|full] [--fault …]
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
import time

import numpy as np
import rclpy
import yaml
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray, String

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))
import cmd_envelope_v3 as E3                            # noqa: E402
import door_bench_start2 as BENCH                       # noqa: E402
import door_geom as DGM                                 # noqa: E402
import door_geom_v3 as G3R                              # noqa: E402
import door_k2_io as KIO                                # noqa: E402
import door_k3b_grip_gate as K3BG                       # noqa: E402
import door_k3m_applied_src as K3MSRC                   # noqa: E402
import door_k3m_corridor as K3MCOR                      # noqa: E402
import door_k3m_mode as K3MMODE                         # noqa: E402
import door_k3m_pause_mode as K3MP                      # noqa: E402
import door_regrasp_exec_core as RGX                    # noqa: E402
import door_regrasp_mode3 as RG3                        # noqa: E402
import door_regrasp_ref_registry as RGREG               # noqa: E402
import door_w2_exec3 as W2X3                            # noqa: E402
import wgmpc_cmd_envelope as ENV                        # noqa: E402
from control_authority import AUTH_WHOLEBODY           # noqa: E402
from dual_source_executor import DualSourceExecutor     # noqa: E402
from park_fixed import HoldMonitor, ParkCfg              # noqa: E402
from physics_contact_hold import PhysicsContactHold     # noqa: E402
from wb_cmd_chain_e2 import CmdChainE2                   # noqa: E402
from wb_wheel_limit import WheelLimitConfig             # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics   # noqa: E402

VERSION = 'door_regrasp_fake_exec/r1'
ARM = [f'joint{i}' for i in range(1, 7)]
FINGERS = ('uflite_finger1', 'uflite_finger2')
BASE_GAIN = (0.9, 0.9, 0.82)
F_GRASP, Q_CONTACT, TAU_F = 3.0, 0.003, 0.05
TOL_P, TOL_R = 0.005, math.radians(3.0)
SYNTH = '合成接觸（替身規則；不是物理接觸，不證明夾持）'


def rot_err(Ra, Rb):
    return math.acos(max(-1.0, min(1.0, (np.trace(Ra.T @ Rb) - 1) / 2)))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Fake(Node):
    def __init__(self, a):
        super().__init__('door_k2_sim')
        self.a = a
        self.declare_parameter('joints', list(ARM))
        lat = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        P = self.create_publisher
        self.clock_pub, self.js_pub, self.odom_pub = P(Clock, '/clock', 10), P(JointState, '/joint_states', 10), P(Odometry, '/odom', 10)
        self.applied_pub, self.sp_pub = P(Float64MultiArray, '/coman/applied_cmd', 10), P(Float64MultiArray, '/coman/arm_setpoint', 10)
        self.sp_meta_pub = P(String, '/coman/arm_setpoint_meta', lat)
        self.drawer_pub, self.handle_pub = P(Float64MultiArray, '/door/state', 10), P(Float64MultiArray, '/door/handle_pose', 10)
        self.sync_pub, self.contacts_pub = P(Float64MultiArray, '/door/sync_state', 10), P(String, '/door/contacts', 10)
        self.ho_pub, self.grip_pub = P(String, '/handover/state', 10), P(Float64MultiArray, '/gripper/state', 10)
        self.applied_src_pub = P(Float64MultiArray, '/door/applied_src', 100)
        self.w2_req_pub, self.w2_state_pub = P(String, '/door/w2_mode_req', lat), P(String, '/door/w2_mode_state', 50)
        self.rg8_c3_pub = P(String, '/door/contact_v3_summary', 50)
        self.k3m_ack_pub, self.k3m_abort_pub = P(String, '/door/base_mode_ack', lat), P(String, '/door/k2_abort', lat)
        self.k3m_pause_ack_pub = P(String, '/door/pause_ack_exec', lat)
        self.park_gate_pub, self.park_vio_pub = P(String, '/park/gate', 1), P(String, '/park/violation', 10)
        self.bench_state_pub = P(String, BENCH.STATE_TOPIC, lat)
        S = self.create_subscription
        self.n_cb = 0
        self.grip_cmd = self.ho_req = self.stop_req = self.k2_abort_msg = None
        self.k3m_req_msg = self.k3m_pause_req_msg = self.bench_first = self.rg8_pause_status = None
        self.w2_inbox, self.w2_req_q, self.n_env3 = collections.deque(), [], 0
        S(Float64, '/gripper/cmd', self._c(lambda m: setattr(self, 'grip_cmd', float(m.data)) if math.isfinite(float(m.data)) else None), 10)
        S(Float64MultiArray, E3.TOPIC3[ENV.ST_ADAPTER], self._c(self._env3), 50)
        S(String, '/door/w2_mode_req_task', self._c(lambda m: self.w2_req_q.append(self._js(m))), lat)
        S(String, '/handover/request', self._c(lambda m: setattr(self, 'ho_req', self._js(m))), 10)
        S(String, '/room/stop', self._c(lambda m: setattr(self, 'stop_req', str(m.data))), 10)
        S(String, '/door/k2_abort', self._c(lambda m: setattr(self, 'k2_abort_msg', str(m.data))), lat)
        S(String, '/door/base_mode_req', self._c(lambda m: setattr(self, 'k3m_req_msg', self._js(m))), lat)
        S(String, '/door/pause_req', self._c(lambda m: setattr(self, 'k3m_pause_req_msg', self._js(m))), lat)
        S(String, BENCH.RELEASE_TOPIC, self._c(lambda m: setattr(self, 'bench_first', m.data)), lat)
        S(String, '/wgmpc/pause_status', self._c(lambda m: setattr(self, 'rg8_pause_status', self._js(m))), lat)

    def _c(self, f):
        def g(m):
            self.n_cb += 1
            f(m)
        return g

    @staticmethod
    def _js(m):
        try:
            return json.loads(m.data)
        except ValueError:
            return {'_unparsable': str(m.data)[:200]}

    def _env3(self, m):
        try:
            e = E3.decode3(m.data)
        except ValueError:
            return
        self.n_env3 += 1
        self.w2_inbox.append((e, time.monotonic()))

    def stamp(self, t):
        from builtin_interfaces.msg import Time
        s = Time()
        s.sec, s.nanosec = int(t), int(round((t - int(t)) * 1e9))
        return s

    def spin_all(self):
        for _ in range(256):
            n = self.n_cb
            rclpy.spin_once(self, timeout_sec=0.0)
            if self.n_cb == n:
                break


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--params', required=True)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--regrasp-ref', required=True)
    ap.add_argument('--grasp-basis', default=os.path.join(HERE, 'results', 'door', 'coopen_core_grasp_basis_r4.json'))
    ap.add_argument('--corridor-table', required=True)
    ap.add_argument('--envelope', required=True)
    ap.add_argument('--spawn', default='1.02,-0.5,3.034362')
    ap.add_argument('--ident', default=os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json'))
    ap.add_argument('--door-spec', default=os.path.join(WS, 'src', 'my_omnibot_description', 'config', 'door_local_v3.yaml'))
    ap.add_argument('--scenario', default='full', choices=('start', 'full'))
    ap.add_argument('--fault', default='none', choices=('none', 'lose_force_hold', 'drop_c3_mode4', 'abort_at_commit6', 'stop_spike', 'stop_gap',
                                                         'newer_before_step1'))
    ap.add_argument('--sim-limit', type=float, default=330.0)
    ap.add_argument('--rtf', type=float, default=1.0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dt = 0.01
    prm = json.load(open(a.params))
    bench_cfg = BENCH.BenchConfig(prm['bench'], 0.05)
    s_init, q_nom, d_true = list(bench_cfg.s_init), list(bench_cfg.q_nominal), np.asarray(bench_cfg.d_init, float)
    ident = json.load(open(a.ident))
    alpha = np.asarray(ident['alpha'], float)
    pkg = json.load(open(a.regrasp_ref))
    ok, why = RGREG.validate(pkg)
    if not ok:
        print(f'[fake] 參考包登錄核驗不過：{why}', flush=True)
        return 66
    D3 = yaml.safe_load(open(a.door_spec))
    gb = json.load(open(a.grasp_basis))['args']                                  # 同任務節點 v10 的讀法
    if 'T_HG_full' in gb:
        T_HG_out = np.asarray(gb['T_HG_full'], float)
    else:
        T_HG_out = np.eye(4)
        T_HG_out[:3, :3] = np.asarray(gb['R_grasp'], float)
        T_HG_out[:3, 3] = [0.0, float(gb['grasp_depth_m']), 0.0]
    T_HG_in = np.asarray(pkg['inner_grasp']['T_HG'], float).reshape(4, 4)
    K = WholeBodyKinematics.from_urdf_file(os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
    lim = np.array(K.joint_limits())
    H_out = (lambda q: DGM.handle_pose(D3, q, D3['handle']['height']))
    H_in = (lambda q: G3R.handle_pose_inner(D3, q))

    def low_speed_bound(vx, vy, wz):
        if math.hypot(vx, vy) > 0.05:
            return (False, 'lin')
        if abs(wz) > 0.20:
            return (False, 'ang')
        return (True, None)
    chain = CmdChainE2(max_cmd_age_s=0.2, arm_rate_max=1.0, wheel_ok=low_speed_bound,
                       joint_lower=tuple(float(x) for x in lim[0, 3:]), joint_upper=tuple(float(x) for x in lim[1, 3:]),
                       joint_margin=0.05, mode='solver_drawer', wheel_cfg=WheelLimitConfig(arm_rate_max=1.0, dt_max=max(5.0 * dt, 0.05)),
                       base_accel_max_lin=0.5, base_accel_max_ang=2.0)
    ex = DualSourceExecutor(chain, initial=AUTH_WHOLEBODY, stow_setpoint=tuple(q_nom), max_seed_age_s=max(5.0 * dt, 0.05),
                            v_box_lin=0.035255, v_box_ang=0.1999, v_min_lin=0.0)
    chain.seed_from_handover([0.0] * 9, list(s_init), 0.0, -1, source='bench_start')
    ex.wb_seeded = True
    env_in = K3MSRC.EnvIntake(run_id_num=ENV.run_id_num(a.run_id), stage_id=ENV.ST_ADAPTER, kinds=(ENV.K_NORMAL,))
    k2_latch = KIO.K2AbortLatch()
    auth = RG3.RegraspAuthority(a.run_id, run_id_num=ENV.run_id_num(a.run_id), stage_id=ENV.ST_ADAPTER, cand_timeout_s=1.0, cmd_max_age_s=0.2,
                                  post_commit_timeout_s=0.25, dt=dt, a_lin=0.5, a_ang=2.0, arm_rate_max=1.0)
    w2x = W2X3.W2Exec3(auth, env_in, filt=lambda u: k2_latch.filter_wb(u))
    grip = K3BG.GripGate(0.0089, 0.0)
    core = RGX.RegraspExecCore(auth, pkg, H_in, dt=dt)
    rt = {'step': 0}
    auth.extra_state = lambda: {'gates': core.gates_state(rt['step']), 'regrasp_abort': core.abort}
    k3m_auth, k3m_pauth = K3MMODE.ExecutorModeAuthority(a.run_id), K3MP.ExecutorPauseAuthority(a.run_id)
    k3m_cor = K3MCOR.Corridor.from_json(json.load(open(a.corridor_table))['table'])
    k3m_env = [float(v) for v in a.envelope.split(',')]
    k3m = {'active': False, 'violation': None, 'events': []}
    park_cfg = ParkCfg(park_x=0.6, park_y=-0.5332, park_yaw=1.570796, hold_s=0.5, hold_mode=True, dev_pos_max=0.020, dev_yaw_max=math.radians(2.0))
    import dataclasses
    hold_cfg = dataclasses.replace(park_cfg, v_lin_max=math.inf, w_max=math.inf)
    contact_hold = PhysicsContactHold(physics_dt=dt, contact_min_n=0.5)
    # ---- 受控體 ----
    sp = [float(v) for v in a.spawn.split(',')]
    W = {'base': np.array(sp, float), 'q': np.asarray(q_nom, float).copy(), 's': np.asarray(s_init, float).copy(), 'qf': np.array([0.0089, 0.0089]),
         'door': 0.0, 'G_T_H': None, 'G_T_H_in': None, 'grasp': None}
    anchor = [float(W['base'][0]), float(W['base'][1]), float(W['base'][2])]
    park_hold = HoldMonitor(hold_cfg, tuple(anchor), 0, 0.0, prev=None)
    rclpy.init()
    nd = Fake(a)
    rec = {'version': VERSION, 'synthetic': SYNTH, 'scenario': a.scenario, 'fault': a.fault, 'events': [], 'steps_cols':
           ['step', 'sim_t', 'owner(0=nav,1=wb,2=glide)', 'kind', 'base_cmd_body', 'meas_vb_body', 'applied_arm', 'base_xyth', 'q_arm_meas', 'door_q_rad',
            'door_qdot_rad_s', 'q_finger', 'finger_contact_n', 'contact_held_s', 'leaf_contacts', 'wall_contacts', 'wrist_force_n', 'park_mode', 'applied_src',
            'auth_mode'],
           'steps': [], 'bench': {}, 'faults': [], 'steps_note': '同 ev9r8 欄位（逐物理步）；auth_mode 為替身附加；leaf_contacts 為合成舊分類'}
    c3f = open(os.path.join(a.out, 'contact_v3.jsonl'), 'w')

    def tcp():
        return np.asarray(K.fk(list(W['base']) + list(W['q']), 'link_tcp'), float)

    def near(T, G):
        return float(np.linalg.norm(T[:3, 3] - G[:3, 3])) <= TOL_P and rot_err(T[:3, :3], G[:3, :3]) <= TOL_R

    def publish_state(t, step):
        js = JointState()
        js.header.stamp = nd.stamp(t)
        js.name, js.position = list(ARM), [float(x) for x in W['q']]
        nd.js_pub.publish(js)
        od = Odometry()
        od.header.stamp = nd.stamp(t)
        od.header.frame_id, od.child_frame_id = 'odom', 'base_footprint'
        od.pose.pose.position.x, od.pose.pose.position.y = float(W['base'][0]), float(W['base'][1])
        od.pose.pose.orientation.w, od.pose.pose.orientation.z = math.cos(W['base'][2] / 2), math.sin(W['base'][2] / 2)
        nd.odom_pub.publish(od)
        Hm = H_out(W['door'])
        nd.drawer_pub.publish(Float64MultiArray(data=[float(t), float(W['door']), 0.0]))
        nd.handle_pub.publish(Float64MultiArray(data=[float(t), float(W['door'])] + [float(x) for x in Hm.reshape(-1)]))
        nd.sync_pub.publish(Float64MultiArray(data=[float(t), float(step), float(W['door'])] + [float(x) for x in Hm.reshape(-1)]
                                              + [float(v) for v in W['base']] + [float(x) for x in W['q']] + [0.0]))

    # ---- 台架凍結前置 ----
    nd.sp_meta_pub.publish(String(data=json.dumps({'cols': ['physics_step_id', 'sim_t', 'ready'] + [f'sp_{j}' for j in ARM] + ['exec_mode_code', 'api_applied'],
                                                   'joint_order': ARM, 'sampling_instant': '**本步寫入後**（apply_action 之後）', 'physics_dt_s': dt,
                                                   'physics_dt_nominal_s': dt})))
    bench_rel = BENCH.BenchRelease(a.run_id, 0.0, bench_cfg.release_timeout_wall_s, time.monotonic())
    rel_sel = None
    nd.bench_state_pub.publish(String(data=json.dumps({'state': 'frozen', 't0': 0.0, 'run_id': a.run_id})))
    w_start = time.monotonic()
    while rclpy.ok():
        nd.spin_all()
        nd.clock_pub.publish(Clock())
        if nd.bench_first is not None:
            try:
                bench_rel.on_announce(json.loads(nd.bench_first))
            except ValueError:
                pass
            nd.bench_first = None
        publish_state(0.0, 0)
        nd.sp_pub.publish(Float64MultiArray(data=[0.0, 0.0, 1.0] + [float(x) for x in s_init] + [0.0, 1.0]))
        nd.applied_pub.publish(Float64MultiArray(data=[0.0, 0.0] + [0.0] * 9 + [0.0, float('nan'), float(chain.n_recv), 0.0, 0.0, -1.0, float('nan')]))
        nd.park_gate_pub.publish(String(data=json.dumps({'step': 0, 'sim_t': 0.0, 'passed': False, 'ok_now': False, 'why_now': 'bench', 'pass_step': None,
                                                         'anchor': None, 'mode': 'hold', 'violation': False, 'bench': True, 'bench_anchor': anchor})))
        nd.ho_pub.publish(String(data=json.dumps({'owner': ex.owner, 'step': 0, 'sim_t': 0.0, 'chain_seeded': True, 'bench': 'frozen',
                                                  **ex.handover_pending_info(), 'nav_applied': None}, default=float)))
        es = max((e.get('source_seq') for e, _ in nd.w2_inbox if e.get('mode_seq') == 0), default=None)
        if bench_rel.try_release(es, time.monotonic()) != 'frozen':
            if bench_rel.state == 'released':
                aseq = (bench_rel.announced or {}).get('source_seq')
                e8 = next((x for x, _ in nd.w2_inbox if x.get('source_seq') == aseq), None)
                keep = [(x, w) for x, w in nd.w2_inbox if not (x.get('source_seq') is not None and aseq is not None and x['source_seq'] <= aseq)]
                nd.w2_inbox.clear()
                nd.w2_inbox.extend(keep)
                ok, why, irec, rel_sel = RGX.bench_release_w2(w2x, chain, e8, aseq, 1, 0.0, nd.k2_abort_msg is not None or nd.stop_req is not None)   # 同 ev9r8：步號＝台架首步 1
                rec['bench']['first_intake'] = {'ok': ok, 'why': why, **(irec or {}), 'n_inbox_kept': len(keep)}
                if not ok:
                    bench_rel.state, bench_rel.released = 'aborted', {'why': f'release_intake：{why}'}
            break
        if time.monotonic() - w_start > bench_cfg.release_timeout_wall_s + 5:
            break
        time.sleep(0.01)
    rec['bench']['release'] = bench_rel.record()
    if bench_rel.state != 'released':
        nd.bench_state_pub.publish(String(data=json.dumps({'state': bench_rel.state, 't0': 0.0, 'run_id': a.run_id, **bench_rel.record()}, default=float)))
        json.dump(rec, open(os.path.join(a.out, 'room_run.json'), 'w'), ensure_ascii=False, default=float)
        return 2
    if a.fault == 'newer_before_step1':                                        # 情境 i：首步前收到較新封裝（模擬：再等一輪求解端發布）
        t_w = time.monotonic()
        while time.monotonic() - t_w < 0.3:
            nd.spin_all()
            time.sleep(0.01)
        rec['faults'].append({'kind': 'newer_before_step1', 'n_inbox_before_step1': len(nd.w2_inbox),
                              'seqs': [e.get('source_seq') for e, _ in nd.w2_inbox][:10]})
    # ---- 主迴圈 ----
    step, t = 1, 0.0                                                          # 同 ev9r8：台架模式主迴圈首步 step_id＝1、t＝t0（不推進）
    first_pending = True
    prev_pose = None
    cw = {'n': 0, 'invalid': 0, 'fb': [0.0, 0.0], 'fb_min': [math.inf, math.inf]}
    zero_t = None
    spike_done = gap_done = False
    t_mode = {}
    w0, t0 = time.monotonic(), t
    while rclpy.ok() and t < a.sim_limit:
        nd.spin_all()
        nd.clock_pub.publish(Clock(clock=nd.stamp(t)))
        if nd.stop_req is not None:
            rec['stop_reason'] = nd.stop_req
            break
        if nd.k2_abort_msg is not None and k2_latch.latched is None:
            gnow = 0.0089 if nd.grip_cmd is None else min(max(float(nd.grip_cmd), 0.0), 0.0089)
            k2_latch.on_msg(nd.k2_abort_msg, step, t, gnow)
            k2_latch.expire_chain(chain)
            ex.cancel_handover('K2 中止閂鎖')
            rec['events'].append({'sim_t': t, 'step': step, 'k2_abort_latched': k2_latch.latched})
        ib = []
        while nd.w2_inbox and not first_pending:
            ib.append(nd.w2_inbox.popleft())
        rq, nd.w2_req_q = list(nd.w2_req_q), []
        known = (f"k2_latch:{k2_latch.latched.get('why')}" if k2_latch.latched is not None else
                 (f"k3m_violation:{k3m['violation'].get('why')}" if k3m['violation'] is not None else None))
        if first_pending and rel_sel is not None:                               # 首步：釋放時已 select（步 0），不再 select（否則 confirm_missing）
            sel, relay, nv = rel_sel, [], []
            nd.w2_req_q = rq + nd.w2_req_q
        else:
            sel, relay, nv = w2x.step_begin(step, t, chain, known_abort=known, inbox=ib, requests=rq)
        if (sel.get('event') or {}).get('kind') == 'commit':
            core.on_commit(step, t, auth.mode_seq, tcp())
            t_mode[auth.mode_seq] = t
            if a.fault == 'abort_at_commit6' and auth.mode_seq == 6:
                nd.k3m_abort_pub.publish(String(data=json.dumps({'why': 'fault_abort_at_commit6', 't': t, 'source': 'fake_fault'})))
                rec['faults'].append({'kind': 'abort_at_commit6', 'step': step, 't': t})
        for r_ in relay:
            nd.w2_req_pub.publish(String(data=json.dumps(r_, default=float)))
        if nv:
            rec['events'].append({'sim_t': t, 'step': step, 'w2_entry_violation': nv[0]})
        if nd.ho_req is not None:
            r = nd.ho_req
            nd.ho_req = None
            if isinstance(r, dict) and r.get('action') != 'cancel':
                ex.request_handover(r.get('to', AUTH_WHOLEBODY), int(r.get('at_step', step + 5)), t, window_steps=int(r.get('window_steps', 0)))
        q_arm = [float(x) for x in W['q']]
        mvb = None
        if prev_pose is not None:
            c_, s_ = math.cos(W['base'][2]), math.sin(W['base'][2])
            dx, dy = W['base'][0] - prev_pose[0], W['base'][1] - prev_pose[1]
            mvb = [(dx * c_ + dy * s_) / dt, (-dx * s_ + dy * c_) / dt, wrap(W['base'][2] - prev_pose[2]) / dt]
        res = ex.step(step, t, dt, nav_cmd=None, glide_cmd=None, q_arm_measured=q_arm, meas_vb=mvb)
        src4, mc4 = K3MSRC.applied_source(chain, env_in.map, res.owner == AUTH_WHOLEBODY)
        nd.applied_src_pub.publish(Float64MultiArray(data=[float(step), float(t), float(src4), float(mc4)] + [float(x) for x in res.base_cmd]
                                                     + [float(x) for x in res.u_applied[3:]]))
        if first_pending:
            first_pending = False
            okf, whyf, frec = BENCH.verify_first_step(res.owner, res.kind, src4, mc4, res.u_applied, res.arm_setpoint, s_init, dt,
                                                      (bench_rel.announced or {}).get('source_seq'), owner_wb=AUTH_WHOLEBODY)
            rec['bench']['first_step'] = {'ok': okf, 'why': whyf, 'step': step, 'sim_t': t, **frec}
            if not okf:
                nd.bench_state_pub.publish(String(data=json.dumps({'state': 'aborted', 't0': 0.0, 'run_id': a.run_id, 'why': f'bench_first_step：{whyf}'})))
                break
            nd.bench_state_pub.publish(String(data=json.dumps({'state': 'released', 't0': 0.0, 'run_id': a.run_id, **bench_rel.record(),
                                                                'first_step': {'ok': True, 'applied_source_seq': src4}}, default=float)))
            if a.scenario == 'start':
                break
        w2x.post_step(step, t, chain, res.owner == AUTH_WHOLEBODY, [float(x) for x in res.base_cmd] + [float(x) for x in res.u_applied[3:]], sel)
        w2x.sync_after(step, t, known_abort=known)
        if step % 2 == 0:
            nd.w2_state_pub.publish(String(data=json.dumps(auth.state_msg(step, t), default=float)))
            s8 = core.summary()
            if s8 is not None:
                nd.rg8_c3_pub.publish(String(data=json.dumps(s8, default=float)))
        fcmd = k2_latch.grip(0.0089 if nd.grip_cmd is None else min(max(float(nd.grip_cmd), 0.0), 0.0089))
        fcmd = grip.command(step, t, k2_latch, auth.mode_seq, auth.finger_target(), fcmd)
        # ---- 受控體推進（本步命令）----
        prev_pose = (float(W['base'][0]), float(W['base'][1]), float(W['base'][2]))
        vb = [float(x) * g for x, g in zip(res.base_cmd, BASE_GAIN)]
        c_, s_ = math.cos(W['base'][2]), math.sin(W['base'][2])
        W['base'] = W['base'] + dt * np.array([c_ * vb[0] - s_ * vb[1], s_ * vb[0] + c_ * vb[1], vb[2]])
        if res.arm_setpoint is not None:
            W['s'] = np.asarray(res.arm_setpoint, float)
        W['q'] = W['q'] + alpha * (W['s'] - W['q']) + alpha * d_true
        T = tcp()
        Go, Gi = H_out(W['door']) @ T_HG_out, H_in(W['door']) @ T_HG_in
        out_c = fcmd < Q_CONTACT and near(T, Go)
        in_c = fcmd < Q_CONTACT and near(T, Gi) and not out_c
        tgt_f = max(fcmd, Q_CONTACT) if (out_c or in_c) else fcmd
        W['qf'] = W['qf'] + (dt / TAU_F) * (tgt_f - W['qf'])
        f_out = [F_GRASP if (out_c and W['qf'][i] <= Q_CONTACT + 1e-4) else 0.0 for i in range(2)]
        f_in = [F_GRASP if (in_c and W['qf'][i] <= Q_CONTACT + 1e-4) else 0.0 for i in range(2)]
        if a.fault == 'lose_force_hold' and getattr(core.hold, 'E0', None) is not None:
            tb = next((e['t'] for e in core.hold.events if e.get('baseline')), None)
            if tb is not None and t >= tb + 0.5:
                f_in[1] = 0.3
                if not rec['faults']:
                    rec['faults'].append({'kind': 'lose_force_hold', 'step': step, 't': t})
        if min(f_out) > 0 and W['G_T_H'] is None:
            W['G_T_H'] = np.linalg.inv(T) @ H_out(W['door'])
            rec['events'].append({'sim_t': t, 'step': step, 'grasp_outer_coupled': True})
        if W['G_T_H'] is not None and min(f_out) > 0:
            Hm = T @ W['G_T_H']
            h0 = H_out(0.0)
            a0 = math.atan2(h0[1, 3], h0[0, 3])
            W['door'] = wrap(a0 - math.atan2(Hm[1, 3], Hm[0, 3]))
        elif W['G_T_H'] is not None and max(f_out) == 0:
            W['G_T_H'] = None
            rec['events'].append({'sim_t': t, 'step': step, 'grasp_outer_released': True})
        if min(f_in) > 0 and W['G_T_H_in'] is None:                           # 內側夾持成立 ⇒ 門隨 TCP（mode 7 拉關）
            W['G_T_H_in'] = np.linalg.inv(T) @ H_in(W['door'])
            rec['events'].append({'sim_t': t, 'step': step, 'grasp_inner_coupled': True})
        if W['G_T_H_in'] is not None and min(f_in) > 0:
            Hm = T @ W['G_T_H_in']
            h0 = H_in(0.0)
            W['door'] = wrap(math.atan2(h0[1, 3], h0[0, 3]) - math.atan2(Hm[1, 3], Hm[0, 3]))
        elif W['G_T_H_in'] is not None and max(f_in) == 0:
            W['G_T_H_in'] = None
            rec['events'].append({'sim_t': t, 'step': step, 'grasp_inner_released': True})
        # ---- 停止准入故障注入（ZERO_PUB 宣告後）----
        z8 = (((nd.rg8_pause_status or {}).get('outcome') or {}).get('transitions') or {}).get('ZERO_PUB')
        if z8 is not None and zero_t is None:
            zero_t = float(z8)
        base_rep = W['base'].copy()
        if a.fault == 'stop_spike' and zero_t is not None and not spike_done and t >= zero_t + 3.0:
            base_rep[0] += 0.0002
            spike_done = True
            rec['faults'].append({'kind': 'stop_spike', 'step': step, 't': t})
        if a.fault == 'stop_gap' and zero_t is not None and not gap_done and t >= zero_t + 3.0:
            gap_done = True
            rec['faults'].append({'kind': 'stop_gap', 'skipped_step': step, 't': t})
            step += 1
            t = round((step - 1) * dt, 6)
        # ---- 合成 v3 紀錄（同一物理步）----
        pts = [{'robot_link': FINGERS[i], 'class': 'bar_out', 'f': f_out[i]} for i in range(2) if f_out[i] > 0] + \
              [{'robot_link': FINGERS[i], 'class': 'bar_in', 'f': f_in[i]} for i in range(2) if f_in[i] > 0]
        c3r = {'status': 'ok', 'complete': True, 'step': step, 't': round(t, 6), 'points': pts, 'synthetic': True}
        if a.fault == 'drop_c3_mode4' and 4 in t_mode and t_mode[4] + 2.0 <= t < t_mode[4] + 2.3:
            c3r = {'status': 'query_failed', 'step': step, 't': round(t, 6), 'synthetic': True}
            if not any(f['kind'] == 'drop_c3_mode4' for f in rec['faults']):
                rec['faults'].append({'kind': 'drop_c3_mode4', 'step': step, 't': t})
        c3f.write(json.dumps(c3r) + '\n')
        # ---- K3-M 模式與暫停權威、走廊（交接前）----
        if nd.k3m_req_msg is not None:
            r_, nd.k3m_req_msg = nd.k3m_req_msg, None
            if not k3m['active'] and park_hold.end is not None:
                kind, ack = 'reject', {'run_id': a.run_id, 'seq': r_.get('seq') if isinstance(r_, dict) else None, 'why': 'hold_monitor_not_active', 'step': step}
            else:
                kind, ack = k3m_auth.on_request(r_, step, t, aborted=k2_latch.latched is not None, owner_wholebody=(res.owner == AUTH_WHOLEBODY))
            if kind == 'switch':
                park_hold.close(step, t)
                k3m['active'] = True
            k3m['events'].append({'step': step, 't': t, 'kind': kind, 'msg': ack})
            nd.k3m_ack_pub.publish(String(data=json.dumps(ack, default=float)))
        if park_hold.end is None:
            park_hold.update(step, t, float(W['base'][0]), float(W['base'][1]), float(W['base'][2]), [float(x) for x in res.base_cmd], phase=res.kind)
        if nd.k3m_pause_req_msg is not None:
            r_, nd.k3m_pause_req_msg = nd.k3m_pause_req_msg, None
            kp, ackp = k3m_pauth.on_request(r_, step, t, aborted=k2_latch.latched is not None, base_mode_free=bool(k3m['active']))
            k3m['events'].append({'step': step, 't': t, 'kind': 'pause_' + kp, 'msg': ackp})
            nd.k3m_pause_ack_pub.publish(String(data=json.dumps(ackp, default=float)))
        if core.handover is not None and 'regrasp_end' not in k3m:
            k3m['regrasp_end'] = dict(core.handover)
        if k3m['active'] and k3m['violation'] is None and 'regrasp_end' not in k3m:
            ok3, e3, why3 = k3m_cor.check(tuple(W['base']), math.degrees(W['door']), 0.020, math.radians(0.8))
            fw = math.cos(anchor[2]) * (W['base'][0] - anchor[0]) + math.sin(anchor[2]) * (W['base'][1] - anchor[1])
            lt = -math.sin(anchor[2]) * (W['base'][0] - anchor[0]) + math.cos(anchor[2]) * (W['base'][1] - anchor[1])
            dyaw = math.degrees(wrap(W['base'][2] - anchor[2]))
            if ok3 and not (k3m_env[0] <= fw <= k3m_env[1] and k3m_env[2] <= lt <= k3m_env[3] and abs(dyaw) <= k3m_env[4]):
                ok3, why3 = False, 'envelope'
            if not ok3:
                k3m['violation'] = {'step': step, 't': t, 'why': f'K3-M 執行端監看：{why3}', 'dev': e3}
                if k2_latch.latched is None:
                    k2_latch.on_msg(k3m['violation']['why'], step, t, fcmd)
                    k2_latch.expire_chain(chain)
                nd.k3m_abort_pub.publish(String(data=json.dumps({'why': k3m['violation']['why'], 't': t, 'source': 'executor_k3m'}, default=float)))
        # ---- 換抓核心（同一物理步資料）----
        rt['step'] = step
        if zero_t is not None:
            core.g[2].set_zero(zero_t)
        why8 = core.step(step, t, sel.get('mode_seq'), c3r, T, list(base_rep), q_arm, list(W['qf']), float(W['door']),
                         base_cmd=[float(x) for x in res.base_cmd], arm_cmd=[float(x) for x in res.u_applied[3:]])
        if why8 is not None and not rec.get('regrasp_abort'):
            rec['regrasp_abort'] = {'step': step, 't': t, 'why': why8}
            if k2_latch.latched is None:
                k2_latch.on_msg(f'regrasp：{why8}', step, t, fcmd)
                k2_latch.expire_chain(chain)
            nd.k3m_abort_pub.publish(String(data=json.dumps({'why': f'regrasp：{why8}', 't': t, 'source': 'executor_rg8'}, default=float)))
        # ---- 舊分類、夾爪狀態、接觸窗口 ----
        held = contact_hold.update(step, t, f_out[0] + f_in[0], f_out[1] + f_in[1], closing=(nd.grip_cmd is not None and nd.grip_cmd < 0.0089 - 1e-12))
        cw['n'] += 1
        fb = [f_out[0] + f_in[0], f_out[1] + f_in[1]]
        cw['fb'] = [max(cw['fb'][i], fb[i]) for i in range(2)]
        cw['fb_min'] = [min(cw['fb_min'][i], fb[i]) for i in range(2)]
        if step % 2 == 0:
            nd.grip_pub.publish(Float64MultiArray(data=[t, float(W['qf'][0]), float(W['qf'][1]), fb[0], fb[1], float(step), float(held), 0.5]))
        nd.applied_pub.publish(Float64MultiArray(data=[float(step), float(t)] + [float(x) for x in res.u_applied]
                                                 + [0.0, float('nan'), float(chain.n_recv), float(chain.n_rejected), 1.0, -1.0, float('nan')]))
        if res.arm_setpoint is not None:
            nd.sp_pub.publish(Float64MultiArray(data=[float(step), float(t), 1.0] + [float(x) for x in res.arm_setpoint] + [0.0, 1.0]))
        if step % 2 == 0:
            nd.park_gate_pub.publish(String(data=json.dumps({'step': step, 'sim_t': t, 'passed': False, 'ok_now': False, 'why_now': 'bench', 'pass_step': None,
                                                             'anchor': None, 'mode': 'hold' if park_hold.end is None else 'post', 'violation': False,
                                                             'bench': True, 'bench_anchor': anchor})))
        lc = {FINGERS[i]: {'bar': f_out[i] + f_in[i], 'panel': 0.0, 'ambiguous': 0.0} for i in range(2)}
        rec['steps'].append([step, round(t, 4), 1 if res.owner == AUTH_WHOLEBODY else 2, res.kind, [round(float(x), 6) for x in res.base_cmd],
                             None if mvb is None else [round(x, 6) for x in mvb], [round(float(x), 6) for x in res.u_applied[3:]],
                             [round(float(v), 6) for v in base_rep], [round(float(v), 6) for v in q_arm], round(float(W['door']), 8), 0.0,
                             [round(float(v), 6) for v in W['qf']], [round(x, 4) for x in fb], float(held), lc, {}, 0.0,
                             'hold' if park_hold.end is None else 'post', [step, round(t, 6), int(src4), int(mc4)], auth.mode_seq])
        step += 1
        t = round((step - 1) * dt, 6)
        if step % 2 == 0:
            publish_state(t, step)
            nd.contacts_pub.publish(String(data=json.dumps({'sim_t': t, 'step': step, 'n_steps': cw['n'], 'n_invalid': cw['invalid'],
                                                            'finger_bar_max_n': cw['fb'], 'finger_bar_min_n': cw['fb_min'], 'unexpected_max_n': 0.0,
                                                            'unexpected_who': None, 'wall_max_n': 0.0, 'wrist_max_n': 0.0}, default=float)))
            cw = {'n': 0, 'invalid': 0, 'fb': [0.0, 0.0], 'fb_min': [math.inf, math.inf]}
            nd.ho_pub.publish(String(data=json.dumps({'owner': ex.owner, 'step': step, 'sim_t': t, 'chain_seeded': ex.wb_seeded,
                                                      **ex.handover_pending_info(), 'nav_applied': ex.nav_applied_report()}, default=float)))
        lag = (t - t0) / a.rtf - (time.monotonic() - w0)
        if lag > 0:
            time.sleep(lag)
    rec['t_end'] = t
    rec['door_final_deg'] = math.degrees(W['door'])
    rec['base_final'] = [float(v) for v in W['base']]
    rec['arm_final'] = [float(v) for v in W['q']]
    rec['t_mode'] = t_mode
    rec['k3m'] = k3m
    rec['w2'] = {'auth': auth.stats(), 'rows_tail': w2x.rows[-50:], 'auth_rows_n': len(auth.applied_rows), 'n_env3': nd.n_env3}
    rec['regrasp'] = {'core': core.report(), 'grip_gate': grip.report()}
    rec['k2_latch'] = k2_latch.latched
    c3f.close()
    json.dump(rec, open(os.path.join(a.out, 'room_run.json'), 'w'), ensure_ascii=False, default=float)
    nd.destroy_node()
    rclpy.shutdown()
    print(f'[fake] 結束 t {t:.2f}；mode {auth.mode_seq}；abort {core.abort}；k2 {k2_latch.latched}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
