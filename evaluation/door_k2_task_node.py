#!/usr/bin/env python3
"""K2 任務節點：把量測組成 door_k2_policy.Meas、執行相位機動作、寫紀錄。決策全部在 door_k2_policy（已離線測試）。

輸入：/door/sync_state（29 欄）、/door/contacts（JSON）、/gripper/state（8 欄）、/handover/state、/wholebody/state、
      /wgmpc/status、/coman/applied_fail、/park/gate。
輸出：/drawer/tcp_target（4×4 列優先，求解端 --target-topic）、/gripper/cmd、/drawer/solver_start、/wgmpc/stop、
      /handover/request（停車閘門通過後送一次）、/door/k2_phase、/room/stop（結束時要求執行端受控停止並封存）。
斜坡速率以**模擬時間**差計算（RTF < 1 時不按牆鐘推進）。牆鐘上限：全趟 600 s、中止後觀察 60 s。

    python3 evaluation/door_k2_task_node.py --out <run_dir>/task.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float64, Float64MultiArray, String

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics   # noqa: E402
import door_k2_policy as P                                                # noqa: E402
from physics_contact_hold import contact_established                     # noqa: E402

ARM = [f'joint{i}' for i in range(1, 7)]


class Node2(Node):
    def __init__(self):
        super().__init__('door_k2_task')
        lat = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.tgt_pub = self.create_publisher(Float64MultiArray, '/drawer/tcp_target', 10)
        self.grip_pub = self.create_publisher(Float64, '/gripper/cmd', 10)
        self.start_pub = self.create_publisher(Bool, '/drawer/solver_start', lat)
        self.stop_pub = self.create_publisher(String, '/wgmpc/stop', 10)
        self.ho_pub = self.create_publisher(String, '/handover/request', 10)
        self.phase_pub = self.create_publisher(String, '/door/k2_phase', 10)
        self.room_stop_pub = self.create_publisher(String, '/room/stop', 10)
        self.sync = None          # (t, step, q, H, pose(3), q_arm(6), qdot)
        self.contacts = None      # dict＋接收時的模擬時間
        self.gs = None
        self.hs = None
        self.wb = None
        self.ws_t = None          # 最後一筆 /wgmpc/status 的模擬時間
        self.n_ws = 0
        self.fail = None
        self.gate = None
        self.create_subscription(Float64MultiArray, '/door/sync_state', self._sync, 1)
        self.create_subscription(String, '/door/contacts', self._ct, 10)
        self.create_subscription(Float64MultiArray, '/gripper/state', self._gs, 1)
        self.create_subscription(String, '/handover/state', self._hs, 1)
        self.create_subscription(String, '/wholebody/state', self._wb, 10)
        self.create_subscription(Float64MultiArray, '/wgmpc/status', self._ws, 1)
        self.create_subscription(String, '/coman/applied_fail', self._af, 10)
        self.create_subscription(String, '/park/gate', self._gate, 1)

    def _sync(self, m):
        d = [float(x) for x in m.data]
        if len(d) != 29:
            return
        self.sync = (d[0], d[1], d[2], np.asarray(d[3:19]).reshape(4, 4), d[19:22], np.asarray(d[22:28]), d[28])

    def _ct(self, m):
        try:
            self.contacts = json.loads(m.data)
        except Exception:
            self.contacts = None

    def _gs(self, m):
        d = list(m.data)
        self.gs = tuple(float(x) for x in d) if len(d) == 8 else None

    def _hs(self, m):
        try:
            self.hs = json.loads(m.data)
        except Exception:
            pass

    def _wb(self, m):
        try:
            self.wb = json.loads(m.data)
        except Exception:
            pass

    def _ws(self, m):
        self.n_ws += 1
        self.ws_t = self.sync[0] if self.sync is not None else None

    def _af(self, m):
        self.fail = str(m.data)

    def _gate(self, m):
        try:
            self.gate = json.loads(m.data)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--urdf', default=os.path.join(WS, 'evaluation/models/omni_bot_wholebody_expanded.urdf'))
    ap.add_argument('--tcp', default='link_tcp')
    ap.add_argument('--grasp-basis', default=os.path.join(HERE, 'runs', 'phf_01_M', 'task.json'))
    ap.add_argument('--rate', type=float, default=20.0)
    ap.add_argument('--wall-total-s', type=float, default=600.0)
    ap.add_argument('--wall-post-abort-s', type=float, default=60.0)
    a = ap.parse_args()
    if os.path.exists(a.out):
        print(f'[k2] {a.out} 已存在 ⇒ 拒絕覆寫', flush=True)
        return 2
    gb = json.load(open(a.grasp_basis))['args']
    T_HG = np.eye(4)
    T_HG[:3, :3] = np.asarray(gb['R_grasp'], float)
    T_HG[:3, 3] = [0.0, float(gb['grasp_depth_m']), 0.0]
    cfg = P.K2Config()
    pol = P.K2Policy(cfg, T_HG)
    K = WholeBodyKinematics.from_urdf_file(a.urdf)
    rep = {'schema': 'door_k2_task/1', 'args': vars(a), 'config': cfg.__dict__,
           'grasp_basis': {'path': os.path.relpath(a.grasp_basis, WS), 'sha256': hashlib.sha256(open(a.grasp_basis, 'rb').read()).hexdigest(),
                           'T_HG': T_HG.tolist()},
           'policy_sha256': hashlib.sha256(open(P.__file__, 'rb').read()).hexdigest(), 'trace': [], 'events': []}

    def save():
        rep['policy_events'] = pol.events
        rep['final_phase'], rep['abort'], rep['max_abs_q_door_rad'] = pol.phase, pol.abort, pol.max_abs_q
        tmp = a.out + '.tmp'
        json.dump(rep, open(tmp, 'w'), ensure_ascii=False, indent=1, default=float)
        os.replace(tmp, a.out)

    rclpy.init()
    nd = Node2()
    w0 = time.monotonic()
    t_prev = None
    ho_sent = False
    w_abort = None
    dt_w = 1.0 / a.rate
    last_phase = None
    try:
        while rclpy.ok():
            for _ in range(64):
                rclpy.spin_once(nd, timeout_sec=0.0)
            if time.monotonic() - w0 > a.wall_total_s:
                rep['wall_timeout'] = True
                pol._abort('wall_total_timeout', t_prev or 0.0)
            if nd.sync is None:
                time.sleep(dt_w)
                continue
            t, step, q, H, pose, q_arm, qdot = nd.sync
            ct = nd.contacts
            gate = nd.gate or {}
            anchor = gate.get('anchor')
            if anchor is not None:
                bd = math.hypot(pose[0] - anchor[0], pose[1] - anchor[1])
                byaw = math.atan2(math.sin(pose[2] - anchor[2]), math.cos(pose[2] - anchor[2]))
            else:
                bd = byaw = None
            T_tcp = K.fk(np.array(list(pose) + list(q_arm), float), a.tcp)
            gs = nd.gs
            ev = contact_established(gs, now=t, close_started=pol.close_t0, contact_min_n=cfg.contact_min_n,
                                     hold_s=cfg.attach_hold_s) if (gs is not None and pol.close_t0 is not None) else False
            m = P.Meas(t=t, owner=(nd.hs or {}).get('owner'),
                       unfold_done=(nd.wb or {}).get('phase') in ('UNFOLD_DONE', 'RELEASED'),
                       solver_up=nd.n_ws > 0, solver_failed=nd.fail,
                       solver_status_age=(None if nd.ws_t is None else t - nd.ws_t), state_age=0.0,
                       contacts_age=(None if ct is None else t - float(ct.get('sim_t', -1e9))),
                       q_door=q, qdot_door=qdot, H=H, T_tcp=T_tcp,
                       finger_bar_min=None if ct is None else tuple(ct['finger_bar_min_n']),
                       finger_bar_max=None if ct is None else tuple(ct['finger_bar_max_n']),
                       unexpected_max=None if ct is None else ct['unexpected_max_n'], unexpected_who=None if ct is None else ct['unexpected_who'],
                       wall_max=None if ct is None else ct['wall_max_n'], wrist_max=None if ct is None else ct['wrist_max_n'],
                       contacts_invalid=None if ct is None else ct['n_invalid'], contact_evidence_ok=bool(ev),
                       finger_q=None if gs is None else (gs[1], gs[2]), base_drift_m=bd, base_drift_yaw=byaw)
            dt = 0.0 if t_prev is None else max(0.0, t - t_prev)
            t_prev = t
            act = pol.step(m, dt)
            if act['request_handover'] and not ho_sent and gate.get('passed') and gate.get('ok_now'):
                nd.ho_pub.publish(String(data=json.dumps({'to': 'wholebody', 'at_step': int(step) + 5, 'window_steps': 50})))
                ho_sent = True
                rep['events'].append({'t': t, 'handover_requested_at_step': int(step) + 5, 'gate': gate})
            if act['target'] is not None:
                nd.tgt_pub.publish(Float64MultiArray(data=[float(x) for x in np.asarray(act['target']).reshape(-1)]))
            nd.grip_pub.publish(Float64(data=float(act['grip'])))
            if act['solver_start']:
                nd.start_pub.publish(Bool(data=True))
            if act['stop_solver']:
                nd.stop_pub.publish(String(data=f'door_k2 {pol.phase}：{pol.abort or "done"}'))
            if pol.phase != last_phase:
                nd.phase_pub.publish(String(data=pol.phase))
                print(f'[k2] 相位 → {pol.phase} @ sim {t:.2f}' + (f'（中止：{pol.abort}）' if pol.abort else ''), flush=True)
                last_phase = pol.phase
            if pol.phase == 'ABORTED' and w_abort is None:
                w_abort = time.monotonic()
            rep['trace'].append({'t': t, 'step': step, 'phase': pol.phase, 'q_door': q, 'qdot_door': qdot, 's': pol.s, 'grip': act['grip'],
                                 'err_p': (None if pol.last_target is None else float(np.linalg.norm(T_tcp[:3, 3] - pol.last_target[:3, 3]))),
                                 'base_drift_m': bd, 'base_drift_yaw': byaw, 'contacts': ct, 'evidence': bool(ev)})
            if len(rep['trace']) % 100 == 0:
                save()
            if act['finish'] or (w_abort is not None and time.monotonic() - w_abort > a.wall_post_abort_s):
                if w_abort is not None and not act['finish']:
                    rep['post_abort_wall_cap_hit'] = True
                break
            time.sleep(dt_w)
    finally:
        rep['wall_s'] = round(time.monotonic() - w0, 2)
        save()
        try:
            nd.stop_pub.publish(String(data='door_k2 結束'))
            nd.room_stop_pub.publish(String(data=f'door_k2 結束：{pol.phase} {pol.abort or ""}'))
            for _ in range(10):
                rclpy.spin_once(nd, timeout_sec=0.05)
        except Exception:
            pass
        print(f'[k2] 結束：{pol.phase}' + (f'（中止 {pol.abort}）' if pol.abort else ''), flush=True)
        nd.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0 if pol.phase == 'DONE' else 1


if __name__ == '__main__':
    sys.exit(main())
