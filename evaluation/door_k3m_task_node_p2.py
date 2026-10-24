#!/usr/bin/env python3
# K3-M 暫停版 p2 任務節點（door_k3m_task_node_p.py 的版本化複本；Codex 20261024_101421 (a)）：
#   策略換為 door_k3m_policy_p2.K3MP2Policy（只換 SETTLE 完成判斷的門速來源）；/door/sync_state 每筆另餵 door_k3m_settle_sig.SettleRateWindow
#   （50 Hz 同步流的 0.1 s 因果後向門角差分），每輪 m.settle_sig＝窗口值（最新樣本必須是本輪步號）；m.qdot_door 不改。
#   trace 每列另記 settle_sig（窗口兩端步號／門角／時間、rate、why）與 settle（各子條件、連續時間）；task.json 另記 k3m_settle。
# K3-M 暫停版任務節點（door_k3m_task_node.py 的版本化複本；規格 K3M_pause_integration_spec_draft1.md）：
#   策略換為 door_k3m_policy_p.K3MPPolicy（完整 SETTLE ⇒ PAUSE、暫停期間監看照舊、求解端 DONE 才 DONE、FAIL ⇒ 中止）；
#   另發 /door/pause_req（latched；策略 pause_outbox，已中止不發），訂閱 /door/pause_ack_exec（執行端，權威）、/door/pause_ack_solver（求解端）、
#   /wgmpc/pause_status（求解端監督狀態）；每輪把目前已發布目標序號交給策略（凍結目標 seq）；task.json 另記 k3m_pause。
#   實驗保護（同長窗版）：/door/lw_guard（獨立觀察器觸發）⇒ 立即結束並記錄；觀察器心跳就緒前不推進，就緒後心跳 > 1.0 s ⇒ 記 lw_observer_unhealthy、有界結束。其餘同原檔。
"""K3-M 任務節點（door_k3a_task_node.py 的版本化複本；K3-A 檔不改；相位機換成 door_k3m_policy：OPEN 起底盤可動協同開門）。
另發 /door/base_mode_req（latched；策略 outbox），訂閱 /door/base_mode_ack（執行端，權威）、/wgmpc/base_mode_ack（求解器）、/park/violation（執行端越界）；
Meas 另帶 state_step、wall、base_pose、qf（底盤＋手臂）、q_finger（兩指平均）、executor_violation；ack 的 t_recv＝收到時最新狀態的模擬時間。
trace 每列另記 k3m_mode、k3m_corr（走廊偏差）、k3m_geom（決策、各類距離與預測、計算耗時）；task.json 另存 k3m（請求、ack、到達時刻、結果）。
原說明：K3-A 任務節點（door_k2_task_node.py 的版本化複本；K2 檔不改）：把量測組成 Meas、執行 door_k3a_policy 動作、寫紀錄。
決策全部在 door_k3a_policy（已離線測試）。K3-A＝夾持 → 開到 8° → SETTLE → 保留抓持受控停止；不張爪、不退離。
話題沿用 K2（執行端與展開端訂閱 /door/k2_abort；不改凍結檔）。

輸入：/door/sync_state（29 欄）、/door/contacts（JSON）、/gripper/state（8 欄）、/handover/state、/wholebody/state、
      /wgmpc/status、/coman/applied_fail、/park/gate。
輸出：/drawer/tcp_target（4×4 列優先，求解端 --target-topic）、/gripper/cmd、/drawer/solver_start、/wgmpc/stop、
      /handover/request（停車閘門通過後送一次）、/door/k2_phase、/room/stop（結束時要求執行端受控停止並封存）。
斜坡速率以**模擬時間**差計算（RTF < 1 時不按牆鐘推進）。牆鐘上限：全趟 600 s、中止後觀察 60 s。

    python3 evaluation/door_k3m_task_node.py --out <run_dir>/task.json --run-id <RUN_ID>
"""
from __future__ import annotations

import argparse
import dataclasses
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
import door_k2_io as KIO                                                  # noqa: E402
import door_k3m_policy as P                                               # noqa: E402
import door_k3m_policy_p as PP                                            # noqa: E402  暫停版策略
import door_k3m_policy_p2 as PP2                                          # noqa: E402  p2：SETTLE 完成訊號
import door_k3m_settle_sig as SSIG                                        # noqa: E402
import door_k3ab_lw_guard as G                                            # noqa: E402  觀察器心跳監看
import door_k3m_corridor as K3MCOR                                        # noqa: E402
import door_k3m_geom_monitor as K3MGEO                                    # noqa: E402
import door_target_k3 as T3                                              # noqa: E402
from physics_contact_hold import contact_established                     # noqa: E402

ARM = [f'joint{i}' for i in range(1, 7)]


class Node2(Node):
    def __init__(self):
        super().__init__('door_k3m_task')
        lat = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.tgt_pub = self.create_publisher(Float64MultiArray, '/drawer/tcp_target', 10)
        self.grip_pub = self.create_publisher(Float64, '/gripper/cmd', 10)
        self.start_pub = self.create_publisher(Bool, '/drawer/solver_start', lat)
        self.stop_pub = self.create_publisher(String, '/wgmpc/stop', 10)
        self.ho_pub = self.create_publisher(String, '/handover/request', 10)
        self.phase_pub = self.create_publisher(String, '/door/k2_phase', 10)
        self.room_stop_pub = self.create_publisher(String, '/room/stop', 10)
        # **K2 中止閂鎖**（latched）：展開端與執行端都訂閱；送出後不收回
        self.abort_pub = self.create_publisher(String, '/door/k2_abort', lat)
        self.sync_tr = KIO.SyncTracker()
        self.settle_win = SSIG.SettleRateWindow()                      # p2
        self.ct_agg = KIO.ContactsAggregator()
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
        self.pol = None
        self.exec_violation = None
        self.mode_msgs = []
        self.mode_req_pub = self.create_publisher(String, '/door/base_mode_req', lat)
        self.create_subscription(String, '/door/base_mode_ack', self._mode_exec, lat)
        self.create_subscription(String, '/wgmpc/base_mode_ack', self._mode_solver, lat)
        self.create_subscription(String, '/park/violation', self._pviol, 10)
        self.create_subscription(Float64MultiArray, '/gripper/state', self._gs, 1)
        self.create_subscription(String, '/handover/state', self._hs, 1)
        self.create_subscription(String, '/wholebody/state', self._wb, 10)
        self.create_subscription(Float64MultiArray, '/wgmpc/status', self._ws, 1)
        self.create_subscription(String, '/coman/applied_fail', self._af, 10)
        self.create_subscription(String, '/park/gate', self._gate, 1)
        self.pause_msgs = []                                                      # 暫停版
        self.pause_req_pub = self.create_publisher(String, '/door/pause_req', lat)
        self.create_subscription(String, '/door/pause_ack_exec', self._pause_exec, lat)
        self.create_subscription(String, '/door/pause_ack_solver', self._pause_solver, lat)
        self.create_subscription(String, '/wgmpc/pause_status', self._pause_status, lat)
        self.lw_guard = self.lw_guard_wall = None                                 # 實驗保護（獨立觀察器）
        self.hbw = G.HeartbeatWatch()
        self.create_subscription(String, '/door/lw_observer/heartbeat', self._lw_hb, 10)
        self.create_subscription(String, '/door/lw_guard', self._lw_guard, lat)

    def _sync(self, m):
        d = [float(x) for x in m.data]
        if len(d) != 29:
            self.sync_tr.anomalies.append(('bad_length', time.monotonic()))
            return
        if self.sync_tr.on_msg(d[1], d[0], time.monotonic()):       # 只接受步號嚴格遞增的新狀態
            self.sync = (d[0], d[1], d[2], np.asarray(d[3:19]).reshape(4, 4), d[19:22], np.asarray(d[22:28]), d[28])
            self.settle_win.on_sync(d[0], int(d[1]), d[2], time.monotonic())   # p2：只收已被 SyncTracker 接受的新狀態

    def _ct(self, m):
        try:
            d = json.loads(m.data)
        except Exception:
            d = {}
        self.ct_agg.on_msg(d, time.monotonic())                     # 未消費的窗口全部合併（短脈衝不被覆蓋）

    def _t_now(self):
        return float('nan') if self.sync is None else float(self.sync[0])

    def _mode_exec(self, m):
        try:
            d = json.loads(m.data)
        except Exception:
            d = None
        self.mode_msgs.append({'src': 'executor', 't_recv': self._t_now(), 'wall': time.monotonic(), 'msg': d})
        if self.pol is not None:
            self.pol.on_executor_ack(d, self._t_now(), time.monotonic())

    def _mode_solver(self, m):
        try:
            d = json.loads(m.data)
        except Exception:
            d = None
        self.mode_msgs.append({'src': 'solver', 't_recv': self._t_now(), 'wall': time.monotonic(), 'msg': d})
        if self.pol is not None:
            self.pol.on_solver_ack(d, self._t_now(), time.monotonic())

    def _pviol(self, m):
        self._pviol0(m)

    def _lw_hb(self, m):
        try:
            d = json.loads(m.data)
        except Exception:
            d = None
        self.hbw.on_beat(d, time.monotonic())

    def _lw_guard(self, m):
        if self.lw_guard is None:
            self.lw_guard, self.lw_guard_wall = str(m.data), time.monotonic()

    def _pause_msg(self, m, src):                                                 # 暫停版：ack 與狀態
        try:
            d = json.loads(m.data)
        except Exception:
            d = None
        self.pause_msgs.append({'src': src, 't_recv': self._t_now(), 'wall': time.monotonic(), 'msg': d})
        if len(self.pause_msgs) > 2000:
            self.pause_msgs = self.pause_msgs[:200] + self.pause_msgs[-1000:]
        return d

    def _pause_exec(self, m):
        d = self._pause_msg(m, 'executor')
        if self.pol is not None:
            self.pol.on_pause_executor(d, self._t_now(), time.monotonic())

    def _pause_solver(self, m):
        d = self._pause_msg(m, 'solver')
        if self.pol is not None:
            self.pol.on_pause_solver(d, self._t_now(), time.monotonic())

    def _pause_status(self, m):
        try:
            d = json.loads(m.data)
        except Exception:
            d = None
        self.pause_last_status = d
        if self.pol is not None:
            self.pol.on_pause_status(d)

    def _pviol0(self, m):
        try:
            self.exec_violation = json.loads(m.data)
        except Exception:
            self.exec_violation = {'raw': str(m.data)[:200]}

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
        self.ws_t = time.monotonic()                                # 獨立牆鐘

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
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--corridor-table', default=os.path.join(HERE, 'results', 'door', 'door_k3m_corridor_table.json'))
    ap.add_argument('--geom-config', default=os.path.join(HERE, 'results', 'door', 'door_k3m_offline_geom_monitor_r3.json'))
    ap.add_argument('--wall-total-s', type=float, default=600.0)
    ap.add_argument('--wall-post-abort-s', type=float, default=60.0)
    ap.add_argument('--lw-ready-timeout-s', type=float, default=60.0)         # 等觀察器就緒（牆鐘）
    ap.add_argument('--lw-heartbeat-max-age-s', type=float, default=1.0)      # 觀察器心跳最大年齡（牆鐘）
    a = ap.parse_args()
    if os.path.exists(a.out):
        print(f'[k3a] {a.out} 已存在 ⇒ 拒絕覆寫', flush=True)
        return 2
    gb = json.load(open(a.grasp_basis))['args']
    T_HG = np.eye(4)
    T_HG[:3, :3] = np.asarray(gb['R_grasp'], float)
    T_HG[:3, 3] = [0.0, float(gb['grasp_depth_m']), 0.0]
    cfg = PP.K3MPConfig()
    corridor = K3MCOR.Corridor.from_json(json.load(open(a.corridor_table))['table'])
    _gc = json.load(open(a.geom_config))
    gcfg = K3MGEO.GeomMonitorConfig(v_floor=dict(_gc['v_floor_registered']), ds_err=dict(_gc['downsampling']['ds_err_registered']))
    pol = PP2.K3MP2Policy(cfg, T_HG, a.run_id, corridor, K3MGEO.GeometryMonitor(gcfg, K3MGEO.default_backend()))
    K = WholeBodyKinematics.from_urdf_file(a.urdf)
    rep = {'schema': 'door_k3m_task_p2/1', 'args': vars(a), 'config': dataclasses.asdict(cfg), 'geom_config': dataclasses.asdict(gcfg),
           'corridor_sha256': hashlib.sha256(open(a.corridor_table, 'rb').read()).hexdigest(),
           'geom_config_sha256': hashlib.sha256(open(a.geom_config, 'rb').read()).hexdigest(),
           'grasp_basis': {'path': os.path.relpath(a.grasp_basis, WS), 'sha256': hashlib.sha256(open(a.grasp_basis, 'rb').read()).hexdigest(),
                           'T_HG': T_HG.tolist()},
           'policy_sha256': hashlib.sha256(open(P.__file__, 'rb').read()).hexdigest(),
           'target_sha256': hashlib.sha256(open(T3.__file__, 'rb').read()).hexdigest(),
           'parent_policy_sha256': hashlib.sha256(open(P.P.__file__, 'rb').read()).hexdigest(),
           'mode_sha256': hashlib.sha256(open(P.MODE.__file__, 'rb').read()).hexdigest(),
           'corridor_module_sha256': hashlib.sha256(open(K3MCOR.__file__, 'rb').read()).hexdigest(),
           'geom_module_sha256': hashlib.sha256(open(K3MGEO.__file__, 'rb').read()).hexdigest(), 'trace': [], 'events': []}
    rep['policy_p2_sha256'] = hashlib.sha256(open(PP2.__file__, 'rb').read()).hexdigest()
    rep['settle_sig_sha256'] = hashlib.sha256(open(SSIG.__file__, 'rb').read()).hexdigest()
    _sw = SSIG.SettleRateWindow()                                          # 與節點內窗口同預設值
    rep['settle_sig_config'] = {'window_s': _sw.window_s, 'gap_max_s': _sw.gap_max_s, 'stale_wall_s': _sw.stale_wall_s}

    def save():
        rep['policy_events'] = pol.events
        rep['final_phase'], rep['abort'], rep['max_abs_q_door_rad'] = pol.phase, pol.abort, pol.max_abs_q
        rep.setdefault('n_target_published', 0)
        rep['open_entry'], rep['max_grasp_drift'] = pol.open_entry, {'pos_m': pol.max_drift[0], 'rot_rad': pol.max_drift[1]}
        W = pol.waiter
        rep['k3m'] = {'mode_state': pol.mode_state, 'request': W.request, 'exec_ack': W.exec_ack, 'solver_ack': W.solver_ack,
                      't_exec_recv': W.t_exec, 't_solver_recv': W.t_solver, 'wall_req': W.wall_req, 'wall_exec_recv': W.wall_exec,
                      'wall_solver_recv': W.wall_solver, 't_recv_semantics': '收到時最新同步狀態的來源模擬時間（非獨立量測）',
                      'result': W.result, 'n_geom_warn': pol.n_geom_warn,
                      'mode_msgs': nd.mode_msgs, 'executor_violation': nd.exec_violation}
        PW = pol.pwaiter                                                          # 暫停版
        rep['k3m_pause'] = {'pause_state': pol.pause_state, 'request': PW.request, 'exec_ack': PW.exec_ack, 'solver_ack': PW.solver_ack,
                            't_exec_recv': PW.t_exec, 't_solver_recv': PW.t_solver, 'wall_req': PW.wall_req,
                            'wall_exec_recv': PW.wall_exec, 'wall_solver_recv': PW.wall_solver, 'result': PW.result,
                            'pause_t0': pol.pause_t0, 'outcome': pol.pause_outcome,
                            'last_status': getattr(nd, 'pause_last_status', None), 'msgs': nd.pause_msgs[-300:]}
        rep['k3m_settle'] = {'resets': pol.settle_resets[-500:], 'n_resets': len(pol.settle_resets),
                             'sig_anomalies': dict(nd.settle_win.anomalies)}
        tmp = a.out + '.tmp'
        json.dump(rep, open(tmp, 'w'), ensure_ascii=False, indent=1, default=float)
        os.replace(tmp, a.out)

    rclpy.init()
    nd = Node2()
    nd.pol = pol
    w0 = time.monotonic()
    t_prev = None
    ho_sent = False
    w_abort = None
    dt_w = 1.0 / a.rate
    last_phase = None
    abort_sent = False
    sync_once = False
    last_ct = None
    n_pub = 0                              # 已發布目標的序號（判定器對帳用）
    try:
        while rclpy.ok():
            for _ in range(64):
                rclpy.spin_once(nd, timeout_sec=0.0)
            wnow = time.monotonic()
            if nd.lw_guard is not None:                                       # 實驗保護 ⇒ 提前結束（不是成功停住）
                rep['lw_guard'] = {'msg': nd.lw_guard, 'wall_recv': nd.lw_guard_wall}
                break
            nd.hbw.max_age_s = a.lw_heartbeat_max_age_s
            if not nd.hbw.ready():
                if wnow - w0 > a.lw_ready_timeout_s:
                    rep['lw_observer_not_ready'] = {'wall_waited_s': wnow - w0}
                    break
                time.sleep(dt_w)
                continue
            if not nd.hbw.healthy(wnow):
                rep['lw_observer_unhealthy'] = {'wall': wnow, 'last_beat_wall': nd.hbw.last_wall, 'last_seq': nd.hbw.last_seq,
                                                'age_s': wnow - nd.hbw.last_wall, 'phase': pol.phase}
                break
            if wnow - w0 > a.wall_total_s and pol.phase not in ('ABORTED', 'DONE'):
                rep['wall_timeout'] = True
                pol._abort('wall_total_timeout', t_prev if t_prev is not None else float('nan'))
                w_abort = w_abort or wnow
            if pol.phase == 'ABORTED' and not abort_sent:
                nd.abort_pub.publish(String(data=json.dumps({'why': pol.abort, 't': pol.t_abort}, default=float)))
                nd.stop_pub.publish(String(data=f'door_k3a 中止：{pol.abort}'))
                abort_sent = True
                w_abort = w_abort or wnow
            if nd.sync is None:
                # 從未收到同步狀態：只能等牆鐘上限；中止後的觀察亦受牆鐘上限
                if w_abort is not None and wnow - w_abort > min(a.wall_post_abort_s, 5.0):
                    rep['ended_without_sync'] = True
                    break
                time.sleep(dt_w)
                continue
            sync_once = True
            t, step, q, H, pose, q_arm, qdot = nd.sync
            _ctn = nd.ct_agg.consume()
            if _ctn is not None:
                last_ct = _ctn                     # 新窗口（已合併所有未消費窗口）
            ct = last_ct                           # 無新窗口 ⇒ 沿用上一份；是否過期由獨立牆鐘年齡判斷（contacts_age）
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
            st_age = nd.sync_tr.age(wnow)
            if not nd.sync_tr.ok():
                st_age = float('inf')                                 # 倒退／非有限 ⇒ 監看失效
            m = P.Meas(t=t, owner=(nd.hs or {}).get('owner'),
                       unfold_done=(nd.wb or {}).get('phase') in ('UNFOLD_DONE', 'RELEASED'),
                       solver_up=nd.n_ws > 0, solver_failed=nd.fail,
                       solver_status_age=(None if nd.ws_t is None else wnow - nd.ws_t), state_age=st_age,
                       contacts_age=nd.ct_agg.age(wnow),
                       q_door=q, qdot_door=qdot, H=H, T_tcp=T_tcp,
                       finger_bar_min=None if ct is None else tuple(ct['finger_bar_min_n']),
                       finger_bar_max=None if ct is None else tuple(ct['finger_bar_max_n']),
                       unexpected_max=None if ct is None else ct['unexpected_max_n'], unexpected_who=None if ct is None else ct['unexpected_who'],
                       wall_max=None if ct is None else ct['wall_max_n'], wrist_max=None if ct is None else ct['wrist_max_n'],
                       contacts_invalid=None if ct is None else ct['n_invalid'], contact_evidence_ok=bool(ev),
                       finger_q=None if gs is None else (gs[1], gs[2]), base_drift_m=bd, base_drift_yaw=byaw)
            m.state_step, m.wall = int(step), wnow                   # K3-M：模式請求、走廊與幾何監看所需
            m.base_pose = tuple(float(x) for x in pose)
            m.qf = np.array(list(pose) + list(q_arm), float)
            m.q_finger = None if gs is None else 0.5 * (float(gs[1]) + float(gs[2]))
            m.executor_violation = nd.exec_violation
            m.settle_sig = nd.settle_win.value(int(step), wnow)      # p2：SETTLE 完成訊號（只在 SETTLE 使用）
            dt = 0.0 if t_prev is None else max(0.0, t - t_prev)
            t_prev = t
            pol.pub_seq = n_pub                                # 暫停版：凍結目標序號＝目前已發布序號
            act = pol.step(m, dt)
            for _rq in pol.take_outbox():                      # 已中止 ⇒ take_outbox 清除、不發布
                nd.mode_req_pub.publish(String(data=json.dumps(_rq, default=float)))
                rep['events'].append({'t': t, 'k3m_mode_request_published': _rq})
            for _rq in pol.take_pause_outbox():                # 暫停版：已中止 ⇒ 清除、不發布
                nd.pause_req_pub.publish(String(data=json.dumps(_rq, default=float)))
                rep['events'].append({'t': t, 'k3m_pause_request_published': _rq})
            _g = pol.geom_last
            if act['request_handover'] and not ho_sent and gate.get('passed') and gate.get('ok_now'):
                nd.ho_pub.publish(String(data=json.dumps({'to': 'wholebody', 'at_step': int(step) + 5, 'window_steps': 50})))
                ho_sent = True
                rep['events'].append({'t': t, 'handover_requested_at_step': int(step) + 5, 'gate': gate})
            if pol.phase == 'ABORTED' and not abort_sent:
                nd.abort_pub.publish(String(data=json.dumps({'why': pol.abort, 't': pol.t_abort}, default=float)))
                abort_sent = True
                w_abort = w_abort or wnow
            pub_seq = None
            if act['target'] is not None and pol.phase != 'ABORTED':
                nd.tgt_pub.publish(Float64MultiArray(data=[float(x) for x in np.asarray(act['target']).reshape(-1)]))
                n_pub += 1
                pub_seq = n_pub
                rep['n_target_published'] = n_pub
            nd.grip_pub.publish(Float64(data=float(act['grip'])))
            if act['solver_start'] and pol.phase != 'ABORTED':
                nd.start_pub.publish(Bool(data=True))
            if act['stop_solver']:
                nd.stop_pub.publish(String(data=f'door_k3a {pol.phase}：{pol.abort or "done"}'))
            if pol.phase != last_phase:
                nd.phase_pub.publish(String(data=pol.phase))
                print(f'[k3a] 相位 → {pol.phase} @ sim {t:.2f}' + (f'（中止：{pol.abort}）' if pol.abort else ''), flush=True)
                last_phase = pol.phase
            if pol.H_frozen is not None and 'H_frozen' not in rep:
                rep['H_frozen'] = pol.H_frozen.tolist()
                rep['H_frozen_t'] = t
            if pol.G_T_H is not None and 'G_T_H' not in rep:
                rep['G_T_H'] = pol.G_T_H.tolist()
                rep['G_T_H_t'] = t
            rep['trace'].append({'t': t, 'step': step, 'phase': pol.phase, 'q_door': q, 'qdot_door': qdot, 's': pol.s, 'grip': act['grip'],
                                 'target': (None if act['target'] is None or pol.phase == 'ABORTED' else np.asarray(act['target']).reshape(-1).tolist()),
                                 'err_p': (None if pol.last_target is None else float(np.linalg.norm(T_tcp[:3, 3] - pol.last_target[:3, 3]))),
                                 'pub_seq': pub_seq, 'q_ref': pol.q_ref, 'ref': (None if pol.ref_info is None else {k: pol.ref_info.get(k) for k in ('q_ref_prev', 'rate', 'lo', 'hi', 'init')}),
                                 'base_drift_m': bd, 'base_drift_yaw': byaw, 'contacts': ct, 'evidence': bool(ev),
                                 'state_age_wall': st_age, 'contacts_age_wall': nd.ct_agg.age(wnow),
                                 'state_step': int(step), 'k3m_mode': pol.mode_state,
                                 'settle_sig': m.settle_sig, 'settle': None if pol.settle_last is None or pol.phase not in ('SETTLE', 'PAUSE') else
                                 {k: v for k, v in pol.settle_last.items() if k != 'sig'},
                                 'k3m_corr': None if pol.corr_last is None else [float(v) for v in pol.corr_last],
                                 'k3m_geom': None if _g is None else {'dec': _g[0], 'why': _g[1].get('why'), 'compute_s': _g[1].get('compute_s'),
                                                                     'cls': {k: [v['d'], v['d_pred']] for k, v in (_g[1].get('classes') or {}).items()}}})
            if len(rep['trace']) % 100 == 0:
                save()
            if act['finish'] or (w_abort is not None and wnow - w_abort > a.wall_post_abort_s):
                if w_abort is not None and not act['finish']:
                    rep['post_abort_wall_cap_hit'] = True
                break
            time.sleep(dt_w)
    finally:
        rep['wall_s'] = round(time.monotonic() - w0, 2)
        rep['lw_heartbeat'] = {'ready_wall': nd.hbw.ready_wall, 'last_seq': nd.hbw.last_seq, 'last_wall': nd.hbw.last_wall, 'bad': nd.hbw.bad[:20]}
        rep['sync_anomalies'] = {'n_dup': nd.sync_tr.n_dup, 'n_back': nd.sync_tr.n_back, 'list': nd.sync_tr.anomalies[:20]}
        rep['received_any_sync'] = sync_once
        save()
        try:
            nd.stop_pub.publish(String(data='door_k3a 結束'))
            nd.room_stop_pub.publish(String(data=f'door_k3a 結束：{pol.phase} {pol.abort or ""}'))
            for _ in range(10):
                rclpy.spin_once(nd, timeout_sec=0.05)
        except Exception:
            pass
        print(f'[k3a] 結束：{pol.phase}' + (f'（中止 {pol.abort}）' if pol.abort else ''), flush=True)
        nd.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0 if pol.phase == 'DONE' else 1


if __name__ == '__main__':
    sys.exit(main())
