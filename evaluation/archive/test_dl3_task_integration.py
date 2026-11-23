#!/usr/bin/env python3
"""DL3 任務節點（dl3_drawer_task_node.py --handle-source vision）整合測試：不開 Isaac，以合成 /drawer/sync_state 餵狀態
（框架同 test_park_abort_integration.py）。真值把手（sync_state 內）刻意與視覺估計／地圖先驗不同，用來檢查目標不讀真值。

A 無 /dl3/handle_est ⇒ ALIGN 進入時鎖定失敗 ⇒ A0 受控停止：abort＝vision_latch_failed、發 /wgmpc/stop、從未 solver_start、**從未發任何目標**。
B 進 ALIGN 前送 4 筆一致估計 ⇒ 鎖定＋GC2＋固定底盤 IK 前檢通過；**鎖定前零目標、第一個目標先於 solver_start 且即鎖定目標**；
  之後目標全等於鎖定目標（不是真值）；手臂角設為鎖定目標的 IK 解 ⇒ pre 到位保持 0.5 s ⇒ PREGRASP_COMPLETE、/wgmpc/stop、
  停止後紀錄完整、結束碼 0、未轉 approach。
C 缺 --stop-at-pregrasp ⇒ 拒絕啟動（結束碼 3）。
D 同 B，但 PREGRASP_COMPLETE 後停止送 sync_state（時鐘停更）⇒ 在牆鐘上限內結束、標 post_stop_record_incomplete、結果仍是 PREGRASP_COMPLETE。

    source /opt/ros/jazzy/setup.bash; python3 evaluation/test_dl3_task_integration.py
"""
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))
os.environ['ROS_DOMAIN_ID'] = os.environ.get('DL3_TEST_DOMAIN', '123')

import rclpy                                                            # noqa: E402
from rclpy.node import Node                                             # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile                      # noqa: E402
from std_msgs.msg import Bool, Float64MultiArray, String                # noqa: E402

import dl3_task_vision as DV                                            # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402

PARK = (-0.136412, 0.560, 1.297349)
STOW = [0.0, 0.0, 0.0, 0.0, -1.5707963, 0.0]
QG = [-0.0321, 0.9177, 1.4853, -0.3580, -1.0325, -1.3813]
MAP = (0.0, 1.165, 0.55)                      # 地圖先驗
TRUTH = (0.0, 1.195, 0.55)                    # sync_state 內的「真值」：與地圖／估計差 30 mm
EST = (0.002, 1.167, 0.551)                   # 視覺估計
fails = []


def check(name, cond, detail=''):
    print(('PASS ' if cond else 'FAIL ') + name + ('' if cond else f'  {detail}'))
    if not cond:
        fails.append(name)


class Probe(Node):
    def __init__(self):
        super().__init__('dl3_task_probe')
        self.sync = self.create_publisher(Float64MultiArray, '/drawer/sync_state', 10)
        self.est = self.create_publisher(String, '/dl3/handle_est', 10)
        self.ws = self.create_publisher(Float64MultiArray, '/wgmpc/status', 10)
        lat = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.tgt, self.stop, self.start = [], [], []
        self.create_subscription(Float64MultiArray, '/drawer/tcp_target',
                                 lambda m: self.tgt.append((time.monotonic(), np.array(m.data).reshape(4, 4))), 10)
        self.create_subscription(String, '/wgmpc/stop', lambda m: self.stop.append((time.monotonic(), m.data)), 10)
        self.create_subscription(Bool, '/drawer/solver_start',
                                 lambda m: self.start.append((time.monotonic(), m.data)), lat)


def H_of(p):
    H = np.eye(4)
    H[:3, 3] = p
    return H


def run(extra, n_est, q_after_align=None, dur=25.0, stall_after_stop=False):
    out = os.path.join(tempfile.mkdtemp(prefix='dl3_task_'), 'task.json')
    cmd = [sys.executable, '-u', os.path.join(HERE, 'dl3_drawer_task_node.py'), '--handle-source', 'vision',
           '--grasp-depth-m', '0.0068', '--pre-settle-s', '0.5',
           '--post-stop-record-s', '1.0', '--open-m', '0.200', '--out', out] + extra
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    nd = Probe()
    t_sim, t0, sent_est, t_align_feed = 10.0, time.monotonic(), 0, None
    Hs = H_of(TRUTH)
    while time.monotonic() - t0 < dur:
        el = time.monotonic() - t0
        # 0–3 s 收攏（先送估計），3 s 起手臂到 QG ⇒ ALIGN；之後若給 q_after_align 再換成鎖定目標的 IK 解
        if el >= 1.0 and sent_est < n_est:
            T = H_of(EST)
            nd.est.publish(String(data=json.dumps({'n': sent_est + 1, 't_obs': t_sim - 0.5, 'est_ok': True, 'est_why': None,
                                                   'T_WH': T.tolist(), 'geometry_contract': 'unchecked'})))
            sent_est += 1
        q = STOW if el < 3.0 else QG
        if el >= 3.0 and t_align_feed is None:
            t_align_feed = time.monotonic()
        if q_after_align is not None and nd.start and any(d for _, d in nd.start):
            q = q_after_align
            m = Float64MultiArray()
            m.data = [t_sim, 0.0, 0.0, 0.0, 0.0]
            nd.ws.publish(m)
        if not (stall_after_stop and nd.stop):
            m = Float64MultiArray()
            m.data = [t_sim, 0.0, 0.0] + Hs.reshape(-1).tolist() + list(PARK) + list(q)
            nd.sync.publish(m)
        t_sim += 0.02
        rclpy.spin_once(nd, timeout_sec=0.02)
        if proc.poll() is not None:
            break
    for _ in range(20):
        rclpy.spin_once(nd, timeout_sec=0.02)
    if proc.poll() is None:
        proc.kill()
    log = proc.stdout.read()
    rep = json.load(open(out)) if os.path.exists(out) else {}
    res = (proc.returncode, rep, nd.tgt, nd.stop, nd.start, log, time.monotonic())
    nd.destroy_node()
    return res


rclpy.init()
K = WholeBodyKinematics.from_urdf_file(os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
Rg = K.fk(np.array(list(PARK) + QG, float), 'link_tcp')[:3, :3]
T_HG, a_H = DV.grasp_params(Rg, 0.0068)
T_lock = DV.target_from_H(H_of(EST), 0.03, T_HG, a_H)
T_truth = DV.target_from_H(H_of(TRUTH), 0.03, T_HG, a_H)

# ---- A：沒有視覺估計 ----
rc, rep, tgt, stop, start, log, _ = run(['--stop-at-pregrasp'], n_est=0, dur=12.0)
check('A_abort_latch_failed', str(rep.get('abort', '')).startswith('vision_latch_failed:too_few_estimates'), (rc, rep.get('abort'), log[-600:]))
check('A_stop_sent', len(stop) >= 1)
check('A_never_solver_start', not any(d for _, d in start), start)
t_ab = stop[0][0] if stop else None
check('A_never_any_target', not tgt, len(tgt))
check('A_exit_nonzero', rc == 1, rc)

# ---- B：一致估計 ⇒ 鎖定 ⇒ 到位 ⇒ PREGRASP_COMPLETE ----
ik = DV.ik_precheck(K, PARK, QG, T_lock)
rc, rep, tgt, stop, start, log, _ = run(['--stop-at-pregrasp'], n_est=4, q_after_align=ik['q_arm'], dur=30.0)
lc = (rep.get('vision') or {}).get('latch_check') or {}
check('B_latch_and_prechecks_ok', lc.get('ok') and lc.get('gc2', {}).get('ok') and lc.get('ik', {}).get('ok'), (lc.get('why'), log[-800:]))
check('B_gc2_on_estimated_object', lc.get('gc2', {}).get('object') == 'estimated'
      and abs(np.array(lc['gc2']['T_WO_est'])[1, 3] - (EST[1] + 0.285)) < 1e-9 if lc.get('gc2') else False)
check('B_solver_started', any(d for _, d in start))
ts = start[0][0] if start else 1e18
check('B_first_target_before_solver_start_and_is_lock', tgt and tgt[0][0] <= ts and np.abs(tgt[0][1] - T_lock).max() < 1e-9,
      (tgt[0][0] - ts) if tgt else None)
check('B_all_targets_are_lock_derived', tgt and all(np.abs(T - T_lock).max() < 1e-9 for _, T in tgt), len(tgt))
post = [T for t, T in tgt if t > ts + 0.2]
check('B_targets_after_start_equal_lock_not_truth', post and all(np.abs(T - T_lock).max() < 1e-9 for T in post)
      and np.abs(T_lock[:3, 3] - T_truth[:3, 3]).max() > 0.02, len(post))
check('B_result_pregrasp_complete', rep.get('result') == 'PREGRASP_COMPLETE' and rc == 0, (rc, rep.get('result'), rep.get('abort'), log[-800:]))
pc = rep.get('pregrasp_complete') or {}
check('B_criteria_recorded', pc.get('pos_err_m', 1) <= 0.005 and pc.get('rot_err_rad', 1) <= 0.02 and pc.get('held_s', 0) >= 0.5, pc)
check('B_stop_sent_and_no_approach', len(stop) >= 1 and not any(e.get('align') == 'approach' for e in rep.get('events', [])))
check('B_post_stop_trace_complete', len(rep.get('post_stop_trace', [])) >= 3 and rep.get('post_stop_record_complete') is True
      and 'post_stop_record_incomplete' not in rep)
check('B_fk_offset_marked_truth_diagnostic', 'truth_diagnostic_only' in str(rep.get('tcp_offset_fk_design_m_note')))

# ---- D：PREGRASP_COMPLETE 後時鐘停更 ⇒ 牆鐘上限內結束、標不完整 ----
rc, rep, tgt, stop, start, log, t_end = run(['--stop-at-pregrasp', '--post-stop-wall-cap-s', '3.0'], n_est=4,
                                            q_after_align=ik['q_arm'], dur=40.0, stall_after_stop=True)
check('D_result_still_pregrasp_complete', rep.get('result') == 'PREGRASP_COMPLETE' and rc == 0, (rc, rep.get('result'), log[-500:]))
check('D_incomplete_flagged', 'post_stop_record_incomplete' in rep and rep.get('post_stop_record_complete') is False, rep.get('post_stop_record_incomplete'))
check('D_exit_within_wall_cap', stop and t_end - stop[0][0] < 3.0 + 4.0, (t_end - stop[0][0]) if stop else None)

# ---- C：缺 --stop-at-pregrasp ⇒ 拒絕啟動 ----
r = subprocess.run([sys.executable, os.path.join(HERE, 'dl3_drawer_task_node.py'), '--handle-source', 'vision'],
                   capture_output=True, text=True, timeout=60)
check('C_refuse_without_stop_at_pregrasp', r.returncode == 3, r.stdout[-300:])

rclpy.shutdown()
print(f'{len(fails)} 失敗')
sys.exit(1 if fails else 0)
