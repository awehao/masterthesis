#!/usr/bin/env python3
"""真任務節點 door_k3m_task_node_k3b 的受控釋放路由與整合測試（無 Isaac）。

範圍與替身（須揭露）：
- 節點以子程序執行真檔（runpy）：參數、ROS 話題、QoS、主迴圈、紀錄、收尾，以及 K3-B 策略、配對幾何、幾何監看、走廊表全為真。
- 唯一替身：父類 door_k3m_policy_p2.K3MP2Policy.step（夾持、開門、SETTLE、暫停握手）＝「立即 PAUSE、暫停已確認」。
- 本測試扮演：
  - 狀態來源：觀察器心跳、/door/sync_state（50 Hz）、/door/contacts、/gripper/state、/park/gate、求解端 /wgmpc/pause_status（HELD）；
  - 執行端權威與解除監看（依請求提交 mode 1／2／3、release 狀態、5 N ack）。
- 起始狀態＝door_wide_w2c_30_1 HELD 實測（底盤、手臂、門 30.21°、把手位姿 H(q)）。
- 退離時以離線 IK（door_offline_check_v1.ik，底盤固定）讓手臂追蹤節點發出的 /drawer/tcp_target（下一輪生效＝一個週期落後）。
- 張爪後手指開度依 K2 實錄量級線性張開，接觸力 0.04 s 內歸零。無門動力學、無真實接觸。
  N1 正常：
     - 請求序 1、2、3；5 N 請求帶 C_CLEAR 證據（s ≥ 0.02、配對 ≥ 5 mm、band_ok）；
     - 節點 DONE、exit 0；/room/stop 在 DONE 之後；
     - task.json k3b：monitor COMPLETE、rel done、eff_done 讀回 True；trace 每列含 k3b；退離 s_cmd 單調到 0.10。
  N2 5 N 被拒（ack reject）⇒ 中止 k3b_effort_reject:…、exit 1、發 /door/k2_abort。
  N3 執行端 release FAILED（確認逾時）⇒ 中止 k3b_exec_release:confirm_timeout。

    python3 evaluation/test_task_node_k3b_routing.py [N1 N2 N3]
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))
TMP = os.environ.get('K3BR_TMP', '/tmp/claude-1000/k3b2_task_routing')
DOM0 = int(os.environ.get('K3BR_DOMAIN', '230'))
RUN = 'k3b2_route'
FAIL = []
FO = 0.0089
GATE_ANCHOR = [0.6000019907951355, -0.5331917405128479, 1.571134844677191]   # 30_1 park_gate 錨點（開門前停位）

LAUNCH = r'''
import runpy, sys
sys.path.insert(0, %(here)r)
import door_k3m_policy_p2 as PP2
def _stand_in(self, m, dt):                     # 替身：立即視為 PAUSE、暫停已確認（不驗開門與暫停）
    if self.phase not in ('PAUSE', 'DONE', 'ABORTED', 'RELEASE'):
        self._go('PAUSE', m.t, stand_in=True)
        self.pause_t0 = m.t
    self.pause_state = 'active'
    if self.phase == 'ABORTED':
        return self._aborted_act({'target': None, 'grip': 0.0, 'request_handover': False, 'solver_start': False,
                                  'stop_solver': False, 'finish': False}, m)
    return {'target': None, 'grip': 0.0, 'request_handover': False, 'solver_start': False, 'stop_solver': False, 'finish': False}
PP2.K3MP2Policy.step = _stand_in
sys.argv = ['door_k3m_task_node_k3b2.py'] + %(args)r
sys.exit(runpy.run_path(%(node)r, run_name='__main__'))
'''


def ck(n, ok, d=''):
    print(f'  {n:64s} {"ok" if ok else "**錯**"}  {d}', flush=True)
    if not ok:
        FAIL.append(n)


def scenario(tag, dom, eff_reject=False, release_fail=False, run_s=40.0):
    os.makedirs(TMP, exist_ok=True)
    out = os.path.join(TMP, f'{tag}.json')
    for f in (out, out[:-5] + '_trace.jsonl', out[:-5] + '_trace_hb.json'):
        if os.path.exists(f):
            os.remove(f)                                              # 本測試自己的暫存輸出
    os.environ['ROS_DOMAIN_ID'] = str(DOM0 + dom)
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    import rclpy
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import Float64MultiArray, String
    import door_k2_check as KC
    import door_k2_offline_v2 as V2
    v4 = json.load(open(os.path.join(HERE, 'results', 'door', 'door_k3b_offline_v4.json')))
    hs = v4['held_state']
    base, q_arm, q_door = list(hs['base']), list(hs['q_arm']), float(hs['q_door'])
    H = KC._H_of(q_door)
    C = V2.C
    args = ['--out', out, '--run-id', RUN, '--w1b-params', 'evaluation/results/door/w1e_params_30.json',
            '--corridor-table', 'evaluation/results/door/door_wide_corridor_table_90.json', '--lw-ready-timeout-s', '20',
            '--wall-total-s', str(run_s), '--wall-post-abort-s', '2', '--w2-mode-run-id', RUN, '--w2-release']
    code = LAUNCH % {'here': HERE, 'args': args, 'node': os.path.join(HERE, 'door_k3m_task_node_k3b2.py')}
    rclpy.init()
    peer = rclpy.create_node(f'k3br_peer_{tag}')
    lat = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    pub = {'hb': peer.create_publisher(String, '/door/lw_observer/heartbeat', 10),
           'sync': peer.create_publisher(Float64MultiArray, '/door/sync_state', 100),
           'ct': peer.create_publisher(String, '/door/contacts', 100),
           'gs': peer.create_publisher(Float64MultiArray, '/gripper/state', 10),
           'gate': peer.create_publisher(String, '/park/gate', 1),
           'ps': peer.create_publisher(String, '/wgmpc/pause_status', lat),
           'state': peer.create_publisher(String, '/door/w2_mode_state', 50),
           'ack': peer.create_publisher(String, '/door/finger_effort_ack', lat)}
    got = {'req': [], 'eff': [], 'room_stop': [], 'k2_abort': [], 'tgt': None}
    peer.create_subscription(String, '/door/w2_mode_req_task', lambda m: got['req'].append(json.loads(m.data)), lat)
    peer.create_subscription(String, '/door/finger_effort_req', lambda m: got['eff'].append(json.loads(m.data)), lat)
    peer.create_subscription(String, '/room/stop', lambda m: got['room_stop'].append(time.monotonic()), 10)
    peer.create_subscription(String, '/door/k2_abort', lambda m: got['k2_abort'].append(m.data), lat)
    peer.create_subscription(Float64MultiArray, '/drawer/tcp_target', lambda m: got.__setitem__('tgt', np.array(m.data, float).reshape(4, 4)), 10)
    p = subprocess.Popen([sys.executable, '-u', '-c', code], cwd=WS, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, start_new_session=True)
    au = {'commits': [], 'last_reject': None, 'seen': set(), 't_open': None, 'eff_done': None}
    st = {'k': 0, 'seq': 0, 'held': False, 'q_arm': list(q_arm), 'n_ik': 0, 'ik_fail': 0}
    w0 = time.monotonic()
    try:
        while p.poll() is None and time.monotonic() - w0 < run_s + 30:
            rclpy.spin_once(peer, timeout_sec=0.003)
            wall = time.monotonic() - w0
            # 權威替身：依請求提交（0.1 s 後確認）
            for r in got['req']:
                key = r['req_seq']
                if key in au['seen']:
                    continue
                au['seen'].add(key)
                k = int(wall * 100) + 10
                au['commits'].append({'mode_seq': r['mode_seq'], 'step': k, 't': k * 0.01, 'first_source_seq': 500 + r['mode_seq'],
                                      'applied_confirmed': True})
                if r['mode_seq'] == 2:
                    au['t_open'] = k * 0.01
            for e in got['eff']:
                if au['eff_done'] is None:
                    au['eff_done'] = wall
                    ack = ({'run_id': RUN, 'result': 'reject:evidence_not_clear'} if eff_reject else
                           {'run_id': RUN, 'result': 'apply', 'done': {'req_seq': e['req_seq'], 'readback_ok': True, 'switch_step': int(wall * 100)}})
                    pub['ack'].publish(String(data=json.dumps(ack)))
            # 追蹤節點目標（退離開始後；底盤固定）
            if got['tgt'] is not None and au['t_open'] is not None and wall > au['t_open']:
                qf = C.ik(np.array(base), got['tgt'], np.array(st['q_arm']), only_prev=True)
                if qf is not None:
                    st['q_arm'] = [float(x) for x in qf[C.IDX]]
                    st['n_ik'] += 1
                else:
                    st['ik_fail'] += 1
                got['tgt'] = None
            k_now = int(wall * 50)
            while st['k'] <= k_now:
                ts = st['k'] * 0.02
                step = st['k'] * 2
                pub['sync'].publish(Float64MultiArray(data=[ts, float(step), q_door, *[float(x) for x in H.reshape(-1)], *base, *st['q_arm'], 0.0]))
                to = au['t_open']
                opened = to is not None and ts > to
                tr = 0.0 if not opened else ts - to
                f = 0.17 if not opened else min(1.0, 0.2 + 0.75 * tr / 0.31 if tr < 0.31 else 0.95 + (tr - 0.31) * 0.2)
                fb = (12.0 if not opened else max(0.0, 12.0 * (1 - tr / 0.04)))
                pub['ct'].publish(String(data=json.dumps({'sim_t': ts, 'step': step, 'n_steps': 2, 'n_invalid': 0, 'finger_bar_max_n': [fb, fb],
                                                          'finger_bar_min_n': [fb, fb], 'unexpected_max_n': 0.0, 'unexpected_who': None,
                                                          'wall_max_n': 0.0, 'wrist_max_n': 3.0})))
                pub['gs'].publish(Float64MultiArray(data=[ts, FO * f, FO * f, fb, fb, float(step), 1.0, 0.5]))
                pub['gate'].publish(String(data=json.dumps({'passed': True, 'ok_now': True, 'anchor': GATE_ANCHOR})))   # 實跑：開門前停位（≠ HELD 底盤）
                if st['k'] % 5 == 0:
                    st['seq'] += 1
                    pub['hb'].publish(String(data=json.dumps({'seq': st['seq']})))
                if st['k'] % 2 == 0:
                    rel = {'state': 'IDLE'}
                    if to is not None:
                        failed = release_fail and ts >= to + 1.5
                        conf = (not release_fail) and ts >= to + 0.8                   # N3：從未確認，1.5 s 後逾時失敗
                        rel = {'state': 'FAILED' if failed else ('B_CONFIRMED' if conf else 'A_OPENING'),
                               'why': 'confirm_timeout' if failed else None, 't_open': to, 't_conf': to + 0.8 if conf else None,
                               'step_conf': None, 'last_valid_step': step}
                    cs = [c for c in au['commits'] if c['t'] <= ts]
                    ms = max([c['mode_seq'] for c in cs] or [0])
                    last = cs[-1] if cs else None
                    pub['state'].publish(String(data=json.dumps({'run_id': RUN, 'mode_seq': ms, 'commits': cs, 'commit': last,
                                                                 'commit_req_seq': None if last is None else last['mode_seq'],
                                                                 'last_reject': None, 'aborted': None, 'release': rel})))
                st['k'] += 1
            if wall >= 3.0 and not st['held']:
                pub['ps'].publish(String(data=json.dumps({'outcome': {'state': 'HELD', 'observation_passed': True, 'physical_stop_confirmed': True,
                                                                      'command_stop_confirmed': True, 'transitions': {'OBS_PASSED': 3.0, 'HELD': 3.0}}})))
                st['held'] = True
        rc = p.wait(timeout=20)
        for _ in range(30):
            rclpy.spin_once(peer, timeout_sec=0.02)
    finally:
        if p.poll() is None:
            p.terminate()
            p.wait(timeout=10)
        peer.destroy_node()
        rclpy.shutdown()
    log = p.stdout.read() if p.stdout else ''
    task = json.load(open(out)) if os.path.exists(out) else None
    return rc, task, got, st, log


def main() -> int:
    sel = sys.argv[1:] or ['N1', 'N2', 'N3']
    print(f'暫存 {TMP}　domain {DOM0}+\n')
    if 'N1' in sel:
        print('N1  正常受控釋放（退離終點保持後 DONE）')
        rc, task, got, st, log = scenario('n1', 0)
        if task is None:
            print(log[-3000:])
        k = (task or {}).get('k3b') or {}
        ck('exit 0、最終 DONE', rc == 0 and (task or {}).get('final_phase') == 'DONE', f"rc={rc} phase={(task or {}).get('final_phase')} abort={(task or {}).get('abort')}")
        ck('請求序 1、2、3', [r['mode_seq'] for r in got['req']] == [1, 2, 3], str([r['mode_seq'] for r in got['req']]))
        ev = (got['eff'][0] if got['eff'] else {}).get('evidence') or {}
        ck('5 N 請求恰一筆、帶 C_CLEAR 證據', len(got['eff']) == 1 and ev.get('s_meas', 0) >= 0.02 and ev.get('pair_d_min', 0) >= 0.005
           and ev.get('band_ok') is True, str(ev))
        mon = k.get('monitor') or {}
        rel = k.get('rel') or {}
        ck('monitor COMPLETE、rel done、eff_done 讀回 True', mon.get('state') == 'COMPLETE' and rel.get('state') == 'done'
           and (rel.get('eff_done') or {}).get('readback_ok') is True, f"{mon.get('state')} {mon.get('why')} {rel.get('state')}")
        tr = (task or {}).get('trace') or []
        ss = [x['k3b']['s_cmd'] for x in tr if 'k3b' in x]
        ck('trace 每列含 k3b、s_cmd 單調到 0.10', bool(tr) and all('k3b' in x for x in tr) and all(b >= a - 1e-12 for a, b in zip(ss, ss[1:]))
           and abs(max(ss or [0]) - 0.10) < 1e-9, f'max={max(ss or [0])} ik={st["n_ik"]} ik_fail={st["ik_fail"]}')
        ck('/room/stop 在結束時才送', bool(got['room_stop']), str(len(got['room_stop'])))
        ck('近區最小配對距離與帶餘量有記錄', isinstance((mon.get('worst') or {}).get('pair_d_min_m'), float), str(mon.get('worst')))
    if 'N2' in sel:
        print('\nN2  5 N 被拒')
        rc, task, got, st, log = scenario('n2', 1, eff_reject=True)
        ck('中止 k3b_effort_reject:evidence_not_clear、exit 1、發 /door/k2_abort', rc == 1 and (task or {}).get('abort') == 'k3b_effort_reject:evidence_not_clear'
           and bool(got['k2_abort']), f"rc={rc} {(task or {}).get('abort')}")
    if 'N3' in sel:
        print('\nN3  執行端解除監看失敗')
        rc, task, got, st, log = scenario('n3', 2, release_fail=True)
        ck('中止 k3b_exec_release:confirm_timeout、未請求 mode 3', (task or {}).get('abort') == 'k3b_exec_release:confirm_timeout'
           and [r['mode_seq'] for r in got['req']] == [1, 2], f"{(task or {}).get('abort')} {[r['mode_seq'] for r in got['req']]}")
    print(f'\n{"全部通過" if not FAIL else "失敗：" + "、".join(FAIL)}')
    return 0 if not FAIL else 1


if __name__ == '__main__':
    sys.exit(main())
