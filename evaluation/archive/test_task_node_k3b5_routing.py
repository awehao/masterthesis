#!/usr/bin/env python3
"""真任務節點 k3b5 路由測試（test_task_node_k3b4_routing.py〔凍結 k3b r4〕的版本化複本；差異＝節點改 door_k3m_task_node_k3b5（核心 release5）、
替身每輪處理完積壓回呼、IK 只在成功後記下目標，見下）。以下為原說明：
真任務節點 door_k3m_task_node_k3b4（--w2-unload，策略 r4）的路由與整合測試（無 Isaac）。test_task_node_k3b3_routing.py 的版本化複本。
替身與範圍同 k3b3 路由測試（唯一策略替身：父類 PAUSE 立即成立）；差異：
- 權威狀態另帶 t、unload（state_brief 形：mode 1 提交起 RAMPING → t1＋1.0 SETTLING〔t_settling〕→ t1＋1.7 UNLOADED；mode 2 提交後 HANDED_TO_RELEASE）、
  finger_effort（最近定期讀回：每 1 s；卸載前 20、斜坡中＝斜坡值、t_settling 後 5）；
- /wgmpc/pause_status 每 0.1 s 發布（sim_t 與子狀態：held → unload → t1＋1.75 unloaded → release_open → t_open＋1.2 release_settled）。
  R1 正常：請求序 1、2、3；mode 2 請求晚於兩端完成；不送 5 N 請求；eff_verify 讀回 5；mode 3 於 release_settled 後；DONE、exit 0。
  R2 執行端 unload FAILED（contact_lost）⇒ 中止 k3b_unload:unload_failed:contact_lost、未請求 mode 2。
  R3 C_CLEAR 時讀回仍 20 ⇒ 中止 k3b_effort_verify:readback_not_5。
  R4 求解端未曾 release_settled ⇒ 未請求 mode 3、中止（近區退出期限）。
離線 IK 追蹤：目標未變時不重解（k3b3 路由測試每次重解；IK 接受 1 mm 位置誤差，重解會在固定目標下產生 ≤ 1 mm 假抖動，
與 core r4 三維收斂門檻 1 mm 同量級，造成 RETREAT_DONE 間歇延後或不成立——替身假象，k3b3 測試檔已凍結不改）。
替身每輪處理完積壓回呼（單次 spin_once 在 IK 慢時會飢餓、間歇漏收 /drawer/tcp_target，手臂停在中途 ⇒ retreat_timeout；替身假象）。
目標未變不重解只在前一次 IK 成功後適用（k3b4 版在 IK 失敗時也記下目標，終點目標不再重試 ⇒ 手臂停在中途；替身假象）。

    python3 evaluation/test_task_node_k3b5_routing.py [R1 R2 R3 R4]
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
TMP = os.environ.get('K3BR_TMP', '/tmp/claude-1000/k3b5_task_routing')
DOM0 = int(os.environ.get('K3BR_DOMAIN', '100'))
RUN = 'k3b4_route'
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
sys.argv = ['door_k3m_task_node_k3b5.py'] + %(args)r
sys.exit(runpy.run_path(%(node)r, run_name='__main__'))
'''


def ck(n, ok, d=''):
    print(f'  {n:64s} {"ok" if ok else "**錯**"}  {d}', flush=True)
    if not ok:
        FAIL.append(n)


def scenario(tag, dom, eff_reject=False, release_fail=False, run_s=40.0, unload_fail=False, rb_stuck=False, never_settled=False):
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
            '--wall-total-s', str(run_s), '--wall-post-abort-s', '2', '--w2-mode-run-id', RUN, '--w2-release', '--w2-unload']
    code = LAUNCH % {'here': HERE, 'args': args, 'node': os.path.join(HERE, 'door_k3m_task_node_k3b5.py')}
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
            for _ in range(30):                                             # r4：每輪處理完積壓回呼（IK 一輪約 70 ms，單次 spin 會飢餓、漏收目標）
                rclpy.spin_once(peer, timeout_sec=0.0)
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
            if got['tgt'] is not None and st.get('tgt_last') is not None and np.allclose(got['tgt'], st['tgt_last'], atol=1e-12):
                got['tgt'] = None                                           # r4：目標未變 ⇒ 不重解 IK（IK 容差 1 mm 會造成假抖動，與 1 mm 收斂門檻同量級）
            if got['tgt'] is not None and au['t_open'] is not None and wall > au['t_open']:
                qf = C.ik(np.array(base), got['tgt'], np.array(st['q_arm']), only_prev=True)
                if qf is not None:
                    st['tgt_last'] = np.array(got['tgt'], float)                 # r5：只在解成功時記下（失敗須於下一筆重試）
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
                    t1 = next((c['t'] for c in cs if c['mode_seq'] == 1), None)
                    t2 = next((c['t'] for c in cs if c['mode_seq'] == 2), None)
                    ub = {'state': 'IDLE', 'why': None, 't_settling': None}
                    if t1 is not None:
                        if unload_fail and ts >= t1 + 0.5:
                            ub = {'state': 'FAILED', 'why': 'contact_lost', 't_settling': None}
                        elif t2 is not None:
                            ub = {'state': 'HANDED_TO_RELEASE', 'why': None, 't_settling': t1 + 1.0}
                        else:
                            ub = {'state': 'UNLOADED' if ts >= t1 + 1.7 else ('SETTLING' if ts >= t1 + 1.0 else 'RAMPING'), 'why': None,
                                  't_settling': t1 + 1.0 if ts >= t1 + 1.0 else None}
                    tp = math.floor(ts)
                    fv = 20.0 if (t1 is None or tp < t1) else (5.0 if (tp >= t1 + 1.0 and not rb_stuck) else
                                                                (20.0 if rb_stuck else 20.0 + (5.0 - 20.0) * (tp - t1) / 1.0))
                    pub['state'].publish(String(data=json.dumps({'run_id': RUN, 't': ts, 'mode_seq': ms, 'commits': cs, 'commit': last,
                                                                 'commit_req_seq': None if last is None else last['mode_seq'],
                                                                 'commit_req_seqs': [c['mode_seq'] for c in cs],
                                                                 'last_reject': None, 'aborted': None, 'release': rel, 'unload': ub,
                                                                 'finger_effort': {'readback': [fv, fv], 't': float(tp), 'ok': True}})))
                    st['t1'], st['t2'] = t1, t2
                st['k'] += 1
            if wall >= 3.0 and wall >= st.get('ps_next', 0.0):                           # r4：求解端狀態每 0.1 s（sim_t＋子狀態）
                st['ps_next'] = wall + 0.1
                t1, t2, tsn = st.get('t1'), st.get('t2'), (st['k'] - 1) * 0.02
                sub = 'held'
                if t1 is not None and tsn >= t1 + 0.02:
                    sub = 'unloaded' if tsn >= t1 + 1.75 else 'unload'
                if t2 is not None and tsn >= t2 + 0.02:
                    sub = 'release_settled' if (tsn >= t2 + 1.2 and not never_settled) else 'release_open'
                pub['ps'].publish(String(data=json.dumps({'sim_t': tsn, 'outcome': {'state': 'HELD', 'sub': sub, 'observation_passed': True,
                                                                                     'physical_stop_confirmed': True, 'command_stop_confirmed': True,
                                                                                     'transitions': {'OBS_PASSED': 3.0, 'HELD': 3.0}}})))
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
    sel = sys.argv[1:] or ['R1', 'R2', 'R3', 'R4']
    print(f'暫存 {TMP}　domain {DOM0}+\n')
    if 'R1' in sel:
        print('R1  正常（閉爪卸載後張爪、讀回核對、退離終點保持後 DONE）')
        rc, task, got, st, log = scenario('r1', 0)
        if task is None:
            print(log[-3000:])
        k = (task or {}).get('k3b') or {}
        rel, mon = k.get('rel') or {}, k.get('monitor') or {}
        ck('exit 0、最終 DONE', rc == 0 and (task or {}).get('final_phase') == 'DONE', f"rc={rc} phase={(task or {}).get('final_phase')} abort={(task or {}).get('abort')}")
        ck('請求序 1、2、3', [r['mode_seq'] for r in got['req']] == [1, 2, 3], str([r['mode_seq'] for r in got['req']]))
        ug = rel.get('unload_gate') or {}
        ck('mode 2 請求晚於兩端完成（gate go）', ug.get('gate') == 'go' and (ug.get('exec_unload') or {}).get('state') == 'UNLOADED'
           and ug.get('solver_sub') == 'unloaded', str(ug)[:200])
        ck('未送 5 N 恢復請求', not got['eff'] and rel.get('eff_req') is None, str(got['eff']))
        ve = rel.get('eff_verify') or {}
        ck('eff_verify 讀回 5、晚於 C_CLEAR', ve.get('readback') == [5.0, 5.0] and (mon.get('t') or {}).get('C_CLEAR') is not None
           and ve.get('t', -1) >= mon['t']['C_CLEAR'], str(ve))
        ck('mode 3 於 release_settled 後請求', rel.get('solver_sub_at_req3') == 'release_settled', str(rel.get('solver_sub_at_req3')))
        ck('monitor COMPLETE', mon.get('state') == 'COMPLETE', f"{mon.get('state')} n_ik={st['n_ik']} ik_fail={st['ik_fail']}")
    if 'R2' in sel:
        print('\nR2  執行端 unload FAILED')
        rc, task, got, st, log = scenario('r2', 1, unload_fail=True)
        ck('中止 k3b_unload:unload_failed:contact_lost、未請求 mode 2', (task or {}).get('abort') == 'k3b_unload:unload_failed:contact_lost'
           and 2 not in [r['mode_seq'] for r in got['req']], f"{(task or {}).get('abort')} {[r['mode_seq'] for r in got['req']]}")
    if 'R3' in sel:
        print('\nR3  C_CLEAR 時讀回仍 20')
        rc, task, got, st, log = scenario('r3', 2, rb_stuck=True)
        ck('中止 k3b_effort_verify:readback_not_5', (task or {}).get('abort') == 'k3b_effort_verify:readback_not_5', str((task or {}).get('abort')))
    if 'R4' in sel:
        print('\nR4  求解端未曾 release_settled')
        rc, task, got, st, log = scenario('r4', 3, never_settled=True)
        ck('未請求 mode 3、中止', 3 not in [r['mode_seq'] for r in got['req']] and (task or {}).get('final_phase') == 'ABORTED',
           f"{[r['mode_seq'] for r in got['req']]} {(task or {}).get('abort')}")
    print(f'\n{"全部通過" if not FAIL else "失敗：" + "、".join(FAIL)}')
    return 0 if not FAIL else 1


if __name__ == '__main__':
    sys.exit(main())
