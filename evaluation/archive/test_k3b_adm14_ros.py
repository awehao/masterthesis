#!/usr/bin/env python3
"""K3-B ADM14 假世界 ROS 測試（無 Isaac；Codex 20261026_023226：交接 r2＋閉爪卸載候選）。
test_k3b_adm13_ros.py 的版本化複本。執行端角色改為 ModeAuthority7（gate2＝ExecUnloadMonitor.confirmed_fresh）＋W2Exec2＋真 CmdChainE2＋EnvIntake；
卸載啟用時，mode 1 提交步起 ExecUnloadMonitor 以合成讀回（＝設定）與合成接觸力（設定值等比）逐物理步餵入，mode 2 於 task_gate 'go'
（執行端 UNLOADED 且求解端 /wgmpc/pause_status 子狀態 unloaded）才請求。
同步狀態（/door/sync_state）的卸載瞬態仿 door_wide_k3b_30_1 W1e 5→20 N 切換量級：門 +0.24°、底盤 x +3 mm、手臂 j1 +0.004 rad，τ 0.5 s。
  M0 ADM14（r7、無卸載）、門 1°/s、狀態延遲 0.1 s：release_open、不停發、1→2→3、交接。
  M1 ADM14＋卸載：mode 1 → unload → unloaded（執行端 UNLOADED）→ mode 2（門 1°/s）→ release_open → 3、交接；mode 1 提交後無停發。
  M2 Codex 跨趟重現：mode 2 提交後 0.5 s 內本趟狀態去掉 commits，另發他趟（run_id 不同、帶提交時刻）狀態 ⇒ 不切換、暫緩逾時 FAIL moved_in_held、mode 3 未提交。
  M3 卸載中底盤偏離 25 mm：執行端 unload FAILED base_dev、任務端不請求 mode 2；求解端 FAIL base_dev_in_unload。
  M4 ADM12（對照）＋卸載瞬態：mode 1 提交後 FAIL moved_in_held（說明需要卸載子狀態）。
替身無門、無接觸物理：只驗模式、准入、監督子狀態與命令流機制。**合成資料，不是實跑結果。**

    python3 evaluation/test_k3b_adm14_ros.py [M0 M1 …]
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
TMP = os.environ.get('K3B_TMP', '/tmp/claude-1000/k3b_adm14_ros')
DOM0 = int(os.environ.get('K3B_DOMAIN', '120'))
RUN = 'k3b_adm14_ros'
FAIL = []
ARM = [f'joint{i}' for i in range(1, 7)]
READY_S, T_FREE, T_PAUSE = 3.0, 4.0, 6.0
OFFSET = (0.05, 0.02, 0.01)
FO = 0.0089


def ck(n, ok, d=''):
    print(f'  {n:64s} {"ok" if ok else "**錯**"}  {d}', flush=True)
    if not ok:
        FAIL.append(n)


def req(seq, ms):
    import door_w2_mode5 as M5
    return {'run_id': RUN, 'req_seq': seq, 'from_mode': M5.MODE_OF_SEQ5[ms - 1], 'to_mode': M5.MODE_OF_SEQ5[ms], 'mode_seq': ms}


def scenario(tag, dom, retreat=True, finger_final=1.0, arm_rate_max=1.0, max_mode=3, run_s=40.0, node_ver=14, door_rate=0.0, state_gap=None,
             unload=False, base_dev=0.003, cross_trip=None):
    os.makedirs(TMP, exist_ok=True)
    out = os.path.join(TMP, f'{tag}.json')
    if os.path.exists(out):
        os.remove(out)                                            # 本測試自己的暫存輸出
    env = dict(os.environ, ROS_DOMAIN_ID=str(DOM0 + dom), PYTHONUNBUFFERED='1')
    os.environ['ROS_DOMAIN_ID'] = str(DOM0 + dom)
    import rclpy
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray, String
    import cmd_envelope_v3 as E3
    import door_k3b_release2 as KR
    import door_k3m_applied_src as K3MSRC
    import door_w2_exec2 as WX2
    import door_w2_mode5 as M5
    import door_w2_mode7 as M7
    import door_k3b_unload as KU
    import wgmpc_cmd_envelope as ENV
    from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics
    from wb_cmd_chain_e2 import CmdChainE2
    from wb_wheel_limit import WheelLimitConfig
    K = WholeBodyKinematics.from_urdf_file(os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
    T0 = K.fk(np.zeros(9), 'link_tcp')
    T_des = np.eye(4)
    T_des[:3, :3], T_des[:3, 3] = T0[:3, :3], T0[:3, 3] + np.asarray(OFFSET)
    rclpy.init()
    peer = rclpy.create_node(f'k3b_peer_{tag}')
    cons = rclpy.create_node('w2_consumer')
    cons.declare_parameter('joints', ARM)
    lat = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    pub = {'gate': peer.create_publisher(String, '/park/gate', 1),
           'free': peer.create_publisher(String, '/door/base_mode_ack', lat),
           'pause': peer.create_publisher(String, '/door/pause_ack_exec', lat),
           'req': peer.create_publisher(String, '/door/w2_mode_req', lat),
           'state': peer.create_publisher(String, '/door/w2_mode_state', 50),
           'sync': peer.create_publisher(Float64MultiArray, '/door/sync_state', 100),
           'asrc': peer.create_publisher(Float64MultiArray, '/door/applied_src', 200),
           'tgt': peer.create_publisher(Float64MultiArray, '/drawer/tcp_target', 10)}
    xm = KR.ExecReleaseMonitor()
    xu = KU.ExecUnloadMonitor()
    auth = M7.ModeAuthority7(RUN, run_id_num=ENV.run_id_num(RUN), stage_id=ENV.ST_ADAPTER, dt=0.01, arm_rate_max=arm_rate_max,
                             max_mode=max_mode, gate3=lambda step: xm.confirmed_fresh(step),
                             gate2=(lambda step: xu.confirmed_fresh(step)) if unload else None)
    TAU = 0.5

    def trans(t):                                                        # 卸載瞬態（mode 1 提交起）
        if not unload or st.get('t_c1') is None:
            return 0.0
        return 1 - math.exp(-max(0.0, t - st['t_c1']) / TAU)

    def low_speed_bound(vx, vy, wz):
        lin = (vx * vx + vy * vy) ** 0.5
        return (False, f'lin {lin:.4f}') if lin > 0.05 else ((False, f'ang {wz:.4f}') if abs(wz) > 0.20 else (True, None))
    chain = CmdChainE2(max_cmd_age_s=0.2, arm_rate_max=1.0, wheel_ok=low_speed_bound, joint_lower=(-100.0,) * 6, joint_upper=(100.0,) * 6,
                       joint_margin=0.05, mode='solver_drawer', wheel_cfg=WheelLimitConfig(arm_rate_max=1.0, dt_max=0.05),
                       base_accel_max_lin=0.5, base_accel_max_ang=2.0)
    intake = K3MSRC.EnvIntake(run_id_num=ENV.run_id_num(RUN), stage_id=ENV.ST_ADAPTER, kinds=(ENV.K_NORMAL,))
    wx = WX2.W2Exec2(auth, intake)
    st = {'t': 0.0, 't_held': None, 'free_sent': False, 'pause_sent': False, 'last_step': None, 'last_sync': -1.0, 'rows': [],
          'reqs': {}, 'req_q': [], 't_open': None, 'k_open': None, 'fingers': [], 'last_tgt': -1.0, 'xm_open': False, 'known': None,
          'cand3_seen': [], 'intake': {}, 'js': None, 'T_open': None}
    peer.create_subscription(Clock, '/clock', lambda m: st.__setitem__('t', m.clock.sec + m.clock.nanosec * 1e-9), 10)

    def on_status(m):
        s = json.loads(m.data)
        o = s.get('outcome') or {}
        st['solver_o'] = o
        if o.get('state') == 'HELD' and st['t_held'] is None:
            st['t_held'] = st['t']
    peer.create_subscription(String, '/wgmpc/pause_status', on_status, lat)
    peer.create_subscription(JointState, '/joint_states', lambda m: st.__setitem__('js', [float(x) for x in m.position[:6]]), 10)

    def on_env3(m):
        try:
            e = E3.decode3(m.data)
        except ValueError:
            return
        if e['mode_seq'] == 3:
            st['cand3_seen'].append(e['source_seq'])
        act, _ = wx.on_env3(e, st['t'], time.monotonic())
        st['intake'][act] = st['intake'].get(act, 0) + 1
    peer.create_subscription(Float64MultiArray, E3.TOPIC3[ENV.ST_ADAPTER], on_env3, 50)
    common = ['--arm-model', 'setpoint', '--N', '5', '--rate', '20', '--target-offset', *[str(x) for x in OFFSET],
              '--target-topic', '/drawer/tcp_target', '--duration-s', str(run_s - 3.0), '--u-prev-policy', 'strict', '--assume-initial-rest',
              '--park-hold', '--continuous', '--mode-switch-run-id', RUN, '--cmd-env', '--run-id', RUN, '--pause',
              '--pause-arm-zero', '--pause-active-hold', '--pause-arm-win-s', '0.5', '--w2-held', '--w2-held-max-s', '60',
              '--w2-mode-run-id', RUN, '--out', out]
    stub = subprocess.Popen([sys.executable, '-u', os.path.join(HERE, 'wgmpc_sp_stub_endpoint.py'), '--run-s', str(run_s), '--ready-after', str(READY_S)],
                            cwd=WS, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    node = subprocess.Popen([sys.executable, '-u', os.path.join(HERE, f'dl3_wgmpc_wg2_adm{node_ver}_node.py')] + common
                            + (['--w2-retreat'] if retreat else []) + (['--w2-hold-r7'] if node_ver == 14 else [])
                            + (['--w2-unload'] if node_ver == 14 and unload else []),
                            cwd=WS, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    adp = subprocess.Popen([sys.executable, '-u', os.path.join(HERE, 'arm_vel_adapter_w2.py'), '--consumer-node', '/w2_consumer', '--cmd-env-v3'],
                           cwd=WS, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    t0 = time.monotonic()
    try:
        while node.poll() is None and time.monotonic() - t0 < run_s + 60:
            rclpy.spin_once(peer, timeout_sec=0.002)
            rclpy.spin_once(cons, timeout_sec=0.0)
            t = st['t']
            pub['gate'].publish(String(data=json.dumps({'passed': True, 'ok_now': True, 'anchor': [0.0, 0.0, 0.0]})))
            if not st['free_sent'] and t >= T_FREE:
                pub['free'].publish(String(data=json.dumps({'run_id': RUN, 'seq': 1, 'mode': 'free', 'switch_step': int(round(t * 100)),
                                                            'switch_t': round(t, 4)})))
                st['free_sent'] = True
            if st['free_sent'] and not st['pause_sent'] and t >= T_PAUSE:
                pub['pause'].publish(String(data=json.dumps({'run_id': RUN, 'seq': 2, 'mode': 'pause', 'switch_step': int(round(t * 100)),
                                                             'switch_t': round(t, 4)})))
                st['pause_sent'] = True
            # 請求排程：HELD＋1 s → 1；mode 1 確認＋0.5 s → 2；解除確認（或 t_open＋1.6 s）→ 3
            cs = {c['mode_seq']: c for c in auth.commits}
            if st['t_held'] is not None and t >= st['t_held'] + 1.0 and 1 not in st['reqs']:
                st['req_q'].append(req(1, 1))
                st['reqs'][1] = t
            go2 = (KU.task_gate({'unload': {'state': xu.state, 'why': xu.why}}, cs[1]['t'], t, st.get('solver_o')) == 'go') if (unload and 1 in cs) else True
            if 1 in cs and cs[1]['applied_confirmed'] and t >= cs[1]['t'] + 0.5 and 2 not in st['reqs'] and max_mode >= 2 and go2:
                st['req_q'].append(req(2, 2))
                st['reqs'][2] = t
            if max_mode >= 3 and 2 in cs and 3 not in st['reqs'] and (xm.state == 'B_CONFIRMED' or (st['t_open'] and t >= st['t_open'] + 1.6)):
                st['req_q'].append(req(3, 3))
                st['reqs'][3] = t
            # 退離目標（mode 2 提交後）
            if st['t_open'] is not None and t >= st['last_tgt'] + 0.05:
                st['last_tgt'] = t
                d = 0.0 if xm.t.get('B_CONFIRMED') is None else min(0.02, 0.02 * max(0.0, t - xm.t['B_CONFIRMED']))
                if st['T_open'] is None and st['js'] is not None:
                    st['T_open'] = K.fk(np.array([0.0, 0.0, 0.0] + st['js']), 'link_tcp')
                Tr = (st['T_open'] if st['T_open'] is not None else T_des).copy()
                Tr[0, 3] -= d
                pub['tgt'].publish(Float64MultiArray(data=[float(x) for x in Tr.reshape(-1)]))
            while st['last_sync'] + 0.02 <= t + 1e-9:
                ts = round(st['last_sync'] + 0.02, 4) if st['last_sync'] >= 0 else round(t, 4)
                st['last_sync'] = ts
                qd = 0.01 + (door_rate * max(0.0, ts - st['t_open']) if st['t_open'] is not None else 0.0) + math.radians(0.24) * trans(ts)
                pub['sync'].publish(Float64MultiArray(data=[ts, float(int(round(ts * 100))), qd] + [0.0] * 16 + [0.6 + base_dev * trans(ts), -0.53, 1.57]
                                                      + [0.004 * trans(ts)] + [0.0] * 5 + [0.0]))
            k_now = int(round(t * 100))
            k0 = st['last_step'] + 1 if st['last_step'] is not None else k_now
            for k in range(k0, k_now + 1):
                tk = k * 0.01
                if xm.state == 'FAILED' and st['known'] is None:
                    st['known'] = f'release_monitor:{xm.why}'
                rq, st['req_q'] = st['req_q'], []
                sel, relay, _ = wx.step_begin(k, tk, chain, known_abort=st['known'], requests=rq)
                for r in relay:
                    pub['req'].publish(String(data=json.dumps(r)))
                if sel['event'] and sel['event'].get('kind') == 'commit' and auth.mode_seq == 1 and unload:
                    st['t_c1'] = tk
                    xu.start(k, tk, [0.6, -0.53, 1.57])
                if sel['event'] and sel['event'].get('kind') == 'commit' and auth.mode_seq == 2:
                    st['t_open'], st['k_open'] = tk, k
                    xu.handoff(k, tk)
                    xm.open(k, tk)
                if unload and xu.state in ('RAMPING', 'SETTLING', 'UNLOADED'):         # 合成卸載資料（執行端逐物理步）
                    fc = xu.command(tk)
                    xu.step(k, tk, [fc, fc], fc, [fc * 0.64, fc * 0.5], True, 0.01 + math.radians(0.24) * trans(tk),
                            [0.6 + base_dev * trans(tk), -0.53, 1.57], [0.004 * trans(tk)] + [0.0] * 5)
                st['fingers'].append((k, sel.get('finger_target'), sel['mode_seq']))
                chain.step(tk, 0.01, [0.0] * 6)
                u9 = [float(x) for x in chain.u_prev] if chain.u_prev is not None else [0.0] * 9
                row = wx.post_step(k, tk, chain, True, u9, sel)
                pub['asrc'].publish(Float64MultiArray(data=[float(k), tk, float(row['src']), float(row['chain_mode'])] + u9))
                st['rows'].append(row)
                if st['k_open'] is not None and k > st['k_open']:                   # 合成手指資料（執行端逐物理步）
                    tr = tk - st['t_open']
                    f = 0.2 + 0.75 * tr / 0.31 if tr < 0.31 else 0.95 + (tr - 0.31) * 0.2
                    fq = FO * min(finger_final, f)
                    pf = max(0.0, 12.0 * (1 - tr / 0.04))
                    xm.step(k, tk, [fq, fq], [pf, pf], True)
                gap = state_gap is not None and st['t_open'] is not None and st['t_open'] + state_gap[0] <= tk < st['t_open'] + state_gap[1]
                xt = cross_trip is not None and st['t_open'] is not None and st['t_open'] <= tk < st['t_open'] + cross_trip
                if k % 2 == 0 and not gap:
                    sm = dict(auth.state_msg(k, tk), release={'state': xm.state, 'why': xm.why}, unload={'state': xu.state, 'why': xu.why})
                    if xt:                                                            # Codex 跨趟重現：本趟去 commits，另發他趟帶提交時刻
                        other = dict(sm, run_id='other_trip')
                        sm = {kk: vv for kk, vv in sm.items() if kk not in ('commits', 'commit_req_seqs')}
                        pub['state'].publish(String(data=json.dumps(other, default=float)))
                    pub['state'].publish(String(data=json.dumps(sm, default=float)))
            st['last_step'] = k_now
        if node.poll() is None:
            node.terminate()
            node.wait(timeout=10)
        for _ in range(20):
            rclpy.spin_once(peer, timeout_sec=0.02)
    finally:
        for p in (stub, adp):
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        peer.destroy_node()
        cons.destroy_node()
        rclpy.shutdown()
    w = json.load(open(out)) if os.path.exists(out) else None
    if w is None:
        print((node.stdout.read() if node.stdout else '')[-3000:])
    st['xu'] = xu.report()
    return w, st, auth, xm


def nopub_after(w, t0):
    return [e for e in ((w or {}).get('log') or []) if e.get('sim_t', 0) >= t0 and e.get('published') is False]


def main() -> int:
    sel = sys.argv[1:] or ['M0', 'M1', 'M2', 'M3', 'M4']
    print(f'暫存 {TMP}　domain {DOM0}+\n')
    DEG = math.pi / 180.0

    def outc(w):
        s = (w or {}).get('stats') or {}
        return s, (s.get('k3m_pause') or {}).get('supervisor_outcome') or {}

    def full_ok(w, st, a, t_from):
        s, o = outc(w)
        cs = {c['mode_seq']: c for c in a.commits}
        ck('三次提交皆確認', sorted(cs) == [1, 2, 3] and all(c['applied_confirmed'] is True for c in cs.values()),
           str({k: (v['step'], v['applied_confirmed']) for k, v in cs.items()}) + f" aborted={a.aborted} rej={a.last_reject}")
        ck('監督器 release_open 後交接', o.get('sub') == 'release_open' and o.get('state') == 'HANDED_OVER',
           f"sub={o.get('sub')} state={o.get('state')} why={o.get('fail_why')} checks={o.get('checks')}")
        k0 = cs.get(t_from, {}).get('t')
        bad = [] if k0 is None else [e for e in nopub_after(w, k0) if e.get('reason') in ('pause_no_publish', 'pause_observe')]
        ck(f'mode {t_from} 提交後無停發', k0 is not None and not bad, str(bad[:2]))
        return s, o, cs

    if 'M0' in sel:
        print('M0  ADM14（無卸載）、門 1°/s、狀態延遲 0.1 s')
        w, st, a, xm = scenario('m0', 0, door_rate=1.0 * DEG, state_gap=(0.0, 0.1))
        s, o, cs = full_ok(w, st, a, 2)
        ck('得知晚於提交 ≥ 0.1 s', (o.get('known_t') or {}).get('2') is not None and 2 in cs and o['known_t']['2'] >= cs[2]['t'] + 0.1 - 1e-6,
           str(o.get('known_t')))
    if 'M1' in sel:
        print('\nM1  ADM14＋卸載')
        w, st, a, xm = scenario('m1', 1, unload=True, door_rate=1.0 * DEG)
        s, o, cs = full_ok(w, st, a, 1)
        tr_ = o.get('transitions') or {}
        ck('子狀態 unload → unloaded → release_open', all(tr_.get(k) is not None for k in ('UNLOAD', 'UNLOADED', 'RELEASE_OPEN'))
           and tr_['UNLOAD'] <= tr_['UNLOADED'] <= tr_['RELEASE_OPEN'], str({k: tr_.get(k) for k in ('UNLOAD', 'UNLOADED', 'RELEASE_OPEN')}))
        xu = st['xu']
        ck('mode 2 提交晚於求解端 unloaded（task_gate 等兩端）', tr_.get('UNLOADED') is not None and 2 in cs and cs[2]['t'] >= tr_['UNLOADED'],
           f"unloaded={tr_.get('UNLOADED')} c2={cs.get(2, {}).get('t')}")
        ck('執行端 UNLOADED 早於 mode 2 請求、交給釋放監看', xu['t'].get('UNLOADED') is not None and 2 in cs
           and xu['t']['UNLOADED'] <= st['reqs'].get(2, -1) and xu['state'] == 'HANDED_TO_RELEASE', f"{xu['t']} req2={st['reqs'].get(2)}")
        ck('卸載斜坡讀回 20→5', bool(xu['set_log']) and abs(xu['set_log'][0][2] - 20.0) < 0.3 and abs(xu['set_log'][-1][2] - 5.0) < 1e-6,
           f"{xu['set_log'][0][2] if xu['set_log'] else None}→{xu['set_log'][-1][2] if xu['set_log'] else None}")
    if 'M2' in sel:
        print('\nM2  Codex 跨趟重現')
        w, st, a, xm = scenario('m2', 2, door_rate=1.0 * DEG, cross_trip=0.5)
        s, o = outc(w)
        r7 = s.get('w2_r7') or {}
        ck('不切換、暫緩逾時 FAIL moved_in_held', o.get('state') == 'FAIL' and o.get('fail_why') == 'moved_in_held' and o.get('sub') == 'held',
           f"{o.get('state')} {o.get('fail_why')} sub={o.get('sub')} rej={r7.get('rejected')} other={r7.get('n_other_run')}")
        ck('他趟訊息被濾除', r7.get('n_other_run', 0) > 0, str(r7.get('n_other_run')))
        ck('mode 3 未提交', 3 not in {c['mode_seq'] for c in a.commits}, f'mode={a.mode_seq}')
    if 'M3' in sel:
        print('\nM3  卸載中底盤偏離 25 mm')
        w, st, a, xm = scenario('m3', 3, unload=True, base_dev=0.025)
        s, o = outc(w)
        xu = st['xu']
        ck('執行端 unload FAILED base_dev', xu['state'] == 'FAILED' and xu['why'] == 'base_dev', f"{xu['state']} {xu['why']}")
        ck('mode 2 未請求', 2 not in st['reqs'], str(st['reqs']))
        ck('求解端 FAIL base_dev_in_unload', o.get('state') == 'FAIL' and o.get('fail_why') == 'base_dev_in_unload', f"{o.get('state')} {o.get('fail_why')}")
    if 'M4' in sel:
        print('\nM4  ADM12（對照）＋卸載瞬態')
        w, st, a, xm = scenario('m4', 4, node_ver=12, unload=True)
        s, o = outc(w)
        k1 = next((c['t'] for c in a.commits if c['mode_seq'] == 1), None)
        ck('mode 1 提交後 FAIL moved_in_held', o.get('state') == 'FAIL' and o.get('fail_why') == 'moved_in_held' and k1 is not None
           and ((o.get('transitions') or {}).get('FAIL') or 0) >= k1, f"{o.get('state')} {o.get('fail_why')} FAIL@{(o.get('transitions') or {}).get('FAIL')} c1={k1}")
    print(f'\n{"全部通過" if not FAIL else "失敗：" + "、".join(FAIL)}')
    return 0 if not FAIL else 1


if __name__ == '__main__':
    sys.exit(main())
