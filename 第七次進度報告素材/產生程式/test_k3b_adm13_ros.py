#!/usr/bin/env python3
"""K3-B ADM13 假世界 ROS 測試（無 Isaac；Codex 20261026_021025 §2：mode 2 門速超過舊 HELD 停止門檻、未觸及釋放保護 ⇒ 不得停發；真正越界仍須中止）。
test_k3b_adm12_ros.py 的版本化複本；替身與執行端角色相同（ModeAuthority5＋W2Exec2＋真 CmdChainE2＋EnvIntake＋ExecReleaseMonitor）。差異：
- 節點 dl3_wgmpc_wg2_adm13_node（--w2-release-hold）；對照組 ADM12。
- /door/sync_state 門角（第 3 欄）自 mode 2 提交步起以 door_rate（rad/s）增加（模擬張爪後門扇轉動）；手臂、底盤量測不變。
- state_gap：mode 2 提交後 [a, b) 秒內不發 /door/w2_mode_state（模擬權威狀態延遲／遺失）。
  L0 ADM13、門不動：1→2→3 皆確認、交接、release_open 已切換、無停發。
  L1 ADM13、門 1°/s、狀態延遲 0.1 s：暫緩後切換 release_open，提交後無停發，1→2→3 皆確認、交接。
  L2 ADM13、門 0.06 rad/s（> 保護 0.05）：監督器 FAIL door_rate_in_release、停發，mode 3 未提交。
  L3 ADM12（對照）、門 1°/s：重現實跑 FAIL moved_in_held、停發。
  L4 ADM13、門 1°/s、狀態中斷 0.5 s（> 暫緩 0.2 s）：FAIL moved_in_held（保守）、停發。
替身無手臂物理以外的動力學：只驗模式、監督交接與命令流機制，**不驗門的物理**。

    python3 evaluation/test_k3b_adm13_ros.py [L0 L1 …]
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
TMP = os.environ.get('K3B_TMP', '/tmp/claude-1000/k3b_adm13_ros')
DOM0 = int(os.environ.get('K3B_DOMAIN', '214'))
RUN = 'k3b_adm13_ros'
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


def scenario(tag, dom, retreat=True, finger_final=1.0, arm_rate_max=1.0, max_mode=3, run_s=40.0, adm13=True, door_rate=0.0, state_gap=None):
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
    auth = M5.ModeAuthority5(RUN, run_id_num=ENV.run_id_num(RUN), stage_id=ENV.ST_ADAPTER, dt=0.01, arm_rate_max=arm_rate_max,
                             max_mode=max_mode, gate3=lambda step: xm.confirmed_fresh(step))

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
    node = subprocess.Popen([sys.executable, '-u', os.path.join(HERE, 'dl3_wgmpc_wg2_adm13_node.py' if adm13 else 'dl3_wgmpc_wg2_adm12_node.py')] + common
                            + (['--w2-retreat'] if retreat else []) + (['--w2-release-hold'] if adm13 else []),
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
            if 1 in cs and cs[1]['applied_confirmed'] and t >= cs[1]['t'] + 0.5 and 2 not in st['reqs'] and max_mode >= 2:
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
                qd = 0.01 + (door_rate * max(0.0, ts - st['t_open']) if st['t_open'] is not None else 0.0)   # ADM13 測試：張爪後門轉動
                pub['sync'].publish(Float64MultiArray(data=[ts, float(int(round(ts * 100))), qd] + [0.0] * 16 + [0.6, -0.53, 1.57]
                                                      + [0.0] * 6 + [0.0]))
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
                if sel['event'] and sel['event'].get('kind') == 'commit' and auth.mode_seq == 2:
                    st['t_open'], st['k_open'] = tk, k
                    xm.open(k, tk)
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
                if k % 2 == 0 and not gap:
                    pub['state'].publish(String(data=json.dumps(dict(auth.state_msg(k, tk), release={'state': xm.state, 'why': xm.why}),
                                                                default=float)))
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
    return w, st, auth, xm


def nopub_after(w, t0):
    return [e for e in ((w or {}).get('log') or []) if e.get('sim_t', 0) >= t0 and e.get('published') is False]


def main() -> int:
    sel = sys.argv[1:] or ['L0', 'L1', 'L2', 'L3', 'L4']
    print(f'暫存 {TMP}　domain {DOM0}+\n')
    DEG = math.pi / 180.0

    def common_ok(w, st, a, expect_release=True):
        s = (w or {}).get('stats') or {}
        cs = {c['mode_seq']: c for c in a.commits}
        o = (s.get('k3m_pause') or {}).get('supervisor_outcome') or {}
        ck('三次提交皆確認', sorted(cs) == [1, 2, 3] and all(c['applied_confirmed'] is True for c in cs.values()),
           str({k: (v['step'], v['applied_confirmed']) for k, v in cs.items()}) + f" aborted={a.aborted} rej={a.last_reject}")
        ck('監督器：release_open 已切換、之後交接', o.get('sub') == 'release_open' and o.get('state') == 'HANDED_OVER',
           f"sub={o.get('sub')} state={o.get('state')} why={o.get('fail_why')} checks={o.get('release_checks')}")
        if 2 in cs:
            bad = [e for e in nopub_after(w, cs[2]['t']) if e.get('reason') in ('pause_no_publish', 'pause_observe')]
            ck('mode 2 提交後無停發（pause_no_publish／observe 0）', not bad, str(bad[:2]))
        return s, cs, o

    if 'L0' in sel:
        print('L0  ADM13、門不動')
        w, st, a, xm = scenario('l0', 0)
        common_ok(w, st, a)
    if 'L1' in sel:
        print('\nL1  ADM13、門 1°/s、狀態延遲 0.1 s')
        w, st, a, xm = scenario('l1', 1, door_rate=1.0 * DEG, state_gap=(0.0, 0.1))
        s, cs, o = common_ok(w, st, a)
        ck('提交時刻＝權威 mode 2 提交', o.get('release_commit_t') is not None and 2 in cs and abs(o['release_commit_t'] - cs[2]['t']) < 1e-6,
           f"{o.get('release_commit_t')} vs {cs.get(2, {}).get('t')}")
        ck('得知晚於提交（延遲有效）', o.get('release_known_t') is not None and 2 in cs and o['release_known_t'] >= cs[2]['t'] + 0.1 - 1e-6,
           str(o.get('release_known_t')))
    if 'L2' in sel:
        print('\nL2  ADM13、門 0.06 rad/s')
        w, st, a, xm = scenario('l2', 2, door_rate=0.06)
        s = (w or {}).get('stats') or {}
        o = (s.get('k3m_pause') or {}).get('supervisor_outcome') or {}
        ck('FAIL door_rate_in_release', o.get('state') == 'FAIL' and o.get('fail_why') == 'door_rate_in_release', f"{o.get('state')} {o.get('fail_why')}")
        k2 = next((c for c in a.commits if c['mode_seq'] == 2), None)
        ck('之後停發', k2 is not None and bool(nopub_after(w, k2['t'])), '')
        ck('mode 3 未提交', 3 not in {c['mode_seq'] for c in a.commits}, f'mode={a.mode_seq} aborted={a.aborted}')
    if 'L3' in sel:
        print('\nL3  ADM12（對照）、門 1°/s')
        w, st, a, xm = scenario('l3', 3, adm13=False, door_rate=1.0 * DEG)
        s = (w or {}).get('stats') or {}
        o = (s.get('k3m_pause') or {}).get('supervisor_outcome') or {}
        ck('FAIL moved_in_held（重現實跑）', o.get('state') == 'FAIL' and o.get('fail_why') == 'moved_in_held', f"{o.get('state')} {o.get('fail_why')}")
        ck('mode 3 未提交', 3 not in {c['mode_seq'] for c in a.commits}, f'mode={a.mode_seq}')
    if 'L4' in sel:
        print('\nL4  ADM13、門 1°/s、狀態中斷 0.5 s')
        w, st, a, xm = scenario('l4', 4, door_rate=1.0 * DEG, state_gap=(0.0, 0.5))
        s = (w or {}).get('stats') or {}
        o = (s.get('k3m_pause') or {}).get('supervisor_outcome') or {}
        ck('FAIL moved_in_held（暫緩逾時）', o.get('state') == 'FAIL' and o.get('fail_why') == 'moved_in_held' and o.get('sub') == 'held',
           f"{o.get('state')} {o.get('fail_why')} sub={o.get('sub')} defer={o.get('defer_log')}")
        ck('mode 3 未提交', 3 not in {c['mode_seq'] for c in a.commits}, f'mode={a.mode_seq}')
    print(f'\n{"全部通過" if not FAIL else "失敗：" + "、".join(FAIL)}')
    return 0 if not FAIL else 1


if __name__ == '__main__':
    sys.exit(main())
