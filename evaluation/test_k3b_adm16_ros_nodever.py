#!/usr/bin/env python3
"""test_k3b_adm16_ros.py（凍結 k3b dev r5，不改）的複本：唯一差異＝K3B_NODE_VER 環境變數選節點版本（18 時同樣加 --w2-hold-r9），供 ADM18 跑同一組回歸。
K3-B ADM16 假世界 ROS 測試（無 Isaac；Codex 20261026_034842：監督器 r9）。test_k3b_adm15_ros.py 的版本化複本（節點 ADM16、旗標 --w2-hold-r9；另加 N5）。
執行端角色：ModeAuthority8（unload＝True 時 gate2 必要）＋W2Exec2＋真 CmdChainE2＋EnvIntake；卸載監看 door_k3b_unload2（權威狀態帶 state_brief，含 t_settling）。
同步狀態：卸載瞬態同 ADM14 測試；張爪後手臂 j1 回彈（release_arm(t)）、門以 door_rate 轉動。mode 3 請求另等求解端子狀態 release_settled。
  N0 卸載＋張爪後門 1°/s＋手臂回彈 0.004 rad（τ 0.2 s）收斂：unload → unloaded（t_settling 後）→ release_open → release_settled → 3、交接；mode 1 後無停發。
  N1 張爪後手臂持續漂移 0.002 rad/s（不收斂）：FAIL release_transient:timeout、停發、mode 3 未提交。
  N2 Codex 跨趟重現（同 M2）：不切換、逾時 FAIL moved_in_held。
  N3 卸載中底盤偏離 25 mm（τ 0.5 s，初速約 50 mm/s）：執行端 base_dev、mode 2 未請求；求解端 FAIL（瞬態監看的 TCP 為世界座標、含底盤運動，
     故可能先以 unload_transient:tcp_speed 失敗）。
  退離目標：同真任務端，mode 3 提交後才推進（ADM12／14 測試以解除確認起推進）。
  N5 卸載＋回彈＋mode 2 後權威狀態延遲 0.1 s：延遲得知時重放提交後樣本（replay_n > 0）、仍交接。
  N4 權威狀態不帶 t_settling：求解端不起算重新收斂窗 ⇒ unload_timeout；任務端 mode 2 未請求。
替身無門、無接觸物理：只驗機制。**合成資料，不是實跑結果。**

    python3 evaluation/test_k3b_adm16_ros.py [N0 N1 …]
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
TMP = os.environ.get('K3B_TMP', '/tmp/claude-1000/k3b_adm16_ros')
NODE_VER = int(os.environ.get('K3B_NODE_VER', '16'))
DOM0 = int(os.environ.get('K3B_DOMAIN', '130'))
RUN = 'k3b_adm16_ros'
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


def scenario(tag, dom, retreat=True, finger_final=1.0, arm_rate_max=1.0, max_mode=3, run_s=40.0, node_ver=None, door_rate=0.0, state_gap=None,
             unload=False, base_dev=0.003, cross_trip=None, arm_drift=0.0, hide_settling=False):
    node_ver = NODE_VER if node_ver is None else node_ver
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
    import door_w2_mode8 as M8
    import door_k3b_unload2 as KU
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
    auth = M8.ModeAuthority8(RUN, run_id_num=ENV.run_id_num(RUN), stage_id=ENV.ST_ADAPTER, dt=0.01, arm_rate_max=arm_rate_max,
                             max_mode=max_mode, gate3=lambda step: xm.confirmed_fresh(step),
                             gate2=(lambda step: xu.confirmed_fresh(step)) if unload else None, unload=unload)
    TAU = 0.5

    def trans(t):                                                        # 卸載瞬態（mode 1 提交起）
        if not unload or st.get('t_c1') is None:
            return 0.0
        return 1 - math.exp(-max(0.0, t - st['t_c1']) / TAU)

    def release_arm(t):                                                  # 張爪後手臂回彈（j1）
        if st.get('t_open') is None:
            return 0.0
        dt_ = max(0.0, t - st['t_open'])
        return 0.004 * (1 - math.exp(-dt_ / 0.2)) + arm_drift * dt_

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
        st['solver_t'] = s.get('sim_t')
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
                            + (['--w2-retreat'] if retreat else []) + (['--w2-hold-r9'] if node_ver in (16, 18) else [])
                            + (['--w2-unload'] if node_ver in (16, 18) and unload else []),
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
            go2 = (KU.task_gate({'run_id': RUN, 'unload': {'state': xu.state, 'why': xu.why}}, cs[1]['t'], t, st.get('solver_o'), run_id=RUN,
                                state_t=t, solver_t=st.get('solver_t')) == 'go') if (unload and 1 in cs) else True
            if 1 in cs and cs[1]['applied_confirmed'] and t >= cs[1]['t'] + 0.5 and 2 not in st['reqs'] and max_mode >= 2 and go2:
                st['req_q'].append(req(2, 2))
                st['reqs'][2] = t
            if max_mode >= 3 and 2 in cs and 3 not in st['reqs'] and xm.state == 'B_CONFIRMED' and (st.get('solver_o') or {}).get('sub') == 'release_settled':
                st['req_q'].append(req(3, 3))
                st['reqs'][3] = t
            # 退離目標（mode 2 提交後）
            if st['t_open'] is not None and t >= st['last_tgt'] + 0.05:
                st['last_tgt'] = t
                _c3 = next((c['t'] for c in auth.commits if c['mode_seq'] == 3), None)      # ADM15 測試：同真任務端，mode 3 提交後才推進
                d = 0.0 if _c3 is None else min(0.02, 0.02 * max(0.0, t - _c3))
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
                                                      + [0.004 * trans(ts) + release_arm(ts)] + [0.0] * 5 + [0.0]))
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
                    xu.start(k, tk)
                if sel['event'] and sel['event'].get('kind') == 'commit' and auth.mode_seq == 2:
                    st['t_open'], st['k_open'] = tk, k
                    xu.handoff(k, tk)
                    xm.open(k, tk)
                if unload and xu.state in ('RAMPING', 'SETTLING', 'UNLOADED'):         # 合成卸載資料（執行端逐物理步）
                    fc = xu.command(tk)
                    xu.step(k, tk, [fc, fc], fc, [fc * 0.64, fc * 0.5], True, 0.01 + math.radians(0.24) * trans(tk),
                            [0.6 + base_dev * trans(tk), -0.53, 1.57], [0.004 * trans(tk) + release_arm(tk)] + [0.0] * 5)
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
                    ub = xu.state_brief()
                    if hide_settling:
                        ub['t_settling'] = None
                    sm = dict(auth.state_msg(k, tk), release={'state': xm.state, 'why': xm.why}, unload=ub)
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
    sel = sys.argv[1:] or ['N0', 'N1', 'N2', 'N3', 'N4', 'N5']
    print(f'暫存 {TMP}　domain {DOM0}+\n')
    DEG = math.pi / 180.0

    def outc(w):
        s = (w or {}).get('stats') or {}
        return s, (s.get('k3m_pause') or {}).get('supervisor_outcome') or {}

    if 'N0' in sel:
        print('N0  卸載＋張爪後門 1°/s＋手臂回彈收斂')
        w, st, a, xm = scenario('n0', 0, unload=True, door_rate=1.0 * DEG)
        s, o = outc(w)
        cs = {c['mode_seq']: c for c in a.commits}
        ck('三次提交皆確認', sorted(cs) == [1, 2, 3] and all(c['applied_confirmed'] is True for c in cs.values()),
           str({k: (v['step'], v['applied_confirmed']) for k, v in cs.items()}) + f" aborted={a.aborted} rej={a.last_reject}")
        tr_ = o.get('transitions') or {}
        ks = ('UNLOAD', 'UNLOADED', 'RELEASE_OPEN', 'RELEASE_SETTLED', 'HANDOVER')
        ck('子狀態序 unload → unloaded → release_open → release_settled → 交接',
           all(tr_.get(k) is not None for k in ks) and all(tr_[x] <= tr_[y] for x, y in zip(ks, ks[1:])) and o.get('state') == 'HANDED_OVER',
           str({k: tr_.get(k) for k in ks}) + f" state={o.get('state')} why={o.get('fail_why')}")
        ck('重新收斂窗自執行端 t_settling 起算', o.get('t_settling') is not None and tr_.get('UNLOADED') is not None
           and tr_['UNLOADED'] >= o['t_settling'] + 0.5 - 0.06, f"t_settling={o.get('t_settling')} unloaded={tr_.get('UNLOADED')}")
        k1 = cs.get(1, {}).get('t')
        bad = [] if k1 is None else [e for e in nopub_after(w, k1) if e.get('reason') in ('pause_no_publish', 'pause_observe')]
        ck('mode 1 提交後無停發', k1 is not None and not bad, str(bad[:2]))
        tm2 = (o.get('transient') or {}).get('2') or {}
        ck('釋放瞬態收斂、位移在界內', tm2.get('state') == 'CONVERGED' and (tm2.get('worst') or {}).get('disp_m', 1.0) <= 0.005, str(tm2))
        ck('mode 3 請求晚於 release_settled', 3 in st['reqs'] and tr_.get('RELEASE_SETTLED') is not None and st['reqs'][3] >= tr_['RELEASE_SETTLED'],
           f"req3={st['reqs'].get(3)} settled={tr_.get('RELEASE_SETTLED')}")
    if 'N1' in sel:
        print('\nN1  張爪後手臂持續漂移（不收斂）')
        w, st, a, xm = scenario('n1', 1, unload=True, door_rate=1.0 * DEG, arm_drift=0.002)
        s, o = outc(w)
        ck('FAIL release_transient:timeout', o.get('state') == 'FAIL' and o.get('fail_why') == 'release_transient:timeout', f"{o.get('state')} {o.get('fail_why')}")
        ck('mode 3 未提交', 3 not in {c['mode_seq'] for c in a.commits}, f'mode={a.mode_seq} aborted={a.aborted}')
    if 'N2' in sel:
        print('\nN2  Codex 跨趟重現')
        w, st, a, xm = scenario('n2', 2, door_rate=1.0 * DEG, cross_trip=0.5)
        s, o = outc(w)
        r7 = s.get('w2_r7') or {}
        ck('不切換、暫緩逾時 FAIL moved_in_held', o.get('state') == 'FAIL' and o.get('fail_why') == 'moved_in_held' and o.get('sub') == 'held',
           f"{o.get('state')} {o.get('fail_why')} sub={o.get('sub')} rej={r7.get('rejected')} other={r7.get('n_other_run')}")
        ck('mode 3 未提交', 3 not in {c['mode_seq'] for c in a.commits}, f'mode={a.mode_seq}')
    if 'N3' in sel:
        print('\nN3  卸載中底盤偏離 25 mm')
        w, st, a, xm = scenario('n3', 3, unload=True, base_dev=0.025)
        s, o = outc(w)
        xu = st['xu']
        ck('執行端 unload FAILED base_dev', xu['state'] == 'FAILED' and xu['why'] == 'base_dev', f"{xu['state']} {xu['why']}")
        ck('mode 2 未請求', 2 not in st['reqs'], str(st['reqs']))
        ck('求解端 FAIL（base_dev_in_unload 或世界座標 TCP 速度 unload_transient:tcp_speed，先到者）',
           o.get('state') == 'FAIL' and o.get('fail_why') in ('base_dev_in_unload', 'unload_transient:tcp_speed'), f"{o.get('state')} {o.get('fail_why')}")
    if 'N4' in sel:
        print('\nN4  權威狀態不帶 t_settling')
        w, st, a, xm = scenario('n4', 4, unload=True, hide_settling=True)
        s, o = outc(w)
        ck('求解端 unload_timeout（不以斜坡期低速宣告）', o.get('state') == 'FAIL' and o.get('fail_why') == 'unload_timeout', f"{o.get('state')} {o.get('fail_why')}")
        ck('mode 2 未請求', 2 not in st['reqs'], str(st['reqs']))
    if 'N5' in sel:
        print('\nN5  卸載＋張爪後手臂回彈＋mode 2 後權威狀態延遲 0.1 s（延遲得知、重放提交後樣本）')
        w, st, a, xm = scenario('n5', 5, unload=True, door_rate=1.0 * DEG, state_gap=(0.0, 0.1))
        s, o = outc(w)
        cs = {c['mode_seq']: c for c in a.commits}
        ck('三次提交皆確認、交接', sorted(cs) == [1, 2, 3] and o.get('state') == 'HANDED_OVER', f"{sorted(cs)} {o.get('state')} {o.get('fail_why')}")
        ch = ((o.get('checks') or {}).get('2') or {})
        ck('mode 2 得知晚於提交、重放提交後樣本', ch.get('t_known', 0) >= ch.get('t_commit', 1e9) + 0.1 - 1e-6 and (ch.get('replay_n') or 0) > 0,
           str({k: ch.get(k) for k in ('t_commit', 't_known', 'replay_n', 'replay_worst')}))
    print(f'\n{"全部通過" if not FAIL else "失敗：" + "、".join(FAIL)}')
    return 0 if not FAIL else 1


if __name__ == '__main__':
    sys.exit(main())
