#!/usr/bin/env python3
"""ADM18 換抓假世界 ROS 測試（無 Isaac；Codex r50–r52）。結構沿用 test_k3b_adm16_ros_nodever.py（真求解端 ADM18＋轉接器＋樁端點；測試行程內為執行端：
RegraspAuthority＋W2Exec2＋真 CmdChainE2＋EnvIntake；gate3＝door_regrasp_gates2.Gate3Release 吃合成手指資料；gate4／5／6 以時間給證據（本測試只驗機制）；
同步狀態不積分底盤（替身，固定位姿）；不卸載）。
請求排程：HELD＋1 s → 1；1 確認＋0.5 s → 2；gate3 准入且求解端 release_settled → 3；3 提交＋3 s → 4；4 提交＋3 s → 5；5 提交＋2 s → 6。
目標：mode 2 提交起固定 T_open；mode 3 起沿 −x 退 0.02 m；mode 4 再退至 0.05 m；mode 5 於 2 s 內回到 T_open（替身關節不積分，回到手臂所在使命令收斂，
  對應真流程 gate6 要求靜止 0.5 s）；mode 6 保持；coord：mode 3／4 v_body (0.01, 0, 0)、mode 5 為 0（B1 固定）；請求 6 於 5 提交＋4 s。
  G0 正常：六次提交皆 applied_confirmed、無中止；運動候選 3／4／5 皆以首筆承接約束另解（log 有 rg_cand.first_step）；
     各運動模式提交步首筆套用值與前一步套用值差 ≤ a_max×0.01；交接後運動模式底盤自由週期 > 0；mode 6 保持形週期 > 0、提交後套用手臂皆 0。
  G1 mode 2 提交後 1.0 s 起停發 /door/applied_src：運動候選 3 不發（skip 原因 applied_*），mode 3 未提交。
替身無門、無接觸物理：只驗機制。**合成資料，不是實跑結果。**

    python3 evaluation/test_regrasp_adm18_ros.py [G0 G1]
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
TMP = os.environ.get('RG_TMP', '/tmp/claude-1000/regrasp_adm18_ros')
NODE_VER = 18
DOM0 = int(os.environ.get('RG_DOMAIN', '150'))
RUN = 'regrasp_adm18_ros'
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
    import door_regrasp_mode1 as RGM
    return {'run_id': RUN, 'req_seq': seq, 'from_mode': RGM.MODE_OF[ms - 1], 'to_mode': RGM.MODE_OF[ms], 'mode_seq': ms}


def scenario(tag, dom, retreat=True, finger_final=1.0, arm_rate_max=1.0, max_mode=3, run_s=40.0, node_ver=None, door_rate=0.0, state_gap=None,
             unload=False, base_dev=0.003, cross_trip=None, arm_drift=0.0, hide_settling=False, asrc_stop_after_open=None):
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
    import door_k3m_applied_src as K3MSRC
    import door_w2_exec2 as WX2
    import door_k3b_unload2 as KU
    import door_regrasp_gates2 as GT2
    import door_regrasp_mode1 as RGM
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
           'tgt': peer.create_publisher(Float64MultiArray, '/drawer/tcp_target', 10),
           'coord': peer.create_publisher(Float64MultiArray, '/door/regrasp_coord', 10)}
    xm = GT2.Gate3Release()
    xu = KU.ExecUnloadMonitor()

    def _ct(k):
        return next((c['t'] for c in auth.commits if c['mode_seq'] == k and c.get('applied_confirmed')), None)

    def _tg(k, dly):
        return lambda step: (_ct(k - 1) is not None and step * 0.01 >= _ct(k - 1) + dly, 'time_evidence')
    auth = RGM.RegraspAuthority(RUN, run_id_num=ENV.run_id_num(RUN), stage_id=ENV.ST_ADAPTER, dt=0.01, arm_rate_max=arm_rate_max,
                                gates={3: lambda step: xm.confirmed_fresh(step), 4: _tg(4, 2.5), 5: _tg(5, 2.5), 6: _tg(6, 1.5)})
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
              '--w2-mode-run-id', RUN, '--out', out, '--regrasp', '--coord-topic', '/door/regrasp_coord']
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
            for _k, _d in ((4, 3.0), (5, 3.0), (6, 4.0)):
                if _k - 1 in cs and cs[_k - 1]['applied_confirmed'] and t >= cs[_k - 1]['t'] + _d and _k not in st['reqs'] and _k - 1 in st['reqs']:
                    st['req_q'].append(req(_k, _k))
                    st['reqs'][_k] = t
            # 退離目標（mode 2 提交後）
            if st['t_open'] is not None and t >= st['last_tgt'] + 0.05:
                st['last_tgt'] = t
                _c3, _c4, _c5 = _ct(3), _ct(4), _ct(5)
                d = 0.0 if _c3 is None else min(0.02, 0.02 * max(0.0, t - _c3))
                if _c4 is not None:
                    d = 0.02 + min(0.03, 0.01 * max(0.0, t - _c4))
                if _c5 is not None:
                    d = 0.05 - min(0.05, 0.025 * max(0.0, t - _c5))                    # mode 5：2 s 內回到 T_open（替身手臂所在）
                if auth.mode_seq in (3, 4, 5):
                    _v = 0.01 if auth.mode_seq in (3, 4) else 0.0                       # mode 5：底盤參考靜止（同真任務端 B1 固定）
                    pub['coord'].publish(Float64MultiArray(data=[1.0, 1.0, _v, 0.0, 0.0, 0.0] + [float('nan')] * 8))
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
                if asrc_stop_after_open is None or st['t_open'] is None or tk < st['t_open'] + asrc_stop_after_open:
                    pub['asrc'].publish(Float64MultiArray(data=[float(k), tk, float(row['src']), float(row['chain_mode'])] + u9))
                st['rows'].append(row)
                if st['k_open'] is not None and k > st['k_open']:                   # 合成手指資料（執行端逐物理步）
                    tr = tk - st['t_open']
                    f = 0.2 + 0.75 * tr / 0.31 if tr < 0.31 else 0.95 + (tr - 0.31) * 0.2
                    fq = FO * min(finger_final, f)
                    pf = max(0.0, 12.0 * (1 - tr / 0.04))
                    xm.step(k, tk, [fq, fq], [pf, pf], True)
                    st.setdefault('xm_fresh', []).append((k, xm.confirmed_fresh(k + 1)))
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


def main() -> int:
    sel = sys.argv[1:] or ['G0', 'G1']
    print(f'暫存 {TMP}　domain {DOM0}+\n')
    if 'G0' in sel:
        print('G0  正常 1→6')
        w, st, a, xm = scenario('g0', 0, run_s=60.0)
        cs = {c['mode_seq']: c for c in a.commits}
        ck('六次提交皆 applied_confirmed、無中止', sorted(cs) == [1, 2, 3, 4, 5, 6] and all(c['applied_confirmed'] is True for c in cs.values()) and a.aborted is None,
           str({k: (v['step'], v['applied_confirmed']) for k, v in cs.items()}) + f" aborted={a.aborted} rej={a.last_reject}")
        rg = ((w or {}).get('stats') or {}).get('regrasp') or {}
        log = (w or {}).get('log') or []
        fs = {k: [e for e in log if (e.get('rg_cand') or {}).get('k') == k and (e['rg_cand'] or {}).get('first_step')] for k in (3, 4, 5)}
        ck('運動候選 3／4／5 皆以首筆承接約束另解', all(fs[k] for k in (3, 4, 5)), str({k: len(v) for k, v in fs.items()}) + f" skip={rg.get('skip_why')}")
        rows = {r['step']: r for r in st['rows']}
        amax = np.array([0.5, 0.5, 2.0] + [19.984] * 6)
        jumps = {}
        for k in (3, 4, 5):
            c = cs.get(k)
            if c is None:
                continue
            r1, r0 = rows.get(c['step']), rows.get(c['step'] - 1)
            if r1 and r0:
                jumps[k] = float(np.max(np.abs(np.asarray(r1['u'], float) - np.asarray(r0['u'], float)) / (amax * 0.01)))
        ck('運動模式提交步首筆套用與前一步差 ≤ a_max×0.01', len(jumps) == 3 and all(v <= 1.0 + 1e-6 for v in jumps.values()), str(jumps))
        ck('交接後運動模式底盤自由週期 > 0、mode 6 保持形週期 > 0', (rg.get('n_free_cycles') or 0) > 0 and (rg.get('n_hold6_cycles') or 0) > 0,
           f"free={rg.get('n_free_cycles')} hold6={rg.get('n_hold6_cycles')} anchor6={rg.get('anchor6')}")
        c6 = cs.get(6)
        after6 = [r for r in st['rows'] if c6 is not None and r['step'] >= c6['step'] and r['src'] >= 0]
        ck('mode 6 提交後套用手臂皆 0', bool(after6) and all(max(abs(x) for x in r['u'][3:]) <= 1e-6 for r in after6), f'n={len(after6)}')
    if 'G1' in sel:
        print('\nG1  mode 2 提交後 1.0 s 起停發 applied_src')
        w, st, a, xm = scenario('g1', 1, run_s=40.0, asrc_stop_after_open=1.0)
        rg = ((w or {}).get('stats') or {}).get('regrasp') or {}
        whys = [x.get('why', '') for x in (rg.get('skip_why') or [])]
        ck('運動候選 3 不發、原因 applied_*', any(str(x).startswith('applied_') for x in whys) and not any(
            (e.get('rg_cand') or {}).get('first_step') for e in ((w or {}).get('log') or []) if (e.get('rg_cand') or {}).get('k') == 3),
            str(whys[:3]))
        ck('mode 3 未提交', 3 not in {c['mode_seq'] for c in a.commits}, f'mode={a.mode_seq} rej={a.last_reject}')
    print(f'\n{"全部通過" if not FAIL else "失敗：" + "、".join(FAIL)}')
    return 0 if not FAIL else 1


if __name__ == '__main__':
    sys.exit(main())
