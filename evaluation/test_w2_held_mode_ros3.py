#!/usr/bin/env python3
"""W2 HELD／模式權威 假世界 ROS 測試 r3（無 Isaac；Codex 20261025_214144）：test_w2_held_mode_ros.py（凍結 w2a r2）的版本化複本。
r3 差異：執行端角色改為 door_w2_exec.W2Exec＋**真正的 CmdChainE2＋EnvIntake＋applied_source**＋door_w2_mode3.ModeAuthority3；
  逐步 select → 進鏈（鏈接收時刻＝來源 stamp）→ chain.step → applied_source／限幅後九軸 → confirm_step；/door/applied_src 回報鏈的實際值；
  請求先由權威登記（pending）才轉送 /door/w2_mode_req 給求解端（ev6 同一次序）；判定器 door_w2_check_mode3（C0–C11）。
  假世界手臂限位取 ±100 rad：替身的 /door/sync_state 手臂角恆為 0，求解端在自由段持續送關節速度，鏈的設定點積分會在約 6 s 越過
  ±3 rad 而失效（首次實跑即如此）——屬替身無手臂回授的假象；本測試不驗限位，限位保護由 ev5／ev6 既有 URDF 限位負責。
  M9（新增）提交步 K2 過濾擋下候選 ⇒ 鏈仍套用 mode 0 舊快照 ⇒ commit_not_applied、不回切。
以下為 r2 原說明（情境不變）：
真節點：wgmpc_sp_stub_endpoint（替身執行端：時鐘、關節、odom、設定點）、dl3_wgmpc_wg2_adm11_node（--w2-held --w2-mode-run-id）、
arm_vel_adapter_w2（--cmd-env-v3；關節順序由本測試的 w2_consumer 參數節點提供）。
執行端角色＝本測試同程序的 door_w2_mode.ModeAuthority（之後 ev6 共用同一類別）：只吃 adapter 的 v3 輸出，逐物理步決定套用、發 /door/applied_src
與 /door/w2_mode_state。測試模式 RELEASE_PREP_TEST 與 HELD 命令相同（不張爪、不退離）。
  M0 參數相依：--w2-held 缺 --pause-active-hold／arm-win、--w2-mode-run-id 缺 --w2-held／--cmd-env ⇒ exit 2。
  M1 正常：HELD 後 1 s 請求 ⇒ 恰一次提交；door_w2_check_mode C1–C7 全過；HELD 監督器持續發布（observation_passed、仍在 HELD）。
  M2 權威狀態遺失：提交後 1 s 內不發 /door/w2_mode_state（亦無 ack）⇒ 求解端仍由之後的狀態得知提交（committed＝1），證據核對全過。
  M3 延遲舊命令（r2）：扣住請求後到達的一筆 mode 0 命令，提交後 0.3 s 才送達 ⇒ stale、計數、不套用（C5）；提交後每步皆為 mode 1。
  M4 重送／衝突：同請求重送 ⇒ repeat；另一 req_seq ⇒ nack conflict；仍恰一次提交。
  M5 切換前失效：在 HELD 之前送請求 ⇒ 求解端不準備（ignored_not_held）⇒ candidate_timeout、保持 mode 0；HELD 有界超時結束（不算完成）。
  M6 切換後失效：提交 0.5 s 後停止轉送新模式命令 ⇒ post_commit_source_lost 中止、不回切（mode_seq 仍 1）。
  M8 候選延遲 0.5 s（超過命令逾時 0.2 s、未超過候選期限 1 s）⇒ 期間舊模式持續套用、無逾時步，延遲後才提交，C0–C7 全過（r2）。
  R0 ADM11 不帶 --w2-*：比照 ADM9（v2 直接作套用來源）⇒ 暫停照常 DONE（觀察 10 s、停發），stats.w2 為 None，/wgmpc/cmd_env3 無任何訊息（r2）。
  M7 中止優先：請求後立即中止 ⇒ 不提交、無 mode 1 命令被套用（求解端可能在發出候選前即由權威狀態撤回；若已發出則判 stale）、求解端撤回（reverted）。

    python3 evaluation/test_w2_held_mode_ros.py [M1 M2 …]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
TMP = os.environ.get('W2M_TMP', '/tmp/claude-1000/w2_mode_ros3')
DOM0 = int(os.environ.get('W2M_DOMAIN', '175'))
RUN = 'w2_mode_ros3'
FAIL = []
ARM = [f'joint{i}' for i in range(1, 7)]
# 啟動時序（r2）：連續多情境時 DDS 探索較慢，替身就緒前須已收到 /park/gate（否則 ADM9 既有保護 park_mode_refused 拒絕，屬測試時序，與 W2 無關）
READY_S, T_FREE, T_PAUSE = 3.0, 4.0, 6.0


def ck(n, ok, d=''):
    print(f'  {n:62s} {"ok" if ok else "**錯**"}  {d}', flush=True)
    if not ok:
        FAIL.append(n)


def req(seq, ms=1):
    import door_w2_mode as W2M
    return {'run_id': RUN, 'req_seq': seq, 'from_mode': W2M.MODE_HELD, 'to_mode': W2M.MODE_RELEASE_TEST, 'mode_seq': ms}


def scenario(tag, dom, run_s=32.0, req_after_held=1.0, early_req=False, state_blackout=0.0, inject_stale=False,
             repeat_conflict=False, drop_after_commit=None, abort_after_req=False, held_max_s=60.0, cand_delay=0.0, filter_at_commit=False):
    os.makedirs(TMP, exist_ok=True)
    out = os.path.join(TMP, f'{tag}.json')
    if os.path.exists(out):
        os.remove(out)                                            # 本測試自己的暫存輸出
    env = dict(os.environ, ROS_DOMAIN_ID=str(DOM0 + dom), PYTHONUNBUFFERED='1')
    os.environ['ROS_DOMAIN_ID'] = str(DOM0 + dom)
    import rclpy
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from rosgraph_msgs.msg import Clock
    from std_msgs.msg import Float64MultiArray, String
    import cmd_envelope_v3 as E3
    import door_k3m_applied_src as K3MSRC
    import door_w2_exec as WX
    import door_w2_mode3 as M3
    import wgmpc_cmd_envelope as ENV
    from wb_cmd_chain_e2 import CmdChainE2
    from wb_wheel_limit import WheelLimitConfig
    rclpy.init()
    peer = rclpy.create_node(f'w2_peer_{tag}')
    cons = rclpy.create_node('w2_consumer')
    cons.declare_parameter('joints', ARM)
    lat = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    pub = {'gate': peer.create_publisher(String, '/park/gate', 1),
           'free': peer.create_publisher(String, '/door/base_mode_ack', lat),
           'pause': peer.create_publisher(String, '/door/pause_ack_exec', lat),
           'req': peer.create_publisher(String, '/door/w2_mode_req', lat),
           'state': peer.create_publisher(String, '/door/w2_mode_state', 50),
           'sync': peer.create_publisher(Float64MultiArray, '/door/sync_state', 100),
           'asrc': peer.create_publisher(Float64MultiArray, '/door/applied_src', 200)}
    auth = M3.ModeAuthority3(RUN, run_id_num=ENV.run_id_num(RUN), stage_id=ENV.ST_ADAPTER, dt=0.01)   # r3：加速度框以物理步計

    def low_speed_bound(vx, vy, wz):                              # 與 ev5 預設相同（--base-lin-max 0.05、--base-ang-max 0.20）
        lin = (vx * vx + vy * vy) ** 0.5
        return (False, f'lin {lin:.4f}') if lin > 0.05 else ((False, f'ang {wz:.4f}') if abs(wz) > 0.20 else (True, None))
    chain = CmdChainE2(max_cmd_age_s=0.2, arm_rate_max=1.0, wheel_ok=low_speed_bound, joint_lower=(-100.0,) * 6, joint_upper=(100.0,) * 6,
                       joint_margin=0.05, mode='solver_drawer', wheel_cfg=WheelLimitConfig(arm_rate_max=1.0, dt_max=0.05),
                       base_accel_max_lin=0.5, base_accel_max_ang=2.0)
    intake = K3MSRC.EnvIntake(run_id_num=ENV.run_id_num(RUN), stage_id=ENV.ST_ADAPTER, kinds=(ENV.K_NORMAL,))
    filt_state = {'n_blocked': 0}

    def k2_filter(u):                                             # M9：提交步擋下候選（其餘原樣）
        if filter_at_commit and auth.commit is not None and auth.commit['step'] == step_cur[0] and filt_state['n_blocked'] == 0:
            filt_state['n_blocked'] += 1
            return None
        return u
    step_cur = [None]
    wx = WX.W2Exec(auth, intake, filt=k2_filter)
    st = {'t': 0.0, 'status': None, 't_held': None, 't_obs': None, 'free_sent': False, 'pause_sent': False, 'last_step': None,
          'last_sync': -1.0, 'latest': None, 'u_applied': None, 'req_sent': False, 't_req': None, 'rows': [], 'last_pre_commit_env': None,
          'stale_injected': [], 't_commit': None, 'repeat_done': False, 'aborted_sent': False, 'intake': {}, 'n_cand_dropped': 0}
    peer.create_subscription(Clock, '/clock', lambda m: st.__setitem__('t', m.clock.sec + m.clock.nanosec * 1e-9), 10)

    def on_status(m):
        s = json.loads(m.data)
        st['status'] = s
        o = s.get('outcome') or {}
        if o.get('state') == 'HELD' and st['t_held'] is None:
            st['t_held'] = st['t']
            st['t_obs'] = (o.get('transitions') or {}).get('OBS_PASSED')
    peer.create_subscription(String, '/wgmpc/pause_status', on_status, lat)

    def step_now():
        """下一個尚未處理的物理步（本測試每輪批次處理多步；到達的命令與請求只能作用於尚未處理的步，提交步＝該步）。"""
        return st['last_step'] + 1 if st['last_step'] is not None else int(round(st['t'] * 100))

    def on_env3(m):
        try:
            e = E3.decode3(m.data)
        except ValueError:
            return
        if drop_after_commit is not None and st['t_commit'] is not None and st['t'] >= st['t_commit'] + drop_after_commit:
            return                                                    # M6：之後的新模式命令不再轉送（來源中斷）
        if cand_delay and st['t_req'] is not None and e['mode_seq'] == 1 and st['t'] < st['t_req'] + cand_delay:
            st['n_cand_dropped'] += 1
            return                                                    # M8：候選延遲（超過命令逾時、未超過候選期限）
        if inject_stale and st['t_req'] is not None and e['mode_seq'] == 0 and st.get('held_old') is None and st['t_commit'] is None:
            st['held_old'] = e                                        # M3：扣住一筆請求後的舊模式命令，提交後才送達（延遲舊命令）
            return
        act, info = wx.on_env3(e, st['t'], time.monotonic())        # r3：入口驗證與緩衝；接收牆鐘另存
        st['intake'][act] = st['intake'].get(act, 0) + 1
        if act == 'buffered_current' and e['mode_seq'] == 0:
            st['last_pre_commit_env'] = e
    peer.create_subscription(Float64MultiArray, E3.TOPIC3[ENV.ST_ADAPTER], on_env3, 50)
    stub = subprocess.Popen([sys.executable, '-u', os.path.join(HERE, 'wgmpc_sp_stub_endpoint.py'), '--run-s', str(run_s), '--ready-after', str(READY_S)],
                            cwd=WS, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    node = subprocess.Popen([sys.executable, '-u', os.path.join(HERE, 'dl3_wgmpc_wg2_adm11_node.py'),
                             '--arm-model', 'setpoint', '--N', '5', '--rate', '20', '--target-offset', '0.05', '0.02', '0.01',
                             '--duration-s', str(run_s - 3.0), '--u-prev-policy', 'strict', '--assume-initial-rest',
                             '--park-hold', '--continuous', '--mode-switch-run-id', RUN, '--cmd-env', '--run-id', RUN, '--pause',
                             '--pause-arm-zero', '--pause-active-hold', '--pause-arm-win-s', '0.5',
                             '--w2-held', '--w2-held-max-s', str(held_max_s), '--w2-mode-run-id', RUN, '--out', out],
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
            want_req = (early_req and st['pause_sent'] and t >= T_PAUSE + 1.5) or (not early_req and st['t_held'] is not None and t >= st['t_held'] + req_after_held)
            if want_req and not st['req_sent']:
                r = req(1)
                kind, _ = auth.on_request(r, step_now(), t)
                if kind == 'pending':                                     # r3：權威登記後才轉送求解端
                    pub['req'].publish(String(data=json.dumps(r)))
                st['req_sent'], st['t_req'] = True, t
                if abort_after_req:
                    auth.abort('test_abort', step_now(), t)
            if repeat_conflict and st['req_sent'] and not st['repeat_done'] and t >= st['t_req'] + 0.05:
                st['rep_kind'] = auth.on_request(req(1), step_now(), t)[0]
                st['conf'] = auth.on_request(req(2), step_now(), t)
                st['repeat_done'] = True
            if inject_stale and st['t_commit'] is not None and not st['stale_injected'] and t >= st['t_commit'] + 0.3 \
                    and st.get('held_old') is not None:
                e = st['held_old']
                act, info = wx.on_env3(e, t, time.monotonic())
                st['stale_injected'].append(int(e['source_seq']))
                st['stale_act'] = act
            while st['last_sync'] + 0.02 <= t + 1e-9:
                ts = round(st['last_sync'] + 0.02, 4) if st['last_sync'] >= 0 else round(t, 4)
                st['last_sync'] = ts
                pub['sync'].publish(Float64MultiArray(data=[ts, float(int(round(ts * 100))), 0.01] + [0.0] * 16 + [0.6, -0.53, 1.57]
                                                      + [0.0] * 6 + [0.0]))
            k_now = int(round(t * 100))
            k0 = st['last_step'] + 1 if st['last_step'] is not None else k_now
            for k in range(k0, k_now + 1):
                tk = k * 0.01
                step_cur[0] = k
                sel = wx.pre_step(k, tk, chain)                           # r3：選定 → 進真鏈
                if sel['event'] and sel['event'].get('kind') == 'commit':
                    st['t_commit'] = tk
                chain.step(tk, 0.01, [0.0] * 6)
                u9 = [float(x) for x in chain.u_prev] if chain.u_prev is not None else [0.0] * 9
                row = wx.post_step(k, tk, chain, True, u9, sel)           # r3：鏈的實際來源與限幅後九軸 → confirm_step
                pub['asrc'].publish(Float64MultiArray(data=[float(k), tk, float(row['src']), float(row['chain_mode'])] + u9))
                st['rows'].append(row)
                blackout = state_blackout and st['t_commit'] is not None and tk < st['t_commit'] + state_blackout
                if k % 2 == 0 and not blackout:
                    pub['state'].publish(String(data=json.dumps(auth.state_msg(k, tk), default=float)))
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
        print((adp.stdout.read() if adp.stdout else '')[-1500:])
    st['chain'] = {'n_recv': chain.n_recv, 'n_rejected': chain.n_rejected, 'fail': chain.fail, 'n_timeout_decel': chain.n_timeout_decel,
                   'n_base_rate_capped': chain.n_base_rate_capped, 'intake_violations': intake.violations[:20], 'feed': wx.feed,
                   'filt_blocked': filt_state['n_blocked']}
    return w, st, auth


def scenario_off(tag, dom, run_s=26.0):
    """ADM11 不帶 --w2-*：比照 ADM9 假世界（v2 封裝直接作套用來源；無 adapter）；同時計數 /wgmpc/cmd_env3 是否有任何訊息。"""
    os.makedirs(TMP, exist_ok=True)
    out = os.path.join(TMP, f'{tag}.json')
    if os.path.exists(out):
        os.remove(out)
    env = dict(os.environ, ROS_DOMAIN_ID=str(DOM0 + dom), PYTHONUNBUFFERED='1')
    os.environ['ROS_DOMAIN_ID'] = str(DOM0 + dom)
    import rclpy
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from rosgraph_msgs.msg import Clock
    from std_msgs.msg import Float64MultiArray, String
    import cmd_envelope_v3 as E3
    import wgmpc_cmd_envelope as ENV
    rclpy.init()
    peer = rclpy.create_node(f'w2_off_{tag}')
    lat = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    pub = {'gate': peer.create_publisher(String, '/park/gate', 1), 'free': peer.create_publisher(String, '/door/base_mode_ack', lat),
           'pause': peer.create_publisher(String, '/door/pause_ack_exec', lat),
           'sync': peer.create_publisher(Float64MultiArray, '/door/sync_state', 100),
           'asrc': peer.create_publisher(Float64MultiArray, '/door/applied_src', 200)}
    st = {'t': 0.0, 'env': None, 'env_t': None, 'free_sent': False, 'pause_sent': False, 'last_step': None, 'last_sync': -1.0, 'n3': 0}
    peer.create_subscription(Clock, '/clock', lambda m: st.__setitem__('t', m.clock.sec + m.clock.nanosec * 1e-9), 10)

    def on_env(m):
        try:
            st['env'], st['env_t'] = ENV.decode(m.data), st['t']
        except ValueError:
            pass
    peer.create_subscription(Float64MultiArray, ENV.TOPIC[ENV.ST_SOLVER], on_env, 50)
    peer.create_subscription(Float64MultiArray, E3.TOPIC3[ENV.ST_SOLVER], lambda m: st.__setitem__('n3', st['n3'] + 1), 50)
    stub = subprocess.Popen([sys.executable, '-u', os.path.join(HERE, 'wgmpc_sp_stub_endpoint.py'), '--run-s', str(run_s), '--ready-after', str(READY_S)],
                            cwd=WS, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    node = subprocess.Popen([sys.executable, '-u', os.path.join(HERE, 'dl3_wgmpc_wg2_adm11_node.py'),
                             '--arm-model', 'setpoint', '--N', '5', '--rate', '20', '--target-offset', '0.05', '0.02', '0.01',
                             '--duration-s', str(run_s - 3.0), '--u-prev-policy', 'strict', '--assume-initial-rest',
                             '--park-hold', '--continuous', '--mode-switch-run-id', RUN, '--cmd-env', '--run-id', RUN, '--pause',
                             '--pause-arm-zero', '--pause-active-hold', '--pause-arm-win-s', '0.5', '--out', out],
                            cwd=WS, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    t0 = time.monotonic()
    try:
        while node.poll() is None and time.monotonic() - t0 < run_s + 60:
            rclpy.spin_once(peer, timeout_sec=0.005)
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
            while st['last_sync'] + 0.02 <= t + 1e-9:
                ts = round(st['last_sync'] + 0.02, 4) if st['last_sync'] >= 0 else round(t, 4)
                st['last_sync'] = ts
                pub['sync'].publish(Float64MultiArray(data=[ts, float(int(round(ts * 100))), 0.01] + [0.0] * 16 + [0.6, -0.53, 1.57]
                                                      + [0.0] * 6 + [0.0]))
            k_now = int(round(t * 100))
            k0 = st['last_step'] + 1 if st['last_step'] is not None else k_now
            for k in range(k0, k_now + 1):
                tk = k * 0.01
                e = st['env']
                if e is not None and st['env_t'] is not None and tk - st['env_t'] <= 0.2 + 1e-9:
                    src, mode, u = int(e['source_seq']), 0, [float(x) for x in list(e['u'])[:9]]
                else:
                    src, mode, u = -1, 1, [0.0] * 9
                pub['asrc'].publish(Float64MultiArray(data=[float(k), tk, float(src), float(mode)] + u))
            st['last_step'] = k_now
        if node.poll() is None:
            node.terminate()
            node.wait(timeout=10)
    finally:
        stub.terminate()
        try:
            stub.wait(timeout=5)
        except subprocess.TimeoutExpired:
            stub.kill()
        peer.destroy_node()
        rclpy.shutdown()
    return (json.load(open(out)) if os.path.exists(out) else None), st['n3']


def held_outcome(w):
    return (((w.get('stats') or {}).get('k3m_pause') or {}).get('supervisor_outcome')) or {}


def main() -> int:
    import door_w2_check_mode3 as CM3
    sel = sys.argv[1:] or ['M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'R0']
    print(f'暫存 {TMP}　domain {DOM0}+\n')
    if 'M0' in sel:
        print('M0  參數相依')
        for args, msg in ((['--w2-held'], '需要 --pause-active-hold'), (['--w2-mode-run-id', 'x'], '需要 --w2-held 與 --cmd-env')):
            p = subprocess.run([sys.executable, os.path.join(HERE, 'dl3_wgmpc_wg2_adm11_node.py')] + args, cwd=WS, capture_output=True, text=True, timeout=60)
            ck(f'{args[0]} 缺前置 ⇒ 參數錯誤（exit 2）', p.returncode == 2 and msg in p.stderr, p.stderr[-100:])
    if 'M1' in sel:
        print('M1  正常')
        w, st, a = scenario('m1', 0)
        o = held_outcome(w or {})
        ck('M1 有產出且曾進 HELD', w is not None and st['t_held'] is not None, f"t_held={st['t_held']}")
        if w is not None and st['t_held'] is not None:
            r = CM3.check(st['rows'], a.stats(), a.applied_rows, (w.get('stats') or {}).get('w2'),
                         st['t_req'], st['t_obs'])
            ck('C0–C11 逐步證據全過（真鏈）', r['ok'], f"{r['why']} k={r.get('k')} first={r.get('first_source_seq')}")
            ck('鏈無失效、無入口違規、提交步實際首筆＝first_source_seq', st['chain']['fail'] is None and not st['chain']['intake_violations']
               and a.commit['applied_confirmed'] is True, str(st['chain']))
            ck('觀察通過且結束時仍在 HELD（持續發布）', o.get('observation_passed') is True and o.get('state') == 'HELD', str(o.get('state')))
    if 'M2' in sel:
        print('\nM2  權威狀態遺失 1 s')
        w, st, a = scenario('m2', 1, state_blackout=1.0)
        if w is not None and a.commit:
            r = CM3.check(st['rows'], a.stats(), a.applied_rows, (w.get('stats') or {}).get('w2'),
                         st['t_req'], st['t_obs'])
            ck('求解端仍得知提交、證據全過', r['ok'] and (w['stats']['w2'] or {}).get('committed') == 1, str(r['why']))
        else:
            ck('M2 有提交', False)
    if 'M3' in sel:
        print('\nM3  延遲舊命令')
        w, st, a = scenario('m3', 2, inject_stale=True)
        ck('被扣住的請求後舊模式命令於提交後送達 ⇒ stale、計數', st.get('stale_act') == 'stale' and a.n_stale >= 1,
           f"{st.get('stale_act')} n_stale={a.n_stale} intake={st['intake']}")
        if w is not None and a.commit:
            r = CM3.check(st['rows'], a.stats(), a.applied_rows, (w.get('stats') or {}).get('w2'),
                         st['t_req'], st['t_obs'], stale_injected=st['stale_injected'])
            ck('C5 舊命令未套用、證據全過', r['ok'], str(r['why']))
            after = [x for x in st['rows'] if x['step'] >= a.commit['step'] and x['src'] >= 0]
            ck('提交後每步套用皆為 mode 1', bool(after) and all(x['mode_seq'] == 1 for x in after), str(len(after)))
    if 'M4' in sel:
        print('\nM4  重送／衝突')
        w, st, a = scenario('m4', 3, repeat_conflict=True)
        conf = st.get('conf') or (None, {})
        ck('重送 ⇒ repeat、另一請求 ⇒ conflict', st.get('rep_kind') == 'repeat' and conf[0] == 'reject'
           and str(conf[1].get('why', '')).startswith('conflict'), f"{st.get('rep_kind')} {conf}")
        ck('恰一次提交', sum(1 for e in a.events if e.get('kind') == 'commit') == 1)
    if 'M5' in sel:
        print('\nM5  切換前失效（HELD 前請求）＋HELD 有界超時')
        w, st, a = scenario('m5', 4, early_req=True, held_max_s=6.0, run_s=30.0)
        o = held_outcome(w or {})
        ev = ((w or {}).get('stats') or {}).get('w2') or {}
        ck('求解端忽略（ignored_not_held）', any(e.get('kind') == 'ignored_not_held' for e in ev.get('events') or []), str(ev.get('events'))[:160])
        ck('候選逾時拒絕、保持 mode 0、無提交', a.commit is None and a.mode_seq == 0 and (a.last_reject or {}).get('why') == 'candidate_timeout',
           str(a.last_reject))
        ck('HELD 有界超時結束（held_end＝timeout）', o.get('held_end') == 'timeout' and o.get('transitions', {}).get('HELD_TIMEOUT_END') is not None,
           str(o.get('held_end')))
        held = [r for r in st['rows'] if st['t_obs'] is not None and st['t_obs'] + 0.3 <= r['t'] < (o.get('transitions') or {}).get('HELD_TIMEOUT_END', 0)]
        ck('HELD 期間持續發布（無逾時步）', bool(held) and all(r['chain_mode'] == 0 for r in held), str(len(held)))
    if 'M6' in sel:
        print('\nM6  切換後失效')
        w, st, a = scenario('m6', 5, drop_after_commit=0.5)
        ck('post_commit_source_lost 中止', (a.aborted or {}).get('why') == 'post_commit_source_lost', str(a.aborted))
        ck('不回切（mode_seq 仍 1）', a.mode_seq == 1 and a.commit is not None)
        after = [r for r in st['rows'] if a.aborted and r['step'] > a.aborted['step']]
        ck('中止後無新模式以外的套用（無回切）', all(r['src'] < 0 or r['mode_seq'] == 1 for r in after), str(len(after)))
    if 'M7' in sel:
        print('\nM7  中止優先')
        w, st, a = scenario('m7', 6, abort_after_req=True)
        ev = ((w or {}).get('stats') or {}).get('w2') or {}
        ck('不提交、mode 0', a.commit is None and a.mode_seq == 0, str(a.events)[:200])
        nbm = ev.get('n_by_mode') or {}
        ck('無 mode 1 命令被套用；求解端未發候選或候選判 stale', not any(r.get('mode_seq') == 1 for r in st['rows'])
           and ('1' not in nbm or a.n_stale >= 1), f"n_by_mode={nbm} n_stale={a.n_stale}")
        ck('求解端撤回（reverted）', any(e.get('kind') == 'reverted' for e in ev.get('events') or []), str(ev.get('events'))[:200])
    if 'M8' in sel:
        print('\nM8  候選延遲 0.5 s（> 命令逾時 0.2 s、< 候選期限 1 s）')
        w, st, a = scenario('m8', 7, cand_delay=0.5)
        pre = [x for x in st['rows'] if st['t_req'] is not None and st['t_req'] <= x['t'] < (st['t_commit'] or 1e9)]
        ck('候選被延遲期間舊模式持續套用（無逾時步）', bool(pre) and all(x['chain_mode'] == 0 and x['mode_seq'] == 0 for x in pre)
           and st['n_cand_dropped'] > 0, f"pre={len(pre)} dropped={st['n_cand_dropped']}")
        ck('延遲後提交（提交晚於請求＋0.5 s）', a.commit is not None and st['t_commit'] >= st['t_req'] + 0.5, f"{st['t_req']}→{st['t_commit']}")
        if w is not None and a.commit:
            r = CM3.check(st['rows'], a.stats(), a.applied_rows, (w.get('stats') or {}).get('w2'),
                         st['t_req'], st['t_obs'])
            ck('C0–C11 全過', r['ok'], str(r['why']))
    if 'M9' in sel:
        print('\nM9  提交步 K2 過濾擋下候選（真鏈）')
        w, st, a = scenario('m9', 9, filter_at_commit=True)
        ck('過濾生效一次', st['chain']['filt_blocked'] == 1, str(st['chain']))
        ck('commit_not_applied 中止、applied_confirmed False', a.commit is not None and a.commit['applied_confirmed'] is False
           and str((a.aborted or {}).get('why', '')).startswith('commit_not_applied:'), str(a.aborted))
        ck('不回切（mode_seq 仍 1）', a.mode_seq == 1)
        rk = next((x for x in st['rows'] if a.commit and x['step'] == a.commit['step']), {})
        ck('提交步實際套用為 mode 0 舊快照（由鏈讀出，非測試指定）', rk.get('mode_seq') == 0 and rk.get('src', -1) >= 0, str(rk)[:160])
        after = [x for x in st['rows'] if a.aborted and x['step'] > a.aborted['step']]
        ck('中止後權威不再選定任何命令', all(x['sel_src'] is None for x in after), str(len(after)))
    if 'R0' in sel:
        print('\nR0  ADM11 關閉 W2 旗標（ADM9 行為回歸）')
        w, n3 = scenario_off('r0', 8)
        o = held_outcome(w or {})
        ck('暫停照常 DONE（觀察 10 s、停發）', o.get('state') == 'DONE' and o.get('physical_stop_confirmed') is True, str(o.get('state')))
        ck('stats.w2 為 None、無 HELD 轉移', (w or {}).get('stats', {}).get('w2') is None and 'HELD' not in str(o.get('transitions')), '')
        ck('無任何 v3 封裝發布', n3 == 0, str(n3))
    print(f'\n{"全部通過" if not FAIL else "失敗：" + "、".join(FAIL)}')
    return 0 if not FAIL else 1


if __name__ == '__main__':
    sys.exit(main())
