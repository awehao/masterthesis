#!/usr/bin/env python3
"""W1b：求解端協同輸入（--coord-topic）的 ROS 假世界測試（ADM8；無 Isaac；沿用 ADM8 暫停假世界框架，不送暫停 ack）。
任務端（本測試）在 /door/wide_coord 發布 door_wide_guidance 的訊息：
  3.5–6.0 s 開啟（w_vref 1、本體 v_ref 固定）⇒ 求解紀錄 coord.src＝topic、w_vref 1、base_vref＝所送值；
  6.0–8.0 s 停發 ⇒ 0.5 s（--coord-max-age-s 預設）後 coord.src＝stale_off（退回啟動值 w_vref 0）；
  8.0–9.5 s 送關閉訊息（off_coord）⇒ coord.src＝topic、w_vref 0、base_vref None；
  3.5 s 前 ⇒ coord.src＝off。全程求解無失敗。

    python3 evaluation/test_dl3_wgmpc_wg2_adm8_coord_ros.py
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import door_wide_guidance as G                        # noqa: E402
TMP = os.environ.get('ADM8C_TMP', '/tmp/claude-1000/adm8_coord_ros')
DOM0 = int(os.environ.get('ADM8C_DOMAIN', '170'))
FAIL = []
RUN = 'adm8_coord_ros'
VREF = (0.005, 0.0, 0.01)


def ck(n, ok, d=''):
    print(f'  {n:62s} {"ok" if ok else "**錯**"}  {d}', flush=True)
    if not ok:
        FAIL.append(n)


def scenario(tag, run_s, door_rate=0.0, stop_at=None, inject_old_after_stop=False, dom=0):
    coord_on = [1.0, 1.0, *VREF, 0.0] + [float('nan')] * 6 + [float('nan'), float('nan')]
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
    import wgmpc_cmd_envelope as ENV
    rclpy.init()
    peer = rclpy.create_node(f'adm4_peer_{tag}')
    lat = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    pub = {'gate': peer.create_publisher(String, '/park/gate', 1),
           'free': peer.create_publisher(String, '/door/base_mode_ack', lat),
           'pause': peer.create_publisher(String, '/door/pause_ack_exec', lat),
           'stop': peer.create_publisher(String, '/wgmpc/stop', 1),
           'sync': peer.create_publisher(Float64MultiArray, '/door/sync_state', 100),
           'asrc': peer.create_publisher(Float64MultiArray, '/door/applied_src', 200),
           'coord': peer.create_publisher(Float64MultiArray, '/door/wide_coord', 1)}
    st = {'t': 0.0, 'env': None, 'env_t': None, 'acks': [], 'status': [], 'free_sent': False, 'pause_sent': False,
          'last_step': None, 'last_sync': -1.0, 'stopped_t': None, 'injected': False, 'stop_sent': False}
    peer.create_subscription(Clock, '/clock', lambda m: st.__setitem__('t', m.clock.sec + m.clock.nanosec * 1e-9), 10)
    peer.create_subscription(String, '/door/pause_ack_solver', lambda m: st['acks'].append(json.loads(m.data)), lat)
    peer.create_subscription(String, '/wgmpc/pause_status', lambda m: st['status'].append(json.loads(m.data)), lat)

    def on_env(m):
        try:
            st['env'], st['env_t'] = ENV.decode(m.data), st['t']
        except ValueError:
            pass
    peer.create_subscription(Float64MultiArray, ENV.TOPIC[ENV.ST_SOLVER], on_env, 50)
    stub = subprocess.Popen([sys.executable, '-u', os.path.join(HERE, 'wgmpc_sp_stub_endpoint.py'), '--run-s', str(run_s), '--ready-after', '1.0'],
                            cwd=WS, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    node = subprocess.Popen([sys.executable, '-u', os.path.join(HERE, 'dl3_wgmpc_wg2_adm8_node.py'),
                             '--arm-model', 'setpoint', '--N', '5', '--rate', '20', '--target-offset', '0.05', '0.02', '0.01',
                             '--duration-s', str(run_s - 3.0), '--u-prev-policy', 'strict', '--assume-initial-rest',
                             '--park-hold', '--continuous', '--mode-switch-run-id', RUN, '--cmd-env', '--run-id', RUN, '--pause', '--pause-arm-zero', '--pause-active-hold', '--pause-arm-win-s', '0.5', '--coord-topic', '/door/wide_coord',
                             '--out', out],
                            cwd=WS, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    t0 = time.monotonic()
    try:
        while node.poll() is None and time.monotonic() - t0 < run_s + 60:
            rclpy.spin_once(peer, timeout_sec=0.005)
            t = st['t']
            pub['gate'].publish(String(data=json.dumps({'passed': True, 'ok_now': True, 'anchor': [0.0, 0.0, 0.0]})))
            if not st['free_sent'] and t >= 3.0:
                pub['free'].publish(String(data=json.dumps({'run_id': RUN, 'seq': 1, 'mode': 'free', 'switch_step': int(round(t * 100)),
                                                            'switch_t': round(t, 4)})))
                st['free_sent'] = True
            if 3.5 <= t < 6.0:
                pub['coord'].publish(Float64MultiArray(data=coord_on))
            elif 8.0 <= t < 9.5:
                pub['coord'].publish(Float64MultiArray(data=G.off_coord()))
            if False:
                pub['pause'].publish(String(data=json.dumps({'run_id': RUN, 'seq': 2, 'mode': 'pause', 'switch_step': int(round(t * 100)),
                                                             'switch_t': round(t, 4)})))
                st['pause_sent'], st['t_pause'] = True, t
            # 同步狀態 50 Hz（靜止；P2 門持續轉）
            while st['last_sync'] + 0.02 <= t + 1e-9:
                ts = round(st['last_sync'] + 0.02, 4) if st['last_sync'] >= 0 else round(t, 4)
                st['last_sync'] = ts
                q = 0.01 + (door_rate * max(0.0, ts - st.get('t_pause', 1e9)) if st['pause_sent'] else 0.0)
                pub['sync'].publish(Float64MultiArray(data=[ts, float(int(round(ts * 100))), q] + [0.0] * 16 + [0.6, -0.53, 1.57]
                                                      + [0.0] * 6 + [door_rate]))
            # 逐物理步套用來源（連續、不跳步）
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
            last = st['status'][-1] if st['status'] else None
            if last and last['outcome']['state'] == 'STOPPED' and st['stopped_t'] is None:
                st['stopped_t'] = t
            if inject_old_after_stop and st['stopped_t'] is not None and not st['injected'] and t >= st['stopped_t'] + 1.0:
                k = int(round(t * 100)) + 1
                pub['asrc'].publish(Float64MultiArray(data=[float(k), k * 0.01, 3.0, 0.0, 0.002] + [0.0] * 8))
                st['injected'], st['last_step'] = True, k
            if stop_at is not None and st['pause_sent'] and not st['stop_sent'] and t >= st['t_pause'] + stop_at:
                pub['stop'].publish(String(data='test：暫停中停止'))
                st['stop_sent'] = True
        if node.poll() is None:
            node.terminate()
            node.wait(timeout=10)
        for _ in range(20):
            rclpy.spin_once(peer, timeout_sec=0.02)
    finally:
        stub.terminate()
        try:
            stub.wait(timeout=5)
        except subprocess.TimeoutExpired:
            stub.kill()
        peer.destroy_node()
        rclpy.shutdown()
    if not os.path.exists(out):
        print((node.stdout.read() if node.stdout else '')[-3000:])
        return None, st
    return json.load(open(out)), st


def main() -> int:
    print(f'暫存 {TMP}　domain {DOM0}\n')
    w, st = scenario('c1', 13.0, dom=0)
    if w is None:
        ck('有產出', False)
    else:
        log = [x for x in (w.get('log') or []) if isinstance(x.get('coord'), dict)]
        sel = lambda a, b: [x for x in log if a <= x.get('sim_t', -1) < b]
        pre, on, stale, off = sel(1.0, 3.4), sel(4.0, 5.9), sel(6.7, 7.9), sel(8.3, 9.4)
        ck('3.5 s 前 src＝off', bool(pre) and all(x['coord']['src'] == 'off' for x in pre), str(len(pre)))
        ck('開啟段 src＝topic、w_vref 1、base_vref＝所送值', bool(on) and all(x['coord']['src'] == 'topic' and x['coord']['w_vref'] == 1.0
           and all(abs(u - v) < 1e-12 for u, v in zip(x['coord']['base_vref'], VREF)) for x in on), str(len(on)))
        ck('停發 0.5 s 後 src＝stale_off、w_vref 0', bool(stale) and all(x['coord']['src'] == 'stale_off' and x['coord']['w_vref'] == 0.0 for x in stale),
           str([x['coord']['src'] for x in stale][:3]))
        ck('關閉訊息 ⇒ src＝topic、w_vref 0、base_vref None', bool(off) and all(x['coord']['src'] == 'topic' and x['coord']['w_vref'] == 0.0
           and x['coord']['base_vref'] is None for x in off), str(len(off)))
        ck('全程求解無失敗', all(x.get('ok', True) for x in log), str(sum(1 for x in log if not x.get('ok', True))))
    print(f'\n{"全部通過" if not FAIL else "失敗：" + "、".join(FAIL)}')
    return 0 if not FAIL else 1


if __name__ == '__main__':
    sys.exit(main())
