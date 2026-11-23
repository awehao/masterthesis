#!/usr/bin/env python3
"""ADM17r4（--offset-update-in-open：偏移估計保持改由暫停觸發）的 ROS 假世界測試（test_dl3_wgmpc_wg2_adm9_hold_ros.py 的版本化複本；Codex r27；無 Isaac）。
假世界同 ADM9：free ack 3.0 s、pause ack 5.0 s；求解端帶 --offset-free --offset-moving --offset-observer interval --offset-hold-on-free。只核接線、更新邊界與紀錄。
  U1 啟用：free（3 s）到暫停（5 s）之間有 used＝updated；凍結 trigger＝pause、暫停 switch_t ≤ 凍結 sim_t ≤ ＋0.06 s；凍結起每列 blocked:offset_hold、
     d_after＝凍結值、無 updated；每個求解列 solve_in.offset_d_hat＝該列 d_after、arm_bias＝α⊙d_after；報告 n_updates＝逐列 updated 數；
     暫停照常 DONE；並以判定器 r12 check_offset_update_open 核對節點實際輸出 ⇒ 無理由。
  U0 未帶新旗標（r3 行為）：凍結 trigger＝free、凍結時刻在 free 切換起 0.06 s 內、free 後無 updated。
  U2 啟用＋暫停中 /wgmpc/stop：停止原因 stop_topic、停止後無求解紀列；凍結在停止前且其後無 updated。
  U3 --offset-update-in-open 缺 --offset-hold-on-free ⇒ 拒跑。

    python3 evaluation/test_dl3_adm17r4_update_ros.py
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
TMP = os.environ.get('ADM17R4U_TMP', '/tmp/claude-1000/adm17r4_update_ros')
DOM0 = int(os.environ.get('ADM17R4U_DOMAIN', '186'))
FAIL = []
RUN = 'adm17r4_update_ros'
ARM_IDENT = os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')


def ck(n, ok, d=''):
    print(f'  {n:62s} {"ok" if ok else "**錯**"}  {d}', flush=True)
    if not ok:
        FAIL.append(n)


def scenario(tag, run_s, door_rate=0.0, stop_at=None, inject_old_after_stop=False, dom=0, extra=()):
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
           'asrc': peer.create_publisher(Float64MultiArray, '/door/applied_src', 200)}
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
    node = subprocess.Popen([sys.executable, '-u', os.path.join(HERE, 'dl3_wgmpc_wg2_adm17r4_node.py'),
                             '--arm-model', 'setpoint', '--N', '5', '--rate', '20', '--target-offset', '0.05', '0.02', '0.01',
                             '--duration-s', str(run_s - 3.0), '--u-prev-policy', 'strict', '--assume-initial-rest',
                             '--park-hold', '--continuous', '--mode-switch-run-id', RUN, '--cmd-env', '--run-id', RUN, '--pause', '--pause-arm-zero', '--pause-active-hold', '--pause-arm-win-s', '0.5',
                             '--offset-free', '--offset-moving', '--offset-observer', 'interval', *extra,
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
            if st['free_sent'] and not st['pause_sent'] and t >= 5.0:
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


def oi(w):
    return [(x.get('sim_t'), x['offset_interval'], x) for x in (w.get('log') or []) if isinstance(x.get('offset_interval'), dict)]


def main() -> int:
    import door_coopen_check_core12 as J12
    print(f'暫存 {TMP}　domain {DOM0}+\n')
    print('U1  啟用（暫停才凍結）')
    w, st = scenario('u1', 26.0, dom=0, extra=('--offset-hold-on-free', '--offset-update-in-open', '--arm-ident', ARM_IDENT))
    if w is None:
        ck('U1 有產出', False)
    else:
        s = w.get('stats') or {}
        h = s.get('offset_hold')
        rows = oi(w)
        tf = ((s.get('k3m_mode') or {}).get('record') or {}).get('executor_ack', {}).get('switch_t')
        tp = (((s.get('k3m_pause') or {}).get('record') or {}).get('executor_ack') or {}).get('switch_t')
        ck('有凍結紀錄且 trigger＝pause', isinstance(h, dict) and h.get('trigger') == 'pause', str(h)[:160])
        ck('free 與暫停之間有 updated', tf is not None and tp is not None and
           sum(1 for t, r, _ in rows if tf + 0.05 < t < tp - 1e-9 and r.get('used') == 'updated') >= 5,
           f"{sum(1 for t, r, _ in rows if tf is not None and tp is not None and tf + 0.05 < t < tp and r.get('used') == 'updated')}")
        if isinstance(h, dict) and tp is not None:
            th = h['sim_t']
            ck('暫停 switch_t ≤ 凍結 ≤ ＋0.06 s', tp - 1e-9 <= th <= tp + 0.06 + 1e-9, f'{tp} → {th}')
            post = [r for t, r, _ in rows if t >= th - 1e-9]
            ck('凍結起每列 blocked:offset_hold、值不變、無 updated', bool(post) and all(r.get('why') == 'blocked:offset_hold' and
               r.get('d_after') == h['d_frozen'] and r.get('used') != 'updated' for r in post), str({r.get('why') for r in post}))
        why, stt = [], {}
        J12.check_offset_update_open(w, why, stt)
        ck('判定器 r12 U1–U8 核對節點輸出無理由', why == [], str(why))
        ck('報告 n_updates＝逐列 updated 數', (w.get('offset_free') or {}).get('n_updates') == sum(1 for _, r, _ in rows if r.get('used') == 'updated'))
        o = ((s.get('k3m_pause') or {}).get('supervisor_outcome')) or {}
        ck('暫停照常 DONE（命令與實際停止確認）', o.get('state') == 'DONE' and o.get('command_stop_confirmed') and o.get('physical_stop_confirmed'),
           str(o.get('state')))
    print('\nU0  未帶新旗標（r3 行為）')
    w, st = scenario('u0', 14.0, dom=1, extra=('--offset-hold-on-free',))
    if w is None:
        ck('U0 有產出', False)
    else:
        s = w.get('stats') or {}
        h = s.get('offset_hold')
        tf = ((s.get('k3m_mode') or {}).get('record') or {}).get('executor_ack', {}).get('switch_t')
        ck('trigger＝free、凍結在 free 切換 0.06 s 內', isinstance(h, dict) and h.get('trigger') == 'free' and tf is not None and
           tf - 1e-9 <= h['sim_t'] <= tf + 0.06 + 1e-9, f"{tf} → {(h or {}).get('sim_t')}")
        ck('free 後無 updated', isinstance(h, dict) and not any(r.get('used') == 'updated' for t, r, _ in oi(w) if t >= h['sim_t'] - 1e-9))
    print('\nU2  啟用＋暫停中停止（停止優先）')
    w, st = scenario('u2', 14.0, stop_at=1.0, dom=2, extra=('--offset-hold-on-free', '--offset-update-in-open'))
    if w is None:
        ck('U2 有產出', False)
    else:
        s = w.get('stats') or {}
        t_stop = (s.get('stop_rx') or {}).get('sim_t')
        ck('停止原因為 stop_topic', str(s.get('stop_why', '')).startswith('stop_topic'), str(s.get('stop_why')))
        ck('停止後沒有任何求解紀錄列', t_stop is not None and not any(x.get('sim_t', 0) > t_stop + 1e-9 for x in w.get('log') or []), str(t_stop))
        h = s.get('offset_hold')
        ck('停止前已凍結（pause）且其後無 updated', isinstance(h, dict) and h.get('trigger') == 'pause' and t_stop is not None and h['sim_t'] <= t_stop and
           not any(r.get('used') == 'updated' for t, r, _ in oi(w) if t >= h['sim_t'] - 1e-9))
    print('\nU3  參數相依')
    p = subprocess.run([sys.executable, os.path.join(HERE, 'dl3_wgmpc_wg2_adm17r4_node.py'), '--mode-switch-run-id', 'x', '--offset-observer', 'interval',
                        '--offset-update-in-open'], cwd=WS, capture_output=True, text=True, timeout=60)
    ck('缺 --offset-hold-on-free ⇒ 拒跑', p.returncode != 0 and '需要 --offset-hold-on-free' in (p.stderr + p.stdout), (p.stderr + p.stdout)[-120:])
    print(f'\n{"全部通過" if not FAIL else "失敗：" + "、".join(FAIL)}')
    return 0 if not FAIL else 1


if __name__ == '__main__':
    sys.exit(main())
