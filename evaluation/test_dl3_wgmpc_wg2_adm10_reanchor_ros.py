#!/usr/bin/env python3
"""ADM10（W1d：--pause-hold-reanchor 主動保持段伺服錨點一次性重設）的 ROS 假世界測試（無 Isaac；沿用 ADM9 保持假世界框架：
free ack 3.0 s、pause ack 5.0 s；求解端帶 W1c 的 --offset-hold-on-free）。替身端點用 w1d 複本：--odom-dx-at 5.6 起 /odom x 改報 3 mm
（hold 錨點在 5.0 s 附近已以 x＝0 設定 ⇒ 製造非零錨點誤差；同步流仍靜止，監督器窗口不受影響）。替身沒有物理，只核接線、語意與紀錄。
  R1 啟用＋非零錨點誤差：恰一次 applied；old＝hold 錨點(0,0,0)、new＝邊緣當輪快照實測底盤位姿（x≈0.003）；邊緣當輪 pause_anchor_used 仍是舊錨點，
     effective.slot＞邊緣 slot 且自該 slot 起 pause_anchor_used＝新錨點；first_published 對上真實已發布列（slot、序號、publish_sim_t_exact）；
     重設前保持命令非零、重設後保持目標＝0；邊緣起已發布底盤命令逐週期變化 ≤ a_base_lin·dt（0.025 m/s）、每列發布可行；park_anchor 每列不變；d̂ 保持（W1c 語意不受影響）；暫停照常 DONE；n_calls＝1（重送不再呼叫）。
  R0 未啟用：stats.pause_reanchor 為 None、無 pause_anchor_used 欄（與 ADM9 同行為）；暫停 DONE。
  R2 暫停中 ZERO_PUB 之前 /wgmpc/stop：停止優先、無 applied 重設紀錄（中止優先）。
  R3 --pause-hold-reanchor 缺 --pause-active-hold ⇒ 參數錯誤（exit 2）。

    python3 evaluation/test_dl3_wgmpc_wg2_adm10_reanchor_ros.py
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
TMP = os.environ.get('ADM10R_TMP', '/tmp/claude-1000/adm10_reanchor_ros')
DOM0 = int(os.environ.get('ADM10R_DOMAIN', '150'))
FAIL = []
RUN = 'adm10_reanchor_ros'


def ck(n, ok, d=''):
    print(f'  {n:62s} {"ok" if ok else "**錯**"}  {d}', flush=True)
    if not ok:
        FAIL.append(n)


def scenario(tag, run_s, door_rate=0.0, stop_at=None, inject_old_after_stop=False, dom=0, extra=(), stub_extra=('--odom-dx-at', '5.6', '--odom-dx', '0.003')):
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
    stub = subprocess.Popen([sys.executable, '-u', os.path.join(HERE, 'wgmpc_sp_stub_endpoint_w1d.py'), '--run-s', str(run_s), '--ready-after', '1.0', *stub_extra],
                            cwd=WS, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    node = subprocess.Popen([sys.executable, '-u', os.path.join(HERE, 'dl3_wgmpc_wg2_adm10_node.py'),
                             '--arm-model', 'setpoint', '--N', '5', '--rate', '20', '--target-offset', '0.05', '0.02', '0.01',
                             '--duration-s', str(run_s - 3.0), '--u-prev-policy', 'strict', '--assume-initial-rest',
                             '--park-hold', '--continuous', '--mode-switch-run-id', RUN, '--cmd-env', '--run-id', RUN, '--pause', '--pause-arm-zero', '--pause-active-hold', '--pause-arm-win-s', '0.5',
                             '--offset-free', '--offset-moving', '--offset-observer', 'interval', '--offset-hold-on-free', *extra,
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


def P(w):
    return [x for x in (w.get('log') or []) if 'pause_anchor_used' in x]


def done_ok(w):
    o = (((w.get('stats') or {}).get('k3m_pause') or {}).get('supervisor_outcome')) or {}
    return o.get('state') == 'DONE' and bool(o.get('command_stop_confirmed')) and bool(o.get('physical_stop_confirmed')), o.get('state')


def main() -> int:
    print(f'暫存 {TMP}　domain {DOM0}+\n')
    print('R1  啟用＋非零錨點誤差')
    w, st = scenario('r1', 26.0, dom=0, extra=('--pause-hold-reanchor',))
    if w is None:
        ck('R1 有產出', False)
    else:
        s = w.get('stats') or {}
        pr = s.get('pause_reanchor') or {}
        r = pr.get('record') or {}
        ck('恰一次 applied', r.get('status') == 'applied' and pr.get('n_calls') == 1, f"{r.get('status')} {r.get('why')} n_calls={pr.get('n_calls')}")
        ev = [e for e in ((s.get('k3m_pause') or {}).get('events') or []) if e.get('stage') == 'hold']
        ck('old＝hold 錨點', bool(ev) and r.get('old_anchor') == ev[0]['anchor'], f"{r.get('old_anchor')} vs {ev[0]['anchor'] if ev else None}")
        na = r.get('new_anchor') or [None]
        ck('new 為非零錨點誤差（x≈0.003）', isinstance(na[0], float) and abs(na[0] - 0.003) < 1e-9, str(na))
        edge = [x for x in w['log'] if isinstance(x.get('pause_reanchor_edge'), dict)]
        ck('邊緣紀錄恰一列', len(edge) == 1, str(len(edge)))
        rows = P(w)
        if edge and r.get('effective'):
            es, ef = edge[0]['slot'], r['effective']['slot']
            ck('邊緣當輪仍用舊錨點', edge[0].get('pause_anchor_used') == r['old_anchor'], str(edge[0].get('pause_anchor_used')))
            ck('effective 在邊緣之後（下一次求解）', ef > es, f'{es}→{ef}')
            aft = [x for x in rows if x['slot'] >= ef]
            bef = [x for x in rows if x['slot'] < ef and x.get('pause_anchor_used') is not None]
            ck('effective 起每列用新錨點', bool(aft) and all(x['pause_anchor_used'] == r['new_anchor'] for x in aft), str(len(aft)))
            ck('effective 前沒有任何列用新錨點', not any(x['pause_anchor_used'] == r['new_anchor'] for x in bef), str(len(bef)))
            fp = r.get('first_published') or {}
            row = next((x for x in w['log'] if x.get('slot') == fp.get('slot') and x.get('published')), None)
            ck('first_published 對上真實已發布列', row is not None and row.get('pause_source_seq') == fp.get('source_seq')
               and row.get('publish_sim_t_exact') == fp.get('publish_sim_t_exact') and fp.get('slot') >= ef, str(fp))
            ub = (edge[0].get('pause_u_published') or [0.0])[0]
            ga = r['effective']['goal_u']
            ck('重設前保持命令非零、重設後保持目標＝0', abs(ub) > 1e-4 and max(abs(v) for v in ga) < 1e-9 and r.get('anchor_shift', [0])[0] > 2.9e-3,
               f"u_edge={ub} goal_after={ga} shift={r.get('anchor_shift')}")
            ux = [x['pause_u_published'][:3] for x in w['log'] if x.get('published') and x.get('slot', -1) >= es
                  and isinstance(x.get('pause_u_published'), list)]
            dmax = max(abs(ux[k + 1][j] - ux[k][j]) for k in range(len(ux) - 1) for j in range(2)) if len(ux) > 1 else None
            # 既有底盤加速度框 a_base_lin·dt＝0.50×0.05＝0.025 m/s／週期 > 保持界 5 mm/s ⇒ 一週期內回到 0 仍在框內（照實，不另設平滑）
            ck('生效前後已發布底盤命令逐週期變化 ≤ a_base_lin·dt', dmax is not None and dmax <= 0.025 + 1e-12, f'max|Δu|={dmax} {ux[:3]}')
            ck('暫停期每列發布可行性 ok', all((x.get('pause_published_feasible') or {}).get('ok') for x in w['log']
                                       if 'pause_published_feasible' in x))
        else:
            ck('有邊緣列與 effective', False, str(r)[:300])
        pa = {tuple(x['park_anchor']) for x in rows if x.get('park_anchor') is not None}
        ck('park_anchor 每列不變', len(pa) == 1, str(pa))
        h = s.get('offset_hold')
        ck('d̂ 保持不受影響（凍結後無 updated）', isinstance(h, dict) and not any(
            isinstance(x.get('offset_interval'), dict) and x['offset_interval'].get('used') == 'updated'
            for x in w['log'] if x.get('sim_t', -1) >= h['sim_t'] - 1e-9))
        ok, stt = done_ok(w)
        ck('暫停照常 DONE（命令與實際停止確認）', ok, str(stt))
    print('\nR0  未啟用（ADM9 行為）')
    w, st = scenario('r0', 26.0, dom=1)
    if w is None:
        ck('R0 有產出', False)
    else:
        ck('stats.pause_reanchor 為 None', (w.get('stats') or {}).get('pause_reanchor') is None)
        ck('無 pause_anchor_used 欄', not P(w))
        ok, stt = done_ok(w)
        ck('暫停 DONE', ok, str(stt))
    print('\nR2  ZERO_PUB 前停止（中止優先）')
    w, st = scenario('r2', 14.0, stop_at=0.3, dom=2, extra=('--pause-hold-reanchor',))
    if w is None:
        ck('R2 有產出', False)
    else:
        s = w.get('stats') or {}
        r = ((s.get('pause_reanchor') or {}).get('record')) or {}
        ck('停止原因為 stop_topic', str(s.get('stop_why', '')).startswith('stop_topic'), str(s.get('stop_why')))
        ck('沒有 applied 重設', r.get('status') != 'applied', str(r.get('status')))
    print('\nR3  參數相依')
    p = subprocess.run([sys.executable, os.path.join(HERE, 'dl3_wgmpc_wg2_adm10_node.py'), '--pause-hold-reanchor'],
                       cwd=WS, capture_output=True, text=True, timeout=60)
    ck('缺 --pause-active-hold ⇒ 參數錯誤', p.returncode == 2 and '需要 --pause-active-hold' in p.stderr, p.stderr[-120:])
    print(f'\n{"全部通過" if not FAIL else "失敗：" + "、".join(FAIL)}')
    return 0 if not FAIL else 1


if __name__ == '__main__':
    sys.exit(main())
