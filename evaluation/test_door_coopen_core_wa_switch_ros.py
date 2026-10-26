#!/usr/bin/env python3
"""接近段成本切換的求解端 ROS 假世界測試（Codex r13 §3-1；求解節點 ADM9 **不改**，只測既有協同路徑；test_dl3_wgmpc_wg2_adm8_coord_ros.py 框架的複本）。
求解節點以 g3 同組權重啟動（--w-a 0.05 --w-s 0.001 --w-s-arm 0.05、--coord-max-age-s 預設 0.5、停車保持、不送 free），目標偏移使命令非零（u_prev ≠ 0、有暖啟動）。
任務端（本測試）在 /door/wide_coord 發 off_coord，候選段把 [12] 設 0.01（同 door_coopen_approach_cost）：
  S1 normal  ：3.5–6.0 候選；6.0 起 off_coord（正常退出）⇒ ≥ 6.55 s 全為 0.05（topic）；4.05–5.95 全為 0.01。
  S2 dropout ：3.5–6.0 候選後停發 ⇒ ≥ 6.55 s 為 0.05（stale_off）。
  S3 delayed ：任務於 6.0 離開，但 NaN 訊息延到 6.3 才送（候選續送至 6.3）⇒ ≥ 6.85 s 為 0.05；且 ≥ 6.55 s 的候選輪只可能落在延遲期內（記錄，不判失敗）。
  S4 abort   ：候選中 5.0 s 送 /wgmpc/stop ⇒ 停止後不再發布（n_published_after_stop_rx 0、stop_rx 有值）。
各情境共同：候選前 w_a 0.05；求解紀錄每輪有 coord；已發布輪皆 ok（可行性核對未被繞過：未通過的輪不發布）。
假世界執行端不移動手臂 ⇒ 權威 u_prev 為 0：候選段 u_prev／暖啟動只記錄；非零 u_prev＋暖啟動的切換另以 g_1 實錄快照離線測（test_door_coopen_core_wa_switch_offline.py）。
S4 的 stop 持續送（同任務節點收尾行為），求解端只記第一筆。

    python3 evaluation/test_door_coopen_core_wa_switch_ros.py
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
import door_wide_guidance as G                        # noqa: E402
TMP = os.environ.get('WASW_TMP', '/tmp/claude-1000/coopen_wa_switch_ros')
DOM0 = int(os.environ.get('WASW_DOMAIN', '175'))
FAIL = []
RUN = 'coopen_wa_switch_ros'
CAND, LAUNCH = 0.01, 0.05


def ck(n, ok, d=''):
    print(f'  {n:62s} {"ok" if ok else "**錯**"}  {d}', flush=True)
    if not ok:
        FAIL.append(n)


def scenario(tag, run_s, case, dom=0, door_rate=0.0, stop_at=None, inject_old_after_stop=False):
    coord_on = G.off_coord()
    coord_on[12] = CAND
    on_end = {'normal': 6.0, 'dropout': 6.0, 'delayed': 6.3, 'abort': 6.0}[case]
    off_from = {'normal': 6.0, 'dropout': None, 'delayed': 6.3, 'abort': 6.0}[case]
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
    node = subprocess.Popen([sys.executable, '-u', os.path.join(HERE, 'dl3_wgmpc_wg2_adm9_node.py'),
                             '--arm-model', 'setpoint', '--N', '5', '--rate', '20', '--target-offset', '0.05', '0.02', '0.01',
                             '--w-s', '0.001', '--w-a', str(LAUNCH), '--w-s-arm', '0.05',
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
            if 1.0 <= t < 3.5:
                pub['coord'].publish(Float64MultiArray(data=G.off_coord()))
            elif 3.5 <= t < on_end:
                pub['coord'].publish(Float64MultiArray(data=coord_on))
            elif off_from is not None and t >= off_from:
                pub['coord'].publish(Float64MultiArray(data=G.off_coord()))
            if case == 'abort' and t >= 5.0:                          # 持續送（同任務節點收尾；求解端只記第一筆）
                pub['stop'].publish(String(data='test：候選中停止'))
                st['stop_sent'], st['t_stop'] = True, st.get('t_stop', t)
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
            if stop_at is not None and st['pause_sent'] and not st['stop_sent'] and t >= st['t_pause'] + stop_at:   # 未用
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
    for dom, mode in enumerate(('normal', 'dropout', 'delayed', 'abort')):
        print(f'[{mode}]')
        w, st = scenario(mode, 11.0, mode, dom=dom)
        if w is None:
            ck(f'{mode} 有產出', False)
            continue
        log = w.get('log') or []
        ck(f'{mode} 每輪有 coord', bool(log) and all(isinstance(x.get('coord'), dict) and isinstance(x['coord'].get('w_a'), float) for x in log), str(len(log)))
        log = [x for x in log if isinstance(x.get('coord'), dict)]
        sel = lambda a, b: [x for x in log if a <= x.get('sim_t', -1) < b]
        pre, on = sel(1.0, 3.45), sel(4.05, 5.95 if mode != 'abort' else 4.95)
        ck(f'{mode} 候選前 w_a 0.05', bool(pre) and all(x['coord']['w_a'] == LAUNCH for x in pre), str(len(pre)))
        ck(f'{mode} 候選段 w_a 0.01（topic）', bool(on) and all(x['coord']['w_a'] == CAND and x['coord']['src'] == 'topic' for x in on), str(len(on)))
        si = [x.get('solve_in') or {} for x in on]
        print(f'    {mode} 候選段 u_prev 非零輪數 {sum(1 for s_ in si if any(abs(v) > 1e-9 for v in (s_.get("u_prev") or [])))}／{len(si)}、'
              f'有暖啟動 {sum(1 for s_ in si if s_.get("U_warm") is not None)}（假世界執行端不移動手臂 ⇒ 權威 u_prev 為 0；非零 u_prev 見離線 test_door_coopen_core_wa_switch_offline.py）')
        pubd = [x for x in log if x.get('published')]
        ck(f'{mode} 已發布輪皆 ok', bool(pubd) and all(x.get('ok') for x in pubd), f'{len(pubd)}')
        if mode == 'normal':
            post = sel(6.55, 9.0)
            ck('normal ≥ 6.55 s 全為 0.05（topic）', bool(post) and all(x['coord']['w_a'] == LAUNCH and x['coord']['src'] == 'topic' for x in post), str(len(post)))
        elif mode == 'dropout':
            post = sel(6.55, 9.0)
            ck('dropout ≥ 6.55 s 全為 0.05（stale_off）', bool(post) and all(x['coord']['w_a'] == LAUNCH and x['coord']['src'] == 'stale_off' for x in post), str(len(post)))
        elif mode == 'delayed':
            post = sel(6.85, 9.0)
            ck('delayed ≥ 6.85 s 全為 0.05', bool(post) and all(x['coord']['w_a'] == LAUNCH for x in post), str(len(post)))
            mid = sel(6.55, 6.85)
            print(f'    delayed 6.55–6.85 s 各輪 w_a：{[x["coord"]["w_a"] for x in mid]}（延遲期內；記錄）')
        else:
            s_ = (w.get('stats') or {})
            ck('abort stop_rx 有值、停止後未發布', s_.get('stop_rx') is not None and s_.get('n_published_after_stop_rx') == 0,
               json.dumps({k: s_.get(k) for k in ('stop_rx', 'n_published_after_stop_rx', 'last_publish_sim_t')}, ensure_ascii=False, default=float)[:200])
    print(f'\n{"全部通過" if not FAIL else "失敗：" + "、".join(FAIL)}')
    return 0 if not FAIL else 1


if __name__ == '__main__':
    sys.exit(main())
