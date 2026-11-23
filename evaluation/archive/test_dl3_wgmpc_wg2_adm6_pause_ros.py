#!/usr/bin/env python3
"""ADM6（K3-M 暫停 p3b：--pause-arm-zero 手臂歸零＋底盤主動保持）的 ROS 假世界測試＝ADM5 測試的複本，節點改 ADM6＋--pause-arm-zero，P1 另核：
  暫停列手臂發布值向 0 單調不增、自第 2 列起手臂為 0；非零速列標 pause_arm_zero_only；發布值都在既有限制內；P5 兩旗標併用 ⇒ 參數錯誤拒跑。
原說明：ADM4（K3-M 受控暫停）的 **ROS 假世界**接線測試（無 Isaac、無 GPU；規格 K3M_pause_integration_spec_draft1.md）。
替身執行端 wgmpc_sp_stub_endpoint.py 餵 /clock、/joint_states、/odom、/coman/arm_setpoint、/coman/applied_cmd；本測試行程另以 rclpy 節點
扮演執行端其餘部分：/park/gate、free 與 pause 兩段握手的執行端 ack、/door/sync_state（50 Hz）、
收求解端封裝 /wgmpc/cmd_env 後**逐物理步連續**發 /door/applied_src（最新封裝的 source_seq 與值；封裝年齡 > 0.2 s ⇒ −1、零值、模式 1）。
跑**真正的** dl3_wgmpc_wg2_adm4_node.py（--park-hold --continuous --mode-switch-run-id --cmd-env --pause）。替身沒有物理，本檔只核接線與紀錄。

  P1 正常（靜止世界）：pause ack 帶凍結目標；LOWSPEED→ZERO_PUB→CONFIRM→STOPPED→DONE；命令停止與實際停止都確認、觀察 ≥ 10 s；
     STOPPED 後不再有已發布列；零速列的發布值手臂為 0。
  P2 門持續轉（0.004 rad/s）：實際停止條件永不成立 ⇒ FAIL timeout；不報命令或實際停止。
  P3 暫停中 /wgmpc/stop：停止優先，求解器以 stop_topic 結束、不回報 DONE。
  P4 停止發布後注入舊序號非零套用：FAIL old_command_after_stop。

    python3 evaluation/test_dl3_wgmpc_wg2_adm4_pause_ros.py      # 需 ROS 2 環境；只清理本趟程序
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
TMP = os.environ.get('ADM6R_TMP', '/tmp/claude-1000/adm6_pause_ros')
DOM0 = int(os.environ.get('ADM6R_DOMAIN', '140'))
FAIL = []
RUN = 'adm6_pause_ros'


def ck(n, ok, d=''):
    print(f'  {n:62s} {"ok" if ok else "**錯**"}  {d}', flush=True)
    if not ok:
        FAIL.append(n)


def scenario(tag, run_s, door_rate=0.0, stop_at=None, inject_old_after_stop=False, dom=0):
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
    node = subprocess.Popen([sys.executable, '-u', os.path.join(HERE, 'dl3_wgmpc_wg2_adm6_node.py'),
                             '--arm-model', 'setpoint', '--N', '5', '--rate', '20', '--target-offset', '0.05', '0.02', '0.01',
                             '--duration-s', str(run_s - 3.0), '--u-prev-policy', 'strict', '--assume-initial-rest',
                             '--park-hold', '--continuous', '--mode-switch-run-id', RUN, '--cmd-env', '--run-id', RUN, '--pause', '--pause-arm-zero',
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


def main() -> int:
    print(f'暫存 {TMP}　domain {DOM0}+\n')
    print('P1  正常（靜止世界）')
    w, st = scenario('p1', 26.0, dom=0)
    if w is None:
        ck('P1 有產出', False)
    else:
        kp = (w.get('stats') or {}).get('k3m_pause') or {}
        o = kp.get('supervisor_outcome') or {}
        ok_acks = [g for g in st['acks'] if 'why' not in g]
        ck('求解器已切入 pause', kp.get('active') is True, str(kp.get('rejections')))
        ck('恰一筆成功 pause ack，帶凍結目標（位置＋完整 4×4）', len(ok_acks) == 1 and len(ok_acks[0].get('frozen_target_xyz') or []) == 3
           and len(ok_acks[0].get('frozen_target_T') or []) == 16, str(ok_acks)[:200])
        ck('命令停止確認', o.get('command_stop_confirmed') is True, str(o)[:300])
        ck('實際停止確認（觀察 ≥ 10 s）', o.get('physical_stop_confirmed') is True and (o.get('observation_covered_s') or 0) >= 10.0 - 1e-6, str(o)[:300])
        log = w.get('log') or []
        t_stop = (o.get('transitions') or {}).get('STOPPED')
        pub_after = [x for x in log if x.get('published') and t_stop is not None and x.get('sim_t', 0) > t_stop + 0.06]
        ck('STOPPED 後不再發布', t_stop is not None and not pub_after, f'{len(pub_after)} 列')
        zl = [x for x in log if x.get('pause_zero')]
        ck('有零速發布列', bool(zl), str(len(zl)))
        ck('暫停期間底盤等式可行（不可行 0 輪）', kp.get('infeasible_base_hold_cycles') == 0, str(kp.get('infeasible_base_hold_cycles')))
        pr = [x for x in log if x.get('published') and x.get('pause_u_published') is not None]
        arm = [max(abs(v) for v in x['pause_u_published'][3:]) for x in pr]
        base = [max(abs(v) for v in x['pause_u_published'][:3]) for x in pr]
        ck('ADM6：有暫停發布列且都標 pause_all_zero 或零速', bool(pr) and all(x.get('pause_all_zero') or x.get('pause_zero') for x in pr), str(len(pr)))
        ck('ADM6：手臂發布值單調不增、第 2 列起為 0', all(arm[i + 1] <= arm[i] + 1e-12 for i in range(len(arm) - 1)) and all(v == 0.0 for v in arm[1:]),
           str([round(v, 6) for v in arm[:5]]))
        ck('ADM6：非零速列標 pause_arm_zero_only', all(x.get('pause_arm_zero_only') for x in pr if not x.get('pause_zero')), str(len(pr)))
        ck('ADM6：發布值都在既有限制內', all((x.get('pause_published_feasible') or {}).get('ok') for x in pr))
    print('\nP2  門持續轉')
    w, st = scenario('p2', 18.0, door_rate=0.004, dom=1)
    if w is None:
        ck('P2 有產出', False)
    else:
        o = ((w.get('stats') or {}).get('k3m_pause') or {}).get('supervisor_outcome') or {}
        ck('FAIL timeout', o.get('state') == 'FAIL' and o.get('fail_why') == 'timeout', str(o)[:300])
        ck('不報命令或實際停止', not o.get('command_stop_confirmed') and not o.get('physical_stop_confirmed'))
    print('\nP3  暫停中停止（停止優先）')
    w, st = scenario('p3', 14.0, stop_at=1.0, dom=2)
    if w is None:
        ck('P3 有產出', False)
    else:
        s = w.get('stats') or {}
        o = (s.get('k3m_pause') or {}).get('supervisor_outcome') or {}
        ck('停止原因為 stop_topic', str(s.get('stop_why', '')).startswith('stop_topic'), str(s.get('stop_why')))
        ck('未報 DONE', o.get('state') != 'DONE', str(o.get('state')))
    print('\nP4  停止發布後舊命令重新生效')
    w, st = scenario('p4', 16.0, inject_old_after_stop=True, dom=3)
    if w is None:
        ck('P4 有產出', False)
    else:
        o = ((w.get('stats') or {}).get('k3m_pause') or {}).get('supervisor_outcome') or {}
        ck('有注入', st['injected'])
        ck('FAIL old_command_after_stop', o.get('state') == 'FAIL' and o.get('fail_why') == 'old_command_after_stop', str(o)[:300])
    print('\nP5  兩旗標併用')
    p = subprocess.run([sys.executable, os.path.join(HERE, 'dl3_wgmpc_wg2_adm6_node.py'), '--pause', '--pause-arm-zero', '--pause-all-zero'],
                       cwd=WS, capture_output=True, text=True, timeout=60)
    ck('參數錯誤拒跑', p.returncode == 2 and '不可併用' in p.stderr, p.stderr[-120:])
    print(f'\n{"全部通過" if not FAIL else "失敗：" + "、".join(FAIL)}')
    return 0 if not FAIL else 1


if __name__ == '__main__':
    sys.exit(main())
