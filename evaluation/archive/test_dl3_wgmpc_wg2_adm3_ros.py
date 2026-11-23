#!/usr/bin/env python3
"""ADM3（K3-M 底盤模式握手）的 **ROS 假世界**接線測試（無 Isaac、無 GPU；Codex 20261023_210920 第二節）。
替身執行端 wgmpc_sp_stub_endpoint.py 餵 /clock、/joint_states、/odom、/coman/arm_setpoint、/coman/applied_cmd；本測試行程另以 rclpy 節點
扮演執行端的握手側：發 /park/gate（錨點）與 /door/base_mode_ack，收 /wgmpc/base_mode_ack。跑**真正的** dl3_wgmpc_wg2_adm3_node.py
（--park-hold --continuous --mode-switch-run-id，同跑法）。替身沒有物理，本檔只核接線與紀錄。

  M1 正常：執行端 ack ⇒ 求解器在快照 ≥ switch_t 後切入 free；**只在已發布的首筆 free 解之後**回成功 ack（published＝True、時間先後）；
     ack 等於求解器 stats 紀錄；首筆 free 前的已發布列為 hold、之後為 free 且 park_hold 為 None；紀錄承接的 applied u_prev（底盤＝替身套用的 0）。
  M2 錯趟：run_id 不符的 ack ⇒ 求解器回 nack、不切入 free。
  M3 pending 期間停止：送 switch_t 在未來的 ack（求解器會延後套用），隨即 /wgmpc/stop ⇒ 不切入 free、不回成功 ack。

    python3 evaluation/test_dl3_wgmpc_wg2_adm3_ros.py      # 需 ROS 2 環境；只清理本趟程序
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
TMP = os.environ.get('ADM3R_TMP', '/tmp/claude-1000/adm3_ros')
DOM0 = int(os.environ.get('ADM3R_DOMAIN', '206'))
FAIL = []
RUN = 'adm3_ros'


def ck(n, ok, d=''):
    print(f'  {n:62s} {"ok" if ok else "**錯**"}  {d}', flush=True)
    if not ok:
        FAIL.append(n)


def scenario(tag, plan, run_s=14.0, dom=0):
    """plan(peer, sim_t) → 每次輪詢呼叫一次，由它決定何時發 ack／stop。回傳 (solver_json, 收到的求解器 ack 列表)。"""
    os.makedirs(TMP, exist_ok=True)
    out = os.path.join(TMP, f'{tag}.json')
    if os.path.exists(out):
        os.remove(out)                                            # 本測試自己的暫存輸出
    env = dict(os.environ, ROS_DOMAIN_ID=str(DOM0 + dom), PYTHONUNBUFFERED='1')
    os.environ['ROS_DOMAIN_ID'] = str(DOM0 + dom)
    import rclpy
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from rosgraph_msgs.msg import Clock
    from std_msgs.msg import String
    rclpy.init()
    peer = rclpy.create_node(f'adm3_peer_{tag}')
    lat = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    peer.gate = peer.create_publisher(String, '/park/gate', 1)
    peer.ack = peer.create_publisher(String, '/door/base_mode_ack', lat)
    peer.stop = peer.create_publisher(String, '/wgmpc/stop', 1)
    got, clk = [], {'t': 0.0}
    peer.create_subscription(String, '/wgmpc/base_mode_ack', lambda m: got.append(json.loads(m.data)), lat)
    peer.create_subscription(Clock, '/clock', lambda m: clk.__setitem__('t', m.clock.sec + m.clock.nanosec * 1e-9), 10)
    stub = subprocess.Popen([sys.executable, '-u', os.path.join(HERE, 'wgmpc_sp_stub_endpoint.py'), '--run-s', str(run_s), '--ready-after', '1.0'],
                            cwd=WS, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    node = subprocess.Popen([sys.executable, '-u', os.path.join(HERE, 'dl3_wgmpc_wg2_adm3_node.py'),
                             '--arm-model', 'setpoint', '--N', '5', '--rate', '20', '--target-offset', '0.05', '0.02', '0.01',
                             '--duration-s', str(run_s - 3.0), '--u-prev-policy', 'strict', '--assume-initial-rest',
                             '--park-hold', '--continuous', '--mode-switch-run-id', RUN, '--out', out],
                            cwd=WS, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    t0 = time.monotonic()
    try:
        while node.poll() is None and time.monotonic() - t0 < run_s + 60:
            rclpy.spin_once(peer, timeout_sec=0.02)
            peer.gate.publish(String(data=json.dumps({'passed': True, 'ok_now': True, 'anchor': [0.0, 0.0, 0.0]})))
            plan(peer, clk['t'])
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
        print((node.stdout.read() if node.stdout else '')[-2000:])
        return None, got
    return json.load(open(out)), got


def once(fn):
    st = {'done': False}

    def w(peer, t):
        if not st['done'] and fn(peer, t):
            st['done'] = True
    return w


def main() -> int:
    from std_msgs.msg import String
    print(f'暫存 {TMP}　domain {DOM0}+\n')

    def m1(peer, t):
        if t >= 4.0:
            peer.ack.publish(String(data=json.dumps({'run_id': RUN, 'seq': 1, 'mode': 'free', 'switch_step': int(round(t * 100)),
                                                    'switch_t': round(t, 4)})))
            return True
        return False
    print('M1  正常切換')
    w, got = scenario('m1', once(m1), dom=0)
    if w is None:
        ck('M1 有產出', False)
    else:
        km = (w.get('stats') or {}).get('k3m_mode') or {}
        rec = km.get('record') or {}
        ack = rec.get('ack')
        ok_acks = [g for g in got if 'why' not in g]
        ck('求解器已切入 free', km.get('active') is True)
        ck('收到恰一筆成功 ack 且等於 stats 紀錄', len(ok_acks) == 1 and ok_acks[0] == ack, f'{ok_acks} vs {ack}')
        if ack:
            ea = rec.get('executor_ack') or {}
            ck('ack 的 switch_step／switch_t 照抄執行端', ack['switch_step'] == ea.get('switch_step') and ack['switch_t'] == ea.get('switch_t'))
            ck('published＝True 且 switch_t ≤ first_free_sim_t ≤ publish_sim_t',
               ack.get('published') is True and ack['switch_t'] <= ack['first_free_sim_t'] + 1e-9 <= ack['publish_sim_t'] + 2e-9, str(ack))
            pubs = [x for x in w.get('log') or [] if x.get('published')]
            first = next((x for x in pubs if x.get('slot') == ack['first_free_slot']), None)
            ck('首筆 free 解確實已發布', first is not None and first.get('k3m_mode_ack') == ack)
            seen, bad = False, 0
            for x in pubs:
                seen = seen or x.get('slot') == ack['first_free_slot']
                if (not seen and x.get('k3m_mode') != 'hold') or (seen and (x.get('k3m_mode') != 'free' or x.get('park_hold') is not None)):
                    bad += 1
            ck('hold→free 列一致（free 後 park_hold 為 None）', bad == 0 and seen, f'bad={bad}')
            up = rec.get('u_prev_applied') or []
            ck('記錄承接的 applied u_prev（底盤＝替身套用的 0）', len(up) == 9 and max(abs(v) for v in up[:3]) == 0.0, str(up[:3]))
            ev = [e for e in km.get('events') or [] if e.get('applied_free')]
            ck('切入當輪快照 ≥ switch_t', bool(ev) and ev[0]['sim_t'] >= ack['switch_t'] - 1e-9, str(ev[:1]))

    def m2(peer, t):
        if t >= 4.0:
            peer.ack.publish(String(data=json.dumps({'run_id': 'other_run', 'seq': 1, 'mode': 'free', 'switch_step': 400, 'switch_t': 4.0})))
            return True
        return False
    print('\nM2  錯趟 ack')
    w, got = scenario('m2', once(m2), dom=1)
    if w is None:
        ck('M2 有產出', False)
    else:
        km = (w.get('stats') or {}).get('k3m_mode') or {}
        ck('未切入 free', km.get('active') is False)
        ck('回 nack（wrong_run_or_invalid）', any(g.get('why') == 'wrong_run_or_invalid' for g in got), str(got))
        ck('無成功 ack', not any('why' not in g for g in got))

    st3 = {'k': 0}

    def m3(peer, t):
        if st3['k'] == 0 and t >= 4.0:
            peer.ack.publish(String(data=json.dumps({'run_id': RUN, 'seq': 1, 'mode': 'free', 'switch_step': 900, 'switch_t': 9.0})))
            st3['k'], st3['t'] = 1, t
        elif st3['k'] == 1 and t >= st3['t'] + 0.5:
            peer.stop.publish(String(data='test：pending 期間停止'))
            st3['k'] = 2
        return False
    print('\nM3  pending 期間停止')
    w, got = scenario('m3', m3, dom=2)
    if w is None:
        ck('M3 有產出', False)
    else:
        km = (w.get('stats') or {}).get('k3m_mode') or {}
        ck('未切入 free', km.get('active') is False, str(km.get('rejections')))
        ck('無成功 ack', not any('why' not in g for g in got), str(got))
        ck('停止原因為 stop_topic', str((w.get('stats') or {}).get('stop_why', '')).startswith('stop_topic'), str((w.get('stats') or {}).get('stop_why')))

    print(f'\n{"全部通過" if not FAIL else "失敗：" + "、".join(FAIL)}')
    return 0 if not FAIL else 1


if __name__ == '__main__':
    sys.exit(main())
