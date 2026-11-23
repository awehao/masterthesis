#!/usr/bin/env python3
"""真任務節點 door_k3m_task_node_w2 的啟動／請求／狀態路由測試（無 Isaac；Codex 20261025_220903「真任務節點的啟動／請求／狀態路由測試」）。

範圍與替身（須揭露）：
- 節點以子程序執行真檔（runpy），參數、ROS 話題、QoS、主迴圈、紀錄、收尾全部是真的；W2 策略 K3MW2Policy.step 也是真的。
- 唯一替身：父類 door_k3m_policy_p2.K3MP2Policy.step（夾持、開門、SETTLE、暫停握手與 PAUSE 監看）換成「立即視為 PAUSE、暫停已確認」的
  替身——本測試不驗開門與暫停（已有既有測試與 Isaac 實跑），只驗 W2 路由。
- 本測試扮演：觀察器心跳、/door/sync_state（50 Hz）、求解端 /wgmpc/pause_status（HELD）、執行端權威（ModeAuthority4 的狀態訊息）。

  T1 正常：HELD ⇒ 1 s 後恰一筆請求出現在 /door/w2_mode_req_task（latched；內容＝run_id、req_seq 1、PAUSE_HELD→RELEASE_PREP_TEST、mode_seq 1）；
     節點不發 /door/w2_mode_req（求解端話題由執行端轉送）；收到本趟提交狀態（applied_confirmed True）⇒ 保持 3 s 後 DONE、exit 0、
     發 /room/stop；task.json w2.state＝done、trace 每列含 w2；他趟 run_id 的狀態不計入（n_state_msgs 只算本趟）。
     HELD 之前與 committed 期間不發 /room/stop。
  T2 請求被拒：權威狀態 last_reject.req_seq＝1 ⇒ 中止 w2_request_rejected:<why>、發 /door/k2_abort、exit 1、最終 ABORTED。
  T3 關閉旗標：不帶 --w2-mode-run-id ⇒ 不建立 /door/w2_mode_req_task 發布者、task.json 無 w2（w1b2 行為）。

    python3 evaluation/test_task_node_w2_routing.py
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
TMP = os.environ.get('W2R_TMP', '/tmp/claude-1000/w2_task_routing')
DOM0 = int(os.environ.get('W2R_DOMAIN', '190'))
RUN = 'w2_route'
FAIL = []

LAUNCH = r'''
import runpy, sys
sys.path.insert(0, %(here)r)
import door_k3m_policy_p2 as PP2
def _stand_in(self, m, dt):                     # 替身：立即視為 PAUSE、暫停已確認（不驗開門與暫停）
    if self.phase not in ('PAUSE', 'DONE', 'ABORTED'):
        self._go('PAUSE', m.t, stand_in=True)
        self.pause_t0 = m.t
    self.pause_state = 'active'
    if self.phase == 'ABORTED':
        return self._aborted_act({'target': None, 'grip': 0.0, 'request_handover': False, 'solver_start': False,
                                  'stop_solver': False, 'finish': False}, m)
    return {'target': None, 'grip': 0.0, 'request_handover': False, 'solver_start': False, 'stop_solver': False, 'finish': False}
PP2.K3MP2Policy.step = _stand_in
sys.argv = ['door_k3m_task_node_w2.py'] + %(args)r
sys.exit(runpy.run_path(%(node)r, run_name='__main__'))
'''


def ck(n, ok, d=''):
    print(f'  {n:64s} {"ok" if ok else "**錯**"}  {d}', flush=True)
    if not ok:
        FAIL.append(n)


def scenario(tag, dom, w2=True, reject=False, run_s=40.0):
    os.makedirs(TMP, exist_ok=True)
    out = os.path.join(TMP, f'{tag}.json')
    for f in (out, out[:-5] + '_trace.jsonl', out[:-5] + '_trace_hb.json'):
        if os.path.exists(f):
            os.remove(f)                                              # 本測試自己的暫存輸出
    os.environ['ROS_DOMAIN_ID'] = str(DOM0 + dom)
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    import rclpy
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import Float64MultiArray, String
    import door_w2_mode4 as M4
    args = ['--out', out, '--run-id', RUN, '--w1b-params', 'evaluation/results/door/w1e_params_30.json',
            '--corridor-table', 'evaluation/results/door/door_wide_corridor_table_90.json', '--lw-ready-timeout-s', '20',
            '--wall-total-s', str(run_s), '--wall-post-abort-s', '2']
    if w2:
        args += ['--w2-mode-run-id', RUN]
    code = LAUNCH % {'here': HERE, 'args': args, 'node': os.path.join(HERE, 'door_k3m_task_node_w2.py')}
    rclpy.init()
    peer = rclpy.create_node(f'w2r_peer_{tag}')
    lat = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    pub = {'hb': peer.create_publisher(String, '/door/lw_observer/heartbeat', 10),
           'sync': peer.create_publisher(Float64MultiArray, '/door/sync_state', 100),
           'ps': peer.create_publisher(String, '/wgmpc/pause_status', lat),
           'state': peer.create_publisher(String, '/door/w2_mode_state', 50)}
    got = {'req_task': [], 'req_solver': [], 'room_stop': [], 'k2_abort': [], 'room_stop_t': []}
    peer.create_subscription(String, '/door/w2_mode_req_task', lambda m: got['req_task'].append((time.monotonic(), json.loads(m.data))), lat)
    peer.create_subscription(String, '/door/w2_mode_req', lambda m: got['req_solver'].append(m.data), lat)
    peer.create_subscription(String, '/room/stop', lambda m: got['room_stop'].append((time.monotonic(), m.data)), 10)
    peer.create_subscription(String, '/door/k2_abort', lambda m: got['k2_abort'].append(m.data), lat)
    auth = M4.ModeAuthority4(RUN, run_id_num=None, stage_id=2, dt=0.01)
    p = subprocess.Popen([sys.executable, '-u', '-c', code], cwd=WS, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, start_new_session=True)
    st = {'seq': 0, 'k': 0, 't_held_sent': None, 'commit_sent': False, 'wall_held': None, 'n_own': 0, 'n_other': 0}
    w0 = time.monotonic()
    try:
        while p.poll() is None and time.monotonic() - w0 < run_s + 30:
            rclpy.spin_once(peer, timeout_sec=0.005)
            wall = time.monotonic() - w0
            k_now = int(wall * 50)
            while st['k'] <= k_now:                                   # 50 Hz 同步狀態（模擬時間＝牆鐘，RTF 1）
                ts = st['k'] * 0.02
                pub['sync'].publish(Float64MultiArray(data=[ts, float(st['k'] * 2), 0.14, *([1.0, 0, 0, 0.6, 0, 1.0, 0, -0.53, 0, 0, 1.0, 0.85,
                                                             0, 0, 0, 1.0]), 0.6, -0.5332, 1.570796, 0.0, 0.0, 0.035, 0.0, -1.5707963, 0.0, 0.0]))
                if st['k'] % 5 == 0:
                    st['seq'] += 1
                    pub['hb'].publish(String(data=json.dumps({'seq': st['seq']})))
                if st['k'] % 2 == 0:                                  # 執行端權威狀態（每 0.04 s；另夾一筆他趟 run_id）
                    m = auth.state_msg(st['k'], ts)
                    if st['commit_sent']:
                        m.update(mode_seq=1, commit={'step': 1, 't': ts, 'first_source_seq': 7, 'mode_seq': 1, 'applied_confirmed': True},
                                 commit_req_seq=1)
                    if reject and got['req_task']:
                        m.update(last_reject={'run_id': RUN, 'req_seq': 1, 'why': 'candidate_invalid:accel_box'})
                    pub['state'].publish(String(data=json.dumps(m, default=float)))
                    st['n_own'] += 1
                    other = dict(m, run_id='other_run', mode_seq=1, commit={'step': 1, 'mode_seq': 1, 'applied_confirmed': True},
                                 commit_req_seq=1)
                    pub['state'].publish(String(data=json.dumps(other, default=float)))
                    st['n_other'] += 1
                st['k'] += 1
            if wall >= 3.0 and st['t_held_sent'] is None:             # 求解端 HELD（觀察通過）
                pub['ps'].publish(String(data=json.dumps({'outcome': {'state': 'HELD', 'observation_passed': True,
                                                                      'physical_stop_confirmed': True, 'command_stop_confirmed': True,
                                                                      'transitions': {'OBS_PASSED': 3.0, 'HELD': 3.0}}})))
                st['t_held_sent'], st['wall_held'] = wall, time.monotonic()
            if got['req_task'] and not st['commit_sent'] and not reject and time.monotonic() - got['req_task'][0][0] > 0.2:
                st['commit_sent'] = True
                st['wall_commit'] = time.monotonic()
        rc = p.wait(timeout=20)
        for _ in range(30):
            rclpy.spin_once(peer, timeout_sec=0.02)
    finally:
        if p.poll() is None:
            p.terminate()
            p.wait(timeout=10)
        peer.destroy_node()
        rclpy.shutdown()
    log = p.stdout.read() if p.stdout else ''
    task = json.load(open(out)) if os.path.exists(out) else None
    return rc, task, got, st, log


def main() -> int:
    print(f'暫存 {TMP}　domain {DOM0}+\n')
    print('T1  正常路由')
    rc, task, got, st, log = scenario('t1', 0)
    if task is None:
        print(log[-3000:])
    w2 = (task or {}).get('w2') or {}
    ck('exit 0、最終 DONE', rc == 0 and (task or {}).get('final_phase') == 'DONE', f"rc={rc} phase={(task or {}).get('final_phase')}")
    reqs = [r for _, r in got['req_task']]
    ck('恰一筆請求（latched /door/w2_mode_req_task）、內容正確', len(reqs) == 1 and reqs[0] == {
        'run_id': RUN, 'req_seq': 1, 'from_mode': 'PAUSE_HELD', 'to_mode': 'RELEASE_PREP_TEST', 'mode_seq': 1}, str(reqs))
    ck('請求在 HELD 之後約 1 s', bool(got['req_task']) and 0.8 <= got['req_task'][0][0] - st['wall_held'] <= 1.6,
       f"{(got['req_task'][0][0] - st['wall_held']) if got['req_task'] else None}")
    ck('節點不發求解端話題 /door/w2_mode_req', got['req_solver'] == [], str(got['req_solver'])[:80])
    rs = [w for w, _ in got['room_stop']]
    ck('/room/stop 只在保持 3 s 之後', bool(rs) and min(rs) - st['wall_commit'] >= 2.9, f"{(min(rs) - st['wall_commit']) if rs else None}")
    ck('task.json w2.state＝done、提交時刻與保持 3 s', w2.get('state') == 'done' and (w2.get('t') or {}).get('done', 0)
       - (w2.get('t') or {}).get('committed', 1e9) >= 3.0 - 1e-6, str(w2.get('t')))
    tr = (task or {}).get('trace') or []
    ck('trace 每列含 w2', bool(tr) and all('w2' in x for x in tr), str(len(tr)))
    ls = w2.get('last_state') or {}
    ck('他趟 run_id 狀態不計入（計數 ≤ 本趟送出數、最後狀態為本趟）', ls.get('run_id') == RUN and 0 < w2.get('n_state_msgs', 0) <= st['n_own'],
       f"n_state_msgs={w2.get('n_state_msgs')} own_sent={st['n_own']} other_sent={st['n_other']}")
    ck('trace 上 w2 狀態序 none→held_wait→requested→committed→done', [s for i, s in enumerate(x['w2'] for x in tr)
                                                                    if i == 0 or s != tr[i - 1]['w2']][:5]
       == ['none', 'held_wait', 'requested', 'committed', 'done'], '')
    print('\nT2  請求被拒')
    rc, task, got, st, log = scenario('t2', 1, reject=True)
    ck('中止 w2_request_rejected:candidate_invalid:accel_box', (task or {}).get('abort') == 'w2_request_rejected:candidate_invalid:accel_box',
       str((task or {}).get('abort')))
    ck('exit 1、最終 ABORTED、發 /door/k2_abort', rc == 1 and (task or {}).get('final_phase') == 'ABORTED' and bool(got['k2_abort']),
       f"rc={rc} {got['k2_abort'][:1]}")
    print('\nT3  關閉旗標')
    rc, task, got, st, log = scenario('t3', 2, w2=False, run_s=12.0)
    ck('無請求發布、task.json 無 w2', got['req_task'] == [] and task is not None and 'w2' not in task, f"rc={rc}")
    print(f'\n{"全部通過" if not FAIL else "失敗：" + "、".join(FAIL)}')
    return 0 if not FAIL else 1


if __name__ == '__main__':
    sys.exit(main())
