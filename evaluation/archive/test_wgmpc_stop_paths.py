#!/usr/bin/env python3
"""A0 停止路徑測試（**不開 Isaac、不用 GPU**；用 test_wgmpc_wg2_iface 的最小假世界）。

驗兩條路徑，求解節點都跑連續模式（抽屜任務用的那個模式）：
  S  /wgmpc/stop：統計由計數器寫出；收到停止後發布次數 = 0；
     所有已發布輪次的發布時刻都早於（或等於）收到停止的模擬時刻
  T  SIGTERM（ROS 外部關閉）：stop_why = external_shutdown；統計照樣寫出，
     且與逐輪紀錄對帳（published／n_solve_calls）

    source install/setup.bash && python3 evaluation/test_wgmpc_stop_paths.py
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
SC = os.environ.get('WG2_TMP', f'/tmp/wg2stop_{os.getpid()}')
DOMAIN = os.environ.get('WG2_DOMAIN', '163')
URDF = os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf')
_bad = 0


def ck(name, cond, extra=''):
    global _bad
    print(f'  {name:60s} {"ok" if cond else "**錯**"}{extra}')
    if not cond:
        _bad += 1


def run_case(name, action, at_s=10.0, duration=40.0):
    """action = 'stop' 或 'term'；在求解節點啟動 at_s 秒（牆鐘）後執行。"""
    os.makedirs(SC, exist_ok=True)
    out = os.path.join(SC, f'{name}.json')
    env = dict(os.environ, ROS_DOMAIN_ID=DOMAIN)
    procs = []
    log = ''
    try:
        for cmd, wait in (
                ([sys.executable, os.path.join(HERE, 'coman_solver_sched_world.py'),
                  '--duration-s', str(duration + 30), '--stop-js-at', '-1'], 3.0),
                ([sys.executable, os.path.join(HERE, 'wgmpc_wg2_sink.py'),
                  '--node-name', 'wgmpc_wg2_sink'], 2.0),
                (['ros2', 'run', 'ammr_wholebody_mpc', 'wholebody_safety',
                  '--ros-args', '-p', 'use_sim_time:=true',
                  '-p', 'report_frame:=odom', '-p', 'base_frame:=base_link',
                  '-p', f'wholebody_urdf:={URDF}',
                  '-p', 'vmax_base_lin:=0.035255',
                  '-p', 'vmax_base_ang:=0.199900', '-p', 'vmax_arm:=0.999900'], 0.0),
                ([sys.executable, os.path.join(HERE, 'arm_vel_adapter.py'),
                  '--consumer-node', '/wgmpc_wg2_sink'], 4.0)):
            procs.append(subprocess.Popen(cmd, cwd=WS, env=env, stdout=subprocess.DEVNULL,
                                          stderr=subprocess.STDOUT, start_new_session=True))
            time.sleep(wait)
        n = subprocess.Popen(
            [sys.executable, '-u', os.path.join(HERE, 'wgmpc_wg2_node.py'),
             '--N', '5', '--rate', '20', '--target', '10.80', '8.40', '0.60',
             '--continuous', '--duration-s', str(duration), '--out', out,
             '--u-prev-policy', 'diagnostic', '--assume-initial-rest'],
            cwd=WS, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, start_new_session=True)
        procs.append(n)
        time.sleep(at_s)
        if action == 'stop':
            subprocess.run(['ros2', 'topic', 'pub', '--once', '/wgmpc/stop',
                            'std_msgs/msg/String', '{data: test_stop}'],
                           cwd=WS, env=env, stdout=subprocess.DEVNULL,
                           stderr=subprocess.STDOUT, timeout=30)
        else:
            os.kill(n.pid, signal.SIGTERM)
        log = n.communicate(timeout=60)[0]
    finally:
        for p in procs:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        time.sleep(1.5)
        for p in procs:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    d = json.load(open(out)) if os.path.exists(out) else None
    return d, log


def reconcile(d):
    log = d.get('log') or []
    st = d.get('stats') or {}
    n_pub = sum(1 for L in log if L.get('published') is True)
    n_solve = sum(1 for L in log if 'sqp_stop' in L)
    return n_pub, n_solve, st.get('published'), st.get('n_solve_calls')


def main() -> int:
    print(f'暫存 {SC}　domain {DOMAIN}　（不開 Isaac、不用 GPU）\n')

    print('S /wgmpc/stop 路徑')
    d, log = run_case('stop', 'stop')
    ck('寫出結果檔', d is not None)
    if d:
        st = d.get('stats') or {}
        ck('stats 存在（非 None）', bool(st))
        ck('stop_why 為 stop_topic', str(st.get('stop_why', '')).startswith('stop_topic'),
           f"  {st.get('stop_why')}")
        rx = (st.get('stop_rx') or {}).get('sim_t')
        ck('記下收到停止的模擬時刻', rx is not None, f'  {rx}')
        ck('收到停止後發布次數 = 0（獨立計數）', st.get('n_published_after_stop_rx') == 0,
           f"  {st.get('n_published_after_stop_rx')}")
        pubs = [L.get('publish_sim_t_exact') for L in d.get('log', []) if L.get('published')]
        late = [t for t in pubs if t is not None and rx is not None and t > rx + 1e-9]
        ck('所有發布都不晚於收到停止的時刻', rx is not None and not late,
           f'  已發布 {len(pubs)} 輪、晚於停止 {len(late)}')
        ck('停止前確實有在發布（測試有效）', len(pubs) > 0)
        a, b, c, e = reconcile(d)
        ck('對帳：published 列數 = stats', a == c, f'  {a} vs {c}')
        ck('對帳：求解列數 = n_solve_calls', b == e, f'  {b} vs {e}')

    print('\nT SIGTERM（外部關閉）路徑')
    d, log = run_case('term', 'term')
    ck('寫出結果檔', d is not None)
    if d:
        st = d.get('stats') or {}
        ck('stats 存在（非 None）', bool(st))
        ck('stop_why = external_shutdown', st.get('stop_why') == 'external_shutdown',
           f"  {st.get('stop_why')}")
        ck('不是部分統計（正常由計數器收尾）', not st.get('stats_partial'))
        a, b, c, e = reconcile(d)
        un = st.get('n_solve_calls_unlogged')
        ck('關閉前確實有在發布（測試有效）', a > 0, f'  {a}')
        ck('對帳：published 列數 = stats', a == c, f'  {a} vs {c}')
        ck('未寫入的求解輪 ≤ 1（被關閉中斷的那一輪）', un in (0, 1), f'  {un}')
        ck('對帳：求解列數 = n_solve_calls − 未寫入', e is not None and un is not None
           and b == e - un, f'  {b} vs {e} − {un}')
        ck('n_solve_rows_logged = 求解列數', st.get('n_solve_rows_logged') == b,
           f"  {st.get('n_solve_rows_logged')} vs {b}")

    print(f'\n停止路徑測試：{"全部通過" if _bad == 0 else f"**{_bad} 項失敗**"}')
    return 0 if _bad == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
