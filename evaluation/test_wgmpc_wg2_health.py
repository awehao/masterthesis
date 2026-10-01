"""第二趟前的最小核對（**不開 Isaac、不用 GPU**）。

只補三個直接對應的測試：
  H1 **健康零命令不誤判** —— exec_mode=0 的零回報不得停止任務
  H2 **注入鏈失效會中止任務** —— exec_mode=3／applied_fail 必須停止並記錄原因
  H3 **正常收尾能留下完整執行紀錄** —— 受控停止路徑與封存狀態契約

H3 的完整路徑（Isaac 自己落盤）只能在實跑驗證；本檔驗的是
執行端有該停止條件、runner 先等封存再 cleanup、以及狀態檔契約。
"""
from __future__ import annotations
import json
import os
import re
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
SC = os.environ.get('WG2H_TMP', f'/tmp/wg2h_{os.getpid()}')
DOMAIN = os.environ.get('WG2H_DOMAIN', '171')
_bad = 0


def ck(name, cond, extra=''):
    global _bad
    print(f'  {name:58s} {"ok" if cond else "**錯**"}{extra}')
    _bad += not cond


def run(name, stub_mode, fail_at=20, duration=10.0):
    os.makedirs(SC, exist_ok=True)
    out = os.path.join(SC, f'{name}.json')
    env = dict(os.environ, ROS_DOMAIN_ID=DOMAIN)
    ps = []
    try:
        ps.append(subprocess.Popen(
            [sys.executable, os.path.join(HERE, 'coman_solver_sched_world.py'),
             '--duration-s', str(duration + 20), '--stop-js-at', '-1'],
            cwd=WS, env=env, stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT, start_new_session=True))
        time.sleep(3.0)
        ps.append(subprocess.Popen(
            [sys.executable, os.path.join(HERE, 'wgmpc_wg2_applied_stub.py'),
             '--mode', stub_mode, '--fail-at', str(fail_at)],
            cwd=WS, env=env, stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT, start_new_session=True))
        time.sleep(2.0)
        ps.append(subprocess.Popen(
            ['ros2', 'run', 'ammr_wholebody_mpc', 'wholebody_safety',
             '--ros-args', '-p', 'use_sim_time:=true',
             '-p', 'report_frame:=odom', '-p', 'base_frame:=base_link',
             '-p', 'wholebody_urdf:=' + os.path.join(
                 HERE, 'models', 'omni_bot_wholebody_expanded.urdf'),
             '-p', 'vmax_base_lin:=0.035255', '-p', 'vmax_base_ang:=0.199900',
             '-p', 'vmax_arm:=0.999900'],
            cwd=WS, env=env, stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT, start_new_session=True))
        time.sleep(5.0)
        ps.append(subprocess.Popen(
            [sys.executable, os.path.join(HERE, 'arm_vel_adapter.py'),
             '--consumer-node', '/wgmpc_wg2_applied_stub'],
            cwd=WS, env=env, stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT, start_new_session=True))
        time.sleep(4.0)
        n = subprocess.Popen(
            [sys.executable, '-u', os.path.join(HERE, 'wgmpc_wg2_node.py'),
             '--N', '5', '--rate', '20', '--target-offset', '0.25', '0.15',
             '0.05', '--duration-s', str(duration), '--out', out,
             '--u-prev-policy', 'strict', '--assume-initial-rest'],
            cwd=WS, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, start_new_session=True)
        ps.append(n)
        log = n.communicate(timeout=duration + 60)[0]
    finally:
        for p in ps:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        time.sleep(1.5)
        for p in ps:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    return (json.load(open(out)) if os.path.exists(out) else None), (log or '')


def main() -> int:
    print(f'暫存 {SC}　domain {DOMAIN}　（不開 Isaac、不用 GPU）\n')

    # ---------- H1 健康零命令不誤判 ----------
    print('H1 健康零命令（exec_mode=0 的零回報）**不得**停止任務')
    d, log = run('healthy_zero', 'healthy-zero', duration=10.0)
    ck('節點有產出紀錄', d is not None)
    if d is not None:
        lg = d['log']
        ck('**沒有**因鏈失效停止', not d.get('stopped_on_chain_fail'),
           f'  stopped={d.get("stopped_on_chain_fail")}')
        ck('沒有 chain_fail_latched 的輪次',
           not any(r.get('reason') == 'chain_fail_latched' for r in lg))
        pub = [r for r in lg if r.get('published')]
        ck('持續發布（零回報不阻斷任務）', len(pub) >= 20,
           f'  {len(pub)} / {len(lg)} 輪')
        srcs = {r.get('u_prev_src') for r in lg}
        ck('u_prev 仍取權威的 applied', 'applied' in srcs, f'  {sorted(srcs)}')
        st = [r for r in pub if r.get('stages')]
        ck('四段紀錄存在且標為未配對',
           bool(st) and st[0]['stages']['pairing'].startswith('unpaired'),
           f'  {st[0]["stages"]["pairing"] if st else "—"}')
        hs = [r['stages']['applied']['health'] for r in st
              if r['stages'].get('applied')]
        ck('applied 帶執行健康狀態（exec_mode／cmd_age／n_recv）',
           bool(hs) and all('exec_mode' in h and 'cmd_age_s' in h
                            and 'n_recv' in h for h in hs))
        ck('健康零回報的 exec_mode 皆為 0',
           bool(hs) and all(h['exec_mode'] == 0 for h in hs),
           f'  模式集合 {sorted({h["exec_mode"] for h in hs})}' if hs else '')

    # ---------- H2 注入鏈失效會中止任務 ----------
    print('\nH2 注入鏈失效（exec_mode=3 ＋ applied_fail）**必須**中止任務')
    d2, log2 = run('fail_latch', 'fail-at', fail_at=15, duration=20.0)
    ck('節點有產出紀錄', d2 is not None)
    if d2 is not None:
        lg2 = d2['log']
        ck('**因鏈失效停止**', bool(d2.get('stopped_on_chain_fail')),
           f'  stopped={d2.get("stopped_on_chain_fail")}')
        fr = [r for r in lg2 if r.get('reason') == 'chain_fail_latched']
        ck('有 chain_fail_latched 的紀錄', len(fr) == 1, f'  {len(fr)} 筆')
        ck('**記下了明確的失效原因**',
           bool(d2.get('chain_fail_info'))
           and '限位' in str(d2.get('chain_fail_info')),
           f'  {str(d2.get("chain_fail_info"))[:70]}')
        ck('停止後**沒有**繼續發布',
           not any(r.get('published') for r in lg2[len(lg2) - 1:]))
        ck('**不是**當成健康零命令繼續跑',
           len(lg2) < 20 * 20, f'  {len(lg2)} 輪（20 s × 20 Hz = 400）')
        print(f'    停止訊息：{[l for l in log2.splitlines() if "失效閂鎖" in l][:1]}')

    # ---------- H3 正常收尾能留下完整執行紀錄 ----------
    print('\nH3 受控停止與封存（**完整路徑只能在 Isaac 實跑驗證**）')
    ep = open(os.path.join(HERE, 'isaac_wholebody_sim_e2.py'),
              encoding='utf-8').read()
    ck('執行端訂閱 /wb_sim/stop_request', '/wb_sim/stop_request' in ep)
    ck('收到請求後走**正常收尾**（break 到封存，非 SIGTERM）',
       "stop = 'stop_request'; break" in ep)
    ck('執行端本來就有 chain.fail 的停止條件',
       "'cmd_chain_fail'" in ep)
    rn = open(os.path.join(HERE, 'run_wgmpc_wg2_free.sh'),
              encoding='utf-8').read()
    ck('runner 先發受控停止請求', '/wb_sim/stop_request' in rn)
    ck('runner 等封存完成且**有上限**',
       'ARCHIVE_WAIT_S' in rn and 'seq "$ARCHIVE_WAIT_S"' in rn)
    ck('超時才升級並**標記封存不完整**',
       'archive_complete": false' in rn and '升級終止' in rn)
    ck('cleanup 在等待**之後**',
       rn.index('ARCHIVE_WAIT_S') < rn.index('[收尾 3/3]'))
    print('    **H3 的完整路徑（Isaac 自己落盤）需實跑驗證**；'
          '本檔只驗停止條件、等待上限與狀態檔契約。')

    print()
    print('第二趟前最小核對：' + ('全部通過' if _bad == 0 else f'**{_bad} 項失敗**'))
    return 1 if _bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
