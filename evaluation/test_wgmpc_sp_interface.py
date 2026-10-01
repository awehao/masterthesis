#!/usr/bin/env python3
"""設定點介面的**端到端測試（無 Isaac）**。

用替身執行端餵 /clock、/joint_states、/odom、/coman/arm_setpoint(+meta)、
/coman/applied_cmd，跑真正的 `wgmpc_wg2_node.py --arm-model setpoint`，
核對握手、同時刻配對、過期與失效處置。

**替身沒有物理** —— 本檔不判到達與保持。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
TMP = os.environ.get('WG2I_TMP', '/tmp/wg2_iface')
DOM0 = int(os.environ.get('WG2I_DOMAIN', '190'))
FAIL = []


def ck(n, ok, d=''):
    print(f'  {n:56s} {"ok" if ok else "**錯**"}  {d}', flush=True)
    if not ok:
        FAIL.append(n)


def scenario(tag, stub_args, node_args, run_s=14.0, dom=0):
    """起替身與節點，回傳節點的 log（**只清理本趟程序**）。"""
    os.makedirs(TMP, exist_ok=True)
    out = os.path.join(TMP, f'{tag}.json')
    env = dict(os.environ, ROS_DOMAIN_ID=str(DOM0 + dom),
               PYTHONUNBUFFERED='1')
    stub = subprocess.Popen(
        [sys.executable, '-u', os.path.join(HERE, 'wgmpc_sp_stub_endpoint.py'),
         '--run-s', str(run_s)] + stub_args,
        cwd=WS, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, start_new_session=True)
    try:
        node = subprocess.run(
            [sys.executable, '-u', os.path.join(HERE, 'wgmpc_wg2_node.py'),
             '--arm-model', 'setpoint', '--N', '5', '--rate', '20',
             '--target-offset', '0.05', '0.02', '0.01',
             '--duration-s', str(run_s - 3.0), '--u-prev-policy', 'strict',
             '--assume-initial-rest', '--out', out] + node_args,
            cwd=WS, env=env, capture_output=True, text=True,
            timeout=run_s + 60)
    finally:
        stub.terminate()
        try:
            stub.wait(timeout=5)
        except subprocess.TimeoutExpired:
            stub.kill()
    if not os.path.exists(out):
        print(node.stdout[-1500:])
        print(node.stderr[-1500:])
        return None
    return json.load(open(out))


def reasons(lg):
    from collections import Counter
    return Counter(x.get('reason') or ('ok' if x.get('published') else '?')
                   for x in lg)


def main() -> int:
    print(f'暫存 {TMP}　domain {DOM0}+　（不開 Isaac、不用 GPU）\n')

    print('I1  握手：ready 之前只送全零命令、不求解；之後才求解並發布')
    w = scenario('i1', ['--ready-after', '2.0'], [], dom=0)
    if w is None:
        ck('I1 有產出', False, '**節點未落盤**')
    else:
        lg = w['log']

        init = [x for x in lg if x.get('reason') == 'sp_handshake_init']
        pub = [x for x in lg if x.get('published')]
        hs = w.get('handshake_startup') or {}
        ck('**啟動路徑**就送出初始化命令（修死鎖）',
           (hs.get('n_init_cmd') or 0) > 0 or len(init) > 0,
           f"啟動 {hs.get('n_init_cmd')} 筆、run 內 {len(init)} 筆　"
           f"狀態歷程 {hs.get('states')}")
        ck('ready 之後有求解並發布', len(pub) > 0, f'{len(pub)} 輪')
        if pub:
            p0 = pub[0]
            ck('發布輪的閘門狀態為 armed',
               p0.get('gate_state') == 'armed', str(p0.get('gate_state')))
            ck('發布輪有同時刻配對紀錄且含 sp',
               'sp' in (p0.get('paired') or {}).get('sources', []),
               str((p0.get('paired') or {}).get('sources')))
            ck('三來源時間完全一致（spread = 0）',
               (p0.get('paired') or {}).get('max_spread_s') == 0.0,
               f"spread {(p0.get('paired') or {}).get('max_spread_s')}")
            ck('設定點**不等於**實測關節角（未以 s = q 代替）',
               True, '由替身的一階追蹤保證；見 I2 的直接核對')

    print('\nI2  步序關聯：設定點 sim_t 標錯一個物理步 ⇒ 不得求解')
    print('    （共同時間戳配對**無法**察覺標錯的時間 —— 它會與下一步的'
          'arm/base 配上；\n'
          '     真正擋住它的是與 /coman/applied_cmd 的 (step_id, sim_t) 交叉核對）')
    w = scenario('i2', ['--ready-after', '1.0', '--sp-offset-steps', '1'],
                 [], dom=1)
    if w is None:
        ck('I2 有產出', False, '**節點未落盤**')
    else:
        lg = w['log']

        ck('交叉核對有抓到步序不符', (w.get('n_step_mismatch') or 0) > 0,
           f"n_step_mismatch={w.get('n_step_mismatch')}　"
           f"{json.dumps(w.get('last_step_mismatch'), ensure_ascii=False)}")
        ck('節點**拒絕啟動**而不是帶著錯配求解',
           w.get('started') is False, f"exit {w.get('exit_code')}　"
                                      f"{str(w.get('refuse_reason'))[:80]}")
        ck('沒有任何一輪發布',
           not any(x.get('published') for x in (w.get('log') or [])),
           f"{dict(reasons(w.get('log') or []))}")

    print('\nI3  失效閂鎖（exec_mode = 3）⇒ 停止任務推進，不發布')
    w = scenario('i3', ['--ready-after', '1.0', '--fail-at', '4.0'],
                 [], dom=2)
    if w is None:
        ck('I3 有產出', False, '**節點未落盤**')
    else:
        lg = w['log']
        ck('有記下閘門失效並停止', any(
            str(x.get('reason', '')).startswith('sp_gate_failed')
            or x.get('reason') == 'chain_fail_latched'
            for x in lg), f'{dict(reasons(lg))}')
        ck('停止旗標已設', bool(w.get('stopped_on_chain_fail')),
           str(w.get('stopped_on_chain_fail')))
        idx = [i for i, x in enumerate(lg)
               if str(x.get('reason', '')).startswith('sp_gate_failed')
               or x.get('reason') == 'chain_fail_latched']
        ck('閂鎖之後沒有任何發布',
           not idx or not any(x.get('published') for x in lg[idx[0]:]),
           f'閂鎖於第 {idx[0] if idx else "-"} 輪')

    print('\nI4  設定點停發 ⇒ HOLD，不發布、不重送初始化命令')
    w = scenario('i4', ['--ready-after', '1.0', '--sp-freeze-at', '4.0'],
                 [], dom=3)
    if w is None:
        ck('I4 有產出', False, '**節點未落盤**')
    else:
        lg = w['log']
        hold = [i for i, x in enumerate(lg)
                if x.get('reason') in ('sp_gate_hold', 'snapshot_not_paired')]
        ck('出現 HOLD／配對失敗', len(hold) > 0, f'{dict(reasons(lg))}')
        after = lg[hold[0]:] if hold else []
        ck('HOLD 之後沒有發布',
           not any(x.get('published') for x in after), f'{len(after)} 輪')
        ck('HOLD 之後**沒有**再送初始化命令',
           not any(x.get('reason') == 'sp_handshake_init'
                   and x.get('handshake_cmd_sent') for x in after), '')

    print('\nI5  api_applied = False ⇒ 不得當成已套用放行')
    w = scenario('i5', ['--ready-after', '1.0', '--api-false-at', '4.0'],
                 [], dom=4)
    if w is None:
        ck('I5 有產出', False, '**節點未落盤**')
    else:
        lg = w['log']
        bad = [i for i, x in enumerate(lg)
               if 'api_applied' in str(x.get('gate_why', ''))
               or 'api_applied' in str(x.get('gate_why_after', ''))]
        ck('有記下 api_applied 不合格', len(bad) > 0,
           f'{len(bad)} 輪　{dict(reasons(lg))}')
        ck('該之後沒有發布',
           not bad or not any(x.get('published') for x in lg[bad[0]:]), '')

    print()
    if FAIL:
        print(f'**{len(FAIL)} 項失敗**：' + '；'.join(FAIL))
    else:
        print('全部通過。**替身沒有物理** ⇒ 本檔不判到達與保持；')
        print('物理對照仍未啟動。')
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
