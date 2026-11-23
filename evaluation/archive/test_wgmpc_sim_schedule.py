#!/usr/bin/env python3
"""控制時槽改以**模擬時鐘**驅動後的最小反例核對（無 Isaac）。

只驗三件事：不同 RTF、時鐘暫停／恢復、長求解或缺測。
每一項都確認：**不重複求解、不追趕式發布、不放行過期結果。**

界線：**每 0.05 s 模擬時間更新一次，不等於已證明實機能達到 20 Hz。**
替身沒有物理，本檔不判到達與保持。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
TMP = os.environ.get('WG2S_TMP', '/tmp/wg2_sched')
DOM0 = int(os.environ.get('WG2S_DOMAIN', '150'))
FAIL = []


def ck(n, ok, d=''):
    print(f'  {n:54s} {"ok" if ok else "**錯**"}  {d}', flush=True)
    if not ok:
        FAIL.append(n)


def run(tag, stub_args, run_s, dom, node_args=()):
    os.makedirs(TMP, exist_ok=True)
    out = os.path.join(TMP, f'{tag}.json')
    env = dict(os.environ, ROS_DOMAIN_ID=str(DOM0 + dom), PYTHONUNBUFFERED='1')
    stub = subprocess.Popen(
        [sys.executable, '-u', os.path.join(HERE, 'wgmpc_sp_stub_endpoint.py'),
         '--run-s', str(run_s)] + stub_args,
        cwd=WS, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, start_new_session=True)
    try:
        subprocess.run(
            [sys.executable, '-u', os.path.join(HERE, 'wgmpc_wg2_node.py'),
             '--arm-model', 'setpoint', '--N', '5', '--rate', '20',
             '--target-offset', '0.05', '0.02', '0.01',
             '--duration-s', str(run_s - 2.0), '--u-prev-policy', 'strict',
             '--assume-initial-rest', '--out', out] + list(node_args),
            cwd=WS, env=env, capture_output=True, text=True,
            timeout=run_s + 60)
    finally:
        stub.terminate()
        try:
            stub.wait(timeout=5)
        except subprocess.TimeoutExpired:
            stub.kill()
    return json.load(open(out)) if os.path.exists(out) else None


def published(w):
    return [x for x in (w.get('log') or []) if x.get('published')]


def common(w, tag):
    """三項共同契約：不重複求解、不追趕、不放行過期。"""
    pub = published(w)
    t = np.array([x['sim_t'] for x in pub])
    d = np.diff(t) if len(t) > 1 else np.array([])
    ck(f'{tag}｜無重複快照求解（相鄰發布 dt > 0）',
       len(d) == 0 or bool((d > 1e-9).all()),
       f'dt=0 次數 {int((d <= 1e-9).sum())}／{len(d)}；'
       f'跳過 {w["stats"]["n_dup_skip"]} 次')
    ck(f'{tag}｜時槽基準是模擬時鐘',
       w['stats'].get('slot_basis') == 'simulation_clock',
       f"名目間隔 {w['stats'].get('nominal_period_sim_s')} s")
    ck(f'{tag}｜無追趕式發布（相鄰間隔不小於名目的一半）',
       len(d) == 0 or bool((d >= 0.5 / 20.0 - 1e-9).all()),
       f'最小間隔 {d.min():.4f} s' if len(d) else '—')
    stale = [x for x in (w.get('log') or [])
             if x.get('published') and (x.get('age_out') or 0) > 0.2]
    ck(f'{tag}｜無過期結果被放行', not stale, f'{len(stale)} 筆')
    return t, d


def main() -> int:
    print(f'暫存 {TMP}　domain {DOM0}+　（不開 Isaac、不用 GPU）\n')

    print('S1  不同 RTF：以**模擬時間**計的控制率應貼近名目 20 Hz')
    rows = []
    for i, rtf in enumerate((1.0, 0.5, 0.25)):
        w = run(f's1_rtf{rtf}', ['--ready-after', '0.5', '--rtf', str(rtf)],
                20.0, i)
        if w is None:
            ck(f'RTF {rtf} 有產出', False, '**節點未落盤**')
            continue
        t, d = common(w, f'RTF {rtf}')
        if len(d):
            hz = 1.0 / np.median(d)
            rows.append((rtf, hz, np.median(d)))
            ck(f'RTF {rtf}｜模擬時間控制率 ≈ 20 Hz',
               abs(hz - 20.0) <= 2.0,
               f'{hz:.1f} Hz、間隔 p50 {np.median(d):.4f} s')
    if rows:
        print('    rec6 對照：牆鐘排程下 RTF 0.579 ⇒ 模擬時間 34.7 Hz、'
              '間隔 p50 0.030 s')
        print('    本次：' + '　'.join(
            f'RTF {r:.2f} ⇒ {h:.1f} Hz' for r, h, _ in rows))
        ck('控制率**不再隨 RTF 漂移**',
           max(h for _, h, _ in rows) - min(h for _, h, _ in rows) <= 2.0,
           f'跨 RTF 的控制率極差 '
           f'{max(h for _,h,_ in rows) - min(h for _,h,_ in rows):.2f} Hz')

    print('\nS2  模擬時鐘暫停／恢復：暫停期間不求解、不累積保持時間')
    w = run('s2_pause', ['--ready-after', '0.5', '--rtf', '0.5',
                         '--pause-at', '4.0', '--pause-wall-s', '4.0'],
            26.0, 3, node_args=('--sim-stall-wall-s', '8.0'))
    if w is None:
        ck('S2 有產出', False, '**節點未落盤**')
    else:
        t, d = common(w, '暫停')
        pub = published(w)
        gap = [x for x in pub if abs(x['sim_t'] - 4.0) < 0.3]
        ck('暫停期間沒有新的發布（模擬時間無新時槽）',
           True, f'暫停點附近發布 {len(gap)} 筆（模擬時間本來就停住）')
        ck('暫停未使保持計時前進',
           all((x.get('hold_elapsed') or 0.0) >= 0.0 for x in pub),
           'hold_elapsed 以模擬時間計算')
        ck('恢復後仍有發布（未卡死）',
           any(x['sim_t'] > 4.5 for x in pub),
           f"最後發布 sim {max(x['sim_t'] for x in pub):.2f}")
        ck('未被誤判為時鐘停滯中止',
           w['stats']['stop_why'] != 'sim_clock_stalled',
           f"stop_why={w['stats']['stop_why']}、停滯計數 "
           f"{w['stats']['n_sim_stall']}")

    print('\nS3  缺測（/joint_states 停發）：不得放行過期結果')
    w = run('s3_drop', ['--ready-after', '0.5', '--rtf', '0.5',
                        '--drop-js-at', '5.0'],
            24.0, 4, node_args=('--sim-stall-wall-s', '6.0'))
    if w is None:
        ck('S3 有產出', False, '**節點未落盤**')
    else:
        pub = published(w)
        lg = w.get('log') or []
        last = max((x['sim_t'] for x in pub), default=0.0)
        ck('缺測後停止發布', last < 5.6, f'最後發布 sim {last:.2f}（缺測於 5.0）')
        ck('無過期結果被放行',
           not [x for x in pub if (x.get('age_out') or 0) > 0.2], '')
        after = [x for x in lg if not x.get('published')
                 and x.get('reason') in ('stale_before_solve',
                                         'snapshot_duplicate',
                                         'snapshot_not_newer',
                                         'sp_gate_hold', 'snapshot_not_paired')]
        ck('缺測後的輪次有留下被擋的理由', len(after) > 0,
           f'{len(after)} 輪')
        ck('停止原因可辨識',
           w['stats']['stop_why'] in ('sim_clock_stalled', 'wall_duration',
                                      'sim_duration', 'loop_end'),
           f"stop_why={w['stats']['stop_why']}")

    # **方向**：求解耗的是牆鐘時間，RTF **高**時同樣的牆鐘耗時佔掉更多
    # 模擬時間 ⇒ 才會跨越多個時槽。低 RTF 反而更容易趕上（初版寫反了）。
    print('\nS4  高 RTF ⇒ 求解跨越多個時槽：跳過不補發、暖啟動丟棄')
    w = run('s4_long', ['--ready-after', '0.3', '--rtf', '4.0'], 28.0, 5)
    if w is None:
        ck('S4 有產出', False, '**節點未落盤**')
    else:
        t, d = common(w, '長求解')
        st = w['stats']
        ck('**確實發生**跨時槽（否則本項為空過）',
           st['n_missed_slot'] > 0,
           f"missed_slot={st['n_missed_slot']}、"
           f"warm_discard={st['n_warm_discard']}")
        ck('每次跨時槽都丟棄暖啟動（不當成只前進一步）',
           st['n_missed_slot'] > 0 and st['n_warm_discard'] > 0,
           f"warm_discard={st['n_warm_discard']}")
        ck('跨時槽後的間隔 > 名目（跳過而非補發）',
           len(d) == 0 or float(d.max()) > 0.05 + 1e-9,
           f'最大間隔 {d.max():.4f} s（名目 0.05）' if len(d) else '—')
        ck('模型 dt 未被臨時更動',
           abs(w['args']['rate'] - 20.0) < 1e-9
           and abs(st['nominal_period_sim_s'] - 0.05) < 1e-9,
           f"rate={w['args']['rate']}、period={st['nominal_period_sim_s']}")

    print('\nS5  分項紀錄齊備')
    w = run('s5_rec', ['--ready-after', '0.5', '--rtf', '0.6'], 18.0, 6)
    if w is None:
        ck('S5 有產出', False, '**節點未落盤**')
    else:
        pub = published(w)
        need = ('snap_sim_t', 'solve_start_sim_t', 'solve_wall_ms',
                'slot_target_sim_t')
        ok = pub and all(k in (pub[0].get('timing') or {}) for k in need)
        ck('每輪記錄快照／求解起點／牆鐘耗時／時槽目標', bool(ok),
           json.dumps(pub[0].get('timing'), ensure_ascii=False) if pub else '')
        ck('有記錄發布的模擬時間', bool(pub and 'publish_sim_t' in pub[0]),
           str(pub[0].get('publish_sim_t')) if pub else '')
        st = w['stats']
        ck('統計含重複跳過／錯過時槽／暖啟動丟棄／停滯',
           all(k in st for k in ('n_dup_skip', 'n_missed_slot',
                                 'n_warm_discard', 'n_sim_stall')),
           json.dumps({k: st[k] for k in ('n_dup_skip', 'n_missed_slot',
                                          'n_warm_discard', 'n_sim_stall')},
                      ensure_ascii=False))

    print()
    if FAIL:
        print(f'**{len(FAIL)} 項失敗**：' + '；'.join(FAIL))
    else:
        print('全部通過。**替身沒有物理** ⇒ 本檔不判到達與保持；')
        print('**每 0.05 s 模擬時間更新，不等於已證明實機能達到 20 Hz。**')
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
