#!/usr/bin/env python3
"""door_coopen_core_bench_6 暫停段與手指承載分報（只讀；Codex r31：分報兩指承載與完整停止觀察；與 bench_5 同法對照）。
  1. 停止觀察：求解端監督 outcome（ZERO_PUB／CONFIRM／STOPPED／DONE、覆蓋秒數）、判定器 r12 的獨立重算（n_moved、first_moved）。
  2. STOPPED 後（bench_6 至 DONE；bench_5 至 FAIL）：手臂 0.5 s 窗口速度最大／p99（room 逐步重算）、j2 線性斜率、兩指合力平均、每指平均、finger1 標準差。
  3. 夾持窗口（HOLD 起）每指 < 0.5 N 的步：時刻、相位、門角、兩指力；依相位計數；最長連續段（步）。
  4. 暫停窗口（PAUSE 起至 DONE／中止）每指平均、p05、最小、< 0.5 N 步數。
界線：描述；不判定原因。

    python3 evaluation/door_coopen_core_bench_6_pause_summary.py   # 寫 runs/door_coopen_core_bench_6/analysis/door_coopen_core_bench_6_pause_summary.json（不覆寫）
"""
import bisect
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = {k: os.path.join(HERE, 'runs', f'door_coopen_core_{k}') for k in ('bench_5', 'bench_6')}
OUT = os.path.join(RUNS['bench_6'], 'analysis', 'door_coopen_core_bench_6_pause_summary.json')


def one(run):
    r = json.load(open(os.path.join(run, 'room_run.json')))
    t = json.load(open(os.path.join(run, 'task.json')))
    sv = json.load(open(os.path.join(run, 'align_solver.json')))
    j = json.load(open(os.path.join(run, 'analysis', 'door_coopen_check_core12.json')))
    c = {k: i for i, k in enumerate(r['steps_cols'])}
    ph = sorted([(e['t'], e['to']) for e in t['policy_events'] if 'to' in e])
    pt = [p[0] for p in ph]

    def phase_at(x):
        i = bisect.bisect_right(pt, x + 1e-9) - 1
        return ph[i][1] if i >= 0 else None
    T = {p: tt for tt, p in ph}
    o = sv['stats']['k3m_pause']['supervisor_outcome']
    ts = o['transitions']['STOPPED']
    te = o['transitions'].get('DONE') or o['transitions'].get('FAIL')
    S = [s for s in r['steps'] if ts <= s[c['sim_t']] <= te]
    TT = np.array([s[c['sim_t']] for s in S])
    Q = np.array([s[c['q_arm_meas']] for s in S], float)
    F = np.array([s[c['finger_contact_n']] for s in S], float)
    w = []
    for i in range(len(TT)):
        k = int(np.searchsorted(TT, TT[i] - 0.5 - 1e-6))
        if TT[i] - TT[k] > 0.4:
            w.append(float((np.abs(Q[i] - Q[k]) / (TT[i] - TT[k])).max()))
    stop = {'supervisor': o, 'judge_pause': {k: (j['stats'].get('pause') or {}).get(k) for k in ('n_moved', 'first_moved', 'observation_covered_s',
                                                                                                  'command_stop_independent', 'physical_stop_independent')},
            'arm_win_max': max(w) if w else None, 'arm_win_p99': float(np.percentile(w, 99)) if w else None, 'n_arm_win_over': int(sum(x > 0.001 for x in w)),
            'j2_slope_rad_s': float(np.polyfit(TT - TT[0], Q[:, 1], 1)[0]) if len(TT) > 2 else None,
            'f_mean': F.mean(axis=0).tolist(), 'f_sum_mean': float(F.sum(axis=1).mean()), 'f1_std': float(F[:, 0].std())}
    lows, run_len, best = [], 0, 0
    for s in r['steps']:
        f = s[c['finger_contact_n']]
        if s[c['sim_t']] < T['HOLD'] or not isinstance(f, list):
            continue
        if min(f) < 0.5:
            lows.append({'t': s[c['sim_t']], 'phase': phase_at(s[c['sim_t']]), 'q_deg': math.degrees(s[c['door_q_rad']]), 'f': f})
            run_len += 1
            best = max(best, run_len)
        else:
            run_len = 0
    by = {}
    for x in lows:
        by[x['phase']] = by.get(x['phase'], 0) + 1
    tp = T.get('PAUSE')
    P = np.array([s[c['finger_contact_n']] for s in r['steps'] if tp is not None and tp <= s[c['sim_t']] <= te and isinstance(s[c['finger_contact_n']], list)], float)
    pause = None if not len(P) else {'n': int(len(P)), 'mean': P.mean(axis=0).tolist(), 'p05': np.percentile(P, 5, axis=0).tolist(), 'min': P.min(axis=0).tolist(),
                                     'n_below_0p5': (P < 0.5).sum(axis=0).tolist(), 'sum_mean': float(P.sum(axis=1).mean())}
    return {'stop': stop, 'below_min_after_hold': {'n': len(lows), 'by_phase': by, 'longest_run_steps': best, 'rows': lows[:60]}, 'pause_fingers': pause,
            'pause_finger_effort_set': ((r.get('finger_effort') or {}).get('pause') or {}).get('set')}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    out = {'schema': 'door_coopen_core_bench_6_pause_summary/1', **{k: one(v) for k, v in RUNS.items()}, 'note': '描述；不判定原因'}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    for k in RUNS:
        s = out[k]['stop']
        print(k, '上限', out[k]['pause_finger_effort_set'], 'N；監督', s['supervisor']['state'], '覆蓋 %.2f s' % s['supervisor']['observation_covered_s'],
              '判定器 n_moved', s['judge_pause']['n_moved'], '實際停止獨立', s['judge_pause']['physical_stop_independent'])
        print('   STOPPED 後：手臂窗口 max %.5f p99 %.5f 超過 %d；j2 斜率 %.2e；兩指均 %s 合力 %.2f；f1 std %.2f' % (
            s['arm_win_max'], s['arm_win_p99'], s['n_arm_win_over'], s['j2_slope_rad_s'], [round(v, 2) for v in s['f_mean']], s['f_sum_mean'], s['f1_std']))
        b = out[k]['below_min_after_hold']
        print('   HOLD 起 < 0.5 N 步數', b['n'], '依相位', b['by_phase'], '最長連續', b['longest_run_steps'], '步')
        p = out[k]['pause_fingers']
        print('   暫停窗口：每指均 %s p05 %s 最小 %s < 0.5 N 步 %s 合力 %.2f' % ([round(v, 2) for v in p['mean']], [round(v, 2) for v in p['p05']],
              [round(v, 2) for v in p['min']], p['n_below_0p5'], p['sum_mean']))


if __name__ == '__main__':
    main()
