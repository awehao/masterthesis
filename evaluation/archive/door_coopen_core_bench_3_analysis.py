#!/usr/bin/env python3
"""door_coopen_core_bench_3 分項分析（只讀；Codex r23／r24 要求分報：是否釋放、是否建立夾持、實測門角、開門同期出口進展）。
來源：room_run（逐物理步）、task.json（policy_events、trace）、判定器 r11 結果。
報告：
  1. 相位時刻與釋放。
  2. 夾持：ENGAGE→HOLD 的接觸證據；HOLD／OPEN 期間每指手指—橫桿力（room finger_contact_n）的最小值、低於 contact_min 0.5 N 的首時刻與累計時間；
     任務 trace 的 grasp drift（若有）與中止事件細節。
  3. 門角：OPEN 前最大（door_angle_pre_open）、OPEN 期間每 10 s 的實測門角與參考 q_ref、最大值。
  4. 出口進展：沿用判定器 core2 的 D（門角包絡場）報告值，另報底盤位移與偏航。
  5. 手指—橫桿接觸出現在非允許相位的時刻（finger_bar_outside_allowed_phase）。

    python3 evaluation/door_coopen_core_bench_3_analysis.py   # 寫 runs/door_coopen_core_bench_3/analysis/door_coopen_core_bench_3_analysis.json（不覆寫）
"""
import bisect
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, 'runs', 'door_coopen_core_bench_3')
OUT = os.path.join(RUN, 'analysis', 'door_coopen_core_bench_3_analysis.json')


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    r = json.load(open(os.path.join(RUN, 'room_run.json')))
    t = json.load(open(os.path.join(RUN, 'task.json')))
    c = {k: i for i, k in enumerate(r['steps_cols'])}
    st = r['steps']
    ev = t['policy_events']
    ph = [(e['t'], e['to']) for e in ev if 'to' in e]
    pt = [p[0] for p in ph]

    def phase_at(x):
        i = bisect.bisect_right(pt, x) - 1
        return ph[i][1] if i >= 0 else None
    T = {p: tt for tt, p in ph}
    out = {'phases': ph, 'abort': [e for e in ev if e.get('abort')], 'release': (r.get('bench') or {}).get('release'),
           'first_step_ok': ((r.get('bench') or {}).get('first_step') or {}).get('ok')}
    # 2. 夾持
    fb = []
    for s in st:
        f = s[c['finger_contact_n']]
        if isinstance(f, list) and len(f) == 2 and s[c['sim_t']] >= T['HOLD']:
            fb.append((s[c['sim_t']], f[0], f[1]))
    low = [(x, a, b) for x, a, b in fb if min(a, b) < 0.5]
    out['grasp'] = {'n_rows_after_hold': len(fb), 'finger_min_n': [min(x[1] for x in fb), min(x[2] for x in fb)] if fb else None,
                    'first_below_0p5_t': low[0][0] if low else None, 'first_below_phase': phase_at(low[0][0]) if low else None,
                    'n_rows_below_0p5': len(low), 'below_0p5_s': len(low) * 0.01,
                    'engage_to_hold_s': T['HOLD'] - T['ENGAGE'], 'max_grasp_drift': t.get('max_grasp_drift')}
    # 3. 門角
    dq = [(s[c['sim_t']], math.degrees(s[c['door_q_rad']])) for s in st]
    pre = [q for x, q in dq if x < T['OPEN']]
    samples = []
    for k in range(0, 110, 10):
        x = T['OPEN'] + k
        row = min(st, key=lambda s: abs(s[c['sim_t']] - x))
        trr = min(t.get('trace') or [{}], key=lambda y: abs(y.get('t', 1e9) - x))
        samples.append({'t': round(row[c['sim_t']], 2), 'q_meas_deg': round(math.degrees(row[c['door_q_rad']]), 3),
                        'q_ref_deg': None if trr.get('q_ref') is None else round(math.degrees(trr['q_ref']), 3),
                        'base_xyth': row[c['base_xyth']]})
    out['door'] = {'pre_open_max_deg': max(pre) if pre else None, 'pre_open_max_t': max(((x, q) for x, q in dq if x < T['OPEN']), key=lambda z: z[1])[0] if pre else None,
                   'open_max_deg': max(q for x, q in dq if x >= T['OPEN']), 'samples_every_10s': samples}
    # 4. 出口進展（判定器 r11 core2 欄位）＋底盤
    j = os.path.join(RUN, 'analysis', 'door_coopen_check_core11.json')
    jr = json.load(open(j)) if os.path.exists(j) else {}
    out['exit_progress_judge'] = (jr.get('items') or {}).get('exit_progress')
    b0 = next(s[c['base_xyth']] for s in st if s[c['sim_t']] >= T['OPEN'])
    b1 = [s[c['base_xyth']] for s in st if s[c['sim_t']] < T.get('ABORTED', 1e9)][-1]
    out['base_open'] = {'start': b0, 'end': b1, 'disp_m': math.hypot(b1[0] - b0[0], b1[1] - b0[1]), 'dyaw_deg': math.degrees(b1[2] - b0[2])}
    # 5. 非允許相位的手指—橫桿接觸
    allowed = ('ENGAGE', 'HOLD', 'OPEN', 'SETTLE', 'END_HOLD', 'DONE')
    bad = [(s[c['sim_t']], max(s[c['finger_contact_n']])) for s in st if isinstance(s[c['finger_contact_n']], list)
           and max(s[c['finger_contact_n']]) > 0.1 and phase_at(s[c['sim_t']]) not in allowed]
    out['finger_bar_outside'] = {'n': len(bad), 'first': bad[0] if bad else None, 'phase_first': phase_at(bad[0][0]) if bad else None,
                                 'max_n': max((x[1] for x in bad), default=None)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: out[k] for k in ('phases', 'abort', 'grasp', 'finger_bar_outside', 'base_open')}, ensure_ascii=False, default=float)[:2500])
    print('door', json.dumps({k: out['door'][k] for k in ('pre_open_max_deg', 'pre_open_max_t', 'open_max_deg')}))
    for s_ in out['door']['samples_every_10s']:
        print('  ', s_['t'], 'q', s_['q_meas_deg'], 'ref', s_['q_ref_deg'], 'base', [round(v, 3) for v in s_['base_xyth']])


if __name__ == '__main__':
    main()
