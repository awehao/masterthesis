#!/usr/bin/env python3
"""K3-A-U 單趟介入觀察報告（只讀；依 K3_hold_approach_unload_spec.md 修訂 1 §10 的事前固定統計窗）。

窗：卸載前 [HOLD 起＋0.1 s, t_cap)；卸載後 [t_cap＋0.5 s, OPEN)（保持段）；OPEN → 物理首次單指 < 0.5 N；首次低接觸 → 中止；中止後。
每窗：兩指回報切向合量的把手 y 分量合計（中位、範圍、有效筆數；任一指缺值 ⇒ 該步 NA）、兩指 Σ|fn|、
已發布目標（task trace，依來源時間）− FK（把手座標，mm）。另報擷取事件、斜坡／保持／OPEN 時序與 OPEN 首筆連續性。
對照：給第二個 run（無卸載）時，以「HOLD 起＋0.1 s → HOLD 起＋0.5 s」與「HOLD 起＋0.5 s → OPEN」同長度定義對照窗（只描述）。

    python3 evaluation/door_k3au_report.py <run_dir> [對照 run_dir]   # 寫 <run_dir>/analysis/door_k3au_report.json（已存在則拒絕覆寫）
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k2_check as KC                            # noqa: E402
import door_k3a_preload_locate as PL                  # noqa: E402

FINGERS = ('uflite_finger1', 'uflite_finger2')


def load(run):
    t = json.load(open(os.path.join(run, 'task.json')))
    r = json.load(open(os.path.join(run, 'room_run.json')))
    ev_rows = [json.loads(ln) for ln in open(os.path.join(run, 'evidence.jsonl'))] if os.path.exists(os.path.join(run, 'evidence.jsonl')) else []
    return t, r, ev_rows


def series(t, r, ev_rows):
    I = {c: i for i, c in enumerate(r['steps_cols'])}
    room = {s[I['step']]: s for s in r['steps']}
    tr = sorted([e for e in t['trace'] if e.get('target') is not None], key=lambda e: e['t'])
    tr_t = np.array([e['t'] for e in tr])
    out = []
    for x in ev_rows:
        s = room.get(x['step'])
        if s is None or x['t'] is None:
            continue
        R = np.array(x['T_WH']).reshape(4, 4)[:3, :3]
        fs = PL.finger_sums(x, R)
        fa = []
        for b in FINGERS:
            lst = (x.get('contacts') or {}).get(b, [])
            fa.append(sum(abs(c['fn']) for c in lst if c.get('fn') is not None))
        H = KC._H_of(s[I['door_q_rad']])
        Tm = KC._tcp(s[I['base_xyth']], s[I['q_arm_meas']])
        j = int(np.searchsorted(tr_t, x['t'] + 1e-9, side='right') - 1)
        tf = None
        if j >= 0:
            Tg = np.array(tr[j]['target']).reshape(4, 4)
            tf = (H[:3, :3].T @ (Tg[:3, 3] - Tm[:3, 3]) * 1e3).tolist()
        out.append({'t': x['t'], 'q_deg': math.degrees(s[I['door_q_rad']]),
                    'ft_hy_sum': None if fs['sum_ft_handle'] is None else fs['sum_ft_handle'][1],
                    'fn_abs': fa, 'tgt_minus_fk_mm': tf,
                    'bar_fn': [((s[I['leaf_contacts']] or {}).get(b, {}) or {}).get('bar', 0.0) for b in FINGERS]})
    return out


def win(rows, lo, hi):
    w = [x for x in rows if lo <= x['t'] < hi]
    hy = [x['ft_hy_sum'] for x in w if x['ft_hy_sum'] is not None]
    tf = np.array([x['tgt_minus_fk_mm'] for x in w if x['tgt_minus_fk_mm'] is not None]) if w else np.zeros((0, 3))
    fa = np.array([x['fn_abs'] for x in w]) if w else np.zeros((0, 2))
    return {'t0': lo, 't1': hi, 'n_steps': len(w), 'n_valid_ft': len(hy), 'n_na_ft': len(w) - len(hy),
            'ft_hy_sum_N': None if not hy else {'median': float(np.median(hy)), 'min': float(min(hy)), 'max': float(max(hy))},
            'tgt_minus_fk_mm_median': None if len(tf) == 0 else np.median(tf, axis=0).tolist(),
            'fn_abs_median_N': None if len(fa) == 0 else np.median(fa, axis=0).tolist(),
            'fn_abs_min_N': None if len(fa) == 0 else fa.min(axis=0).tolist()}


def main():
    run = sys.argv[1]
    ref = sys.argv[2] if len(sys.argv) > 2 else None
    out = os.path.join(run, 'analysis', 'door_k3au_report.json')
    if os.path.exists(out):
        raise SystemExit(f'{out} 已存在，不覆寫')
    t, r, ev_rows = load(run)
    I = {c: i for i, c in enumerate(r['steps_cols'])}
    ph = {e['to']: e['t'] for e in t['policy_events'] if 'to' in e}
    cap_ev = next((e for e in t['policy_events'] if 'unload_capture' in e), None)
    cap = cap_ev['unload_capture'] if cap_ev else None
    marks = {k: next((e['t'] for e in t['policy_events'] if k in e), None) for k in ('unload_ramp_end', 'unload_hold_end')}
    ab = next((e for e in t['policy_events'] if 'abort' in e), None)
    rows = series(t, r, ev_rows)
    t_open = ph.get('OPEN')
    low = next((s for s in r['steps'] if t_open is not None and s[I['sim_t']] >= t_open
                and min((s[I['leaf_contacts']] or {}).get(f, {}).get('bar', 0.0) for f in FINGERS) < 0.5), None)
    res = {'run': os.path.basename(os.path.normpath(run)), 'phase_t': ph, 'abort': ab,
           'capture': None if cap is None else {k: cap[k] for k in ('t', 'step', 'delta')},
           'marks': marks,
           'timing': None if cap is None else {'capture_tau_s': cap['t'] - ph['HOLD'],
                                               'ramp_end_minus_cap_s': None if marks['unload_ramp_end'] is None else marks['unload_ramp_end'] - cap['t'],
                                               'hold_end_minus_cap_s': None if marks['unload_hold_end'] is None else marks['unload_hold_end'] - cap['t'],
                                               'open_minus_cap_s': None if t_open is None else t_open - cap['t'],
                                               'hold_duration_s': None if t_open is None else t_open - ph['HOLD']},
           'first_low_phys': None if low is None else {'t': low[I['sim_t']], 'q_deg': math.degrees(low[I['door_q_rad']]),
                                                     'since_open_s': low[I['sim_t']] - t_open},
           'abort_door_q_deg': None,
           'windows': {}, 'reference_windows': None,
           'note': '介入觀察（一趟）。切向作用側為推論；HOLD 延長為配置差異；不證實根因。'}
    if ab is not None:
        s = min(r['steps'], key=lambda s: abs(s[I['sim_t']] - ab['t']))
        res['abort_door_q_deg'] = math.degrees(s[I['door_q_rad']])
        res['abort_since_open_s'] = None if t_open is None else ab['t'] - t_open
    if cap is not None and t_open is not None:
        res['windows'] = {'pre_unload': win(rows, ph['HOLD'] + 0.1, cap['t']), 'post_unload_hold': win(rows, cap['t'] + 0.5, t_open),
                          'open_to_first_low': win(rows, t_open, low[I['sim_t']] if low else math.inf),
                          'first_low_to_abort': win(rows, low[I['sim_t']], ab['t']) if (low and ab) else None,
                          'post_abort': win(rows, ab['t'], math.inf) if ab else None}
    if ref:
        t2, r2, e2 = load(ref)
        ph2 = {e['to']: e['t'] for e in t2['policy_events'] if 'to' in e}
        rows2 = series(t2, r2, e2)
        res['reference_windows'] = {'run': os.path.basename(os.path.normpath(ref)),
                                    'hold_0.1_to_0.5': win(rows2, ph2['HOLD'] + 0.1, ph2['HOLD'] + 0.5),
                                    'hold_0.5_to_open': win(rows2, ph2['HOLD'] + 0.5, ph2['OPEN'])}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(res, ensure_ascii=False, default=float))


if __name__ == '__main__':
    main()
