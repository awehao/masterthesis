#!/usr/bin/env python3
"""K3-A-X 單趟介入觀察報告（只讀；依 K3_bar_axis_realign_spec.md §5 與 Codex 20261022_125921）。

窗口全部以**來源物理步號**對齊（task trace 的 step；evidence 的 step）：
  重對齊前＝[HOLD 步＋10, 擷取步)；重對齊後保持＝[擷取步＋50, OPEN 步)；OPEN＝[OPEN 步, 觸發步)；觸發步另列；中止後＝(觸發步, 末]。
主要量（沿桿 x）：已發布目標（task trace 依來源時間）− FK 的把手 x（mm）；兩指回報切向合量 Σft 的把手 x 分量（N；任一指缺 ⇒ NA）。
附帶：Σft_y、兩指 Σ|fn|、首次單指 < 0.5 N（物理步）、觸發步門角。**不直接沿用 U 報告的 y 欄位作為結論。**
對照（給第二個以後的 run）：以「HOLD 步＋10 → HOLD 步＋50」與「HOLD 步＋50 → OPEN 步」同法定窗（只描述）。

    python3 evaluation/door_k3ax_report.py <run_dir> [對照 run_dir ...]   # 寫 <run_dir>/analysis/door_k3ax_report.json（已存在則拒絕覆寫）
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
    ev = [json.loads(ln) for ln in open(os.path.join(run, 'evidence.jsonl'))]
    return t, r, ev


def step_of(t, phase):
    return next((e['step'] for e in t.get('trace') or [] if e.get('phase') == phase), None)


def rows_of(t, r, ev):
    I = {c: i for i, c in enumerate(r['steps_cols'])}
    room = {s[I['step']]: s for s in r['steps']}
    tr = sorted([e for e in t['trace'] if e.get('target') is not None], key=lambda e: e['t'])
    tr_t = np.array([e['t'] for e in tr])
    out = {}
    for x in ev:
        s = room.get(x['step'])
        if s is None or x['t'] is None:
            continue
        R = np.array(x['T_WH']).reshape(4, 4)[:3, :3]
        fs = PL.finger_sums(x, R)
        H = KC._H_of(s[I['door_q_rad']])
        Tm = KC._tcp(s[I['base_xyth']], s[I['q_arm_meas']])
        j = int(np.searchsorted(tr_t, x['t'] + 1e-9, side='right') - 1)
        tf = None if j < 0 else (H[:3, :3].T @ (np.array(tr[j]['target']).reshape(4, 4)[:3, 3] - Tm[:3, 3]) * 1e3).tolist()
        out[x['step']] = {'step': x['step'], 't': x['t'], 'q_deg': math.degrees(s[I['door_q_rad']]),
                          'ft': fs['sum_ft_handle'], 'tgt_fk_mm': tf,
                          'fn_abs': [sum(abs(c['fn']) for c in (x.get('contacts') or {}).get(b, []) if c.get('fn') is not None) for b in FINGERS],
                          'bar': [((s[I['leaf_contacts']] or {}).get(b, {}) or {}).get('bar', 0.0) for b in FINGERS]}
    return out


def win(rows, lo, hi):
    w = [rows[k] for k in sorted(rows) if lo <= k < hi]
    ftx = [x['ft'][0] for x in w if x['ft'] is not None]
    fty = [x['ft'][1] for x in w if x['ft'] is not None]
    tf = np.array([x['tgt_fk_mm'] for x in w if x['tgt_fk_mm'] is not None]) if w else np.zeros((0, 3))
    fa = np.array([x['fn_abs'] for x in w]) if w else np.zeros((0, 2))

    def st(v):
        return None if not v else {'median': float(np.median(v)), 'min': float(min(v)), 'max': float(max(v))}
    return {'step0': lo, 'step1': hi, 'n_steps': len(w), 'n_valid_ft': len(ftx), 'n_na_ft': len(w) - len(ftx),
            'ft_x_sum_N': st(ftx), 'ft_y_sum_N_aux': st(fty),
            'tgt_minus_fk_x_mm': st(tf[:, 0].tolist()) if len(tf) else None,
            'tgt_minus_fk_mm_median': None if len(tf) == 0 else np.median(tf, axis=0).tolist(),
            'fn_abs_median_N': None if len(fa) == 0 else np.median(fa, axis=0).tolist(),
            'fn_abs_min_N': None if len(fa) == 0 else fa.min(axis=0).tolist()}


def main():
    run, refs = sys.argv[1], sys.argv[2:]
    out = os.path.join(run, 'analysis', 'door_k3ax_report.json')
    if os.path.exists(out):
        raise SystemExit(f'{out} 已存在，不覆寫')
    t, r, ev = load(run)
    rows = rows_of(t, r, ev)
    s_hold, s_open, s_trig = step_of(t, 'HOLD'), step_of(t, 'OPEN'), step_of(t, 'ABORTED')
    cap = next((e['unload_capture'] for e in t['policy_events'] if 'unload_capture' in e), None)
    marks = {k: next((e['t'] for e in t['policy_events'] if k in e), None) for k in ('unload_ramp_end', 'unload_hold_end')}
    ph = {e['to']: e['t'] for e in t['policy_events'] if 'to' in e}
    s_cap = cap['step'] if cap else None
    first_low = next((rows[k] for k in sorted(rows) if s_open is not None and k >= s_open and (s_trig is None or k < s_trig)
                      and min(rows[k]['bar']) < 0.5), None)
    res = {'run': os.path.basename(os.path.normpath(run)), 'steps': {'HOLD': s_hold, 'capture': s_cap, 'OPEN': s_open, 'trigger': s_trig},
           'capture': None if cap is None else {k: cap.get(k) for k in ('t', 'step', 'axis_mode', 'delta', 'a')},
           'timing_s': None if cap is None else {'capture_tau': cap['t'] - ph['HOLD'], 'ramp_end_minus_cap': (marks['unload_ramp_end'] or math.nan) - cap['t'],
                                                 'hold_end_minus_cap': (marks['unload_hold_end'] or math.nan) - cap['t'],
                                                 'open_minus_cap': (ph.get('OPEN') or math.nan) - cap['t'], 'hold_duration': (ph.get('OPEN') or math.nan) - ph['HOLD']},
           'first_low_phys': None if first_low is None else {'step': first_low['step'], 't': first_low['t'], 'q_deg': first_low['q_deg'],
                                                           'steps_since_open': first_low['step'] - s_open},
           'trigger_row': None if s_trig is None or s_trig not in rows else {k: rows[s_trig][k] for k in ('step', 't', 'q_deg', 'fn_abs', 'ft')},
           'windows': {}, 'references': [],
           'note': '沿桿單軸介入的一趟觀察；切向作用側為推論；HOLD 延長為配置差異；不證實根因。窗口以來源物理步對齊。'}
    if cap is not None and s_open is not None:
        res['windows'] = {'pre_realign': win(rows, s_hold + 10, s_cap), 'post_realign_hold': win(rows, s_cap + 50, s_open),
                          'open_to_trigger': win(rows, s_open, s_trig if s_trig is not None else 10 ** 9),
                          'post_trigger': None if s_trig is None else win(rows, s_trig + 1, 10 ** 9)}
    for ref in refs:
        t2, r2, e2 = load(ref)
        rows2 = rows_of(t2, r2, e2)
        h2, o2 = step_of(t2, 'HOLD'), step_of(t2, 'OPEN')
        c2 = next((e['unload_capture'] for e in t2['policy_events'] if 'unload_capture' in e), None)
        res['references'].append({'run': os.path.basename(os.path.normpath(ref)), 'has_capture': c2 is not None,
                                  'hold_10_to_50': win(rows2, h2 + 10, h2 + 50),
                                  'hold_50_to_open': win(rows2, (c2['step'] + 50) if c2 else h2 + 50, o2)})
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(res, ensure_ascii=False, default=float))


if __name__ == '__main__':
    main()
