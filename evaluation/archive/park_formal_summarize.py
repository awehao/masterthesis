#!/usr/bin/env python3
"""嚴格停車（PARK_HOLD v2.1）vs MOTM 正式比較彙整（口徑見 results/motm_speed/park_hold_formal_plan.yaml，起跑前鎖定）。只讀實錄。

每趟需已有 analysis/physical_check.json、analysis/replay_check.json；H 組另需 park_gate.json 與 analysis/parked_audit.json。
缺檔 ⇒ 該欄 NA 並列入 missing（不當成通過）。

計時：起點＝mission.json go_sim_t；**獨立終端完成**＝S1 開帶連續段與 S2 關帶連續段都已滿 2 s（HOLD_S）、且控制者已交還導航之後，
第一個底盤距起點 ≤ 0.20 m 的物理步時間（room_run 逐物理步）。DONE 只作交叉證據。

    python3 evaluation/park_formal_summarize.py [排程.tsv] [輸出.json]
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'runs')
HOLD_S, HOME_TOL = 2.0, 0.20


def load(p, missing):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        missing.append(os.path.relpath(p, HERE))
        return None


def run_row(rid, method, missing):
    R = os.path.join(RUNS, rid)
    task = load(os.path.join(R, 'task.json'), missing)
    mission = load(os.path.join(R, 'mission.json'), missing)
    room = load(os.path.join(R, 'room_run.json'), missing)
    pc = load(os.path.join(R, 'analysis', 'physical_check.json'), missing)
    rp = load(os.path.join(R, 'analysis', 'replay_check.json'), missing)
    row = {'rid': rid, 'method': method, 'final_phase': (task or {}).get('final_phase'),
           'abort': (task or {}).get('abort'), 'S': (pc or {}).get('results'),
           'replay': (rp or {}).get('verdict')}
    row['task_ok'] = bool(row['final_phase'] == 'DONE' and row['S'] and all(v is True for v in row['S'].values())
                          and row['replay'] == 'PASS')
    if method == 'PARK_HOLD':
        pg = load(os.path.join(R, 'park_gate.json'), missing)
        au = load(os.path.join(R, 'analysis', 'parked_audit.json'), missing)
        cw = ((au or {}).get('runs') or [{}])[0].get('control_window') or {}
        hold = (pg or {}).get('hold') or {}
        row['park'] = {'gate_pass_t': ((pg or {}).get('gate') or {}).get('pass_t'),
                       'hold_verdict': hold.get('verdict'), 'first_violation': hold.get('first_violation'),
                       'audit_status': cw.get('status'), 'audit_strict': cw.get('strict_stationary_full_window'),
                       'max': hold.get('max')}
        row['mode_ok'] = bool(row['park']['hold_verdict'] == 'PASS' and row['park']['audit_strict'] is True)
    else:
        row['mode_ok'] = True
    go = (mission or {}).get('go_sim_t')
    row['go_sim_t'] = go
    row['T_complete_s'] = None
    row['segments'] = None
    if room is not None and pc is not None and go is not None:
        ci = room['steps_cols'].index
        t = np.array([s[1] for s in room['steps']], float)
        own = np.array([s[ci('owner(0=nav,1=wb,2=glide)')] for s in room['steps']])
        xy = np.array([s[ci('base_xyth')][:2] for s in room['steps']], float)
        sp = [float(x) for x in room['config']['start_pose'].split(',')][:2]
        s1 = (pc.get('S1_open_hold') or {}).get('at_sim_t')
        s2 = (pc.get('S2_close_hold') or {}).get('at_sim_t')
        wb = np.flatnonzero(own == 1)
        t_xfer = float(t[wb[0]]) if len(wb) else None
        hb = np.flatnonzero((own == 0) & (np.arange(len(own)) > (wb[-1] if len(wb) else 10**9)))
        t_hb = float(t[hb[0]]) if len(hb) else None
        s1_done = (s1[0] + HOLD_S) if (s1 and (s1[1] - s1[0]) >= HOLD_S - 1e-9) else None
        s2_done = (s2[0] + HOLD_S) if (s2 and (s2[1] - s2[0]) >= HOLD_S - 1e-9) else None
        t_end = None
        if None not in (s1_done, s2_done, t_hb):
            t0 = max(s1_done, s2_done, t_hb)
            dist = np.hypot(xy[:, 0] - sp[0], xy[:, 1] - sp[1])
            k = np.flatnonzero((t >= t0) & (dist <= HOME_TOL))
            t_end = float(t[k[0]]) if len(k) else None
        attach = next((float(e['sim_t']) for e in (task or {}).get('events', []) if e.get('attached') is True), None)
        row['T_complete_s'] = None if t_end is None else round(t_end - go, 3)
        row['segments'] = {k: (None if v is None else round(v - go, 3)) for k, v in (
            ('transfer_to_wholebody', t_xfer), ('grasp_attached', attach), ('open_hold_done', s1_done),
            ('close_hold_done', s2_done), ('handback_to_nav', t_hb), ('terminal_complete', t_end))}
        row['DONE_cross_check'] = (None if 'DONE' not in {e.get('phase') for e in (task or {}).get('events', [])}
                                   else round(next(e['sim_t'] for e in task['events'] if e.get('phase') == 'DONE') - go, 3))
        # 夾持品質（OPEN 相位）
        ph = {e['phase']: e['sim_t'] for e in (task or {}).get('events', []) if 'phase' in e}
        if 'OPEN' in ph:
            end = ph.get('OPEN_HOLD', ph['OPEN'] + 12)
            F = np.array([s[ci('finger_contact_n')] for s in room['steps']
                          if ph['OPEN'] <= s[1] <= end and s[ci('finger_contact_n')]], float)
            if len(F):
                row['grip_open'] = {'both_ge_0p5N_frac': round(float(np.mean((F[:, 0] >= 0.5) & (F[:, 1] >= 0.5))), 3),
                                    'f1_median': round(float(np.median(F[:, 0])), 2),
                                    'f2_median': round(float(np.median(F[:, 1])), 2),
                                    'peak_N': round(float(F.max()), 2)}
        s4 = pc.get('S4_drift')
        row['S4_drift_max_mm'] = s4.get('max_mm') if isinstance(s4, dict) else None
    return row


def main():
    plan = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'results', 'motm_speed', 'park_hold_formal_schedule.tsv')
    rows = list(csv.DictReader(open(plan), delimiter='\t'))
    missing = []
    table = [run_row(r['rid'], r['method'], missing) for r in rows]
    by = {t['rid']: t for t in table}
    pairs = {}
    for pr in sorted({r['pair'] for r in rows}):
        rm = next(r['rid'] for r in rows if r['pair'] == pr and r['method'] == 'MOTM')
        rh = next(r['rid'] for r in rows if r['pair'] == pr and r['method'] == 'PARK_HOLD')
        a, b = by[rm], by[rh]
        usable = (a['task_ok'] and b['task_ok'] and b['mode_ok']
                  and a['T_complete_s'] is not None and b['T_complete_s'] is not None)
        why = None if usable else [k for k, v in (('M 任務', a['task_ok']), ('H 任務', b['task_ok']),
                                                   ('H 模式合規', b['mode_ok']),
                                                   ('M 終端', a['T_complete_s'] is not None),
                                                   ('H 終端', b['T_complete_s'] is not None)) if not v]
        d = round(a['T_complete_s'] - b['T_complete_s'], 3) if usable else None
        pairs[pr] = {'M': rm, 'H': rh, 'order': next(r['order'] for r in rows if r['pair'] == pr),
                     'usable': usable, 'excluded_because': why, 'T_M_minus_T_H_s': d,
                     'pct_of_H': (None if d is None else round(100 * d / b['T_complete_s'], 2))}
    out = {'schema': 'park_hold_formal_results/1', 'plan': os.path.relpath(plan, HERE), 'runs': table,
           'pairs': pairs, 'missing_inputs': missing,
           'rules': ['只有兩組皆完成且 H 組模式合規的配對計算時間差', '全部六趟結果都列', 'n = 3，不做顯著性',
                     '比較兩個完整策略，不宣稱唯一差別是底盤是否移動']}
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, 'results', 'motm_speed', 'park_hold_formal_results.json')
    json.dump(out, open(dst, 'w'), ensure_ascii=False, indent=1, default=float)
    for t in table:
        print(t['rid'], t['method'], 'task_ok' if t['task_ok'] else f"任務未通過({t['final_phase']}, {t['abort']})",
              'mode_ok' if t['mode_ok'] else 'MODE_VIOLATION', 'T', t['T_complete_s'], 'DONE', t.get('DONE_cross_check'))
    for k, v in pairs.items():
        print('pair', k, v['order'], 'usable' if v['usable'] else f"排除 {v['excluded_because']}", v['T_M_minus_T_H_s'])
    if missing:
        print('缺檔：', missing)
    return 0


if __name__ == '__main__':
    sys.exit(main())
