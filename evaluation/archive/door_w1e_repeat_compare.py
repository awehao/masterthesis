#!/usr/bin/env python3
"""door_wide_w1e_30_1 與 door_wide_w1e_30_2（同凍結重複趟；Codex 20261025_025410）逐項並列，另附 w1c_30_1 基準（只讀）。
各段同 door_w1e_pause_compare.seg：PAUSE→STOPPED、STOPPED 後同長（w1c STOPPED→中止長度 5.32 s）、STOPPED→結束。
另列：時序、門角（SETTLE／PAUSE／結束）、上限生效紀錄、保護極值（判定器 w1e2 stats）。單趟並列，只描述，不作統計結論。
輸出：results/door/w1e_repeat_compare.json。

    python3 evaluation/door_w1e_repeat_compare.py
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_w1e_pause_compare as P                     # noqa: E402

RUNS = ['door_wide_w1c_30_1', 'door_wide_w1e_30_1', 'door_wide_w1e_30_2']
OUT = os.path.join(HERE, 'results', 'door', 'w1e_repeat_compare.json')


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    data = {rid: P.load(rid) for rid in RUNS}
    L = data[RUNS[0]][4] - data[RUNS[0]][3]['STOPPED']
    res = {'schema': 'w1e_repeat_compare/1', 'same_len_s': L, 'runs': {}}
    for rid, (r, ev, pe, tr, t_end, geo) in data.items():
        I = {c: i for i, c in enumerate(r['steps_cols'])}
        qat = (lambda tt: math.degrees(min(r['steps'], key=lambda x: abs(x[I['sim_t']] - tt))[I['door_q_rad']]))
        ts = tr['STOPPED']
        jp = os.path.join(HERE, 'runs', rid, 'analysis', 'door_k3m_check_w1e2.json')
        jw = os.path.join(HERE, 'runs', rid, 'analysis', 'door_k3m_check_w1c.json')
        j = json.load(open(jp)) if os.path.exists(jp) else (json.load(open(jw)) if os.path.exists(jw) else {})
        sj = j.get('stats') or {}
        fe = (r.get('finger_effort') or {}).get('pause') or {}
        res['runs'][rid] = {
            'phases': {k: pe.get(k) for k in ('OPEN', 'SETTLE', 'PAUSE', 'DONE')}, 'transitions': tr, 't_end': t_end,
            'door_deg': {'SETTLE': qat(pe['SETTLE']), 'PAUSE': qat(pe['PAUSE']), 'end': qat(t_end)},
            'judge': {'file': os.path.basename(jp if os.path.exists(jp) else jw), 'pass': j.get('pass'), 'why': j.get('why')},
            'protection': {k: sj.get(k) for k in ('max_unexpected_n', 'max_wall_n', 'max_wrist_n', 'max_drift_m', 'max_drift_rad',
                                                  'corridor_worst_mm', 'corridor_worst_deg', 'min_grasp_finger_bar_n')},
            'finger_effort': {'step': fe.get('step'), 'old': fe.get('old'), 'readback': fe.get('readback'), 'ok': fe.get('ok'),
                              'n_periodic': len(fe.get('periodic') or [])} if fe else None,
            'PAUSE→STOPPED': P.seg(r, ev, geo, pe['PAUSE'], ts), 'STOPPED+同長': P.seg(r, ev, geo, ts, ts + L),
            'STOPPED→結束': P.seg(r, ev, geo, ts, t_end)}
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    keys = ('F_bar_median_n', 'door_deg_change', 'tcp_rel_handle_total_mm', 'clearance_mm_first_last_min', 'base_disp_mm',
            'base_max_speed_0p1s_mm_s', 'wrist_median_n', 'arm_meas_change_peak_mrad')
    for rid, d in res['runs'].items():
        print(rid, d['judge']['pass'], d['transitions'].get('STOPPED'), d['transitions'].get('DONE'), {k: round(v, 3) for k, v in d['door_deg'].items()})
        print('   prot', d['protection'])
        for sname in ('PAUSE→STOPPED', 'STOPPED+同長'):
            sg = d[sname]
            print('  ', sname, {k: (round(sg[k], 3) if isinstance(sg[k], float) else [round(v, 3) for v in sg[k]] if isinstance(sg[k], list) else sg[k]) for k in keys})


if __name__ == '__main__':
    main()
