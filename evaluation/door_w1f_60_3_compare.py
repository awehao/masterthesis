#!/usr/bin/env python3
"""door_wide_w1f_60_3（W1f2：任務端紀錄修正後的 60° 驗證）與 door_wide_w1f_60_1（同控制、舊紀錄）並列（只讀）。
各段同 door_w1e_pause_compare.seg；另列 trace 時序（每 100 筆邊界間隙 vs 其餘）、迴圈／寫入時序（60_3 才有）。單趟對單趟，只描述。
輸出：results/door/w1f_60_3_compare.json。

    python3 evaluation/door_w1f_60_3_compare.py
"""
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_w1e_pause_compare as P                     # noqa: E402

RUNS = {'door_wide_w1f_60_1': 'door_k3m_check_w1e2.json', 'door_wide_w1f_60_3': 'door_k3m_check_w1f2.json'}
OUT = os.path.join(HERE, 'results', 'door', 'w1f_60_3_compare.json')


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    res = {'schema': 'w1f_60_3_compare/1', 'runs': {}}
    for rid, jf in RUNS.items():
        r, ev, pe, tr, t_end, geo = P.load(rid)
        t = json.load(open(os.path.join(HERE, 'runs', rid, 'task.json')))
        j = json.load(open(os.path.join(HERE, 'runs', rid, 'analysis', jf)))
        tt = t['trace']
        d = [b['t'] - a['t'] for a, b in zip(tt, tt[1:])]
        at100 = {n: round(d[n - 1], 3) for n in range(100, len(tt), 100)}
        other = [x for i, x in enumerate(d, 1) if i % 100]
        ts = tr['STOPPED']
        res['runs'][rid] = {'judge': jf, 'pass': j['pass'], 'why': j['why'], 'n_trace': len(tt),
                            'trace_gap_at_100s': at100, 'trace_gap_other_median': statistics.median(other), 'trace_gap_other_max': max(other),
                            'max_trace_gap_s': j['stats'].get('max_trace_gap_s'),
                            'loop_timing': t.get('loop_timing'), 'trace_log': {k: (t.get('trace_log') or {}).get(k) for k in ('n_written', 'n_dropped', 'max_depth', 'max_put_s', 'close_wall_s', 'complete')} if t.get('trace_log') else None,
                            'settle_q_deg': j['stats'].get('settle_final_q_deg'), 'q_open_max_deg': j['stats'].get('q_open_max_deg'),
                            'protection': {k: j['stats'].get(k) for k in ('max_unexpected_n', 'max_wall_n', 'max_wrist_n', 'min_grasp_finger_bar_n', 'max_drift_m', 'max_drift_rad', 'corridor_worst_mm', 'corridor_worst_deg')},
                            'transitions': tr,
                            'PAUSE→STOPPED': P.seg(r, ev, geo, pe['PAUSE'], ts), 'STOPPED→結束': P.seg(r, ev, geo, ts, t_end)}
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    for rid, d in res['runs'].items():
        print(rid, d['pass'], d['why'], 'n', d['n_trace'], 'max_gap', round(d['max_trace_gap_s'], 3), '百位', d['trace_gap_at_100s'], '其餘中位／最大', d['trace_gap_other_median'], round(d['trace_gap_other_max'], 3))
        print('   settle', round(d['settle_q_deg'], 3), 'max', round(d['q_open_max_deg'], 3), 'prot', d['protection'])
        for s in ('PAUSE→STOPPED', 'STOPPED→結束'):
            g = d[s]
            print('  ', s, {k: (round(g[k], 3) if isinstance(g[k], float) else [round(v, 3) for v in g[k]] if isinstance(g[k], list) else g[k])
                            for k in ('F_bar_median_n', 'door_deg_change', 'tcp_rel_handle_total_mm', 'clearance_mm_first_last_min', 'base_disp_mm', 'base_max_speed_0p1s_mm_s', 'wrist_median_n')})


if __name__ == '__main__':
    main()
