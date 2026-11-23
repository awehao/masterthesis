#!/usr/bin/env python3
"""door_wide_w1f_60_1（60°）與 w1e_30_1／_2（30°，同控制配置）暫停期並列（只讀；同 door_w1e_pause_compare.seg 方法）。
另列：OPEN 期兩指承載（中位／最小）與 F2−F1、走廊最差、底盤最大位移、門角、判定理由、trace 間隙。單趟並列，只描述。
輸出：results/door/w1f_60_1_compare.json、第七次素材／圖表／w1f_60_1_開門與暫停.png。

    python3 evaluation/door_w1f_60_compare.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_w1e_pause_compare as P                     # noqa: E402

RUNS = ['door_wide_w1e_30_1', 'door_wide_w1e_30_2', 'door_wide_w1f_60_1']
OUT = os.path.join(HERE, 'results', 'door', 'w1f_60_1_compare.json')
FIG = os.path.join(os.path.dirname(HERE), '第七次進度報告素材', '圖表', 'w1f_60_1_開門與暫停.png')


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    res = {'schema': 'w1f_60_compare/1', 'runs': {}}
    data = {}
    for rid in RUNS:
        r, ev, pe, tr, t_end, geo = P.load(rid)
        data[rid] = (r, ev, pe, tr, t_end, geo)
        I = {c: i for i, c in enumerate(r['steps_cols'])}
        op = [x for x in r['steps'] if pe['OPEN'] <= x[I['sim_t']] < pe['SETTLE']]
        fb = np.array([[float(((x[I['leaf_contacts']] or {}).get(f) or {}).get('bar', 0.0)) for f in P.FING] for x in op])
        qd = np.degrees([x[I['door_q_rad']] for x in op])
        bins = []
        for q0 in range(0, 60, 10):
            m = (qd >= q0) & (qd < q0 + 10)
            if m.any():
                bins.append({'q': [q0, q0 + 10], 'F_med': np.median(fb[m], 0).tolist(), 'F_min': fb[m].min(0).tolist(),
                             'F2_minus_F1_med': float(np.median(fb[m, 1] - fb[m, 0]))})
        j = json.load(open(os.path.join(HERE, 'runs', rid, 'analysis', 'door_k3m_check_w1e2.json')))
        st = j['stats']
        ts = tr['STOPPED']
        res['runs'][rid] = {'judge_pass': j['pass'], 'why': j['why'], 'max_trace_gap_s': st.get('max_trace_gap_s'),
                            'phases': {k: pe.get(k) for k in ('OPEN', 'SETTLE', 'PAUSE', 'DONE')}, 'transitions': tr,
                            'settle_q_deg': st.get('settle_final_q_deg'), 'q_open_max_deg': st.get('q_open_max_deg'),
                            'protection': {k: st.get(k) for k in ('max_unexpected_n', 'max_wall_n', 'max_wrist_n', 'min_grasp_finger_bar_n',
                                                                  'max_drift_m', 'max_drift_rad', 'corridor_worst_mm', 'corridor_worst_deg', 'max_base_m')},
                            'open_load_bins_10deg': bins,
                            'PAUSE→STOPPED': P.seg(r, ev, geo, pe['PAUSE'], ts), 'STOPPED→結束': P.seg(r, ev, geo, ts, t_end)}
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fp = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(fp)
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name()
    fig, ax = plt.subplots(3, 1, figsize=(9, 8.5))
    col = {'door_wide_w1e_30_1': '#9aa', 'door_wide_w1e_30_2': '#688', 'door_wide_w1f_60_1': '#2a6fdb'}
    for rid, (r, ev, pe, tr, t_end, geo) in data.items():
        I = {c: i for i, c in enumerate(r['steps_cols'])}
        xs = [x for x in r['steps'] if pe['OPEN'] <= x[I['sim_t']] <= t_end][::5]
        qd = [math.degrees(x[I['door_q_rad']]) for x in xs]
        fb = [[float(((x[I['leaf_contacts']] or {}).get(f) or {}).get('bar', 0.0)) for f in P.FING] for x in xs]
        ax[0].plot([x[I['sim_t']] - pe['OPEN'] for x in xs], qd, color=col[rid], lw=1.3, label=rid.replace('door_wide_', ''))
        op = [(q, f) for x, q, f in zip(xs, qd, fb) if x[I['sim_t']] < pe['SETTLE']]
        ax[1].plot([q for q, _ in op], [f[0] for _, f in op], color=col[rid], lw=0.9, label=rid.replace('door_wide_', '') + ' F1')
        ax[1].plot([q for q, _ in op], [f[1] for _, f in op], color=col[rid], lw=0.9, ls='--', label=rid.replace('door_wide_', '') + ' F2')
        cl = [(k - pe['OPEN'], geo[k] * 1e3) for k in sorted(geo) if pe['OPEN'] <= k <= t_end]
        ax[2].plot([u for u, _ in cl], [v for _, v in cl], color=col[rid], lw=1.2, label=rid.replace('door_wide_', ''))
        for a_ in (ax[0], ax[2]):
            a_.axvline(tr['STOPPED'] - pe['OPEN'], color=col[rid], ls=':', lw=0.9)
    ax[2].axhline(5.0, color='#c0392b', ls='--', lw=0.8, label='幾何門檻 5 mm')
    ax[0].set_ylabel('門角（°）')
    ax[0].set_xlabel('OPEN 起時間（s）；點線＝STOPPED')
    ax[1].set_ylabel('OPEN 期兩指法向承載（N）')
    ax[1].set_xlabel('門角（°）')
    ax[2].set_ylabel('殼—桿淨空（mm）')
    ax[2].set_xlabel('OPEN 起時間（s）')
    for a_ in ax:
        a_.grid(alpha=0.25)
        a_.legend(fontsize=7, frameon=False, ncol=3)
    fig.suptitle('W1f 60°（單趟）與 W1e 30°（兩趟）：門角、OPEN 承載、淨空', fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG, dpi=140)
    for rid, d in res['runs'].items():
        print(rid, d['judge_pass'], d['why'], round(d['max_trace_gap_s'], 3), d['settle_q_deg'] and round(d['settle_q_deg'], 2))
        print('  open bins', [(b['q'], [round(v, 1) for v in b['F_med']], [round(v, 1) for v in b['F_min']]) for b in d['open_load_bins_10deg']])
        for sname in ('PAUSE→STOPPED', 'STOPPED→結束'):
            sg = d[sname]
            print('  ', sname, {k: (round(sg[k], 3) if isinstance(sg[k], float) else [round(v, 3) for v in sg[k]] if isinstance(sg[k], list) else sg[k])
                                for k in ('F_bar_median_n', 'door_deg_change', 'tcp_rel_handle_total_mm', 'clearance_mm_first_last_min',
                                          'base_disp_mm', 'base_max_speed_0p1s_mm_s', 'wrist_median_n')})


if __name__ == '__main__':
    main()
