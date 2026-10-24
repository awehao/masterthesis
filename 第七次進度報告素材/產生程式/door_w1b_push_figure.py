#!/usr/bin/env python3
"""door_wide_w1b_30_1：驅動推送、兩指承載差與偏移估計 d̂ 的對齊圖與統計（只讀既有資料；W1b 承載分析附錄）。

驅動推送＝FK(設定點自由平衡 s＋b/α) − FK(實測關節)，轉到把手座標（mm）。b、α 取自 wgmpc_arm_sp_ident_free4.json（已辨識的靜態偏置）。
注意：設定點本身已含求解端對 d̂ 的補償，故「推送」與 d̂ 的高相關接近同義反覆，**不是獨立的物理成因證據**；
獨立的事實是實測 TCP 相對把手幾乎不動（≤ 0.9 mm）而推送與承載差同步增加。
輸出：results/door/w1b_30_1_push.json、第七次素材／圖表／w1b_30_1_推送與承載差.png。

    python3 evaluation/door_w1b_push_figure.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k2_check as KC                             # noqa: E402

RUN = os.path.join(HERE, 'runs', 'door_wide_w1b_30_1')
OUT = os.path.join(HERE, 'results', 'door', 'w1b_30_1_push.json')
FIG = os.path.join(os.path.dirname(HERE), '第七次進度報告素材', '圖表', 'w1b_30_1_推送與承載差.png')


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    idt = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
    al, bi = np.array(idt['alpha'], float), np.array(idt['bias_rad'], float)
    r = json.load(open(os.path.join(RUN, 'room_run.json')))
    I = {c: i for i, c in enumerate(r['steps_cols'])}
    t = json.load(open(os.path.join(RUN, 'task.json')))
    pe = {e['to']: e['t'] for e in t['policy_events'] if 'to' in e}
    t_abort = next(e['t'] for e in t['policy_events'] if e.get('abort'))
    ev = {}
    for line in open(os.path.join(RUN, 'evidence.jsonl')):
        d = json.loads(line)
        ev[d['step']] = d
    rows = []
    for s in r['steps']:
        if s[1] < pe['HOLD'] or s[1] > t_abort:
            continue
        e = ev.get(s[0])
        if not e:
            continue
        H = np.array(e['T_WH'], float).reshape(4, 4)
        Rt = H[:3, :3].T
        b = s[I['base_xyth']]
        Tm = KC._tcp(b, np.array(s[I['q_arm_meas']], float))
        Te = KC._tcp(b, np.array(e['arm_sp'], float) + bi / al)
        push = Rt @ (Te[:3, 3] - Tm[:3, 3]) * 1e3
        lc = s[I['leaf_contacts']]
        fb = [float((lc.get(f) or {}).get('bar', 0.0)) for f in ('uflite_finger1', 'uflite_finger2')]
        rows.append([float(s[1]), math.degrees(float(s[I['door_q_rad']])), fb[1] - fb[0], *push.tolist()])
    A = np.array(rows)
    s = json.load(open(os.path.join(RUN, 'align_solver.json')))
    D = np.array([[x['sim_t']] + list(x['offset_interval']['d_after']) for x in s['log']
                  if isinstance(x.get('offset_interval'), dict) and x['offset_interval'].get('d_after') and pe['HOLD'] <= x['sim_t'] <= t_abort])
    n_upd = sum(1 for x in s['log'] if isinstance(x.get('offset_interval'), dict) and x['offset_interval'].get('used') == 'updated'
                and x['sim_t'] >= pe['ENGAGE'])
    mo = A[:, 0] >= pe['OPEN']
    bins = []
    for q0 in range(0, 14, 2):
        m = mo & (A[:, 1] >= q0) & (A[:, 1] < q0 + 2)
        if m.any():
            bins.append({'q_deg': [q0, q0 + 2], 'f2_minus_f1_n': float(A[m, 2].mean()), 'push_x_mm': float(A[m, 3].mean()),
                         'push_y_mm': float(A[m, 4].mean()), 'push_z_mm': float(A[m, 5].mean())})
    hold = A[:, 0] < pe['OPEN']
    pz = np.interp(D[:, 0], A[:, 0], A[:, 5])
    md = D[:, 0] >= pe['OPEN']
    res = {'schema': 'w1b_push/1', 'definition': 'FK(s＋b/α) − FK(q_meas)，把手座標 mm；s 已含 d̂ 補償（同義反覆注意）',
           'hold': {'f2_minus_f1_n': float(A[hold, 2].mean()), 'push_mm': A[hold, 3:].mean(0).tolist()}, 'open_bins_2deg': bins,
           'corr_open': {'f2mf1_vs_push_z': float(np.corrcoef(A[mo, 2], A[mo, 5])[0, 1]),
                         'f2mf1_vs_push_x': float(np.corrcoef(A[mo, 2], A[mo, 3])[0, 1])},
           'dhat': {'n_updates_from_engage': n_upd, 'at_open_mrad': (D[np.argmin(abs(D[:, 0] - pe['OPEN'])), 1:] * 1e3).tolist(),
                    'at_abort_mrad': (D[-1, 1:] * 1e3).tolist(),
                    'corr_with_push_z_open': [float(np.corrcoef(D[md, 1 + j], pz[md])[0, 1]) for j in range(6)]}}
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fp = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(fp)
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name()
    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    q = A[mo, 1]
    ax[0].plot(q, A[mo, 2], color='#d9822b', lw=1.0)
    ax[0].set_ylabel('F2 − F1（N）')
    ax[1].plot(q, A[mo, 5], color='#2a6fdb', lw=1.0, label='z（向上）')
    ax[1].plot(q, A[mo, 4], color='#4a9a5a', lw=1.0, label='y')
    ax[1].plot(q, A[mo, 3], color='#888', lw=1.0, label='x（沿橫桿）')
    ax[1].set_ylabel('驅動推送（mm，把手座標）')
    ax[1].legend(fontsize=8, frameon=False)
    qd = np.interp(D[md, 0], A[:, 0], A[:, 1])
    ax[2].plot(qd, D[md, 2] * 1e3, color='#2a6fdb', lw=1.2, label='d_hat j2')
    ax[2].plot(qd, D[md, 3] * 1e3, color='#d9822b', lw=1.2, label='d_hat j3')
    ax[2].plot(qd, D[md, 4] * 1e3, color='#888', lw=1.2, label='d_hat j4')
    ax[2].set_ylabel('偏移估計 d_hat（mrad）')
    ax[2].set_xlabel('門角 °')
    ax[2].legend(fontsize=8, frameon=False)
    for a in ax:
        a.grid(alpha=0.25)
    fig.suptitle('door_wide_w1b_30_1 OPEN：兩指承載差、驅動推送與偏移估計（推送含 d_hat 補償，相關不是獨立成因證據）', fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG, dpi=140)
    print(json.dumps(res, ensure_ascii=False, indent=1)[:2500])


if __name__ == '__main__':
    main()
