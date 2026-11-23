#!/usr/bin/env python3
"""嚴格停車（PARK_HOLD v2.1）vs MOTM 正式三對：完成時間配對與分段耗時（讀 park_hold_formal_results.json）。

    python3 evaluation/park_formal_figures.py → results/motm_speed/figures/park_formal_paired.png
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({'font.family': ['Noto Sans CJK JP'], 'axes.unicode_minus': False,
                     'axes.spines.top': False, 'axes.spines.right': False})
SEG = [('transfer_to_wholebody', 'GO→轉給全身（入場與交棒）'), ('grasp_attached', '→夾持成立'),
       ('open_hold_done', '→開啟保持完成'), ('close_hold_done', '→關閉保持完成'),
       ('handback_to_nav', '→收臂並交還導航'), ('terminal_complete', '→回程終端完成')]
COL = ['#94a3b8', '#60a5fa', '#34d399', '#fbbf24', '#f87171', '#a78bfa']


def main():
    d = json.load(open(os.path.join(HERE, 'results', 'motm_speed', 'park_hold_formal_results.json')))
    by = {r['rid']: r for r in d['runs']}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw={'width_ratios': [1, 2.2]})
    for k, (pr, v) in enumerate(sorted(d['pairs'].items())):
        tm, th = by[v['M']]['T_complete_s'], by[v['H']]['T_complete_s']
        a1.plot([0, 1], [tm, th], '-o', color='#374151', mfc=['#ffffff', '#9ca3af', '#374151'][k], ms=7,
                label=f'配對 {pr}（{v["order"]}）：{v["T_M_minus_T_H_s"]:+.2f} s')
    a1.set_xticks([0, 1], ['MOTM', 'PARK_HOLD\n（嚴格停車）'])
    a1.set_xlim(-0.4, 1.4)
    a1.set_ylabel('共同 GO → 獨立終端完成（s；縱軸截短，非由 0 起）')
    a1.legend(frameon=False, fontsize=8, loc='upper left')
    a1.grid(axis='y', color='#e5e7eb', lw=0.6)
    a1.set_title('完成時間（三對；n = 3，不做顯著性）', fontsize=10)
    order = []
    for pr, v in sorted(d['pairs'].items()):
        order += [(f'{pr}-M', by[v['M']]), (f'{pr}-H', by[v['H']])]
    for j, (lab, r) in enumerate(order):
        prev = 0.0
        for (key, name), c in zip(SEG, COL):
            t = r['segments'][key]
            a2.barh(j, t - prev, left=prev, color=c, edgecolor='white', height=0.7,
                    label=name if j == 0 else None)
            prev = t
    a2.set_yticks(range(len(order)), [o[0] for o in order], fontsize=8)
    a2.invert_yaxis()
    a2.set_xlabel('自共同 GO 的模擬時間（s）')
    a2.set_title('分段耗時：差距主要在「GO→全身接手」區段（入場減速、收斂、等待與交棒策略；閘門窗本身約 0.5 s）', fontsize=9)
    a2.legend(frameon=False, fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=3)
    a2.grid(axis='x', color='#e5e7eb', lw=0.6)
    fig.suptitle('嚴格停車（PARK_HOLD v2.1，底盤確實停住並以停車伺服保持）vs 移動中操作（MotM）：兩個完整策略的比較', fontsize=10)
    fig.tight_layout()
    od = os.path.join(HERE, 'results', 'motm_speed', 'figures')
    os.makedirs(od, exist_ok=True)
    fig.savefig(os.path.join(od, 'park_formal_paired.png'), dpi=160)
    print('ok')


if __name__ == '__main__':
    main()
