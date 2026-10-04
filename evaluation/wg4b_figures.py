#!/usr/bin/env python3
"""WG4-B 正式結果圖（只讀實錄與 wg4b_formal_results.json）。

  wg4b_opening.png   6 趟抽屜開度 vs 時間（以 OPEN 起點為 0），標開啟帶 195–205 mm 與 220 mm 行程端
  wg4b_paired.png    共同區段（ALIGN、ENGAGE_WAIT、OPEN）三對 B1 − P 配對差：err_p p95、段長、核心求解 p50

    python3 evaluation/wg4b_figures.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OD = os.path.join(HERE, 'results', 'wg4b', 'figures')
C = {'B1': '#c2410c', 'P': '#1d4ed8'}
plt.rcParams.update({'font.family': ['Noto Sans CJK JP'], 'axes.unicode_minus': False,
                     'axes.spines.top': False, 'axes.spines.right': False})


def main():
    d = json.load(open(os.path.join(HERE, 'results', 'wg4b', 'wg4b_formal_results.json')))
    os.makedirs(OD, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.axhspan(195, 205, color='#9ca3af', alpha=0.25, lw=0)
    ax.text(0.5, 199, '開啟帶 195–205 mm', fontsize=8, color='#374151', va='center')
    ax.axhline(220, color='#6b7280', lw=0.8, ls=':')
    ax.text(0.5, 222, '行程端 220 mm', fontsize=8, color='#374151')
    seen = set()
    for r in d['runs']:
        t = json.load(open(os.path.join(HERE, 'runs', r['rid'], 'task.json')))
        t0 = next(e['sim_t'] for e in t['events'] if e.get('phase') == 'OPEN')
        tr = [x for x in t['trace'] if x['sim_t'] >= t0 - 1]
        m = r['method']
        lab = (('B1（單步 QP，kp 1.0）' if m == 'B1' else 'P（W-GMPC H5）') if m not in seen else None)
        seen.add(m)
        ax.plot([x['sim_t'] - t0 for x in tr], [x['opening_m'] * 1e3 for x in tr],
                color=C[m], lw=1.6, alpha=0.85, label=lab)
    ax.set_xlim(-1, 60)
    ax.set_ylim(-5, 230)
    ax.set_xlabel('自 OPEN 起的模擬時間（s）；B1 實錄持續至約 175 s，圖只畫前 60 s')
    ax.set_ylabel('抽屜開度（mm）')
    ax.set_title('WG4-B 正式 6 趟：P 3/3 完成開→關；B1 0/3（超出開啟帶後停在行程端）', fontsize=10)
    ax.legend(frameon=False, loc='center right')
    ax.grid(axis='y', color='#e5e7eb', lw=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(OD, 'wg4b_opening.png'), dpi=160)
    plt.close(fig)

    phases = ['ALIGN', 'ENGAGE_WAIT', 'OPEN']
    mets = [('err_p_mm_p95', '位置誤差 p95 差（mm）'), ('duration_s', '段長差（s）'),
            ('core_ms_p50', '核心求解 p50 差（ms）')]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, (k, lab) in zip(axs, mets):
        for j, p in enumerate(phases):
            for i, pr in enumerate(sorted(d['pairs'])):
                v = d['pairs'][pr]['diff_B1_minus_P'][p][k]
                if isinstance(v, (int, float)):
                    ax.plot(j + (i - 1) * 0.12, v, 'o', ms=7, color='#374151',
                            mfc=['#ffffff', '#9ca3af', '#374151'][i], label=(f'配對 {pr}' if j == 0 else None))
        ax.axhline(0, color='#6b7280', lw=0.8)
        ax.set_xticks(range(3), phases, fontsize=8)
        ax.set_title(lab, fontsize=9)
        ax.grid(axis='y', color='#e5e7eb', lw=0.6)
    axs[0].legend(frameon=False, fontsize=8)
    fig.suptitle('共同區段 B1 − P 配對差（>0 = B1 較大；n = 3 對，不做顯著性）', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OD, 'wg4b_paired.png'), dpi=160)
    plt.close(fig)
    print('ok')


if __name__ == '__main__':
    main()
