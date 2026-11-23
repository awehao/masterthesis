#!/usr/bin/env python3
"""C1a（c1a_b1）成果圖：可用配對的小倍數圖＋本批完成計數。只讀已封存的彙整，不重算判定。

輸出 results/horizon_ablation/figures/c1a_paired.png
  每格一個指標，H5 → H1 以線連起同一對；只畫證據完整、可用的配對（T100-B、T150-A、T150-B）。
  被排除的 T100-A 不畫連續指標（缺資料發生在任務失敗後，排除不是隨機的），在圖下方註明。

    python3 evaluation/c1a_figures.py
"""
from __future__ import annotations

import json
import os

import c0_figures as F                                            # 字型、配色與版型沿用 C0 圖
import matplotlib.pyplot as plt

A = F.A
OUT = F.OUT
PANELS = [p for p in F.PANELS if p[0] not in ('open_hold_s', 'home_dist_m')] + [
    ('open_hold_s', '開保持（連續）', 1, 's', None),
    ('home_dist_m', '回程終點距起點', 100, 'cm', None)]
CASE_DASH = {'T100': (0, (4, 2)), 'T150': 'solid'}


def main():
    S = json.load(open(os.path.join(A, 'c1a_b1_summary.json')))
    pairs = []                      # (標籤, case, H5 m, H1 m)
    excluded = []
    for case, c in S['by_case'].items():
        for e in c['pairs']:
            if e['usable']:
                pairs.append((f"{e['pair']}（{e['order']}）", case, e['H5'], e['H1']))
            else:
                excluded.append(f"{e['pair']}（H1 任務失敗，缺節點統計 ⇒ 不進連續指標）")
    fig, axes = plt.subplots(2, 5, figsize=(15.5, 6.8))
    for ax, (k, title, sc, unit, _) in zip(axes.flat, PANELS):
        h5s, h1s = [], []
        for lbl, case, m5, m1 in pairs:
            a, b = m5[k] * sc, m1[k] * sc
            h5s.append(a)
            h1s.append(b)
            ax.plot([0, 1], [a, b], color='#b9b8b2', lw=1.5, ls=CASE_DASH[case], zorder=1)
            ax.scatter([0], [a], s=46, color=F.C_H5, zorder=3, edgecolors=F.SURF, linewidths=1.5)
            ax.scatter([1], [b], s=46, color=F.C_H1, zorder=3, edgecolors=F.SURF, linewidths=1.5)
        lo, hi = min(h5s + h1s), max(h5s + h1s)
        pad = (hi - lo) * 0.12 or 0.5
        ax.set_ylim(lo - pad, hi + pad)
        span = (hi - lo) + 2 * pad
        items = sorted(((b, lbl.split('（')[0]) for (lbl, _, _, _), b in zip(pairs, h1s)))
        ys = []
        for b, _ in items:
            ys.append(b if not ys else max(b, ys[-1] + span * 0.085))
        over = ys[-1] - (hi + pad * 0.9)
        if over > 0:
            ys = [y - over for y in ys]
        for (b, name), y in zip(items, ys):
            ax.annotate(name, (1, b), xytext=(1.09, y), textcoords='data',
                        va='center', fontsize=8, color=F.INK2)
        ax.set_xticks([0, 1], ['H5', 'H1'])
        ax.set_xlim(-0.35, 1.55)
        ax.set_title(f'{title}\n({unit})', fontsize=9.5, color=F.INK, loc='left')
        ax.grid(axis='y', color=F.GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s_ in ('top', 'right'):
            ax.spines[s_].set_visible(False)
        t150 = S['by_case']['T150']['direction'].get(k, '')
        ax.text(0.0, -0.2, f'T150：{t150}', transform=ax.transAxes, fontsize=8, color=F.INK2)
    fig.suptitle('C1a 行程 100／150 mm：H1 vs H5 可用配對（c1a_b1；實線 T150、虛線 T100）',
                 x=0.01, ha='left', fontsize=12.5, color=F.INK)
    fig.text(0.01, 0.028,
             '本批完成計數（物理判準、全部 8 趟）：T100 H5 2/2、H1 1/2；T150 H5 2/2、H1 2/2。'
             '排除：' + '；'.join(excluded) + '。',
             fontsize=8.5, color=F.INK2)
    fig.text(0.01, 0.006,
             '連續指標只描述證據完整的趟次，排除不是隨機的；T100 只有一對可用、不判方向。'
             '每案例 2 對，探索性比較；kernel 7.0.0-31、峰值 88–91 °C；計算時間差不完全歸因於 N。',
             fontsize=8.5, color=F.INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94), h_pad=2.6)
    p = os.path.join(OUT, 'c1a_paired.png')
    fig.savefig(p, dpi=170)
    plt.close(fig)
    print(p)


if __name__ == '__main__':
    main()
