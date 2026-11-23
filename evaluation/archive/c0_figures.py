#!/usr/bin/env python3
"""C0（c0b1）成果圖表：三對配對圖＋分相位表。只讀已封存的分析檔，不重算判定。

輸出（results/horizon_ablation/figures/）：
  c0_paired.png     小倍數圖：每格一個指標，H5 → H1 以線連起同一對（p1、p2、p3r）
  c0_by_phase.csv   分相位表：每對 × 每相位 × 方法的時長、err_p p95、手臂相鄰反轉率、du_arm p95
  c0_by_phase.md    同上的摘要：各相位三對配對差是否同號（不同號 = 未呈現一致方向）

    python3 evaluation/c0_figures.py
"""
from __future__ import annotations

import csv
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib import font_manager                              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'runs')
A = os.path.join(HERE, 'results', 'horizon_ablation')
OUT = os.path.join(A, 'figures')
BATCH = 'c0b1'
PAIRS = ['p1', 'p2', 'p3r']
PAIR_ENV = {'p1': '7.0.0-34・63 °C', 'p2': '7.0.0-34・91 °C', 'p3r': '7.0.0-31・91 °C'}
PHASES = ['ALIGN', 'ENGAGE_WAIT', 'OPEN', 'OPEN_HOLD', 'CLOSE', 'CLOSE_HOLD',
          'RELEASE_WAIT', 'RETREAT']

# 參考調色盤（dataviz references/palette.md）第 1、2 槽；文字用墨色，不用系列色
C_H5, C_H1 = '#2a78d6', '#eb6834'
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def _tc_font():
    """由 NotoSansCJK 的 .ttc 取出繁中字面（同 build_sixth_progress_motm.py）。"""
    import tempfile
    from fontTools.ttLib import TTCollection
    out = []
    for w in ('Regular', 'Bold'):
        dst = os.path.join(tempfile.gettempdir(), f'NotoSansCJKtc-{w}.otf')
        if not os.path.exists(dst):
            col = TTCollection(f'/usr/share/fonts/opentype/noto/NotoSansCJK-{w}.ttc')
            for f in col.fonts:
                if f['name'].getDebugName(1).startswith('Noto Sans CJK TC'):
                    f.save(dst)
                    break
        font_manager.fontManager.addfont(dst)
        out.append(dst)
    return font_manager.FontProperties(fname=out[0]).get_name()


plt.rcParams.update({'font.family': _tc_font(), 'font.size': 10,
                     'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF})

# (鍵, 標題, 單位換算, 單位, 「越低越好」或 None)
PANELS = [
    ('window_s', '控制窗長度', 1, 's', True),
    ('disallowed_stops', '不允許停頓', 1, '次', True),
    ('grip_fraction_in_window', '夾持期間雙指 ≥ 0.5 N 比例', 100, '%', False),
    ('drift_max_mm', '夾持漂移 max', 1, 'mm', True),
    ('err_p_p95_m', '位置追蹤誤差 p95（整窗）', 1000, 'mm', True),
    ('arm_rev_adjacent_per_s', '手臂命令相鄰反轉', 1, '次/s', True),
    ('du_arm_p95', '手臂命令變化 p95', 1, 'rad/s', True),
    ('core_ms_p50', '求解核心時間 p50', 1, 'ms', True),
    ('open_hold_s', '開保持（連續）', 1, 's', None),
    ('home_dist_m', '回程終點距起點', 100, 'cm', None),
]


def paired_figure(S):
    P = S['paired']
    fig, axes = plt.subplots(2, 5, figsize=(15.5, 6.6))
    for ax, (k, title, sc, unit, lower_better) in zip(axes.flat, PANELS):
        h5 = [v * sc for v in P[k]['H5']]
        h1 = [v * sc for v in P[k]['H1']]
        for p, a, b in zip(PAIRS, h5, h1):
            ax.plot([0, 1], [a, b], color='#b9b8b2', lw=1.5, zorder=1)
            ax.scatter([0], [a], s=46, color=C_H5, zorder=3, edgecolors=SURF, linewidths=1.5)
            ax.scatter([1], [b], s=46, color=C_H1, zorder=3, edgecolors=SURF, linewidths=1.5)
        ax.set_xticks([0, 1], ['H5', 'H1'])
        ax.set_xlim(-0.35, 1.45)
        ax.set_title(f'{title}\n({unit})', fontsize=9.5, color=INK, loc='left')
        ax.grid(axis='y', color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        d = P[k]['diff_H1_minus_H5']
        if P[k]['same_sign_all_pairs']:
            tag = '三對方向一致'
        else:
            tag = '三對方向不一致'
        ax.text(0.0, -0.2, tag, transform=ax.transAxes, fontsize=8, color=INK2)
        lo, hi = min(h5 + h1), max(h5 + h1)
        pad = (hi - lo) * 0.12 or 0.5
        ax.set_ylim(lo - pad, hi + pad)
        # 對次標籤：同值合併，其餘以最小間距由下往上錯開，避免重疊
        span = (hi - lo) + 2 * pad
        groups = {}
        for p, b in zip(PAIRS, h1):
            key = round(b / span, 3)
            groups.setdefault(key, [b, []])[1].append(p)
        items = sorted(groups.values(), key=lambda g: g[0])
        gap = span * 0.085
        ys = []
        for b, _ in items:
            y = b if not ys else max(b, ys[-1] + gap)
            ys.append(y)
        over = ys[-1] - (hi + pad * 0.9)
        if over > 0:
            ys = [y - over for y in ys]
        for (b, names), y in zip(items, ys):
            ax.annotate('、'.join(names), (1, b), xytext=(1.09, y), textcoords='data',
                        va='center', fontsize=8, color=INK2)
    fig.suptitle('C0 單步（H1）vs 多步（H5）：三對配對結果（c0b1；每條線 = 同一對）',
                 x=0.01, ha='left', fontsize=12.5, color=INK)
    fig.text(0.01, 0.005,
             '配對環境：' + '；'.join(f'{p} {PAIR_ENV[p]}' for p in PAIRS) +
             '。每對固定 H5 先跑。n = 3 對，不做顯著性宣稱；計算時間受熱與順序影響未排除。',
             fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94), h_pad=2.6)
    p = os.path.join(OUT, 'c0_paired.png')
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


def by_phase():
    rows = []
    for p in PAIRS:
        for M in ('H5', 'H1'):
            rid = f'{BATCH}_{p}_{M}'
            hm = json.load(open(os.path.join(RUNS, rid, 'analysis', 'horizon_metrics.json')))
            st = hm['phase_start']
            t_end = hm['window']['to_sim_t']
            seq = [(ph, st[ph]) for ph in PHASES if ph in st]
            for i, (ph, t0) in enumerate(seq):
                t1 = seq[i + 1][1] if i + 1 < len(seq) else t_end
                b = hm['by_phase'].get(ph)
                if b is None:
                    continue
                sm = b['smoothness']
                ep = b['tracking']['err_p_m']
                rows.append({'pair': p, 'method': M, 'phase': ph,
                             'duration_s': round(max(0.0, min(t1, t_end) - t0), 3),
                             'err_p_p95_mm': None if not ep else round(ep['p95'] * 1e3, 3),
                             'arm_rev_adj_per_s': sm['arm_reversals']['adjacent_rate_per_s'],
                             'du_arm_p95': None if not sm['du_arm'] else sm['du_arm']['p95']})
    with open(os.path.join(OUT, 'c0_by_phase.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def get(p, M, ph, k):
        for r in rows:
            if r['pair'] == p and r['method'] == M and r['phase'] == ph:
                return r[k]
        return None
    keys = [('duration_s', '時長 s'), ('err_p_p95_mm', 'err_p p95 mm'),
            ('arm_rev_adj_per_s', '手臂反轉 次/s'), ('du_arm_p95', 'du_arm p95')]
    lines = ['# C0 分相位表（c0b1：p1、p2、p3r）', '',
             '每格：H5 → H1 三對的值（p1／p2／p3r），最後標三對配對差（H1 − H5）的方向。',
             '「一致↑」= 三對 H1 都較高；「一致↓」= 三對 H1 都較低；「不一致」= 未呈現一致方向'
             '（不等於證明沒有差異）。n = 3 對，不做顯著性宣稱。', '']
    lines.append('| 相位 | ' + ' | '.join(lbl for _, lbl in keys) + ' |')
    lines.append('|---|' + '---|' * len(keys))
    for ph in PHASES:
        cells = []
        for k, _ in keys:
            h5 = [get(p, 'H5', ph, k) for p in PAIRS]
            h1 = [get(p, 'H1', ph, k) for p in PAIRS]
            if any(v is None for v in h5 + h1):
                cells.append('—')
                continue
            d = [b - a for a, b in zip(h5, h1)]
            tag = ('一致↑' if all(x > 0 for x in d) else
                   '一致↓' if all(x < 0 for x in d) else '不一致')

            def f(v):
                return f'{v:.3g}'
            cells.append('／'.join(f(v) for v in h5) + ' → ' +
                         '／'.join(f(v) for v in h1) + f'　**{tag}**')
        lines.append(f'| {ph} | ' + ' | '.join(cells) + ' |')
    lines += ['', '來源：runs/c0b1_*/analysis/horizon_metrics.json（horizon_metrics.py，凍結版）。',
              '相位時長以任務事件切分；RETREAT 到求解節點停止為止。']
    p = os.path.join(OUT, 'c0_by_phase.md')
    open(p, 'w').write('\n'.join(lines) + '\n')
    return p, rows


def main():
    os.makedirs(OUT, exist_ok=True)
    S = json.load(open(os.path.join(A, f'{BATCH}_summary.json')))
    print(paired_figure(S))
    p, _ = by_phase()
    print(p)


if __name__ == '__main__':
    main()
