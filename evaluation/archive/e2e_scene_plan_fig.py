#!/usr/bin/env python3
"""由 results/specs/E2E_scene_v1_draft.yaml 畫有尺寸的平面圖（數值只取自登錄檔）。

    python3 evaluation/e2e_scene_plan_fig.py      # 輸出 第七次進度報告素材/圖表/fig_e2e_scene_v1_draft.png
"""
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.patches import Arc, Circle, Rectangle, Wedge  # noqa: E402
import yaml                                            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, '..'))
S = yaml.safe_load(open(os.path.join(HERE, 'results', 'specs', 'E2E_scene_v1_draft.yaml')))
OUT = os.path.join(WS, '第七次進度報告素材', '圖表', 'fig_e2e_scene_v1_draft.png')
for f in ('Noto Sans CJK TC', 'Noto Sans CJK JP', 'Noto Sans TC'):
    if any(f in x.name for x in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = f
        break

INK, MUTED, WALL = '#1f2328', '#6e7781', '#3d434b'
FURN, UNK, DYN, DOOR, PATH = '#c8b48a', '#d1495b', '#2a7ab0', '#8a5a2b', '#2e8b57'


def wall_rect(ax, x, y):
    ax.add_patch(Rectangle((x[0], y[0]), x[1] - x[0], y[1] - y[0], fc=WALL, ec='none'))


def main():
    fig, ax = plt.subplots(figsize=(13, 7.2), dpi=160)
    (x0, x1), (y0, y1) = S['interior']['x'], S['interior']['y']
    t = 0.10
    for x, y in (((x0 - t, x1 + t), (y0 - t, y0)), ((x0 - t, x1 + t), (y1, y1 + t)),
                 ((x0 - t, x0), (y0, y1)), ((x1, x1 + t), (y0, y1))):
        wall_rect(ax, x, y)
    for w in S['walls_internal'].values():
        oy = w['opening']['y']
        wall_rect(ax, w['x'], (y0, oy[0]))
        wall_rect(ax, w['x'], (oy[1], y1))
    for name, r in S['rooms'].items():
        ax.text(sum(r['x']) / 2, y1 + 0.18, {'bedroom': '臥室（起點區）', 'living': '客廳（導航區）', 'study': '書房（抽屜操作區）'}[name],
                ha='center', va='bottom', fontsize=11, color=INK, weight='bold')
    for f in S['furniture_known']:
        if 'x' in f:
            ax.add_patch(Rectangle((f['x'][0], f['y'][0]), f['x'][1] - f['x'][0], f['y'][1] - f['y'][0], fc=FURN, ec='#8c7a55', lw=0.8))
            ax.text(sum(f['x']) / 2, sum(f['y']) / 2, f['name'], ha='center', va='center', fontsize=7, color=INK)
        else:
            px, py = f['pose']['x'], f['pose']['y']
            ax.add_patch(Rectangle((px - 0.575, py - 0.285), 1.15, 0.57, fc=FURN, ec='#8c7a55', lw=0.8))
            ax.text(px, py, '抽屜櫃\n(drawer_unit)', ha='center', va='center', fontsize=7, color=INK)
    for u in S['unknown_static']['items']:
        if True:
            sx, sy = u['size'][0], u['size'][1]
            ax.add_patch(Rectangle((u['x'] - sx / 2, u['y'] - sy / 2), sx, sy, fc=UNK, ec='none', alpha=0.85))
            ax.text(u['x'], u['y'] - sy / 2 - 0.12, f"{u['name']}（未知靜態）", ha='center', va='top', fontsize=7, color=UNK)
    d = S['door']
    hx, hy, L = d['hinge']['x'], d['hinge']['y'], d['leaf']['length']
    ax.add_patch(Wedge((hx, hy), L, 90, 180, fc='#f3e6d6', ec='none', alpha=0.9))
    ax.add_patch(Arc((hx, hy), 2 * L, 2 * L, theta1=90, theta2=180, ec=DOOR, ls='--', lw=1))
    ax.plot([hx - 0.02, hx - 0.02], [hy + 0.025, hy + 0.025 + L], color=DOOR, lw=3, solid_capstyle='butt')
    ax.plot([hx, hx - L], [hy, hy], color=DOOR, lw=1.2, ls=':', alpha=0.8)
    ax.add_patch(Circle((hx, hy), 0.04, fc=DOOR))
    ax.text(hx - L / 2, hy - 0.08, '門扇掃掠區（不放家具）', ha='center', va='top', fontsize=7, color=DOOR)
    ax.text(hx + 0.05, hy + 0.5, '通行門\n淨寬 1.00', ha='left', va='center', fontsize=7, color=DOOR)
    for dyn in S['dynamic']:
        (a, b), (c, e) = dyn['path']
        ax.annotate('', xy=(c, e), xytext=(a, b), arrowprops=dict(arrowstyle='->', color=DYN, lw=1.6))
        ax.add_patch(Circle((a, b), 0.25, fc=DYN, alpha=0.35, ec=DYN))
        ax.text(a + 0.32, b - 0.1, f"{dyn['name']}\n{dyn['speed']} m/s，時間排程", fontsize=7, color=DYN, va='top')
    st = S['start']
    ax.add_patch(Circle((st['x'], st['y']), 0.30, fc='none', ec=PATH, lw=1.5))
    ax.annotate('', xy=(st['x'] + 0.45, st['y']), xytext=(st['x'], st['y']), arrowprops=dict(arrowstyle='->', color=PATH))
    ax.text(st['x'], st['y'] - 0.38, '起點／終點', ha='center', va='top', fontsize=8, color=PATH)
    grp = {}
    for g in S['nav_goals']:
        if isinstance(g.get('x'), (int, float)):
            grp.setdefault((g['x'], g['y']), []).append(g['id'])
    for (gx, gy), ids in grp.items():
        ax.plot(gx, gy, marker='x', color=PATH, ms=6)
        ax.text(gx + 0.06, gy + 0.08, ' / '.join(ids), fontsize=6.5, color=PATH)
    ax.text(8.75, 4.6 - 0.285 - 0.12, '前板 y≈4.315', ha='center', va='top', fontsize=6.5, color=MUTED)

    def dim(xa, xb, y, txt, vertical=False):
        if vertical:
            ax.annotate('', xy=(y, xa), xytext=(y, xb), arrowprops=dict(arrowstyle='<->', color=MUTED, lw=0.8))
            ax.text(y - 0.08, (xa + xb) / 2, txt, rotation=90, ha='right', va='center', fontsize=7, color=MUTED)
        else:
            ax.annotate('', xy=(xa, y), xytext=(xb, y), arrowprops=dict(arrowstyle='<->', color=MUTED, lw=0.8))
            ax.text((xa + xb) / 2, y - 0.06, txt, ha='center', va='top', fontsize=7, color=MUTED)
    dim(x0, x1, y0 - 0.35, f'{x1 - x0:.2f} m')
    for r in S['rooms'].values():
        dim(r['x'][0], r['x'][1], y0 - 0.65, f"{r['x'][1] - r['x'][0]:.2f}")
    dim(y0, y1, x0 - 0.35, f'{y1 - y0:.2f} m', vertical=True)
    oy = S['walls_internal']['W2_living_study']['opening']['y']
    dim(oy[0], oy[1], 7.25, '開口 1.20', vertical=True)
    ax.set_xlim(x0 - 0.9, x1 + 0.3)
    ax.set_ylim(y0 - 1.0, y1 + 0.6)
    ax.set_aspect('equal')
    ax.set_xticks(range(0, 11))
    ax.set_yticks(range(0, 6))
    ax.tick_params(labelsize=7, colors=MUTED)
    ax.grid(color='#e6e8eb', lw=0.5)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title('E2E 住宅整合場景 v1（draft-1）：起點 → 通行門 → 客廳導航 → 書房抽屜 → 返回　〔TBD 尺寸待離線核對〕',
                 fontsize=10, color=INK, loc='left')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT)
    print('寫出', OUT)


if __name__ == '__main__':
    main()
