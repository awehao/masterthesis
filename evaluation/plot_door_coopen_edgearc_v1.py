#!/usr/bin/env python3
"""圖：去程「繞緣夾法、邊開邊繞」俯視（讀 door_coopen_edgearc_v1.json 的 q45 roll 0 種子最佳解）。

    python3 evaluation/plot_door_coopen_edgearc_v1.py <out.png>
"""
from __future__ import annotations

import json
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                         # noqa: E402
from matplotlib.patches import Circle, Rectangle        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams['font.family'] = ['Noto Sans CJK TC', 'DejaVu Sans']
INK, MUTED, HOLD, REL, DOOR = '#1f2328', '#8a9099', '#2a6fdb', '#d9822b', '#6b4f3a'


def main():
    out = sys.argv[1]
    d = json.load(open(os.path.join(HERE, 'results', 'door', 'door_coopen_edgearc_v1.json')))
    s = next(x for x in d['per_seed'] if x['seed_q_deg'] == 45 and x['seed']['roll'] == 0)
    b = s['best']
    T, R = b['trajectory'], b['release_path']
    fig, ax = plt.subplots(figsize=(7.2, 7.6))
    ax.add_patch(Rectangle((-1.2, 0.0), 1.2, 0.12, color=MUTED, alpha=0.5))
    ax.add_patch(Rectangle((1.0, 0.0), 1.0, 0.12, color=MUTED, alpha=0.5))
    for qd, a in ((0, 0.25), (15, 0.35), (30, 0.5), (46, 1.0)):
        q = math.radians(qd)
        ux, uy = math.cos(q), -math.sin(q)
        ax.plot([0, 0.95 * ux], [0, 0.95 * uy], color=DOOR, lw=3.5 if qd == 46 else 2.2, alpha=a, solid_capstyle='butt')
        ax.text(0.98 * ux + 0.02, 0.98 * uy - 0.02, f'{qd}°', color=DOOR, alpha=max(a, 0.6), fontsize=9)
        hx, hy = 0.80 * ux + 0.04 * (-math.sin(q)), 0.80 * uy + 0.04 * (-math.cos(q))
        ax.plot(hx, hy, 'o', color=DOOR, ms=4, alpha=max(a, 0.6))
    xs, ys = [p['base'][0] for p in T], [p['base'][1] for p in T]
    ax.plot(xs, ys, color=HOLD, lw=2.2, label='持握開門（0→46°，底盤中心）')
    for p in T[::5] + [T[-1]]:
        x, y, yaw = p['base']
        ax.annotate('', xy=(x + 0.09 * math.cos(yaw), y + 0.09 * math.sin(yaw)), xytext=(x, y), arrowprops=dict(arrowstyle='->', color=HOLD, lw=1.2))
    ax.plot([r[0] for r in R], [r[1] for r in R], color=REL, lw=2.2, ls='--', label='46° 放手後直通（格點最短路徑）')
    for p, lab in ((T[0], '起點 0°'), (T[-1], '放手 46°')):
        ax.add_patch(Circle(p['base'][:2], 0.30, fill=False, color=INK, lw=1.0, ls=':'))
        ax.text(p['base'][0] + 0.05, p['base'][1] + 0.04, lab, fontsize=9, color=INK)
    ax.add_patch(Circle(R[-1], 0.30, fill=False, color=REL, lw=1.0, ls=':'))
    ax.text(-1.1, 0.2, '門洞另一側（目標）', fontsize=9, color=MUTED)
    ax.plot(0, 0, 's', color=INK, ms=6)
    ax.text(-0.17, -0.07, '鉸鏈', fontsize=9, color=INK)
    hs, rel = b['hold'], b['release']
    ax.set_title(f"去程繞緣夾法（φ 60°）邊開邊繞候選：0→46° 持握，46° 放手\n"
                 f"持握 {hs['path_len_m']:.2f} m／{hs['duration_s']:.0f} s；放手後最短路徑 {rel['len_m']:.2f} m、與持握末段弦方向差 {rel['turn_deg']:.0f}°（貼 20 mm 門檻）",
                 fontsize=10, color=INK)
    ax.set_xlim(-1.2, 1.9)
    ax.set_ylim(-1.5, 0.9)
    ax.set_aspect('equal')
    ax.grid(color='#e3e6ea', lw=0.6)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)（門向 −y 開；機器人起於 −y 側）')
    ax.legend(loc='lower left', fontsize=9, frameon=False)
    fig.text(0.01, 0.005, '離線靜態運動學核驗（剛性夾持、無接觸力）；最寬路徑瓶頸 28 mm 屬另一條未重建路徑；放手後門視為靜止；手臂收攏、張爪後門餘轉未核', fontsize=7.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


if __name__ == '__main__':
    main()
