#!/usr/bin/env python3
"""圖：去程繞緣候選 v2（預算版）俯視——edgehold_v3 持握軌跡、門停後握持前進、放手收臂（底盤轉身 +90°）、通過路徑與交還點。

    python3 evaluation/plot_door_coopen_candidate_v2.py <out.png>
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
INK, MUTED, HOLD, REL, ARM, DOOR = '#1f2328', '#8a9099', '#2a6fdb', '#d9822b', '#7a4fb3', '#6b4f3a'
R = os.path.join(HERE, 'results', 'door')


def main():
    out = sys.argv[1]
    h = json.load(open(os.path.join(R, 'door_coopen_edgehold_v3.json')))
    p1 = json.load(open(os.path.join(R, 'door_coopen_pass_smooth_v1.json')))
    p3 = json.load(open(os.path.join(R, 'door_coopen_pass_smooth_v3.json')))
    T, P = h['trajectory'], p1['path']
    q_rel = h['q_rel_deg']
    fig, ax = plt.subplots(figsize=(7.2, 7.8))
    ax.add_patch(Rectangle((-1.2, 0.0), 1.2, 0.12, color=MUTED, alpha=0.5))
    ax.add_patch(Rectangle((1.0, 0.0), 1.0, 0.12, color=MUTED, alpha=0.5))
    for qd, a in ((0, 0.25), (20, 0.35), (40, 0.5), (q_rel, 1.0)):
        q = math.radians(qd)
        ux, uy = math.cos(q), -math.sin(q)
        ax.plot([0, 0.95 * ux], [0, 0.95 * uy], color=DOOR, lw=3.5 if qd == q_rel else 2.2, alpha=a, solid_capstyle='butt')
        ax.text(0.98 * ux - 0.13, 0.98 * uy - 0.06, f'{qd}°', color=DOOR, alpha=max(a, 0.6), fontsize=9)
    ax.plot([p['base'][0] for p in T], [p['base'][1] for p in T], color=HOLD, lw=2.2, label=f'握著把手開門 0→{q_rel}°（底盤中心）')
    for p in T[::6] + [T[-1]]:
        x, y, yaw = p['base']
        ax.annotate('', xy=(x + 0.09 * math.cos(yaw), y + 0.09 * math.sin(yaw)), xytext=(x, y), arrowprops=dict(arrowstyle='->', color=HOLD, lw=1.1))
    ax.plot([p[0] for p in P], [p[1] for p in P], color=REL, lw=2.2, ls='--', label='門停後：握持前進 → 張爪 → 通過（同一條路徑）')
    adv = p3['base_advance_m']
    k = min(int(round(adv / 0.01)), len(P) - 1)
    ax.plot([p[0] for p in P[:k + 1]], [p[1] for p in P[:k + 1]], color=ARM, lw=4.5, alpha=0.45, label=f'收臂段（底盤前進 {adv:.2f} m、轉身 +{p3["dyaw_deg"]:.0f}°）')
    ax.add_patch(Circle(T[0]['base'][:2], 0.30, fill=False, color=INK, lw=1.0, ls=':'))
    ax.text(T[0]['base'][0] + 0.06, T[0]['base'][1] + 0.05, '起點 0°', fontsize=9, color=INK)
    ax.add_patch(Circle(T[-1]['base'][:2], 0.30, fill=False, color=INK, lw=1.0, ls=':'))
    ax.text(T[-1]['base'][0] + 0.12, T[-1]['base'][1] - 0.12, f'門停 {q_rel}°', fontsize=9, color=INK)
    ax.axhline(0.50, color=MUTED, lw=0.8, ls=':')
    ax.text(1.05, 0.52, '交還導航：底盤 y ≥ 0.50', fontsize=8.5, color=MUTED)
    ax.plot(0, 0, 's', color=INK, ms=6)
    ax.text(-0.17, -0.07, '鉸鏈', fontsize=9, color=INK)
    m = h['margin_min_mm']
    ax.set_title(f"去程繞緣候選（預算版）：開門時底盤後退，門停 {q_rel}° 後轉向前進、張爪、轉身收臂、通過\n"
                 f"持握最小餘裕：底盤 {m['base']:.0f} mm、自碰 {m['self']:.1f} mm；通過路徑在門 {q_rel - 2}–{q_rel + 5}° 下 ≥ {p1['path_clear_min_m'] * 1000:.0f} mm",
                 fontsize=9.5, color=INK)
    ax.set_xlim(-1.0, 1.9)
    ax.set_ylim(-1.5, 0.9)
    ax.set_aspect('equal')
    ax.grid(color='#e3e6ea', lw=0.6)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)（門向 −y 開；機器人起於 −y 側）')
    ax.legend(loc='lower left', fontsize=8.5, frameon=False)
    fig.text(0.01, 0.005, '離線靜態運動學核驗（剛性夾持、無接觸力）；底盤誤差 10 mm＋門餘轉區間包絡；張爪前門須靜止；不是已驗證的不停頓通行', fontsize=7.2, color=MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


if __name__ == '__main__':
    main()
