#!/usr/bin/env python3
"""圖：聯合搜尋候選俯視（door_coopen_joint_v2 單組輸出的 full[0]）——持握開門軌跡、門停後握持前進、出口路徑，標開門末 10° 段。

    python3 evaluation/plot_door_coopen_joint_v2.py <joint_v2_單組.json> <out.png>
"""
from __future__ import annotations

import json
import math
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                         # noqa: E402
from matplotlib.patches import Circle, Rectangle        # noqa: E402

plt.rcParams['font.family'] = ['Noto Sans CJK TC', 'DejaVu Sans']
INK, MUTED, HOLD, TAIL, REL, DOOR = '#1f2328', '#8a9099', '#2a6fdb', '#0f9d58', '#d9822b', '#6b4f3a'


def main():
    d = json.load(open(sys.argv[1]))
    c = d['full'][0]
    T, P, q_rel = c['trajectory'], c['exit_path'], c['q_rel_deg']
    fig, ax = plt.subplots(figsize=(7.2, 7.8))
    ax.add_patch(Rectangle((-1.2, 0.0), 1.2, 0.12, color=MUTED, alpha=0.5))
    ax.add_patch(Rectangle((1.0, 0.0), 1.0, 0.12, color=MUTED, alpha=0.5))
    for qd, a in ((0, 0.25), (20, 0.35), (q_rel - 10, 0.55), (q_rel, 1.0)):
        q = math.radians(qd)
        ux, uy = math.cos(q), -math.sin(q)
        ax.plot([0, 0.95 * ux], [0, 0.95 * uy], color=DOOR, lw=3.5 if qd == q_rel else 2.2, alpha=a, solid_capstyle='butt')
        ax.text(0.98 * ux - 0.14, 0.98 * uy - 0.06, f'{qd:.0f}°', color=DOOR, alpha=max(a, 0.6), fontsize=9)
    head = [p for p in T if p['q_deg'] <= q_rel - 10]
    tail = [p for p in T if p['q_deg'] >= q_rel - 10]
    ax.plot([p['base'][0] for p in head], [p['base'][1] for p in head], color=HOLD, lw=2.2, label=f'握著把手開門 0→{q_rel - 10:.0f}°')
    ax.plot([p['base'][0] for p in tail], [p['base'][1] for p in tail], color=TAIL, lw=3.2,
            label=f'開門末 10°（{q_rel - 10:.0f}→{q_rel:.0f}°）：朝出口推進 {c["D_drop_last10_m"] * 100:.1f} cm')
    for p in T[::6] + [T[-1]]:
        x, y, yaw = p['base']
        ax.annotate('', xy=(x + 0.09 * math.cos(yaw), y + 0.09 * math.sin(yaw)), xytext=(x, y), arrowprops=dict(arrowstyle='->', color=HOLD, lw=1.0))
    ax.plot([p[0] for p in P], [p[1] for p in P], color=REL, lw=2.2, ls='--', label=f'門停後：握持前進 {c["fwd_m"] * 100:.0f} cm → 張爪 → 通過')
    ax.add_patch(Circle(T[0]['base'][:2], 0.30, fill=False, color=INK, lw=1.0, ls=':'))
    ax.text(T[0]['base'][0] + 0.06, T[0]['base'][1] + 0.05, '起點 0°', fontsize=9, color=INK)
    ax.add_patch(Circle(T[-1]['base'][:2], 0.30, fill=False, color=INK, lw=1.0, ls=':'))
    ax.text(T[-1]['base'][0] + 0.12, T[-1]['base'][1] - 0.14, f'門停 {q_rel:.0f}°', fontsize=9, color=INK)
    ax.plot(0, 0, 's', color=INK, ms=6)
    ax.text(-0.17, -0.07, '鉸鏈', fontsize=9, color=INK)
    ax.set_title(f"聯合搜尋候選（夾法 φ{c['grasp'][0]:.0f}°、s {c['grasp'][1]}）：開門末段已朝出口推進，{q_rel:.0f}° 門停後續握持前進\n"
                 f"開門末段 → 握持前進 方向轉角 {c['turn_open_fwd_deg']:.0f}°；規劃底盤 ≥ 30 mm（門檻 20＋誤差 10）",
                 fontsize=9.5, color=INK)
    ax.set_xlim(-1.0, 1.9)
    ax.set_ylim(-1.5, 0.9)
    ax.set_aspect('equal')
    ax.grid(color='#e3e6ea', lw=0.6)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)（門向 −y 開；機器人起於 −y 側）')
    ax.legend(loc='lower left', fontsize=8.3, frameon=False)
    fig.text(0.01, 0.005, '離線靜態運動學（剛性夾持、無接觸力）；有限搜尋範圍內的候選；全身掃掠、距離下界、門餘轉與時間化見 door_coopen_verify_v1', fontsize=7, color=MUTED)
    fig.tight_layout()
    fig.savefig(sys.argv[2], dpi=150)


if __name__ == '__main__':
    main()
