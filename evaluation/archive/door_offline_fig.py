#!/usr/bin/env python3
"""door_offline_v0.json 的俯視圖：門扇各角度、把手圓弧、協同開門的底盤路徑（貪婪延續的存在性示例）、通行門角。

    python3 evaluation/door_offline_fig.py     # 輸出 第七次進度報告素材/圖表/fig_door_offline_v0.png
"""
import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle  # noqa: E402
import numpy as np                                     # noqa: E402
import yaml                                            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
R = json.load(open(os.path.join(HERE, 'results', 'door', 'door_offline_v0.json')))
D = yaml.safe_load(open(os.path.join(WS, R['spec'])))
OUT = os.path.join(WS, '第七次進度報告素材', '圖表', 'fig_door_offline_v0.png')
for f in ('Noto Sans CJK TC', 'Noto Sans CJK JP'):
    if any(f in x.name for x in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = f
        break
INK, MUTED, WALL, LEAF, BASE, PASS = '#1f2328', '#6e7781', '#3d434b', '#8a5a2b', '#2a7ab0', '#2e8b57'


def leaf_poly(q):
    L, t = D['leaf']['length'], D['leaf']['thickness']
    u = np.array([math.cos(q), -math.sin(q)])
    v = np.array([math.sin(q), math.cos(q)])
    return np.array([[0, 0], u * L, u * L + v * t, v * t])


def panel(ax, h, title):
    W, T = D['frame_clear_width'], D['wall']['thickness']
    ax.add_patch(Rectangle((-1.5, 0), 1.5, T, fc=WALL))
    ax.add_patch(Rectangle((W, 0), 1.5, T, fc=WALL))
    p = R['heights'][h]['C_coop']['path']
    qs = [x['q_deg'] for x in p]
    for qd in (0, 30, 60, 80, 90):
        a = 0.25 + 0.15 * (qd / 90)
        ax.add_patch(Polygon(leaf_poly(math.radians(qd)), closed=True, fc=LEAF, alpha=a, ec=LEAF, lw=0.6))
        u = np.array([math.cos(math.radians(qd)), -math.sin(math.radians(qd))])
        ax.text(*(u * 1.0), f'{qd}°', fontsize=7, color=LEAF, ha='center', va='center')
    th = np.radians(np.linspace(0, 100, 50))
    rh = D['handle']['from_hinge']
    ax.plot(rh * np.cos(th) + 0.04 * np.sin(-th) * 0, -rh * np.sin(th), color=LEAF, ls=':', lw=1)
    xs = [x['base'][0] for x in p]
    ys = [x['base'][1] for x in p]
    ax.plot(xs, ys, color=BASE, lw=1.4)
    for x in p:
        if x['q_deg'] in (2.0, 40.0, 80.0, qs[-1]):
            ax.add_patch(Circle((x['base'][0], x['base'][1]), 0.30, fc='none', ec=BASE, lw=0.9))
            yaw = x['base'][2]
            ax.annotate('', xy=(x['base'][0] + 0.25 * math.cos(yaw), x['base'][1] + 0.25 * math.sin(yaw)), xytext=(x['base'][0], x['base'][1]),
                        arrowprops=dict(arrowstyle='->', color=BASE, lw=0.9))
            ax.text(x['base'][0], x['base'][1] - 0.36, f"q={x['q_deg']:.0f}°", fontsize=7, color=BASE, ha='center', va='top')
    pa = R['pass_angle']['clear_0.05']['q_deg']
    ax.add_patch(Polygon(leaf_poly(math.radians(pa)), closed=True, fc='none', ec=PASS, lw=1.2, ls='--'))
    ax.text(0.55, 0.35, f'門洞淨寬 {W:.2f} m', fontsize=7.5, color=INK, ha='center')
    ax.text(-1.45, -1.95, f"通行門角 ≥ {pa}°（收攏包絡半徑 {R['stow_horizontal_envelope_m']:.3f} m＋5 cm）", fontsize=7.5, color=PASS)
    C = R['heights'][h]['C_coop']
    jm = min(x['joint_margin'] for x in p)
    bc = min(x['base_clear'] for x in p)
    ax.set_title(f"{title}　協同至 {C['q_max_deg']:.0f}°；最小關節餘裕 {jm:.3f} rad、底盤–門扇最小間距 {bc:.3f} m", fontsize=8.5, loc='left', color=INK)
    ax.set_xlim(-1.5, 1.6)
    ax.set_ylim(-2.05, 0.5)
    ax.set_aspect('equal')
    ax.grid(color='#e6e8eb', lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7, colors=MUTED)
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.6), dpi=160)
    panel(axs[0], '0.75', '把手高 0.75 m')
    panel(axs[1], '0.85', '把手高 0.85 m')
    fig.suptitle('局部通行門 v0 離線核對（候選參數、底盤圓盤近似；貪婪延續只示範存在一條可行路徑，非最佳）——鉸鏈在原點、機器人在 −y 側、門朝機器人拉開',
                 fontsize=9.5, x=0.01, ha='left', color=INK)
    fig.tight_layout()
    fig.savefig(OUT)
    print('寫出', OUT)


if __name__ == '__main__':
    main()
