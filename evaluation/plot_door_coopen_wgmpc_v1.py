#!/usr/bin/env python3
"""圖：核心控制驗證——q48 參考底盤路徑 vs 真實 W-GMPC 閉迴路（A 無引導、B 引導 1、E 引導 1＋限幅 0.035 m/s）。

    python3 evaluation/plot_door_coopen_wgmpc_v1.py <out.png>
"""
import json
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                         # noqa: E402
from matplotlib.patches import Rectangle               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results', 'door')
plt.rcParams['font.family'] = ['Noto Sans CJK TC', 'DejaVu Sans']
INK, MUTED, DOOR = '#1f2328', '#8a9099', '#6b4f3a'
COL = {'ref': '#1f2328', 'A': '#c0392b', 'B': '#d9822b', 'E': '#2a6fdb'}


def main():
    ref = json.load(open(os.path.join(R, 'door_coopen_joint_v2_phi60_s0.85_q48.json')))['full'][0]
    v1 = {c['name']: c for c in json.load(open(os.path.join(R, 'door_coopen_wgmpc_v1.json')))['configs']}
    ve = {c['name']: c for c in json.load(open(os.path.join(R, 'door_coopen_wgmpc_v1_vcap0.035_wcap0.1.json')))['configs']}
    fig, ax = plt.subplots(figsize=(7.4, 7.0))
    ax.add_patch(Rectangle((-1.2, 0.0), 1.2, 0.12, color=MUTED, alpha=0.5))
    ax.add_patch(Rectangle((1.0, 0.0), 1.0, 0.12, color=MUTED, alpha=0.5))
    for qd, a in ((0, 0.25), (38, 0.5), (48, 1.0)):
        q = math.radians(qd)
        ax.plot([0, 0.95 * math.cos(q)], [0, -0.95 * math.sin(q)], color=DOOR, lw=3 if qd == 48 else 2, alpha=a)
        ax.text(0.98 * math.cos(q) - 0.12, -0.98 * math.sin(q) - 0.06, f'{qd}°', color=DOOR, fontsize=9)
    T = ref['trajectory']
    ax.plot([p['base'][0] for p in T], [p['base'][1] for p in T], color=COL['ref'], lw=1.6, ls=':', label='參考（beam 運動學，q48 候選）')
    for key, c, lab in (('A', v1['A_noguide'], 'A 無引導（底盤撞門框 −30.9 mm）'), ('B', v1['B_guide1'], 'B 引導 w 1、限幅 0.02 m/s（偏差 76 mm）'),
                        ('E', ve['E_guide1_cap'], 'E 引導 w 1、限幅 0.035 m/s（偏差 23 mm）')):
        bp = c['base_path']
        ax.plot([p[0] for p in bp], [p[1] for p in bp], color=COL[key], lw=2.0 if key == 'E' else 1.4, label=lab)
    ax.plot(0, 0, 's', color=INK, ms=6)
    ax.set_title('核心控制驗證：真實 W-GMPC 閉迴路（剛性持握運動學）\nE：門開到 48°、末 10° 朝出口推進 23.5 cm、全身淨空過門檻、底盤 ≤ 0.045 m/s', fontsize=10, color=INK)
    ax.set_xlim(-0.3, 1.5)
    ax.set_ylim(-1.3, 0.3)
    ax.set_aspect('equal')
    ax.grid(color='#e3e6ea', lw=0.6)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.legend(loc='lower left', fontsize=8.3, frameon=False)
    fig.text(0.01, 0.005, '運動學閉迴路、剛性持握、無接觸力與門動力學；夾持閉合誤差 E 5.3 mm（剛性假設下的追蹤落差）；不含放手與收臂；非 Isaac', fontsize=7, color=MUTED)
    fig.tight_layout()
    fig.savefig(sys.argv[1], dpi=150)


if __name__ == '__main__':
    main()
