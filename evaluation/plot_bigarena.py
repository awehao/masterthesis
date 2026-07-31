"""Draw the 20 x 20 m floor: what is on the map, what is not, and the traffic.

    python3 evaluation/plot_bigarena.py [out.png]
"""
import os
import re
import sys

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle, Circle
from PIL import Image

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc')
plt.rcParams.update({'font.family': ['Noto Sans CJK JP'],
                     'axes.unicode_minus': False})

ROOT = '/home/howardchen/masterthesis'
SHARE = f'{ROOT}/src/ammr_bringup'
ORIGIN, RES = (-1.5, -1.5), 0.05
START, GOAL = (0.0, 0.0), (17.0, 17.0)


def models(prefix):
    sdf = open(f'{SHARE}/worlds/bigarena.sdf').read()
    out = []
    for m in re.finditer(r'<model name="(%s\d+)">(.*?)</model>' % prefix,
                         sdf, re.S):
        b = m.group(2)
        po = re.search(r'<pose>([-\d.eE+ ]+)</pose>', b)
        bx = re.search(r'<box><size>([\d.]+) ([\d.]+)', b)
        if po and bx:
            v = po.group(1).split()
            out.append((float(v[0]), float(v[1]),
                        float(bx.group(1)), float(bx.group(2))))
    return out


def mover_parts():
    sdf = open(f'{SHARE}/worlds/bigarena.sdf').read()
    out = {}
    for m in re.finditer(r'<model name="(dyn_obs_\d+)">(.*?)</model>', sdf, re.S):
        parts = []
        for c in re.finditer(r'<collision[^>]*>(.*?)</collision>', m.group(2), re.S):
            cb = c.group(1)
            po = re.search(r'<pose>([-\d.eE+ ]+)</pose>', cb)
            px, py = ((float(po.group(1).split()[0]),
                       float(po.group(1).split()[1])) if po else (0.0, 0.0))
            bx = re.search(r'<box><size>([\d.]+) ([\d.]+)', cb)
            cy = re.search(r'<cylinder><radius>([\d.]+)', cb)
            if bx:
                parts.append(('box', px, py,
                              float(bx.group(1)), float(bx.group(2))))
            elif cy:
                parts.append(('cyl', px, py, float(cy.group(1)), 0.0))
        out[m.group(1)] = parts
    return out


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else \
        f'{ROOT}/evaluation/results/figs/bigarena.png'
    img = np.array(Image.open(f'{SHARE}/maps/bigarena.pgm'))
    occ = (255 - img) / 255.0 > 0.65
    h, w = occ.shape
    extent = [ORIGIN[0], ORIGIN[0] + w * RES, ORIGIN[1], ORIGIN[1] + h * RES]

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(occ, cmap='Greys', origin='upper', extent=extent, alpha=0.85)

    for x, y, sx, sy in models('unknown_obs_'):
        ax.add_patch(Rectangle((x - sx / 2, y - sy / 2), sx, sy,
                               color='#f9a825', alpha=0.95, zorder=5))
    ax.plot([], [], 's', color='#f9a825', ms=10,
            label='未知靜態雜物（不在地圖上，只能靠感知）')
    ax.plot([], [], 's', color='#777777', ms=10, label='已知牆與雜物（在地圖上）')

    obs = yaml.safe_load(
        open(f'{SHARE}/config/dynamic_trajectories_bigarena_traffic.yaml')
    ).get('dynamic_obstacles') or []
    fp = mover_parts()
    cols = ['#c62828', '#6a1b9a', '#00838f', '#2e7d32',
            '#ef6c00', '#1a237e', '#ad1457']
    for i, o in enumerate(obs):
        a, b = np.array(o['start'], float), np.array(o['end'], float)
        col = cols[i % len(cols)]
        ax.plot([a[0], b[0]], [a[1], b[1]], '-', color=col, lw=2.2,
                alpha=0.8, zorder=6)
        ax.plot(*b, 'x', color=col, ms=9, mew=2, zorder=7)
        for kind, px, py, sx, sy in fp.get(o['name'], []):
            if kind == 'box':
                ax.add_patch(Rectangle((a[0] + px - sx / 2, a[1] + py - sy / 2),
                                       sx, sy, color=col, alpha=0.95, zorder=8))
            else:
                ax.add_patch(Circle((a[0] + px, a[1] + py), sx,
                                    color=col, alpha=0.95, zorder=8))
        L = float(np.linalg.norm(b - a))
        ax.plot([], [], '-', color=col, lw=2.2,
                label=f"{o['name']}  v={o['speed']}  穿越 {L:.1f} m  "
                      f"週期 {2*L/max(o['speed'],1e-6):.0f} s")

    ax.plot([START[0], GOAL[0]], [START[1], GOAL[1]], ':', color='#1565c0',
            lw=1.4, alpha=0.6, zorder=4, label='起點→終點直線 24.0 m')
    ax.plot(*START, '^', color='#1565c0', ms=16, zorder=10)
    ax.annotate('起點', START, xytext=(12, -20), textcoords='offset points',
                fontsize=12, fontweight='bold', color='#1565c0')
    ax.plot(*GOAL, '*', color='#2e7d32', ms=26, zorder=10)
    ax.annotate('終點', GOAL, xytext=(-42, 12), textcoords='offset points',
                fontsize=12, fontweight='bold', color='#2e7d32')

    ax.set_xlim(-1.7, 18.7)
    ax.set_ylim(-1.7, 18.7)
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('bigarena：20×20 m,八個區塊、錯開的門口、交通流\n'
                 '交錯的長距離穿越,沒有一條瞄準機器人',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8.5, framealpha=0.93)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=125, bbox_inches='tight')
    print(f'saved -> {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
