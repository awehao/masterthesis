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

    # No fixed start/goal pair is drawn. The batches sample a new traverse per
    # trial from bigarena_poses_big.csv, so a single (0,0) -> (17,17) diagonal
    # would suggest the routes are one repeated corridor when they are 300
    # different ones. A sample of them is drawn instead, faintly, to show the
    # spread without implying any particular run.
    try:
        import csv as _csv
        rts = [r for r in _csv.DictReader(open(f'{ROOT}/evaluation/results/'
                                               'bigarena_poses_big.csv'))]
        for r in rts[:40]:
            ax.plot([float(r['start_x']), float(r['goal_x'])],
                    [float(r['start_y']), float(r['goal_y'])],
                    '-', color='#1565c0', lw=0.7, alpha=0.16, zorder=3)
        ax.plot([], [], '-', color='#1565c0', lw=1.4, alpha=0.5,
                label=f'隨機起訖點（示意 40 / 共 {len(rts)} 條）')
    except Exception:
        pass

    ax.set_xlim(-1.7, 18.7)
    ax.set_ylim(-1.7, 18.7)
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('實驗場景', fontsize=15, fontweight='bold')
    # Outside the axes: at 'lower right' the box sat on top of an unknown
    # static and the end of dyn_obs_0's traverse, hiding scene content in the
    # one figure whose job is to show all of it.
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=9,
              framealpha=0.95, borderaxespad=0)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=125, bbox_inches='tight')
    print(f'saved -> {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
