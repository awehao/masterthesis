"""One panel per arena scenario: ten trajectories, the obstacles, what went wrong.

Reading a table of collision counts does not tell you WHY a scenario is hard.
Seeing ten runs overlaid does: whether the robot picked the same gap every time
or split between them, whether it stopped and waited, where it grazed something.
Each panel therefore carries the geometry that produced the numbers -- known
walls, unknown static cylinders, and every mover's actual shape and track.

Marks:
  green line   a trial that reached the goal without contact
  red line     a trial that made contact
  red circle   a heading reversal of more than 90 degrees
  orange dot   the point of closest approach in that trial

    python3 evaluation/plot_arena.py [out.png]
"""
import glob
import math
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle, Circle
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wiggle import trajectory, resample, DS            # noqa: E402

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc')
plt.rcParams.update({'font.family': ['Noto Sans CJK JP'],
                     'axes.unicode_minus': False})

ROOT = '/home/howardchen/masterthesis'
SHARE = f'{ROOT}/src/ammr_bringup'
ORIGIN, RES = (-1.3, -1.3), 0.05
UNKNOWN_STATIC = [(1.5, 4.8, 0.30), (3.4, 2.4, 0.30)]
START, GOAL = (0.0, 0.0), (6.6, 6.6)

ORDER = ['none', 'crossing', 'gapblock', 'corridor', 'converge', 'overtake',
         'parked', 'shapes', 'occlude', 'stopgo', 'small', 'ell', 'dense',
         'diagonal', 'headon', 'bothgaps', 'fast', 'chase', 'merge', 'wide']


def mover_geometry():
    """Each mover's true shape, so a box is not drawn (or scored) as a circle."""
    sdf = open(f'{SHARE}/worlds/arena.sdf').read()
    geo = {}
    for m in re.finditer(r'<model name="(dyn_obs_\d+)">(.*?)</model>', sdf, re.S):
        body = m.group(2)
        bx = re.search(r'<box><size>([\d.]+) ([\d.]+)', body)
        cy = re.search(r'<cylinder><radius>([\d.]+)', body)
        n_col = len(re.findall(r'<collision', body))
        if bx:
            geo[m.group(1)] = ('ell' if n_col > 1 else 'box',
                               float(bx.group(1)), float(bx.group(2)))
        elif cy:
            geo[m.group(1)] = ('cyl', float(cy.group(1)), float(cy.group(1)))
    return geo


def scenario_movers(name):
    import yaml
    f = f'{SHARE}/config/dynamic_trajectories_arena_{name}.yaml'
    if not os.path.isfile(f):
        return []
    return yaml.safe_load(open(f)).get('dynamic_obstacles') or []


def trial_rows(name):
    import csv
    f = f'{ROOT}/evaluation/results/final_arena_{name}.csv'
    if not os.path.isfile(f):
        return []
    return list(csv.DictReader(open(f)))


def reversal_points(p):
    q = resample(p, DS)
    d = np.diff(q, axis=0)
    keep = np.linalg.norm(d, axis=1) > 1e-4
    idx = np.where(keep)[0]
    d = d[keep]
    if len(d) < 3:
        return []
    h = np.arctan2(d[:, 1], d[:, 0])
    dh = np.degrees(np.abs((np.diff(h) + np.pi) % (2 * np.pi) - np.pi))
    return [q[idx[i + 1]] for i in np.where(dh > 90)[0]]


def draw(ax, name, occ, extent, geo):
    ax.imshow(occ, cmap='Greys', origin='upper', extent=extent, alpha=0.9)
    for x, y, r in UNKNOWN_STATIC:
        ax.add_patch(Circle((x, y), r, color='#f9a825', alpha=0.9, zorder=3))
    for o in scenario_movers(name):
        a, b = np.array(o['start'], float), np.array(o['end'], float)
        kind, sx, sy = geo.get(o['name'], ('cyl', 0.25, 0.25))
        col = {'cyl': '#37474f', 'box': '#c62828', 'ell': '#6a1b9a'}[kind]
        ax.plot([a[0], b[0]], [a[1], b[1]], '-', color=col, lw=2.0, alpha=0.55,
                zorder=4)
        if kind == 'cyl':
            ax.add_patch(Circle(a, sx, color=col, alpha=0.85, zorder=5))
        else:
            ax.add_patch(Rectangle((a[0] - sx / 2, a[1] - sy / 2), sx, sy,
                                   color=col, alpha=0.85, zorder=5))

    rows = trial_rows(name)
    n_rev, n_hit, clr = 0, 0, []
    for r in rows:
        bag = f"{ROOT}/evaluation/bags/archive_arena_{name}/gmpc_cbf__{r['run']}"
        if not os.path.isdir(bag):
            continue
        try:
            p = trajectory(bag)
        except Exception:
            continue
        if len(p) < 20:
            continue
        hit = str(r.get('collided', '')).lower() == 'true'
        n_hit += hit
        clr.append(float(r['min_clearance_m']))
        ax.plot(p[:, 0], p[:, 1], '-', lw=1.3, zorder=6,
                color='#c62828' if hit else '#2e7d32',
                alpha=0.85 if hit else 0.55)
        for q in reversal_points(p):
            ax.plot(*q, 'o', ms=9, mfc='none', mec='#c62828', mew=1.5, zorder=7)
            n_rev += 1
    ax.plot(*START, '^', color='#1565c0', ms=11, zorder=8)
    ax.plot(*GOAL, '*', color='#2e7d32', ms=16, zorder=8)
    ax.set_xlim(-1.4, 7.8); ax.set_ylim(-1.4, 7.8); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    ok = sum(str(r.get('success', '')).lower() == 'true' for r in rows)
    t = np.mean([float(r['arrival_time_s']) for r in rows]) if rows else float('nan')
    sub = (f'到達 {ok}/{len(rows)}   碰撞 {n_hit}   折返 {n_rev/max(len(rows),1):.1f}/趟\n'
           f'淨距中位 {np.median(clr):+.3f}   {t:.0f} s' if rows else '（無資料）')
    ax.set_title(f'{name}\n{sub}', fontsize=9.5, fontweight='bold')


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else \
        f'{ROOT}/evaluation/results/figs/arena20_traj.png'
    img = np.array(Image.open(f'{SHARE}/maps/arena.pgm'))
    occ = (255 - img) / 255.0 > 0.65
    h, w = occ.shape
    extent = [ORIGIN[0], ORIGIN[0] + w * RES, ORIGIN[1], ORIGIN[1] + h * RES]
    geo = mover_geometry()
    have = [n for n in ORDER
            if os.path.isfile(f'{ROOT}/evaluation/results/final_arena_{n}.csv')]
    if not have:
        print('no scenario results yet')
        return 1
    cols = 5
    rows_n = int(np.ceil(len(have) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(3.7 * cols, 4.0 * rows_n))
    axes = np.atleast_1d(axes).flatten()
    for ax in axes:
        ax.axis('off')
    for i, name in enumerate(have):
        axes[i].axis('on')
        draw(axes[i], name, occ, extent, geo)
    fig.suptitle('9×9 m 非結構化競技場：plan3 逐場景軌跡（每格 10 趟）\n'
                 '綠 = 無接觸，紅 = 有接觸，紅圈 = >90° 折返；'
                 '黃 = 未知靜態柱，深灰/紅/紫 = 圓柱/方形/L 型移動障礙',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95 if rows_n > 2 else 0.90])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print(f'saved -> {out}   ({len(have)} scenarios)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
