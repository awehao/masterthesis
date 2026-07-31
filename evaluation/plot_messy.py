"""Draw the messy scenario's layout: what the robot is being asked to cross.

A trajectories YAML is a list of coordinates, which says nothing about whether
the bodies actually sit on the route, whether the big ones fit, or how the
encounters are spaced. This draws all of it against the real map, at each
mover's true footprint rather than a nominal circle.

    python3 evaluation/plot_messy.py [out.png]
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
ORIGIN, RES = (-1.3, -1.3), 0.05
UNKNOWN_STATIC = [(1.75, 4.90, 0.30), (5.00, 5.30, 0.30)]
START, GOAL = (0.0, 0.0), (6.6, 6.6)


def footprints():
    """Every mover's collision boxes/cylinders, in its own body frame."""
    sdf = open(f'{SHARE}/worlds/arena.sdf').read()
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


def route():
    """The recorded unobstructed run the placements were computed against."""
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from nav_msgs.msg import Odometry
    except Exception:
        return None
    b = f'{ROOT}/evaluation/bags/archive_arena_none/gmpc_cbf__scan_seed1'
    if not os.path.isdir(b):
        return None
    sr = rosbag2_py.SequentialReader()
    sr.open(rosbag2_py.StorageOptions(uri=b, storage_id='mcap'),
            rosbag2_py.ConverterOptions('', ''))
    P = []
    while sr.has_next():
        t, d, _ = sr.read_next()
        if t == '/odom':
            m = deserialize_message(d, Odometry)
            P.append((m.pose.pose.position.x, m.pose.pose.position.y))
    return np.array(P) if P else None


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else \
        f'{ROOT}/evaluation/results/figs/arena_messy.png'
    obs = yaml.safe_load(
        open(f'{SHARE}/config/dynamic_trajectories_arena_messy.yaml')
    ).get('dynamic_obstacles') or []
    fp = footprints()

    img = np.array(Image.open(f'{SHARE}/maps/arena.pgm'))
    occ = (255 - img) / 255.0 > 0.65
    h, w = occ.shape
    extent = [ORIGIN[0], ORIGIN[0] + w * RES, ORIGIN[1], ORIGIN[1] + h * RES]

    fig, ax = plt.subplots(figsize=(9.5, 9.5))
    ax.imshow(occ, cmap='Greys', origin='upper', extent=extent, alpha=0.9)

    P = route()
    if P is not None:
        ax.plot(P[:, 0], P[:, 1], '--', color='#1565c0', lw=1.6, alpha=0.7,
                zorder=4, label='無障礙時錄到的行駛路徑')

    for x, y, r in UNKNOWN_STATIC:
        ax.add_patch(Circle((x, y), r, color='#f9a825', alpha=0.95, zorder=5))
    ax.plot([], [], 'o', color='#f9a825', ms=10, label='未知靜態柱（地圖上沒有）')

    cols = ['#c62828', '#6a1b9a', '#00838f', '#2e7d32',
            '#ef6c00', '#4527a0', '#ad1457']
    for i, o in enumerate(obs):
        a, b = np.array(o['start'], float), np.array(o['end'], float)
        col = cols[i % len(cols)]
        ax.plot([a[0], b[0]], [a[1], b[1]], '-', color=col, lw=2.4,
                alpha=0.75, zorder=6)
        ax.plot(*b, 'x', color=col, ms=8, mew=2, zorder=7)
        for kind, px, py, sx, sy in fp.get(o['name'], []):
            if kind == 'box':
                ax.add_patch(Rectangle((a[0] + px - sx / 2, a[1] + py - sy / 2),
                                       sx, sy, color=col, alpha=0.9, zorder=7))
            else:
                ax.add_patch(Circle((a[0] + px, a[1] + py), sx,
                                    color=col, alpha=0.9, zorder=7))
        L = float(np.linalg.norm(b - a))
        ax.annotate(f"{o['name'][-1]}  {o['speed']} m/s",
                    xy=(a + b) / 2, fontsize=8.5, color=col,
                    fontweight='bold', zorder=9,
                    xytext=(6, 6), textcoords='offset points')
        ax.plot([], [], '-', color=col, lw=2.4,
                label=f"{o['name']}  v={o['speed']}  巡邏 {L:.1f} m  "
                      f"週期 {2*L/max(o['speed'],1e-6):.0f} s")

    ax.plot(*START, '^', color='#1565c0', ms=15, zorder=10)
    ax.annotate('起點', START, xytext=(10, -18), textcoords='offset points',
                fontsize=11, fontweight='bold', color='#1565c0')
    ax.plot(*GOAL, '*', color='#2e7d32', ms=24, zorder=10)
    ax.annotate('終點', GOAL, xytext=(-38, 10), textcoords='offset points',
                fontsize=11, fontweight='bold', color='#2e7d32')

    ax.set_xlim(-1.4, 7.8)
    ax.set_ylim(-1.4, 7.8)
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('arena_messy：七個移動障礙同時上場\n'
                 '灰 = 已知牆（在地圖上）  黃 = 未知靜態柱  '
                 '彩色 = 移動障礙的真實外形與巡邏線（× 為另一端）',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8, framealpha=0.92)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f'saved -> {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
