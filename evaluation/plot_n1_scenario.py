#!/usr/bin/env python3
"""畫 N1 情境俯視圖：plot_n1_scenario.py <out.png> <情境 yaml> [<情境 yaml> ...]（同名 _meta.json 需在旁邊；地圖與世界檔取 src/ammr_bringup）。"""
import json
import math
import os
import re
import sys

import matplotlib
import numpy as np
import yaml

matplotlib.use('Agg')
import matplotlib.pyplot as plt                                     # noqa: E402
from matplotlib.patches import Circle, Polygon                      # noqa: E402

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'ammr_bringup')
NAME = dict(n1_bigarena='L1 多房間', n1_office='L2 辦公室走廊', n1_warehouse='L3 倉儲貨架')


def draw(ax, yf):
    m = json.load(open(yf[:-5] + '_meta.json')); Y = yaml.safe_load(open(yf))
    lay = m['layout']
    my = yaml.safe_load(open(os.path.join(B, 'maps', lay + '.yaml'))); f = open(os.path.join(B, 'maps', my['image']), 'rb'); f.readline(); ln = f.readline()
    while ln.startswith(b'#'):
        ln = f.readline()
    w, h = map(int, ln.split()); f.readline(); img = np.frombuffer(f.read(), np.uint8).reshape(h, w); r = my['resolution']; ox, oy = my['origin'][:2]
    ax.imshow(img, cmap='gray', extent=[ox, ox + w * r, oy, oy + h * r], alpha=0.9)
    sdf = open(os.path.join(B, 'worlds', lay + '.sdf')).read()
    size = {mm.group(1): [float(v) for v in re.search(r'<box><size>([\d.]+) ([\d.]+)', mm.group(2)).groups()]
            for mm in re.finditer(r'<model name="(unknown_obs_\d+)">(.*?)</model>', sdf, re.S)}
    for u in Y.get('unknown_static') or []:
        sx, sy = size[u['name']]; c, s = math.cos(u['yaw']), math.sin(u['yaw'])
        pts = [(u['x'] + c * dx - s * dy, u['y'] + s * dx + c * dy) for dx, dy in ((-sx / 2, -sy / 2), (sx / 2, -sy / 2), (sx / 2, sy / 2), (-sx / 2, sy / 2))]
        ax.add_patch(Polygon(pts, color='#c53030'))
    role = {o['name']: o['role'] for o in m.get('movers', [])}
    for d in Y['dynamic_obstacles']:
        if d['speed'] <= 0:
            continue
        col = '#2f855a' if role.get(d['name'], '').startswith('cross') else '#8a8f98'
        ax.annotate('', xy=d['end'], xytext=d['start'], arrowprops=dict(arrowstyle='<->', color=col, lw=1.7))
        ax.text((d['start'][0] + d['end'][0]) / 2, (d['start'][1] + d['end'][1]) / 2 + 0.3, f"{d['speed']:.2f}", color=col, fontsize=7.5, ha='center')
    P = np.array(m['path']); ax.plot(P[:, 0], P[:, 1], '--', color='#2b6cb0', lw=2)
    ax.plot(*m['start'], 'o', color='#2b6cb0', ms=9); ax.plot(*m['goal'], '*', color='#c05621', ms=15)
    for c in (m['start'], m['goal']):
        ax.add_patch(Circle(c, 2.5, fill=False, ls=':', color='#c05621'))
    ax.set_xlim(ox, ox + w * r); ax.set_ylim(oy, oy + h * r)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{NAME.get(lay, lay)}・{'交會' if m['cond'] == 'crossing' else '靜態'}・種子 {m['seed']}（名目路長 {m['path_len_m']} m）", fontsize=10.5, loc='left')


def main():
    out, files = sys.argv[1], sys.argv[2:]
    fig, ax = plt.subplots(1, len(files), figsize=(6 * len(files), 6.9), squeeze=False)
    for a, yf in zip(ax[0], files):
        draw(a, yf)
    fig.text(0.01, 0.015, '黑＝地圖上已知　紅＝未知靜態障礙（不在地圖）　綠＝穿越路線中段的行人　灰＝背景行人（數字＝m/s，0.10–0.25）　藍虛線＝名目路線　點線圈＝起終點 2.5 m 內不放行人',
             fontsize=9.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97)); fig.savefig(out, dpi=100)


if __name__ == '__main__':
    main()
