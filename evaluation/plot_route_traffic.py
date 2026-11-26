#!/usr/bin/env python3
"""畫路途穿越交通情境俯視圖：plot_route_traffic.py <情境 yaml> <out.png>（同名 _meta.json 需在旁邊）。"""
import json
import sys

import matplotlib
import numpy as np
import yaml

matplotlib.use('Agg')
import matplotlib.pyplot as plt                                     # noqa: E402

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'Noto Sans CJK JP', 'WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
Y = yaml.safe_load(open(sys.argv[1])); m = json.load(open(sys.argv[1][:-5] + '_meta.json'))
S = 'install/ammr_bringup/share/ammr_bringup/maps/bigarena'
my = yaml.safe_load(open(S + '.yaml')); res = my['resolution']; ox, oy = my['origin'][:2]
f = open(S + '.pgm', 'rb'); f.readline(); ln = f.readline()
while ln.startswith(b'#'):
    ln = f.readline()
w, h = map(int, ln.split()); f.readline(); img = np.frombuffer(f.read(), np.uint8).reshape(h, w)
fig, ax = plt.subplots(figsize=(7, 7)); ax.imshow(img, cmap='gray', extent=[ox, ox + w * res, oy, oy + h * res])
P = np.array(m['path']); ax.plot(P[:, 0], P[:, 1], '-', color='#2b6cb0', lw=2, label='機器人名目路徑（地圖 A*）')
z = m['params']['zone']
for c, lab in ((m['start'], '起點'), (m['goal'], '終點')):
    ax.add_patch(plt.Circle(c, z, fill=False, ls='--', color='#c05621')); ax.plot(*c, 'o', color='#c05621'); ax.text(c[0] + 0.3, c[1] + 0.3, lab, color='#c05621')
info = {o['name']: o for o in m['obstacles']}
for o in Y['dynamic_obstacles']:
    i = info[o['name']]; col = '#2f855a' if i['role'] == 'cross' else '#8a8f98'
    ax.plot([o['start'][0], o['end'][0]], [o['start'][1], o['end'][1]], '-', color=col, lw=2.5)
    if i['role'] == 'cross':
        ax.plot(*i['cross_xy'], 'x', color='#22543d', ms=8)
        ax.text(i['cross_xy'][0] + 0.2, i['cross_xy'][1] - 0.55, '%s・%.0f s' % (i['kind'], i['t_pass_s']), fontsize=7, color='#22543d')
ax.plot([], [], '-', color='#2f855a', lw=2.5, label='穿越行人來回路段（×＝穿越點、時刻＝預定通過）')
ax.plot([], [], '-', color='#8a8f98', lw=2.5, label='背景行人')
ax.plot([], [], '--', color='#c05621', label='起終點 %.1f m 內不放穿越點' % z)
ax.legend(loc='upper left', fontsize=7, framealpha=0.9)
ax.set_title('路線 %d・%s・%s・種子 %d（路徑 %.1f m）' % (m['route'], m['speed'], m['pattern'], m['seed'], m['path_len_m']))
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=120)
