"""Instantaneous KF speed against net-displacement speed, per track per cycle.

The net-displacement gate exists because instantaneous KF speed cannot separate
a stationary object from a mover: occlusion and centroid slide inflate a static
body's apparent velocity. Plotting both axes at once tests that claim rather
than asserting it -- if the gate works, the static cloud spreads along x but
stays pinned near zero on y.

Labelled from Gazebo ground truth, which the controller never sees, and ONLY
where the label is unambiguous: a track within MATCH_M of a mover is a mover,
one sitting on a known unknown_obs is static, and everything else is dropped.
Counting "not a mover" as static would sweep in map-subtraction leaks off
walls, which jitter by construction and would make both gates look worse than
they are.

    python3 evaluation/plot_static_dynamic.py [out.png]
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

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc')
plt.rcParams.update({'font.family': ['Noto Sans CJK JP'],
                     'axes.unicode_minus': False})

import rosbag2_py
from rclpy.serialization import deserialize_message
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray

ROOT = '/home/howardchen/masterthesis'
SHARE = f'{ROOT}/install/ammr_bringup/share/ammr_bringup'
TRK = ['tid', 'x', 'y', 'vx', 'vy', 'age', 'misses', 'n_frag',
       'coast_s', 'innov', 'confirmed', 'reject']
WINDOW = 2.0          # s, the net-displacement window the tracker uses
V_INST, V_NET = 0.10, 0.05
MATCH_M = 0.9         # m, track-to-truth association for LABELLING only
N_BAGS = 12


def unknown_statics():
    sdf = open(f'{SHARE}/worlds/bigarena.sdf').read()
    out = []
    for m in re.finditer(r'<model name="unknown_obs_?\d*">(.*?)</model>', sdf, re.S):
        p = re.search(r'<pose>([-\d.]+) ([-\d.]+)', m.group(1))
        if p:
            out.append((float(p.group(1)), float(p.group(2))))
    return out


def one_bag(path, statics):
    topics = {'/gmpc/tracks_debug': Float32MultiArray}
    for i in range(10):
        topics[f'/model/dyn_obs_{i}/pose'] = PoseStamped
    rd = rosbag2_py.SequentialReader()
    try:
        rd.open(rosbag2_py.StorageOptions(uri=path, storage_id='mcap'),
                rosbag2_py.ConverterOptions('', ''))
    except Exception:
        return []
    have = {t.name for t in rd.get_all_topics_and_types()}
    if '/gmpc/tracks_debug' not in have:
        return []
    rd.set_filter(rosbag2_py.StorageFilter(
        topics=[k for k in topics if k in have]))
    trk, mv = [], {i: [] for i in range(10)}
    while rd.has_next():
        tp, buf, t = rd.read_next()
        ts = t * 1e-9
        if tp == '/gmpc/tracks_debug':
            d = list(deserialize_message(buf, Float32MultiArray).data)
            if len(d) >= len(TRK):
                trk.append((ts, np.array(d).reshape(-1, len(TRK))))
        else:
            i = int(tp.split('dyn_obs_')[1].split('/')[0])
            m = deserialize_message(buf, PoseStamped)
            mv[i].append((ts, m.pose.position.x, m.pose.position.y))

    hist = {}          # tid -> [(t, x, y)]
    rows = []
    for ts, arr in trk:
        for r in arr:
            tid = int(r[0])
            hist.setdefault(tid, []).append((ts, r[1], r[2]))
            h = hist[tid]
            # oldest sample still inside the window
            t0 = x0 = y0 = None
            for (tt, xx, yy) in h:
                if ts - tt <= WINDOW:
                    t0, x0, y0 = tt, xx, yy
                    break
            if t0 is None or ts - t0 < 0.3:
                continue                      # not enough history to judge
            v_net = math.hypot(r[1] - x0, r[2] - y0) / (ts - t0)
            v_inst = math.hypot(r[3], r[4])

            # Label from ground truth, and only where ground truth is
            # unambiguous. "Everything that is not a mover" would sweep in
            # map-subtraction leaks off walls, which jitter by construction and
            # would flatter neither gate fairly. A track counts as static only
            # if it sits on one of the seven KNOWN unknown_obs.
            lab = None
            best = MATCH_M
            for i in range(10):
                s = mv[i]
                if not s:
                    continue
                k = int(np.clip(np.searchsorted([a[0] for a in s], ts) - 1,
                                0, len(s) - 1))
                d = math.hypot(r[1] - s[k][1], r[2] - s[k][2])
                if d < best:
                    best, lab = d, 'mover'
            if lab is None:
                for (sx, sy) in statics:
                    if math.hypot(r[1] - sx, r[2] - sy) < MATCH_M:
                        lab = 'static'
                        break
            if lab is not None:
                rows.append((v_inst, v_net, lab))
    return rows


statics = unknown_statics()
rows = []
for bag in sorted(glob.glob(f'{ROOT}/evaluation/bags/archive_shield100/'
                            'gmpc_cbf__scan_seed*'))[:N_BAGS]:
    rows += one_bag(bag, statics)
if not rows:
    print('no data'); sys.exit(1)

vi = np.array([r[0] for r in rows])
vn = np.array([r[1] for r in rows])
is_m = np.array([r[2] == 'mover' for r in rows])
print(f'  samples {len(rows)}  movers {is_m.sum()}  static {(~is_m).sum()}')

fig, ax = plt.subplots(figsize=(9.0, 7.0))
ax.scatter(vi[~is_m], vn[~is_m], s=5, alpha=0.25, c='#8c8c8c',
           label=f'靜態物體（真值）  n={(~is_m).sum()}', rasterized=True)
ax.scatter(vi[is_m], vn[is_m], s=6, alpha=0.35, c='#ed7d31',
           label=f'移動體（真值）  n={is_m.sum()}', rasterized=True)

ax.axvline(V_INST, color='#c00000', lw=1.8, ls='--')
ax.axhline(V_NET, color='#2f75b5', lw=1.8, ls='--')

# how much each gate alone would misclassify
fp_inst = int((~is_m & (vi >= V_INST)).sum())
fp_both = int((~is_m & (vi >= V_INST) & (vn >= V_NET)).sum())
n_s = max(int((~is_m).sum()), 1)
ax.set_xlabel('瞬時 KF 速度  $\\|\\hat v_i\\|$   [m/s]')
ax.set_ylabel('滑動窗淨位移速度  $\\bar v_{net,i}$   [m/s]')
ax.set_title('靜態／動態分流：兩道門檻各自能分開多少',
             fontsize=13, fontweight='bold')
ax.set_xlim(0, min(0.6, float(np.percentile(vi, 99.5))))
ax.set_ylim(0, min(0.5, float(np.percentile(vn, 99.5))))
ax.grid(alpha=0.25)
# after the limits exist, so the labels land inside the axes
ax.text(V_INST + 0.004, ax.get_ylim()[1] * 0.97, f'瞬時門檻 {V_INST}',
        color='#c00000', fontsize=11, va='top')
ax.text(ax.get_xlim()[1] * 0.99, V_NET + 0.004, f'淨位移門檻 {V_NET}',
        color='#2f75b5', fontsize=11, ha='right')
ax.legend(loc='upper left', fontsize=10, framealpha=0.95)

txt = (f'靜態物體被誤判為移動體\n'
       f'  僅瞬時門檻：{fp_inst}/{n_s}  ({fp_inst/n_s:.1%})\n'
       f'  加上淨位移：{fp_both}/{n_s}  ({fp_both/n_s:.1%})')
ax.text(0.98, 0.03, txt, transform=ax.transAxes, ha='right', va='bottom',
        fontsize=11, bbox=dict(fc='#eef5fb', ec='#9dc3e6', boxstyle='round,pad=0.5'))

out = sys.argv[1] if len(sys.argv) > 1 else \
    f'{ROOT}/evaluation/results/figs/static_dynamic_gate.png'
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.tight_layout()
plt.savefig(out, dpi=125, bbox_inches='tight')
print(f'saved -> {out}')
print(f'  誤判：僅瞬時 {fp_inst}/{n_s} ({fp_inst/n_s:.1%})  '
      f'加淨位移 {fp_both}/{n_s} ({fp_both/n_s:.1%})')
