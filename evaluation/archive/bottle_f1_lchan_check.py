#!/usr/bin/env python3
"""L 型走廊場景的離線幾何核對（只讀 F1_WALLS 離線規劃結果與 bottle_f1_3 紀錄，不跑模擬）：路徑到牆與桌腳淨空、走廊入口／出口弧長對收近時序、全身包絡，並畫平面圖。
    python3 evaluation/bottle_f1_lchan_check.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.argv = ['test_bottle_f1_task.py', 'nominal']
import test_bottle_f1_task as T                         # noqa: E402,F401
import bottle_s1_geom as G                              # noqa: E402
from scipy.spatial.transform import Rotation as Rot     # noqa: E402

BPT = G.bottle_points()
import bottle_f1_lchan_scene as LS                      # noqa: E402
SC = json.load(open(LS.PATH, encoding='utf-8'))
SW = LS.widths(SC['boxes'])
assert abs(SW['ns'] - 1.2) < 1e-6 and abs(SW['ew'] - 1.2) < 1e-6, SW
DX = float(SC.get('table2_dx', 0.0))
PL = json.load(open(os.path.join(HERE, 'results', 'bottle', SC['plan_file'])))
WALLS = {k: (np.array(v[:3], float), np.array(v[3:], float)) for k, v in SC['boxes'].items()}
T2 = np.array([2.4 + DX, -1.0, 0.0])
LEGS = {'t1_' + n: (c, sz) for n, c, sz in G.table_boxes() if n != 'table_top'}
LEGS.update({'t2_' + n: (c + T2, sz) for n, c, sz in G.table_boxes() if n != 'table_top'})
BOX = {'box': (np.array([-2.0, 0.0, 0.25]), np.array([0.4, 0.4, 0.5]))}


def d_box(P, c, sz):
    return np.linalg.norm(np.maximum(np.abs(P[:, :2] - c[:2]) - sz[:2] / 2.0, 0.0), axis=1)


def arc(P):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1))]


def path_report(P):
    s = arc(P)
    dw = {k: float(d_box(P, *v).min()) for k, v in WALLS.items()}
    dl = min(float(d_box(P, *v).min()) for v in LEGS.values())
    in_v = (P[:, 0] > -0.30) & (P[:, 0] < 0.90) & (P[:, 1] < -1.35) & (P[:, 1] > -2.90)
    in_h = (P[:, 0] >= 0.90) & (P[:, 0] < 1.45) & (P[:, 1] < -1.70) & (P[:, 1] > -2.90)
    inside = in_v | in_h
    k = np.flatnonzero(inside)
    i = int(np.searchsorted(s, s[-1] - 0.3))
    d_ = P[-1, :2] - P[i, :2]
    return dict(end_dir_deg=round(float(np.degrees(np.arctan2(d_[1], d_[0]))), 1), L=round(float(s[-1]), 3), wall_min={a: round(b, 3) for a, b in dw.items()}, leg_min=round(dl, 3), box_min=round(float(d_box(P, *BOX['box']).min()), 3), corridor=None if not len(k) else dict(s_in=round(float(s[k[0]]), 3), s_out=round(float(s[k[-1]]), 3), rem_after=round(float(s[-1] - s[k[-1]]), 3), L_in=round(float(s[k[-1]] - s[k[0]]), 3), y_min=round(float(P[:, 1].min()), 3), x_max=round(float(P[:, 0].max()), 3)))


out = dict(widths=SW, scene=SC['name'], paths={n: path_report(np.asarray(PL[n]['P_region'], float)) for n in ('carry_far', 'return_short', 'carry_near')}, compact_arc_measured={}, envelope={})
Z = np.load(os.path.join(HERE, 'runs', 'bottle_f1_3', 's1_steps.npz'), allow_pickle=True)
col = {str(n): Z['rows'][:, i] for i, n in enumerate(Z['cols'])}
t = col['t']
Q = np.column_stack([col[n] for n in ('base_x', 'base_y', 'base_yaw', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6')])


def extent(k):
    ex = []
    for j in k[::10]:
        Pn, Pf = T.split(G.link_world(Q[j], float(col['finger_cmd'][j])))
        W = np.vstack([Pn, Pf])
        c, s = np.cos(Q[j, 2]), np.sin(Q[j, 2])
        d = W[:, :2] - Q[j, :2]
        ex.append(np.column_stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]]))
    E = np.vstack(ex)
    Eb = []
    for j in k[::10]:
        R_ = Rot.from_quat([col['bottle_qx'][j], col['bottle_qy'][j], col['bottle_qz'][j], col['bottle_qw'][j]]).as_matrix()
        B_ = (R_ @ BPT.T).T + np.array([col['bottle_base_x'][j], col['bottle_base_y'][j], col['bottle_base_z'][j]])
        c, s = np.cos(Q[j, 2]), np.sin(Q[j, 2])
        d = B_[:, :2] - Q[j, :2]
        Eb.append(np.column_stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]]))
    Eb = np.vstack(Eb)
    return dict(x=[round(float(E[:, 0].min()), 3), round(float(E[:, 0].max()), 3)], y=[round(float(E[:, 1].min()), 3), round(float(E[:, 1].max()), 3)], r_max=round(float(np.hypot(E[:, 0], E[:, 1]).max()), 3), r_max_bottle=round(float(np.hypot(Eb[:, 0], Eb[:, 1]).max()), 3), z_note='實測姿態取樣：連桿與手指 link_world、持瓶 bottle_points；bottle 欄位對應當趟持瓶')


for nm, fn, st in (('r1', 's1_report_r0.json', 0.0), ('r2', 's1_report.json', None)):
    r = json.load(open(os.path.join(HERE, 'runs', 'bottle_f1_3', fn)))
    ev = {e['what']: e['t'] for e in r['events'] if str(e.get('what', '')).startswith('compact_')}
    tl = [e['t'] for e in r['events'] if e.get('what') == 'carry_plan_request'][0]
    k = np.flatnonzero((t >= tl) & (t <= ev['compact_reached']))
    k = k[np.abs(t[k] * 20 - np.round(t[k] * 20)) < 1e-6]
    out['compact_arc_measured'][nm] = dict(from_plan_request=round(float(np.linalg.norm(np.diff(Q[k, :2], axis=0), axis=1).sum()), 3), t_compact=round(ev['compact_reached'] - tl, 2))
    k2 = np.flatnonzero((t >= ev['compact_reached'] - 0.3) & (t <= ev['compact_raise_start'] + 0.05))
    out['envelope']['compact_' + nm] = extent(k2)
    k3 = np.flatnonzero((t >= ev['compact_raise_start']) & (t <= ev['compact_out_start']))
    out['envelope']['raise_' + nm] = extent(k3)
r0 = json.load(open(os.path.join(HERE, 'runs', 'bottle_f1_3', 's1_report_r0.json')))
evr = {e['to']: e['t'] for e in r0['events'] if e.get('what') == 'return_state'}
kr = np.flatnonzero((t >= evr['NAV']) & (t <= evr['ROT1']))
out['envelope']['return_hold'] = {k_: v_ for k_, v_ in extent(kr).items() if k_ != 'r_max_bottle'}   # 返回段沒有持瓶
out['corridor_center_margin'] = dict(half_width=0.6, base_radius=0.39, disc_gap=0.21, static_keepout_margin=0.47, center_to_keepout=0.13, dynamic_margin=0.69)
print(json.dumps(out, ensure_ascii=False, default=float))
json.dump(out, open(os.path.join(HERE, 'results', 'bottle', 'f1_lchan_check.json'), 'w'), ensure_ascii=False, indent=1, default=float)

import matplotlib                                       # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                         # noqa: E402
from matplotlib.patches import Rectangle               # noqa: E402
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots(figsize=(9, 7.2))
for c, sz in [v for v in WALLS.values()]:
    ax.add_patch(Rectangle((c[0] - sz[0] / 2, c[1] - sz[1] / 2), sz[0], sz[1], color='#3a4150'))
for nm, x0 in (('第一桌', 0.0), ('第二桌', 2.4 + DX)):
    y0 = 0.36 + (0.0 if x0 == 0.0 else -1.0)
    ax.add_patch(Rectangle((x0 - 0.4, y0), 0.8, 0.5, fill=False, ec='#6b7280', lw=1.5))
    ax.text(x0, y0 + 0.25, nm, ha='center', va='center', color='#374151', fontsize=11)
c_, sz_ = BOX['box']
ax.add_patch(Rectangle((c_[0] - sz_[0] / 2, c_[1] - sz_[1] / 2), sz_[0], sz_[1], color='#9ca3af'))
for nm, col_, lab in (('carry_far', '#1f6feb', '第一輪運送（遠瓶）'), ('carry_near', '#0e9f6e', '第二輪運送（近瓶）'), ('return_short', '#d9480f', '短返回')):
    P = np.asarray(PL[nm]['P_region'], float)
    ax.plot(P[:, 0], P[:, 1], color=col_, lw=2, label=f'{lab} {PL[nm]["length_m"]:.2f} m')
ax.plot([0.15, -0.15], [0.42, 0.42], 'o', color='#111827', ms=4)
ax.plot([2.55 + DX, 2.25 + DX], [-0.58, -0.58], 's', color='#111827', ms=4)
ax.set_xlim(-3.2, 3.6)
ax.set_ylim(-3.1, 1.2)
ax.set_aspect('equal')
ax.grid(alpha=0.25)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.legend(loc='upper left', frameon=False)
ax.set_title('F1 L 型走廊場景：真規劃器路徑（內淨寬 1.20 m；離線，未實跑）')
fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(HERE), '第七次進度報告素材', '圖表', 'F1_L型走廊_離線路徑_20261103_v3_淨寬1.20.png'), dpi=150)
