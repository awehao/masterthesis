#!/usr/bin/env python3
"""視覺 V1 選相機（Codex 20261101_045309）：沿既有實跑軌跡（s1_steps.npz 的量測底盤、關節、瓶位姿），逐步算兩台相機看瓶子的幾何可見度。純幾何、非渲染。
相機外參取 URDF（omni_bot_manip.urdf）：底盤相機 base_link→(0.25,0,0.3)…color_optical；手腕相機 link_eef→camera_link…color_optical（光學座標 z 前、x 右、y 下）。
內參：底盤 hfov 1.518 rad 640×480、深度 0.3–8 m（URDF sensor）；手腕 hfov 69° 640×480、fx 465.6（門任務 CP1 讀回）、裁切 0.05–10 m。
可見＝瓶軸取樣點在影像內、深度在範圍內、相機到點的線段不穿過第一桌桌面、桌腳、方塊、第二桌（軸對齊盒近似，瓶身自遮不算）。
    python3 evaluation/bottle_v1_cam_visibility.py bottle_p1_3
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_s1_cand as SC                             # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import test_bottle_place as TP                          # noqa: E402

PHASES = ('CRUISE', 'PREGRASP', 'INSERT', 'CLOSE', 'LIFT', 'CARRY')


def rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])


def tf(xyz, r=(0.0, 0.0, 0.0)):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = rpy(*r), xyz
    return T


OPT = tf((0, 0, 0), (-math.pi / 2, 0, -math.pi / 2))
CAMS = dict(base=dict(parent='base_link', T=tf((0.25, 0, 0.3)) @ tf((0.0106, 0.0175, 0.0125)) @ tf((0, 0.015, 0)) @ OPT, hfov=1.518, near=0.3, far=8.0),
            wrist=dict(parent='link_eef', T=tf((0.06746, -0.0175, 0.0237), (math.pi, -math.pi / 2, 0)) @ tf((0, 0.015, 0)) @ OPT, hfov=math.radians(69.0), near=0.05, far=10.0))
W_, H_ = 640, 480


def seg_hits_box(p0, p1, c, sz):
    d = p1 - p0
    lo, hi = c - sz / 2, c + sz / 2
    t0, t1 = 0.0, 1.0
    for k in range(3):
        if abs(d[k]) < 1e-12:
            if p0[k] < lo[k] or p0[k] > hi[k]:
                return False
            continue
        a, b = (lo[k] - p0[k]) / d[k], (hi[k] - p0[k]) / d[k]
        a, b = min(a, b), max(a, b)
        t0, t1 = max(t0, a), min(t1, b)
        if t0 > t1:
            return False
    return t1 > 1e-6 and t0 < 1.0 - 1e-6


def main():
    run = sys.argv[1]
    Z = np.load(os.path.join(HERE, 'runs', run, 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    names = [str(x) for x in Z['phases']]
    boxes = [(cc, sz) for _, cc, sz in G.table_boxes()] + [(SC.OBST['c'], SC.OBST['sz'])] + [(cc, sz) for _, cc, sz in TP.table2_boxes()]
    K = G.C.K
    res = {}
    for name, cam in CAMS.items():
        fx = (W_ / 2) / math.tan(cam['hfov'] / 2)
        per = {}
        for i in range(0, len(c['t']), 10):
            ph = names[int(c['phase_index'][i])]
            if ph not in PHASES[:4]:
                continue
            q = np.array([c[k][i] for k in ('base_x', 'base_y', 'base_yaw', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6')])
            Tc = K.fk(q, cam['parent']) @ cam['T']
            w, x, y, z = (c[k][i] for k in ('bottle_qw', 'bottle_qx', 'bottle_qy', 'bottle_qz'))
            ax = np.array([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)])
            b0 = np.array([c['bottle_base_x'][i], c['bottle_base_y'][i], c['bottle_base_z'][i]])
            pts = b0 + np.outer(np.linspace(0.01, G.BOTTLE['h'] - 0.01, 10), ax)
            Ti = np.linalg.inv(Tc)
            pc = (Ti[:3, :3] @ pts.T).T + Ti[:3, 3]
            vis = []
            for pw, pcam in zip(pts, pc):
                if not (cam['near'] <= pcam[2] <= cam['far']):
                    vis.append(False)
                    continue
                u, v = fx * pcam[0] / pcam[2] + W_ / 2, fx * pcam[1] / pcam[2] + H_ / 2
                if not (0 <= u < W_ and 0 <= v < H_):
                    vis.append(False)
                    continue
                vis.append(not any(seg_hits_box(Tc[:3, 3], pw, bc, bs) for bc, bs in boxes))
            frac = float(np.mean(vis))
            depth = float(np.linalg.norm(Tc[:3, 3] - (b0 + 0.5 * G.BOTTLE['h'] * ax)))
            key = 'CRUISE_nav' if ph == 'CRUISE' and c['base_x'][i] < -1.55 else ('CRUISE_unfold' if ph == 'CRUISE' else ph)
            e = per.setdefault(key, dict(n=0, seen=0, frac=[], px=[], depth=[]))
            e['n'] += 1
            e['seen'] += int(frac >= 0.5)
            e['frac'].append(frac)
            e['depth'].append(depth)
            e['px'].append(2 * G.BOTTLE['r'] * fx / max(depth, 1e-6))
        res[name] = {k: dict(n=v['n'], seen_ratio=round(v['seen'] / v['n'], 3), axis_vis_mean=round(float(np.mean(v['frac'])), 3),
                             depth=[round(min(v['depth']), 2), round(max(v['depth']), 2)], width_px=[round(min(v['px']), 1), round(max(v['px']), 1)]) for k, v in per.items()}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    json.dump(dict(run=run, cams={k: {kk: (vv if not isinstance(vv, np.ndarray) else vv.tolist()) for kk, vv in v.items() if kk != 'T'} for k, v in CAMS.items()}, result=res,
                   note='純幾何可見度（軸取樣、盒遮擋近似），非渲染'), open(os.path.join(HERE, 'results', 'bottle', 'v1_cam_visibility.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    return 0


if __name__ == '__main__':
    sys.exit(main())
