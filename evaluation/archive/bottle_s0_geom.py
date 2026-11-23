#!/usr/bin/env python3
"""S0a 抓取幾何核對（無 Isaac）：真實手指網格點雲對瓶身。TCP 越過瓶軸 GRASP_DEPTH、手指全開（INSERT）與閉到 30 mm（CLOSE）時，
報告指片、手指根部、夾爪殼、其他連桿對瓶身的最小距離，並掃描抓取深度 0–8 mm。距離為名目構型（無 TCP 誤差、無接觸變形）。
    python3 evaluation/bottle_s0_geom.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_reach_scan as RS                          # noqa: E402
import bottle_ref as BR                                 # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import door_offline_check_v1 as C                       # noqa: E402

BASE = np.array([-0.10, 0.0, 0.0])
TOP, H, BX, BY = 0.60, 0.70, 0.0, 0.42
R = G.BOTTLE['r']


def dists(q, q_f):
    W = G.link_world(q, q_f)
    out = {}
    for n, P in W.items():
        d = G.cyl_dist(P, BX, BY, TOP, TOP + G.BOTTLE['h'], R)
        if n in C.FINGERS:
            z_f = C.CL[n][:, 2]                         # 手指座標 z：≤ 8 mm 根部、> 8 mm 指片（finger_collision.Z_SPLIT_M）
            out[f'{n}:root'] = float(d[z_f <= 0.008].min())
            out[f'{n}:pad'] = float(d[z_f > 0.008].min())
        else:
            out[n] = float(d.min())
    return out


def main():
    res = {}
    for depth in (0.0, 0.002, 0.004, 0.006, 0.008):
        T = BR.grasp_pose(BX, BY, H)
        T[:3, 3] += depth * T[:3, 2]
        g = RS.ik(BASE, T, RS.SEEDS)
        q = np.r_[BASE, g[2]]
        do, dc = dists(q, 0.0089), dists(q, 0.0038)
        nf = min(v for k, v in dc.items() if 'finger' not in k)
        res[f'{depth * 1e3:.0f}mm'] = dict(ik_ok=bool(g[0]), open=do, closed=dc, nonfinger_min=nf)
        print(f"深度 {depth * 1e3:.0f} mm：全開 指片 {min(do['uflite_finger1:pad'], do['uflite_finger2:pad']) * 1e3:.2f}、根部 {min(do['uflite_finger1:root'], do['uflite_finger2:root']) * 1e3:.2f} mm；"
              f"閉到 30 mm 指片 {min(dc['uflite_finger1:pad'], dc['uflite_finger2:pad']) * 1e3:.2f}、根部 {min(dc['uflite_finger1:root'], dc['uflite_finger2:root']) * 1e3:.2f} mm；非手指最近 {nf * 1e3:.2f} mm")
    out = os.path.join(HERE, 'results', 'bottle', 's0_geom.json')
    json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    sys.exit(main())
