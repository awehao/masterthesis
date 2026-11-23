#!/usr/bin/env python3
"""受控碰觸（S0c）幾何：TCP 姿態固定、由起點垂直下降，逐 1 mm 解 IK，算各連桿對桌面距離，找第一接觸部件與下降量（連桿點雲名目值，非物理凸包）。
    python3 evaluation/bottle_probe_geom.py
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
import bottle_s0_ctrl as C                              # noqa: E402
import bottle_s1_geom as G                              # noqa: E402

BASE = np.array([-0.10, 0.0, 0.0])
TOP = [b for b in G.table_boxes() if b[0] == 'table_top'][0]


def main():
    Tg = C.grasp_target(0.0, 0.42, 0.70)
    Tp = Tg.copy()
    Tp[:3, 3] -= 0.06 * Tg[:3, 2]
    seed = RS.ik(BASE, Tp, RS.SEEDS)[2]
    res = []
    for xs, ys, zs in ((0.18, 0.46, 0.70), (0.18, 0.46, 0.68), (0.20, 0.44, 0.70), (0.14, 0.50, 0.70)):
        Ts = BR.grasp_pose(xs, ys, zs)
        g = RS.ik(BASE, Ts, [seed])
        if not g[0]:
            print('起點不可達', xs, ys, zs)
            continue
        q = g[2]
        first, hist = None, []
        for mm in range(0, 121):
            Td = Ts.copy()
            Td[2, 3] -= mm * 1e-3
            h = RS.ik(BASE, Td, [q], iters=60)
            if not h[0]:
                hist.append((mm, 'ik_fail'))
                break
            q = h[2]
            W = G.link_world(np.r_[BASE, q], C.FINGER_OPEN)
            d = {n: float(G.box_dist(P, TOP[1], TOP[2]).min()) for n, P in W.items()}
            n0 = min(d, key=d.get)
            hist.append((mm, n0, round(d[n0] * 1e3, 1)))
            if d[n0] <= 0.0 and first is None:
                first = dict(descent_mm=mm, link=n0, dist=d, q=[float(v) for v in q])
                break
        res.append(dict(start=[xs, ys, zs], first=first, tail=hist[-3:]))
        print(f'起點 ({xs}, {ys}, {zs})：', ('第一接觸 ' + first['link'] + f" 下降 {first['descent_mm']} mm；此時次近 " + str(sorted(((round(v * 1e3, 1), k) for k, v in first['dist'].items()))[:3])) if first else f'未接觸 {hist[-1]}')
    json.dump(res, open(os.path.join(HERE, 'results', 'bottle', 'probe_geom.json'), 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    sys.exit(main())
