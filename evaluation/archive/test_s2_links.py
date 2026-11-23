#!/usr/bin/env python3
"""link3–5 擴展驗收：任意連桿 Jacobian 對差分、包圍盒涵蓋來源點雲、同場景離線一次並分連桿記錄。假受控對象，非物理。
    python3 evaluation/test_s2_links.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bottle_s1_cand as SC                             # noqa: E402
import bottle_s1_ctrl as S1                             # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import bottle_s1_groups_check as CK                     # noqa: E402
import bottle_s2_check as S2                            # noqa: E402
import test_bottle_s0b as T                             # noqa: E402
from ammr_wholebody_mpc import gmpc_wb as GW            # noqa: E402

K, LINKS = T.K, ('link3', 'link4', 'link5')
rng = np.random.default_rng(9)
worst = 0.0
for n in LINKS:
    fc = GW.FastChain.of(K, n)
    for _ in range(20):
        q = np.r_[rng.uniform(-1, 1, 2), rng.uniform(-3, 3), rng.uniform(-2.5, 2.5, 6)]
        off = rng.uniform(-0.1, 0.1, 3)
        Tc, J = fc.fk_jac(q)
        assert np.abs(Tc - K.fk(q, n)).max() < 1e-12
        p, Jp = GW.point_jac(Tc, J, off)
        Jn = np.zeros((3, 9))
        for i in range(9):
            d = np.zeros(9)
            d[i] = 1e-6
            Jn[:, i] = (GW.point_jac(*fc.fk_jac(q + d), off)[0] - GW.point_jac(*fc.fk_jac(q - d), off)[0]) / 2e-6
        worst = max(worst, float(np.abs(Jp - Jn).max()))
assert worst < 1e-7, worst
print(f'ok 任意連桿 FK 同共用函式、Jacobian 對差分誤差峰 {worst:.1e}')
for n in LINKS:
    A = np.array(S1.link_aabb(n))
    P = G.C.CL[n]
    assert (P >= A.min(axis=0) - 1e-12).all() and (P <= A.max(axis=0) + 1e-12).all()
    print(f'ok {n} 包圍盒涵蓋 {len(P)} 點，尺寸 {np.round(A.max(axis=0) - A.min(axis=0), 3).tolist()} m')

c = SC.make_lidar(body_boxes=[S2.S2_BOX], body_links=LINKS)
f = CK.fake(c, 50.0)
r = CK.summary('S2＋link3–5', c, f)
g = S2.box_min(f, S2.S2_BOX)
per = {}
for h in f.hist[::5]:
    for n, P in G.link_world(h['q'], 0.0038).items():
        per[n] = min(per.get(n, 9.9), float(G.box_dist(P, *S2.S2_BOX).min()))
print('  對方塊最小距離 mm', {k: round(v * 1e3, 1) for k, v in sorted(per.items(), key=lambda x: x[1])})
print('  總', g, '列峰', r['body_rows_max'], '分本體 ticks／列峰', r['body_rows_by'], '分隔面', r['body_planes'], r['qp_counts'], 'GI 未採用', r['gi_fail'])
