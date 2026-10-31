#!/usr/bin/env python3
"""第二桌放置第一包的離線幾何核對（假受控對象名目軌跡，非物理）：
持瓶路線上手臂／瓶對兩張桌與方塊的淨空、停車後各相位的關節餘裕、下降段手臂與手指對第二桌、退離段手臂全連桿對瓶。
未鬆爪時手指↔瓶是指定接觸，不列入；瓶底↔第二桌面自 LOWER 起是指定支撐。只報數，跨線的門檻在設計內，不在這裡改。
    python3 evaluation/bottle_place_geom_check.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_s1_cand as SC                             # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import test_bottle_place as TP                          # noqa: E402
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE    # noqa: E402


def main():
    c, f = TP.build()
    H = []
    TP.run(c, f, hook=lambda t, s: H.append((t, s.q.copy(), s.Tb.copy(), float(c.f_cmd), c.pph, c.st.phase)))
    r = c.report()
    t1 = TP.table2_boxes()
    t0 = [(n, c_, sz) for n, c_, sz in G.table_boxes()]
    BP = G.bottle_points()
    out = {}

    def upd(k, d, t, ph):
        if d < out.get(k, (9.9,))[0]:
            out[k] = (float(d), round(t, 2), ph)
    lo, hi = np.array(LITE6_SAFE.lower), np.array(LITE6_SAFE.upper)
    marg, trace = {}, {}
    for t, q, Tb, fc, pph, ph in H[::2]:
        tag = pph or ph
        W = G.link_world(q, fc)
        nonf = np.vstack([P for n, P in W.items() if n not in G.C.FINGERS])
        fing = np.vstack([P for n, P in W.items() if n in G.C.FINGERS])
        allp = np.vstack(list(W.values()))
        Pb = (Tb[:3, :3] @ BP.T).T + Tb[:3, 3]
        if pph is not None or ph == 'CARRY':
            for n, c_, sz in t1:
                upd(f'nonfinger↔table2_{"top" if n == "table_top" else "leg"}', float(G.box_dist(nonf, c_, sz).min()), t, tag)
                upd(f'finger↔table2_{"top" if n == "table_top" else "leg"}', float(G.box_dist(fing, c_, sz).min()), t, tag)
                if n != 'table_top':
                    upd('bottle↔table2_leg', float(G.box_dist(Pb, c_, sz).min()), t, tag)
            if pph in (None, 'CARRY_NAV', 'PLACE_APPROACH', 'PARKED'):
                upd('bottle↔table2_top（LOWER 前）', float(G.box_dist(Pb, *[x for n, *x in t1 if n == 'table_top'][0]).min()), t, tag)
            for n, c_, sz in t0:
                upd('bottle↔table1（抬離後）', float(G.box_dist(Pb, c_, sz).min()), t, tag)
            upd('nonfinger↔obstacle', float(G.box_dist(nonf, SC.OBST['c'], SC.OBST['sz']).min()), t, tag)
            upd('bottle↔obstacle', float(G.box_dist(Pb, SC.OBST['c'], SC.OBST['sz']).min()), t, tag)
        if pph in ('RETREAT', 'HOLD'):
            for nm, PP in (('nonfinger', nonf), ('finger', fing)):
                Pl = (Tb[:3, :3].T @ (PP - Tb[:3, 3]).T).T
                d = float(G.cyl_dist(Pl, 0.0, 0.0, 0.0, G.BOTTLE['h'], G.BOTTLE['r']).min())
                upd(f'{nm}↔bottle（退離與保持）', d, t, tag)
                if pph == 'RETREAT':
                    trace.setdefault(nm, []).append((round(t - c.place_ev['RETREAT']['t'], 2), round(d * 1e3, 1)))
        if pph in ('PARKED', 'LOWER', 'RELEASE', 'RETREAT', 'HOLD'):
            m = float(np.minimum(q[3:] - lo, hi - q[3:]).min())
            marg[pph] = min(marg.get(pph, 9.9), m)
    for k in sorted(out):
        print(f'  {k:<36} {out[k][0] * 1e3:8.1f} mm @ {out[k][1]} s {out[k][2]}')
    for nm, tr in trace.items():
        print(f'  退離中 {nm}↔瓶（相位內秒、mm）', tr[::12], '末', tr[-1])
    print('  關節餘裕（rad，各相位最小）', {k: round(v, 3) for k, v in marg.items()})
    print('  持瓶路徑', r['place']['carry'] and {k: v for k, v in r['place']['carry'].items() if k != 'plan'}, '停車', np.round(r['place']['park'], 3).tolist())
    json.dump(dict(mins={k: [round(v[0] * 1e3, 1), v[1], v[2]] for k, v in out.items()}, joint_margin=marg, success=r['mission_done'], carry=r['place']['carry'], park=r['place']['park'],
                   note='假受控對象名目軌跡、合成支撐；非物理'), open(os.path.join(HERE, 'results', 'bottle', 'place_geom_check.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    return 0


if __name__ == '__main__':
    sys.exit(main())
