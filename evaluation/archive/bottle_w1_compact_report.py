#!/usr/bin/env python3
"""緊湊持物名目物理趟 vs 連續展臂基線、前持（只讀紀錄）。包絡＝瓶身（讀回位姿取樣點）＋夾爪包圍盒（實測關節 FK，同 S1._obb_tcp 的連桿集合），
相對底盤中心的前向／側向投影與水平半徑；分期：側持（離桌到收近／前持開始）、過渡、穩態、轉回；另報瓶底高度、底盤偏航與 q1 範圍、持瓶對第二桌（到 LOWER 前）、傾角、滑移、放置。
    python3 evaluation/bottle_w1_compact_report.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), '第七次進度報告素材', '產生程式'))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_mainline_table as M                       # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import bottle_w1_compact_pose as CP                     # noqa: E402
import test_bottle_place as TP                          # noqa: E402

RUNS = (('連續展臂（側持）', 'bottle_w1_cont_nominal1', None), ('前持', 'bottle_w1_front_nominal1', 'front_carry'), ('緊湊', 'bottle_w1_compact_nominal1', 'compact'))


def one(run, kind):
    D = os.path.join(HERE, 'runs', run)
    r = json.load(open(os.path.join(D, 's1_report.json')))
    Z = np.load(os.path.join(D, 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    ev = {}
    for e in r['events']:
        ev.setdefault(e['what'] if e['what'] != 'place_phase' else 'pp_' + e.get('to', ''), e['t'])
    names = {'front_carry': ('front_carry_start', 'front_carry_reached', 'front_carry_return', 'front_carry_returned'),
             'compact': ('compact_start', 'compact_reached', 'compact_return', 'compact_returned')}.get(kind)
    tt = [ev.get(n) for n in names] if names else [None] * 4
    t_low = ev.get('pp_LOWER', 1e9)
    GC = CP.grip_corners(np.array([c[f'q{j}'][0] for j in range(1, 7)]) * 0 + np.array(json.load(open(os.path.join(HERE, 'results', 'bottle', 'w1_compact_pose_v2.json')))['reference']['q_side']))
    BP = G.bottle_points()
    t2 = [(c_, sz) for n, c_, sz in TP.table2_boxes()]

    def seg(t):
        if tt[0] is None:
            return 'side_carry'
        if t < tt[0]:
            return 'side_carry'
        if t < (tt[1] or 1e9):
            return 'transition_in'
        if t < (tt[2] or 1e9):
            return 'steady'
        if t < (tt[3] or 1e9):
            return 'transition_back'
        return 'side_place'
    out, t2min, tilt = {}, 9.9, 0.0
    for i in range(0, len(c['t']), 10):
        t = float(c['t'][i])
        if r['t_lift_ok'] is None or t < r['t_lift_ok'] or t > t_low:
            continue
        q = np.array([c['base_x'][i], c['base_y'][i], c['base_yaw'][i]] + [c[f'q{j}'][i] for j in range(1, 7)])
        Tt = G.C.K.fk(q, 'link_tcp')
        w, x, y, z = (c[k][i] for k in ('bottle_qw', 'bottle_qx', 'bottle_qy', 'bottle_qz'))
        R = CP.quat(w, x, y, z)
        Pb = (R @ BP.T).T + np.array([c['bottle_base_x'][i], c['bottle_base_y'][i], c['bottle_base_z'][i]])
        Pg = (Tt[:3, :3] @ GC.T).T + Tt[:3, 3]
        P = np.vstack([Pb, Pg])
        cth, sth = math.cos(q[2]), math.sin(q[2])
        dx, dy = P[:, 0] - q[0], P[:, 1] - q[1]
        fw, sd = cth * dx + sth * dy, -sth * dx + cth * dy
        k = seg(t)
        o = out.setdefault(k, dict(fwd=-9.0, side=0.0, rad=0.0, bottom=9.9))
        o.update(fwd=max(o['fwd'], float(fw.max())), side=max(o['side'], float(np.abs(sd).max())), rad=max(o['rad'], float(np.hypot(dx, dy).max())), bottom=min(o['bottom'], float(Pb[:, 2].min())))
        for c_, sz in t2:
            t2min = min(t2min, float(G.box_dist(Pb, c_, sz).min()))
        tilt = max(tilt, math.degrees(math.acos(min(1.0, abs(float(R[2, 2]))))))
    e, _ = M.metrics(run)
    m = (c['t'] >= (r['t_lift_ok'] or 0)) & (c['t'] <= t_low)
    return dict(run=run, ok=e['ok'], t_done=e['t_done'], events=tt, parked=ev.get('pp_PARKED'), env={k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in out.items()},
                bottle_t2_mm=round(t2min * 1e3, 1), tilt_deg=round(tilt, 2), yaw_max_deg=round(float(np.degrees(np.abs(c['base_yaw'][m]).max())), 2),
                q1_range=[round(float(c['q1'][m].min()), 3), round(float(c['q1'][m].max()), 3)], slip_mm=round(e['slip_mm'], 2),
                place_mm=round((r['place']['metrics'] or {}).get('offset', float('nan')) * 1e3, 1), shield=e['shield']['active'], over50=e['ms']['over50'])


def main():
    out = {nm: one(run, kind) for nm, run, kind in RUNS}
    json.dump(out, open(os.path.join(HERE, 'results', 'bottle', 'w1_compact_physical.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    for nm, o in out.items():
        print(nm, json.dumps(o, ensure_ascii=False))


if __name__ == '__main__':
    sys.exit(main())
