#!/usr/bin/env python3
"""前持候選兩個移動物案例的物理趟包絡與結果（只讀紀錄）。連桿點雲由實測關節（G.link_world）、瓶由讀回位姿（bottle_points）、移動物由讀回位置（鉛直圓柱 z 0–高度）；
到 LOWER 前持瓶對第二桌；另報前持時序、傾角、外伸、交會時窗安全層（bottle_w1_sr_pair_report.window_metrics）、滑移與耗時。
與先前 SHROW＋候選 2 預測組（dyn1sr_trk1／dyn2sr_trk1）並列，但兩者差連續展臂與前持兩項，不作歸因。
    python3 evaluation/bottle_w1_frontcarry_dyn_report.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_s1_geom as G                              # noqa: E402
import bottle_w1_sr_pair_report as PR                   # noqa: E402
import test_bottle_place as TP                          # noqa: E402

PAIRS = (('w1_dyn_pass1', 'bottle_w1_dyn1sr_trk1', 'bottle_w1_dyn1fc1'), ('w1_dyn_overtake1', 'bottle_w1_dyn2sr_trk1', 'bottle_w1_dyn2fc1'))


def quat(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def one(run):
    D = os.path.join(HERE, 'runs', run)
    r = json.load(open(os.path.join(D, 's1_report.json')))
    Z = np.load(os.path.join(D, 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    CA = r['dyn']['case']
    ev = {}
    for e in r['events']:
        ev.setdefault(e['what'] if e['what'] != 'place_phase' else 'pp_' + e.get('to', ''), e['t'])
    t_low = ev.get('pp_LOWER', 1e9)
    BP = G.bottle_points()
    t2 = [(c_, sz) for n, c_, sz in TP.table2_boxes()]
    link_dyn, bot_dyn, bot_t2, tilt, fwd = (9.9, None, None), (9.9, None), (9.9, None), 0.0, -9.0
    H = float(CA['height'])
    for i in range(0, len(c['t']), 10):
        t = float(c['t'][i])
        q = np.array([c['base_x'][i], c['base_y'][i], c['base_yaw'][i]] + [c[f'q{j}'][i] for j in range(1, 7)])
        o = np.array([c['dyn_x'][i], c['dyn_y'][i]])
        for n, P in G.link_world(q, float(c['finger1_q'][i])).items():
            dz = np.maximum(np.maximum(-P[:, 2], P[:, 2] - H), 0.0)
            d = float(np.hypot(np.maximum(np.hypot(P[:, 0] - o[0], P[:, 1] - o[1]) - CA['radius'], 0.0), dz).min())
            if d < link_dyn[0]:
                link_dyn = (d, round(t, 2), n)
        if r['t_lift_ok'] is not None and r['t_lift_ok'] <= t <= t_low:
            R = quat(c['bottle_qw'][i], c['bottle_qx'][i], c['bottle_qy'][i], c['bottle_qz'][i])
            Pw = (R @ BP.T).T + np.array([c['bottle_base_x'][i], c['bottle_base_y'][i], c['bottle_base_z'][i]])
            d = float(np.maximum(np.hypot(Pw[:, 0] - o[0], Pw[:, 1] - o[1]) - CA['radius'], 0.0).min())
            if d < bot_dyn[0]:
                bot_dyn = (d, round(t, 2))
            for c_, sz in t2:
                d2 = float(G.box_dist(Pw, c_, sz).min())
                if d2 < bot_t2[0]:
                    bot_t2 = (d2, round(t, 2))
            tilt = max(tilt, math.degrees(math.acos(min(1.0, abs(float(R[2, 2]))))))
            th = q[2]
            fwd = max(fwd, math.cos(th) * (Pw[:, 0].mean() - q[0]) + math.sin(th) * (Pw[:, 1].mean() - q[1]))
    e, _ = PR.M.metrics(run)
    w = PR.window_metrics(run, [k for k, v in PR.DC.CASES.items() if v['name'] == CA['name']][0])
    return dict(run=run, ok=e['ok'], t_done=e['t_done'], stop=e['stop'], fc=[ev.get(k) for k in ('front_carry_start', 'front_carry_reached', 'front_carry_return', 'front_carry_returned')],
                link_dyn_mm=(round(link_dyn[0] * 1e3, 1), link_dyn[1], link_dyn[2]), bottle_dyn_mm=(round(bot_dyn[0] * 1e3, 1), bot_dyn[1]), bottle_t2_mm=(round(bot_t2[0] * 1e3, 1), bot_t2[1]),
                base_dyn_m=round(w['surf_min_run_m'], 3), tilt_deg=round(tilt, 2), fwd_m=round(fwd, 3), slip_mm=round(e['slip_mm'], 2), shield=e['shield']['active'],
                slack=(w['slack_n'], w['solve_n'], round(w['slack_max'], 3)), over50=e['ms']['over50'], u0_bind=w['u0_bind_total'], near_run=round(e['near_run_s'], 2))


def main():
    out = {}
    for case, base, fc in PAIRS:
        for nm, run in (('SHROW＋候選2（側持）', base), ('前持候選', fc)):
            o = one(run)
            out[run] = dict(case=case, group=nm, **o)
            print(case, nm, o)
    json.dump(out, open(os.path.join(HERE, 'results', 'bottle', 'w1_frontcarry_dyn_physical.json'), 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    sys.exit(main())
