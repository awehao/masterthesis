#!/usr/bin/env python3
"""緊湊持物修正版 v2 物理趟（只讀紀錄）：分期包絡、瓶底、偏航、q1 等沿用 bottle_w1_compact_report.one；另報 v2 閘（起點餘裕、位姿數、收近中最小、單拍耗時）、
收近起點（或注入時刻）之後到 LOWER（或結束）前：瓶身（讀回位姿取樣點）＋夾爪格點（實測關節 FK）對第一桌、第二桌（桌面、桌腳）的三維最小距離；第一阻點、停止有效、掃描中斷注入。
    python3 evaluation/bottle_w1_compact2_report.py RUN_ID [RUN_ID ...]
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
import bottle_w1_compact_pose as CP                     # noqa: E402
import bottle_w1_compact_report as CR                   # noqa: E402
import bottle_w1_compactcarry2_ctrl as C2               # noqa: E402
import test_bottle_place as TP                          # noqa: E402


def clear_after(run, t0, t1):
    D = os.path.join(HERE, 'runs', run)
    Z = np.load(os.path.join(D, 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    q_side = np.array(json.load(open(os.path.join(HERE, 'results', 'bottle', 'w1_compact_pose_v2.json')))['reference']['q_side'])
    GG, BP = C2.grip_grid_tcp(q_side), G.bottle_points()
    boxes = [('table1', c_, sz) for _, c_, sz in G.table_boxes()] + [('table2', c_, sz) for _, c_, sz in TP.table2_boxes()]
    best, tilt = {}, 0.0
    for i in range(0, len(c['t']), 2):
        t = float(c['t'][i])
        if t < t0 - 1e-9 or t > t1 + 1e-9:
            continue
        q = np.array([c['base_x'][i], c['base_y'][i], c['base_yaw'][i]] + [c[f'q{j}'][i] for j in range(1, 7)])
        Tt = G.C.K.fk(q, 'link_tcp')
        R = CP.quat(*(c[k][i] for k in ('bottle_qw', 'bottle_qx', 'bottle_qy', 'bottle_qz')))
        Pb = (R @ BP.T).T + np.array([c['bottle_base_x'][i], c['bottle_base_y'][i], c['bottle_base_z'][i]])
        Pg = (Tt[:3, :3] @ GG.T).T + Tt[:3, 3]
        tilt = max(tilt, math.degrees(math.acos(min(1.0, abs(float(R[2, 2]))))))
        for nm, c_, sz in boxes:
            for who, P in (('bottle', Pb), ('gripper', Pg)):
                d = float(G.box_dist(P, c_, sz).min())
                k = f'{who}_{nm}'
                if k not in best or d < best[k][0]:
                    best[k] = (d, round(t, 2))
    return {k: (round(v[0] * 1e3, 1), v[1]) for k, v in best.items()}, round(tilt, 2)


def one(run):
    r = json.load(open(os.path.join(HERE, 'runs', run, 's1_report.json')))
    o = CR.one(run, 'compact')
    ev = {e['what']: e['t'] for e in r['events'] if e['what'] in ('compact_start', 'compact_reached', 'compact_return', 'compact_raised', 'compact_returned')}
    t_low = next((e['t'] for e in r['events'] if e['what'] == 'place_phase' and e.get('to') == 'LOWER'), 1e9)
    sc = r.get('scan_cut') or {}
    t0 = sc.get('t_cut') or ev.get('compact_start') or 0.0
    cl, tilt = clear_after(run, ev.get('compact_start', 0.0), t_low)
    cl_cut, tilt_cut = clear_after(run, t0, t_low) if sc.get('t_cut') else (None, None)
    g = (r.get('compact_carry') or {}).get('gate') or {}
    se = r.get('stop_effect') or {}
    o.update(version=(r.get('compact_carry') or {}).get('version'), events_v2=ev, stop=r.get('stop'), stop_effect=dict(ok=se.get('ok'), t=se.get('t')) if se else None, scan_cut=sc or None,
             gate=dict(trigger=g.get('trigger'), monitor_min=g.get('monitor_min'), ms_max=g.get('ms_max'), ms_mean=g.get('ms_mean'), n=g.get('n')),
             clear_after_start_mm=cl, tilt_after_start=tilt, clear_after_cut_mm=cl_cut, tilt_after_cut=tilt_cut)
    return o


def main():
    runs = sys.argv[1:]
    p = os.path.join(HERE, 'results', 'bottle', 'w1_compact2_physical.json')
    out = json.load(open(p)) if os.path.exists(p) else {}
    for run in runs:
        out[run] = one(run)
        print(run, json.dumps(out[run], ensure_ascii=False, default=float))
    json.dump(out, open(p, 'w'), ensure_ascii=False, indent=1, default=float)
    return 0


if __name__ == '__main__':
    sys.exit(main())
