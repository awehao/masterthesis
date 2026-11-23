#!/usr/bin/env python3
"""緊湊 v3 名目物理 vs v1 名目物理（只讀紀錄；Codex 20261101_204716 第 4 項要求比較）：停車前近停（平移 < 0.01 m/s 連續 ≥ 0.1 s，抓後到 PARKED）、DONE、
外伸持續時間（持瓶 LIFT 起到 LOWER 前，瓶身＋夾爪水平半徑 > 0.30 m 的累計秒數；另分「離桌後到收近完成」與「轉回起到 LOWER」兩段）、升高後近身持瓶秒數、滑移、
v2.1／v3 閘紀錄（起點、收近中、停止軌跡、外伸閘、耗時）、Shield、>50 ms。v2 沒有名目物理趟（只有掃描中斷與兩個移動物案例），不列入名目比較。
    python3 evaluation/bottle_w1_compact3_report.py
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
import bottle_w1_compact2_report as CR2                 # noqa: E402
import bottle_w1_compact_pose as CP                     # noqa: E402
import bottle_w1_compactcarry2_ctrl as C2               # noqa: E402

RUNS = (('v1 名目', 'bottle_w1_compact_nominal1'), ('v3 名目', 'bottle_w1_compact3_nominal1'))


def one(run):
    D = os.path.join(HERE, 'runs', run)
    r = json.load(open(os.path.join(D, 's1_report.json')))
    Z = np.load(os.path.join(D, 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    t = c['t']
    ev = {}
    for e in r['events']:
        ev.setdefault(e['what'] if e['what'] != 'place_phase' else 'p_' + e.get('to', ''), e['t'])
    tg, tp, tl = r['t_grasp'], ev.get('p_PARKED', 1e9), ev.get('p_LOWER', 1e9)
    v = np.r_[0.0, np.hypot(np.diff(c['base_x']), np.diff(c['base_y'])) / np.diff(t)]
    m = (t >= tg) & (t < tp) & (v < 0.01)
    stops, i = [], 0
    while i < len(t):
        if m[i]:
            j = i
            while j + 1 < len(t) and m[j + 1]:
                j += 1
            if t[j] - t[i] >= 0.1:
                stops.append((round(float(t[i]), 2), round(float(t[j]), 2)))
            i = j + 1
        else:
            i += 1
    q_side = np.array(json.load(open(os.path.join(HERE, 'results', 'bottle', 'w1_compact_pose_v2.json')))['reference']['q_side'])
    GG, BP = C2.grip_grid_tcp(q_side), G.bottle_points(12, 5)
    t_ret = ev.get('compact_raise_start', ev.get('compact_return'))
    dt = float(np.median(np.diff(t)))
    reach = dict(total=0.0, before_compact=0.0, from_return=0.0)
    for k in range(len(t)):
        if t[k] < (r['t_lift_ok'] or 1e9) or t[k] > tl:
            continue
        q = np.array([c['base_x'][k], c['base_y'][k], c['base_yaw'][k]] + [c[f'q{j}'][k] for j in range(1, 7)])
        Tt = G.C.K.fk(q, 'link_tcp')
        R = CP.quat(c['bottle_qw'][k], c['bottle_qx'][k], c['bottle_qy'][k], c['bottle_qz'][k])
        P = np.vstack([(R @ BP.T).T + np.array([c['bottle_base_x'][k], c['bottle_base_y'][k], c['bottle_base_z'][k]]), (Tt[:3, :3] @ GG.T).T + Tt[:3, 3]])
        if float(np.hypot(P[:, 0] - q[0], P[:, 1] - q[1]).max()) > 0.30:
            reach['total'] += dt
            if t[k] < ev.get('compact_reached', 1e9):
                reach['before_compact'] += dt
            elif t_ret is not None and t[k] >= t_ret:
                reach['from_return'] += dt
    e, _ = M.metrics(run)
    raised = (ev['compact_out_start'] - ev['compact_raised']) if 'compact_out_start' in ev and 'compact_raised' in ev else None
    g = (r.get('compact_carry') or {}).get('gate') or {}
    w_park = sum(b - a for a, b in stops if b >= tp - 0.1)
    out = dict(run=run, version=(r.get('compact_carry') or {}).get('version', 'v1'), t_done=e['t_done'], parked=ev.get('p_PARKED'), near_stops=stops, park_wait_s=round(w_park, 2),
               reach_s={k_: round(v_, 2) for k_, v_ in reach.items()}, raised_close_s=None if raised is None else round(raised, 2),
               events={k_: ev.get(k_) for k_ in ('compact_start', 'compact_reached', 'compact_raise_start', 'compact_raised', 'compact_return', 'compact_out_start', 'compact_returned', 'p_PLACE_APPROACH')},
               slip_mm=round(e['slip_mm'], 2), place_mm=round((r['place']['metrics'] or {}).get('offset', float('nan')) * 1e3, 1), shield=e['shield']['active'], over50=e['ms']['over50'],
               gate=dict(trigger=g.get('trigger'), monitor_min=g.get('monitor_min'), stop_traj={k_: v_ for k_, v_ in (g.get('stop_traj') or {}).items() if k_ != 'rule'}, ms_max=g.get('ms_max')),
               reach_gate={k_: v_ for k_, v_ in ((r.get('compact_carry') or {}).get('reach_gate') or {}).items() if k_ != 'log'})
    try:
        cl, tilt = CR2.clear_after(run, ev.get('compact_start', 0.0), tl)
        out.update(clear_after_start_mm=cl, tilt_after_start=tilt)
    except Exception as ex:                                  # noqa: BLE001
        out.update(clear_error=repr(ex))
    return out


def main():
    out = {nm: one(run) for nm, run in RUNS}
    json.dump(out, open(os.path.join(HERE, 'results', 'bottle', 'w1_compact3_physical.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    for nm, o in out.items():
        print(nm, json.dumps(o, ensure_ascii=False, default=float))
    return 0


if __name__ == '__main__':
    sys.exit(main())
