#!/usr/bin/env python3
"""第八次第一包：假世界短程閉迴路的評估（m8fw_truth.npz＋bag；只讀紀錄）。
距離：底盤圓盤（0.39）對行人表面、未知箱體、已知牆家具與門扇的最近距離（真值；負＝安全包絡侵入；假世界不模擬接觸力）。
Shield：/shield/diag 逐週期（序號、active、輸入、輸出），計介入週期數與輸出對輸入的最大差。
CBF：/gmpc/diag_v2 的 eps0、min_h、cbf_active。
變化率：/cmd_vel_pre_mux（平滑器輸出）與 /cmd_vel（Shield 後）相鄰訊息的逐軸加速度，對照平滑器限制 [1.5, 1.0, 2.0]。
衝突層：m8nav.json 的 conflict／conflict_layer_log（r2 才有）。
    source /opt/ros/jazzy/setup.bash; python3 evaluation/m8_fake_report.py RUN_ID
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m8_bag as MB  # noqa: E402
import mission_room_obstacles as MO  # noqa: E402

R_BASE = 0.39
A_LIM = np.array([1.5, 1.0, 2.0])


def num(v, d=3):
    return None if v is None or not np.isfinite(v) else round(float(v), d)


def main():
    run = sys.argv[1]
    D = os.path.join(HERE, 'runs', run)
    nav = json.load(open(os.path.join(D, 'm8nav.json')))
    with np.load(os.path.join(D, 'm8fw_truth.npz')) as Z:
        rows, cols = Z['rows'], [str(c) for c in Z['cols']]
        walkers, static, meta = json.loads(str(Z['walkers'])), json.loads(str(Z['static'])), json.loads(str(Z['meta']))
    c = {k: rows[:, i] for i, k in enumerate(cols)}
    seg = {s_['seg']: s_['t'] for s_ in nav['segments']}
    t0, t1 = seg.get('CRUISE'), seg.get('ZERO', c['t'][-1])
    k = (c['t'] >= (t0 or 0)) & (c['t'] <= t1)
    P = np.c_[c['base_x'][k], c['base_y'][k]]
    tt, tw = c['t'][k], c['tw'][k]
    out = dict(run=run, nav_node=nav.get('version'), result=nav.get('result'), arrive=nav.get('arrive'), t_cruise=t0, t_end=t1,
               t_cruise_world=num(float(np.interp(t0, c['t'], c['tw'])), 2) if t0 else None, t_end_world=num(float(np.interp(t1, c['t'], c['tw'])), 2),
               path_len_m=num(np.sum(np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1])))), meta=meta)
    clr = {}
    for w in walkers:
        d = np.hypot(P[:, 0] - c[f"{w['name']}_x"][k], P[:, 1] - c[f"{w['name']}_y"][k]) - w['radius'] - R_BASE
        i = int(np.argmin(d))
        clr[w['name']] = dict(disc_min=num(d[i]), t_world=num(tw[i], 2), n_intrusion_samples=int((d < 0).sum()))
    for u in static:
        d = MB.poly_dist(P, MB.box_poly(u)) - R_BASE
        i = int(np.argmin(d))
        clr[u['name']] = dict(disc_min=num(d[i]), t_world=num(tw[i], 2), n_intrusion_samples=int((d < 0).sum()))
    kb = [MB.poly_dist(P, np.array([(r[0], r[2]), (r[1], r[2]), (r[1], r[3]), (r[0], r[3])])) for r in MO.known_boxes()]
    kb.append(MB.poly_dist(P, np.array([(0.0, 0.0), (0.95, 0.0), (0.95, 0.04), (0.0, 0.04)])))
    dk = np.min(kb, axis=0) - R_BASE
    i = int(np.argmin(dk))
    clr['known_walls_furniture_door'] = dict(disc_min=num(dk[i]), t_world=num(tw[i], 2), n_intrusion_samples=int((dk < 0).sum()))
    out['clearance'] = clr
    out['safety'] = dict(intrusion=any(v['disc_min'] is not None and v['disc_min'] < 0 for v in clr.values()),
                         note='假世界不模擬接觸力；負值＝安全包絡侵入（幾何重疊）')
    B = MB.read_bag(os.path.join(D, 'bag'), ['/shield/diag', '/gmpc/diag_v2', '/cmd_vel_pre_mux', '/cmd_vel', '/odom'])
    od = MB.odom_series(B['/odom'])
    rec = np.array([x[0] for x in B['/odom']])

    def to_sim(r):
        return float(np.interp(r, rec, od[:, 0]))
    sd = [(to_sim(r), np.asarray(m.data, float)) for r, m in B['/shield/diag']]
    sd = [x for x in sd if (t0 or 0) <= x[0] <= t1]
    if sd:
        A = np.array([x[1] for x in sd])
        dv = np.abs(A[:, 8:11] - A[:, 5:8]).max(axis=1)
        out['shield'] = dict(n_cycles=len(A), n_active=int((A[:, 1] > 0.5).sum()), max_dv=num(dv.max(), 4), n_unresolved=int((A[:, 17] > 0.5).sum()),
                             cycle_monotonic=bool(np.all(np.diff(A[:, 0]) > 0)), note='/shield/diag 逐週期；active＝Shield 改寫')
    dg = []
    for r, m in B['/gmpc/diag_v2']:
        try:
            x = json.loads(m.data)
        except ValueError:
            continue
        if x.get('t') is not None and (t0 or 0) <= float(x['t']) <= t1:
            dg.append(x)
    eps = np.array([x['eps0'] for x in dg if x.get('eps0') is not None], float)
    mh = np.array([x['min_h'] for x in dg if x.get('min_h') is not None], float)
    out['cbf'] = dict(n=len(dg), eps0_max=num(eps.max() if len(eps) else np.nan, 4), min_h_min=num(mh.min() if len(mh) else np.nan, 4))

    def acc(topic):
        S = sorted((to_sim(r), (m.linear.x, m.linear.y, m.angular.z)) for r, m in B[topic])
        S = [x for x in S if (t0 or 0) <= x[0] <= t1]
        if len(S) < 3:
            return None
        T = np.array([x[0] for x in S])
        U = np.array([x[1] for x in S])
        dt = np.diff(T)
        ok = dt > 1e-3
        a = np.abs(np.diff(U, axis=0)[ok]) / dt[ok, None]
        return dict(n=int(ok.sum()), a_max=[num(v) for v in a.max(axis=0)], n_over=[int((a[:, j] > A_LIM[j] * 1.05).sum()) for j in range(3)], lim=A_LIM.tolist())
    out['rate'] = dict(smoother_out=acc('/cmd_vel_pre_mux'), final=acc('/cmd_vel'), note='相鄰訊息差分；時間戳取 bag 接收換算，會有抖動')
    out['conflict'] = nav.get('conflict')
    out['conflict_layer_log'] = nav.get('conflict_layer_log')
    out['safety_node'] = nav.get('safety')
    json.dump(out, open(os.path.join(D, 'm8_fake_report.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    show = {k_: out[k_] for k_ in ('run', 'nav_node', 'result', 't_cruise_world', 't_end_world', 'path_len_m', 'safety', 'shield', 'cbf', 'rate')}
    show['clearance'] = {k_: v for k_, v in clr.items() if k_ in ('walker_a1', 'unknown_a1', 'unknown_a2', 'known_walls_furniture_door')}
    show['conflict_modes'] = [(x.get('t'), x.get('mode'), (x.get('plan') or {}).get('kind')) for x in (out['conflict'] or []) if x.get('mode')]
    print(json.dumps(show, ensure_ascii=False, indent=1, default=float))


if __name__ == '__main__':
    main()
