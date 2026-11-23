#!/usr/bin/env python3
"""第八次：主動保持功能核對的分析（Codex 20261110_200117 核對 1–3；只讀假世界紀錄）。
1 保持期間 GMPC 狀態（/gmpc/diag_v2 state）：應連續 running（有解 QP），不出現 no_plan／arrived／holding_stuck 代替求解。
2 約束是否改變保持命令：行人遠（中心距 > 2.5 m）與近（< 1.5 m）時，解 u 的大小、cbf_active、eps0；保持期間圓盤對行人最小距離。
3 進出保持：前後 0.5 s 內 /cmd_vel_pre_mux 與 /cmd_vel 相鄰訊息的逐軸變化率（對照平滑器限制 1.5／1.0／2.0）；
  感知過期（停掉追蹤器）後狀態應為 stale_obstacles、輸出為零。
    source /opt/ros/jazzy/setup.bash; python3 evaluation/m8_hold_check.py RUN_ID
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m8_bag as MB  # noqa: E402

A_LIM = np.array([1.5, 1.0, 2.0])


def main():
    run = sys.argv[1]
    D = os.path.join(HERE, 'runs', run)
    pr = json.load(open(os.path.join(D, 'm8nav.json')))
    ev = [e for e in pr['events'] if 'hold' in e]
    t_on = next(e['t'] for e in ev if e['hold'])
    t_off = next((e['t'] for e in ev if not e['hold']), None)
    B = MB.read_bag(os.path.join(D, 'bag'), ['/odom', '/gmpc/diag_v2', '/cmd_vel_pre_mux', '/cmd_vel'])
    od = MB.odom_series(B['/odom'])
    rec = np.array([x[0] for x in B['/odom']])

    def to_sim(r):
        return float(np.interp(r, rec, od[:, 0]))
    with np.load(os.path.join(D, 'm8fw_truth.npz')) as Z:
        c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
        walkers = json.loads(str(Z['walkers']))
    dg = sorted(((float(json.loads(m.data).get("t") or to_sim(r)), json.loads(m.data)) for r, m in B["/gmpc/diag_v2"]), key=lambda z: z[0])
    win = [(t, x) for t, x in dg if t_on + 0.2 <= t <= (t_off if t_off else 1e9) - 0.05]
    states = {}
    for _, x in win:
        states[str(x.get('state'))] = states.get(str(x.get('state')), 0) + 1

    def dmin_w(t):
        k = int(np.argmin(np.abs(c['t'] - t)))
        return min(math.hypot(c['base_x'][k] - c[f"{w['name']}_x"][k], c['base_y'][k] - c[f"{w['name']}_y"][k]) for w in walkers)
    far = [x for t, x in win if x.get('u') is not None and dmin_w(t) > 2.5]
    near = [x for t, x in win if x.get('u') is not None and dmin_w(t) < 1.5]

    def summ(L):
        if not L:
            return None
        U = np.array([np.linalg.norm(x['u'][:2]) for x in L])
        return dict(n=len(L), u_med=round(float(np.median(U)), 4), u_max=round(float(U.max()), 4), cbf_active_med=float(np.median([x.get('cbf_active') or 0 for x in L])),
                    eps0_max=round(max(x.get('eps0') or 0 for x in L), 4))
    k = (c['t'] >= t_on) & (c['t'] <= (t_off if t_off else c['t'][-1]))
    dw = np.full(int(k.sum()), np.inf)
    for w in walkers:
        dw = np.minimum(dw, np.hypot(c['base_x'][k] - c[f"{w['name']}_x"][k], c['base_y'][k] - c[f"{w['name']}_y"][k]) - w['radius'] - 0.39)
    disp = float(np.hypot(c['base_x'][k] - c['base_x'][k][0], c['base_y'][k] - c['base_y'][k][0]).max()) if k.any() else None

    def jumps(topic, t0):
        S = sorted(((to_sim(r), np.array([m.linear.x, m.linear.y, m.angular.z])) for r, m in B[topic]), key=lambda z: z[0])
        S = [x for x in S if t0 - 0.5 <= x[0] <= t0 + 0.5]
        if len(S) < 3:
            return None
        T = np.array([x[0] for x in S])
        U = np.array([x[1] for x in S])
        dt = np.diff(T)
        ok = dt > 1e-3
        a = np.abs(np.diff(U, axis=0)[ok]) / dt[ok, None]
        return dict(a_max=[round(float(v), 3) for v in a.max(axis=0)], over=[int((a[:, j] > A_LIM[j] * 1.05).sum()) for j in range(3)])
    out = dict(run=run, t_hold_on=t_on, t_hold_off=t_off, states_in_hold=states, n_diag_in_hold=len(win), u_when_walker_far=summ(far), u_when_walker_near=summ(near),
               disc_min_to_walker_in_hold=None if not len(dw) else round(float(dw.min()), 3), hold_displacement_max_m=None if disp is None else round(disp, 3),
               jumps={'enter_pre_mux': jumps('/cmd_vel_pre_mux', t_on), 'enter_final': jumps('/cmd_vel', t_on),
                      'exit_pre_mux': None if t_off is None else jumps('/cmd_vel_pre_mux', t_off), 'exit_final': None if t_off is None else jumps('/cmd_vel', t_off)})
    rl = open(os.path.join(D, 'run.log'), encoding='utf-8').read()
    if '已停追蹤器' in rl:
        post = [x for t, x in win if str(x.get('state')).startswith('stale')]
        out['stale_check'] = dict(n_stale_diag=len(post), states=states, note='停掉追蹤器後應轉 stale_obstacles 並輸出零（失效關閉）')
    json.dump(out, open(os.path.join(D, 'm8_hold_check.json'), 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
