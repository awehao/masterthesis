#!/usr/bin/env python3
"""第八次第一包 C 項核對（Codex 20261110_194328 第 C 點；只讀假世界紀錄）。
1 Shield 配對：/shield/diag 每週期的輸入（欄 5–7）是否等於某筆 /cmd_vel_pre_shield，輸出（欄 8–10）是否等於某筆 /cmd_vel（逐筆以數值與順序配對）。
2 停距模型：Shield 參數 a_brake 6.25、tau 0.15 的停距 v·tau＋v²/(2·a_brake)，對照假世界實際停止軌跡（最終命令由 ≥ 0.15 m/s 降到 0 後，
  實測速度降到 < 0.02 m/s 前走過的距離）。假世界被控模型：一階延遲 0.10 s＋逐軸加速度 1.5／1.0 m/s²。
3 侵入時 min_h：首次安全包絡侵入前後 2 s，GMPC 狀態（diag_v2 state）、min_h、cbf_active，與 /gmpc/obstacles 中落在行人表面 0.15 m 內的點數。
    source /opt/ros/jazzy/setup.bash; python3 evaluation/m8_shield_check.py RUN_ID
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m8_bag as MB  # noqa: E402

A_BRAKE, TAU_S = 6.25, 0.15


def main():
    run = sys.argv[1]
    D = os.path.join(HERE, 'runs', run)
    B = MB.read_bag(os.path.join(D, 'bag'), ['/shield/diag', '/cmd_vel_pre_shield', '/cmd_vel', '/odom', '/gmpc/diag_v2', '/gmpc/obstacles'])
    od = MB.odom_series(B['/odom'])
    rec = np.array([x[0] for x in B['/odom']])

    def to_sim(r):
        return float(np.interp(r, rec, od[:, 0]))
    sd = [(r, np.asarray(m.data, float)) for r, m in B['/shield/diag']]
    pre = np.array([(m.linear.x, m.linear.y, m.angular.z) for _, m in B['/cmd_vel_pre_shield']])
    post = np.array([(m.linear.x, m.linear.y, m.angular.z) for _, m in B['/cmd_vel']])
    def matched(vals, pool):
        return int(sum(1 for v in vals if len(pool) and np.min(np.abs(pool - v).max(axis=1)) < 1e-5))
    ins = np.array([x[1][5:8] for x in sd]) if sd else np.zeros((0, 3))
    outs = np.array([x[1][8:11] for x in sd]) if sd else np.zeros((0, 3))
    pairing = dict(n_diag=len(sd), n_pre=len(pre), n_post=len(post), in_matched=matched(ins, pre), out_matched=matched(outs, post),
                   cycle_monotonic=bool(np.all(np.diff([x[1][0] for x in sd]) > 0)) if len(sd) > 1 else None)
    with np.load(os.path.join(D, 'm8fw_truth.npz')) as Z:
        c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
        walkers = json.loads(str(Z['walkers']))
    t, vx, vy, cx, cy = c['t'], c['v_x'], c['v_y'], c['cmd_x'], c['cmd_y']
    sp, cs = np.hypot(vx, vy), np.hypot(cx, cy)
    stops = []
    i = 1
    while i < len(t):
        if cs[i] < 1e-6 and cs[i - 1] >= 1e-6 and sp[i] >= 0.15:
            v0, j = sp[i], i
            while j < len(t) and sp[j] >= 0.02 and cs[j] < 1e-6:
                j += 1
            if j < len(t) and sp[j] < 0.02:
                dist = float(np.hypot(c['base_x'][j] - c['base_x'][i], c['base_y'][j] - c['base_y'][i]))
                stops.append(dict(t=round(float(t[i]), 2), v0=round(float(v0), 3), actual_m=round(dist, 3), shield_model_m=round(v0 * TAU_S + v0 * v0 / (2 * A_BRAKE), 3),
                                  t_stop_s=round(float(t[j] - t[i]), 2)))
            i = j
        i += 1
    for s_ in stops:
        s_['ratio'] = round(s_['actual_m'] / max(s_['shield_model_m'], 1e-6), 2)
    d_w = np.full(len(t), np.inf)
    for w in walkers:
        d_w = np.minimum(d_w, np.hypot(c['base_x'] - c[f"{w['name']}_x"], c['base_y'] - c[f"{w['name']}_y"]) - w['radius'] - 0.39)
    k_in = np.flatnonzero(d_w < 0)
    intr = None
    if len(k_in):
        t_in = float(t[k_in[0]])
        dg = [(float(json.loads(m.data).get('t') or to_sim(r)), json.loads(m.data)) for r, m in B['/gmpc/diag_v2']]
        win = [x for x in dg if t_in - 2.0 <= x[0] <= t_in + 2.0]
        states = {}
        for _, x in win:
            states[str(x.get('state'))] = states.get(str(x.get('state')), 0) + 1
        mh = [x.get('min_h') for _, x in win if x.get('min_h') is not None]
        obs_near = 0
        n_obs = 0
        for r, m in B['/gmpc/obstacles']:
            ts = to_sim(r)
            if not (t_in - 2.0 <= ts <= t_in + 2.0):
                continue
            n_obs += 1
            P = np.asarray(m.data, float).reshape(-1, 5)[:, :2] if len(m.data) else np.zeros((0, 2))
            k = int(np.argmin(np.abs(t - ts)))
            for w in walkers:
                wc = np.array([c[f"{w['name']}_x"][k], c[f"{w['name']}_y"][k]])
                if len(P) and (np.abs(np.hypot(P[:, 0] - wc[0], P[:, 1] - wc[1]) - w['radius']) <= 0.15).any():
                    obs_near += 1
                    break
        intr = dict(t_first=round(t_in, 2), gmpc_states=states, n_diag=len(win), min_h_min=None if not mh else round(min(mh), 4), n_min_h=len(mh),
                    obstacles_msgs=n_obs, msgs_with_walker_pts=obs_near)
    out = dict(run=run, pairing=pairing, stops=stops, stop_ratio_max=max((s_['ratio'] for s_ in stops), default=None), intrusion_window=intr,
               note='停距：Shield 模型 v·0.15＋v²/(2·6.25)；假世界一階延遲 0.10 s＋加速度 1.5／1.0')
    json.dump(out, open(os.path.join(D, 'm8_shield_check.json'), 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(dict(pairing=pairing, n_stops=len(stops), stops=stops[:5], stop_ratio_max=out['stop_ratio_max'], intrusion_window=intr), ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
