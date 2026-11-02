#!/usr/bin/env python3
"""F1 實跑後的移動物交互分析（只讀紀錄，不動控制）：讀兩輪 s1_cbf*.npz，用同型控制器的導航 GMPC 逐拍重解（bottle_f1_dyn_report.analyse；重現差逐拍核對），再從逐步紀錄量距離、車速、全身到障礙淨空與返回段求解耗時。
    python3 evaluation/bottle_f1_dyn_replay.py RUN_ID
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RUN = sys.argv[1]
sys.argv = ['test_bottle_f1_task.py', 'nominal']
import test_bottle_f1_task as T                         # noqa: E402  先載入：它會設好 ammr_wholebody_mpc 的路徑
import bottle_f1_dyn_report as FD                       # noqa: E402
import bottle_s1_geom as G                              # noqa: E402

D = os.path.join(HERE, 'runs', RUN)
CA = dict(json.load(open(os.path.join(HERE, 'experiments', '20261103_f1_dyn_case.json')))['case'])
for k in ('start', 'end', 'meet_robot', 'robot_dir'):
    CA[k] = tuple(CA[k])
r0 = json.load(open(os.path.join(D, 's1_report_r0.json')))
r1 = json.load(open(os.path.join(D, 's1_report.json')))
ev = {e['to']: e['t'] for e in r0['events'] if e.get('what') == 'return_state'}
nav = (ev.get('NAV', 0.0), ev.get('ROT1', 1e9))
c, _ = T.build_ctrl()
rows = FD.load_rows(os.path.join(D, 's1_cbf_r0.npz')) + FD.load_rows(os.path.join(D, 's1_cbf.npz'))
SL = np.load(os.path.join(D, 'scan_log.npz'))
ts_, rr_ = SL['t'], SL['ranges']


def before_solve(e):
    """還原該拍的 Shield 列輸入：控制器當下的 t 與最新一幀掃描（時刻 ≤ t）。"""
    c.t = float(e['t'])
    i_ = int(np.searchsorted(ts_, c.t + 1e-9) - 1)
    c.scan_last = None if i_ < 0 else (float(ts_[i_]), np.array(rr_[i_], float))


out = dict(run=RUN, case=CA['name'], nav_window=nav, return_states=ev, interaction=FD.analyse(rows, c.mpc_nav, CA['t_start'], nav, before_solve=before_solve))
n_d = np.array([len(FD._dyn(e['obs'])) for e in rows if nav[0] <= float(e['t']) <= nav[1]])
t_d = np.array([float(e['t']) for e in rows if nav[0] <= float(e['t']) <= nav[1]])
on = t_d[n_d > 0]
gaps = [[round(float(on[i_]), 2), round(float(on[i_ + 1]), 2)] for i_ in range(len(on) - 1) if on[i_ + 1] - on[i_] > 0.051]
out['tracking'] = dict(n_dyn_hist={int(k_): int((n_d == k_).sum()) for k_ in np.unique(n_d)}, t_first=round(float(on[0]), 2), t_last=round(float(on[-1]), 2), gaps=gaps, n_gap_ticks=int(round(sum(b_ - a_ for a_, b_ in gaps) / 0.05)) - len(gaps))
Z = np.load(os.path.join(D, 's1_steps.npz'), allow_pickle=True)
cols = [str(x) for x in Z['cols']]
R = Z['rows']
col = {n: R[:, i] for i, n in enumerate(cols)}
t = col['t']
Q = np.column_stack([col[n] for n in ('base_x', 'base_y', 'base_yaw', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6')])
ob = np.column_stack([col['dyn_x'], col['dyn_y']])
d = np.hypot(Q[:, 0] - ob[:, 0], Q[:, 1] - ob[:, 1]) - CA['radius']
i = int(np.argmin(d))
out['actual'] = dict(base_center_surface=round(float(d[i]), 4), base_disc=round(float(d[i] - 0.39), 4), at_t=round(float(t[i]), 2), phase_stage=int(col['f1_stage'][i]),
                     truth_err_max_m=round(float(np.hypot(ob[:, 0] - col['dyn_tx'], ob[:, 1] - col['dyn_ty']).max()), 4))
k = np.flatnonzero((t >= nav[0]) & (t <= nav[1]))
k = k[np.abs(t[k] * 20 - np.round(t[k] * 20)) < 1e-6]
v = np.hypot(np.diff(Q[k, 0]), np.diff(Q[k, 1])) / 0.05
wz = np.abs(np.arctan2(np.sin(np.diff(Q[k, 2])), np.cos(np.diff(Q[k, 2])))) / 0.05
near = d[k[1:]] < 1.0
out['motion'] = dict(nav_v_mean=round(float(v.mean()), 4), v_mean_near=round(float(v[near].mean()), 4) if near.any() else None, v_min_near=round(float(v[near].min()), 4) if near.any() else None, v_max_near=round(float(v[near].max()), 4) if near.any() else None, w_max_near=round(float(wz[near].max()), 4) if near.any() else None, near_s=round(float(near.sum() * 0.05), 2))
wd = {}
fq = col['finger_cmd'] if 'finger_cmd' in col else np.zeros(len(t))
sel = np.flatnonzero((d - 0.39 < 1.5) & (np.abs(t * 20 - np.round(t * 20)) < 1e-6))
for j in sel:
    W = G.link_world(Q[j], float(fq[j]))
    Pn, Pf = T.split(W)
    for nm, P in (('nonfinger', Pn), ('finger', Pf)):
        P = P[P[:, 2] <= CA['height']]
        if len(P):
            wd[nm] = min(wd.get(nm, 9.9), float((np.hypot(P[:, 0] - ob[j, 0], P[:, 1] - ob[j, 1]) - CA['radius']).min()))
out['body_clearance'] = {k_: round(v_, 4) for k_, v_ in wd.items()}
S_all = list(r0.get('solve_ms_all') or []) + list(r1.get('solve_ms_all') or [])
if len(S_all) == len(rows):
    S_ret = np.array([s for s, e in zip(S_all, rows) if float(e['w_p']) == 0.0])
    S_nav = np.array([s for s, e in zip(S_all, rows) if nav[0] <= float(e['t']) <= nav[1]])
    out['solve_return'] = dict(n=len(S_ret), p50=round(float(np.percentile(S_ret, 50)), 2), p95=round(float(np.percentile(S_ret, 95)), 2), max=round(float(S_ret.max()), 2), n_over_50=int((S_ret > 50).sum()), nav_max=round(float(S_nav.max()), 2))
else:
    out['solve_return'] = dict(note=f'耗時筆數 {len(S_all)} 與紀錄列 {len(rows)} 不等，未對齊', n_solve=len(S_all), n_rows=len(rows))
out['events'] = sorted({e.get('what') for e in r0['events']} | {e.get('what') for e in r1['events']}, key=str)
print(json.dumps(out, ensure_ascii=False, default=float))
