#!/usr/bin/env python3
"""F1-L 實跑的走廊與求解指標（只讀紀錄，真值只供事後評估）：三段走廊的時段與車速、全身與持瓶對牆淨空、手臂排程、求解耗時與後備、鬆弛、追蹤誤判。
    python3 evaluation/bottle_f1l_run_metrics.py RUN_ID
"""
import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RUN = sys.argv[1]
sys.argv = ['test_bottle_f1_task.py', 'nominal']
import test_bottle_f1_task as T                         # noqa: E402
import bottle_f1_dyn_report as FD                       # noqa: E402
import bottle_f1_lchan_scene as LS                      # noqa: E402
import bottle_s1_geom as G                              # noqa: E402

D = os.path.join(HERE, 'runs', RUN)
SCN = LS.load()
WALLS = [(c_, s_) for _, c_, s_ in SCN['walls']]
BPT = G.bottle_points()
r0, r1 = json.load(open(os.path.join(D, 's1_report_r0.json'))), json.load(open(os.path.join(D, 's1_report.json')))
Z = np.load(os.path.join(D, 's1_steps.npz'), allow_pickle=True)
col = {str(n): Z['rows'][:, i] for i, n in enumerate(Z['cols'])}
t = col['t']
Q = np.column_stack([col[n] for n in ('base_x', 'base_y', 'base_yaw', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6')])
ins = LS.in_corridor(Q[:, :2])
ev0 = {e['what']: e['t'] for e in r0['events'] if str(e.get('what', '')).startswith('compact_')}
ev1 = {e['what']: e['t'] for e in r1['events'] if str(e.get('what', '')).startswith('compact_')}
rs = {e['to']: e['t'] for e in r0['events'] if e.get('what') == 'return_state'}
t_done0 = next(e['t'] for e in r0['events'] if e.get('what') == 'mission_done')
t_sw = next((float(t[i]) for i in range(1, len(t)) if col['f1_stage'][i] != col['f1_stage'][i - 1]), None)
SEG = (('carry_r1', 0.0, t_done0, ev0), ('return', t_done0, t_sw, None), ('carry_r2', t_sw, 1e9, ev1))
out = dict(run=RUN, scene=r1.get('scene'), widths=r1.get('scene_widths'), corridor={}, return_states=rs)
k20 = np.abs(t * 20 - np.round(t * 20)) < 1e-6
for nm, a, b, ev in SEG:
    k = np.flatnonzero((t >= a) & (t < b) & ins & k20)
    if not len(k):
        out['corridor'][nm] = dict(reached=False)
        continue
    v = np.linalg.norm(np.diff(Q[k, :2], axis=0), axis=1) / np.maximum(np.diff(t[k]), 1e-6)
    v = v[np.diff(t[k]) < 0.06]
    d = dict(reached=True, t_in=round(float(t[k[0]]), 2), t_out=round(float(t[k[-1]]), 2), dur=round(float(t[k[-1]] - t[k[0]]), 2), v_mean=round(float(v.mean()), 3), v_min=round(float(v.min()), 3), v_max=round(float(v.max()), 3), v_b_mean=round(float(col['v_b'][k].mean()), 3))
    if ev:
        d.update(compact_reached=round(ev['compact_reached'], 2), raise_start=round(ev['compact_raise_start'], 2), out_start=round(ev['compact_out_start'], 2), in_after_compact_s=round(float(t[k[0]]) - ev['compact_reached'], 2), raise_after_out_s=round(ev['compact_raise_start'] - float(t[k[-1]]), 2), out_after_out_s=round(ev['compact_out_start'] - float(t[k[-1]]), 2))
    out['corridor'][nm] = d
W_ = {}
for j in np.flatnonzero(k20):
    x = Q[j]
    Pn, Pf = T.split(G.link_world(x, float(col['finger_cmd'][j])))
    st = int(col['f1_stage'][j])
    seg = 'carry_r1' if t[j] < t_done0 else ('return' if st == 0 else 'carry_r2')
    W_.setdefault(seg, {})
    upd = lambda k_, v_, seg_=seg, tt=float(t[j]): (W_[seg_].__setitem__(k_, (v_, round(tt, 2))) if v_ < W_[seg_].get(k_, (9.9, 0))[0] else None)   # noqa: E731
    upd('base_disc', min(float(LS.d2_box(x[:2], c_, s_)[0]) for c_, s_ in WALLS) - 0.39)
    upd('nonfinger', min(float(G.box_dist(Pn, c_, s_).min()) for c_, s_ in WALLS))
    upd('finger', min(float(G.box_dist(Pf, c_, s_).min()) for c_, s_ in WALLS))
    if seg != 'return' and np.isfinite(col['bottle_base_x'][j]):
        R_ = Rot.from_quat([col['bottle_qx'][j], col['bottle_qy'][j], col['bottle_qz'][j], col['bottle_qw'][j]]).as_matrix()
        P_b = (R_ @ BPT.T).T + np.array([col['bottle_base_x'][j], col['bottle_base_y'][j], col['bottle_base_z'][j]])
        upd('active_bottle', min(float(G.box_dist(P_b, c_, s_).min()) for c_, s_ in WALLS))
out['wall_clearance'] = {s_: {k_: dict(m=round(v_[0], 4), t=v_[1]) for k_, v_ in d_.items()} for s_, d_ in W_.items()}
out['forbidden_peak'] = [{k_: v_['peak'] for k_, v_ in r_['forbidden_pulses'].items() if v_['peak'] > 0} for r_ in (r0, r1)]
out['forbidden_keys'] = sorted(r0['forbidden_pulses'])
rows = [FD.load_rows(os.path.join(D, f_)) for f_ in ('s1_cbf_r0.npz', 's1_cbf.npz')]
for nm, L_, r_ in (('r1', rows[0], r0), ('r2', rows[1], r1)):
    tt = np.array([float(e['t']) for e in L_])
    eps = np.array([float(e['eps0']) for e in L_])
    wp = np.array([float(e['w_p']) for e in L_])
    ms = np.asarray(r_.get('solve_ms_all') or [], float)
    q_ = np.array([np.asarray(e['qp'], float)[:2] for e in L_])
    ic = LS.in_corridor(q_)
    d = dict(n=len(L_), eps_over=[(round(float(tt[i]), 2), round(float(eps[i]), 4), 'hold' if wp[i] == 0 else 'active', bool(ic[i])) for i in np.flatnonzero(eps > 1e-4)], min_h=round(float(np.min([float(e['min_h']) for e in L_])), 4), dyn_ticks=[round(float(e['t']), 2) for e in L_ if FD._dyn(e['obs'])][:3] + [sum(1 for e in L_ if FD._dyn(e['obs']))])
    if len(ms) == len(L_):
        i_o = np.flatnonzero(ms > 50.0)
        d.update(solve=dict(p50=round(float(np.percentile(ms, 50)), 2), p95=round(float(np.percentile(ms, 95)), 2), max=round(float(ms.max()), 2)), over50=[(round(float(tt[i]), 2), round(float(ms[i]), 1), 'hold' if wp[i] == 0 else 'active') for i in i_o])
    else:
        d['solve_len_mismatch'] = (len(ms), len(L_))
    out['solver_' + nm] = d
s_ = (r0.get('f1l') or {}).get('ret_solver') or {}
out['ret_solver'] = dict(n=s_.get('n'), n_fb=s_.get('n_fb'), n_fb_ok=s_.get('n_fb_ok'), total_ms=s_.get('total_ms'), n_over_50=s_.get('n_over_50'), fb=[{k_: (round(v_, 4) if isinstance(v_, float) else v_) for k_, v_ in x_.items() if k_ in ('t', 'gi_ok', 'gi_steps', 'osqp_ms', 'gi_ms', 'total_ms', 'accept', 'eps0', 'resid_noslack', 'min_h')} for x_ in s_.get('fb', [])])
out['profile'] = [((r_.get('place') or {}).get('carry') or {}).get('profile') for r_ in (r0, r1)]
out['carry_plan'] = [{k_: v_ for k_, v_ in (((r_.get('place') or {}).get('carry') or {}).items()) if k_ in ('turn_max_deg', 'conn_turn_deg', 'lateral', 'n')} for r_ in (r0, r1)]
out['shield'] = (r1.get('nav_chain') or {}).get('shield')
out['dynamic_dropouts'] = None
print(json.dumps(out, ensure_ascii=False, default=float))
json.dump(out, open(os.path.join(HERE, 'results', 'bottle', f'f1l_metrics_{RUN}.json'), 'w'), ensure_ascii=False, indent=1, default=float)
