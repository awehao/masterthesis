#!/usr/bin/env python3
"""door_alongpath_check 的延伸：經過把手前方（腕部奇異帶）時允許 TCP 目標姿態偏轉 ψ（繞把手接近軸 a_n 或繞世界鉛直軸 z），後退量 r 與 ψ 逐點自選，
要求同一分支連續（Δq＜0.3）、σ≥下限、限位、淨空；由窗口起點（ψ＝0、r＝S_PRE）往回延拓，往回走 |ψ| 與 r 只能不減（往前走只會收回）。
運動學層離線，只回答「沿導航門前線是否存在連續、不換分支的手臂排程」。
    python3 door_alongpath_orient.py RUN_DIR PROBE_JSON AXIS(an|z) OUT.json"""
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.argv_ = list(sys.argv); sys.argv = [sys.argv[0], sys.argv[1], '/dev/null']
import numpy as np
import door_turn_search as TS
sys.argv = sys.argv_
AX = sys.argv[3]
P = json.load(open(sys.argv[2]))
path = np.array([p[:2] for p in P[-1]['plan']], float) if 'start' in P[0] else np.array([p[:2] for p in P[0]['plan']], float)
rel = path - TS.h_xy
A, SB = -(rel @ TS.d), rel @ TS.n_
on = np.abs(SB - SB[-1]) < 0.03
sb_line = float(np.median(SB[on]))
a_w = -0.09
axis = TS.a_n / np.linalg.norm(TS.a_n) if AX == 'an' else np.array([0.0, 0.0, 1.0])


def rot(ax, th):
    ax = ax / np.linalg.norm(ax); K_ = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + math.sin(th) * K_ + (1 - math.cos(th)) * K_ @ K_


def target(r, psi):
    T = TS.target(r).copy()
    T[:3, :3] = rot(axis, psi) @ T[:3, :3]
    return T


def solve(b, r, psi, seed):
    q, res, sm = TS.DFK.ik_closure_fc(target(r, psi), b, seed)
    if not (res < 1e-6 and np.isfinite(q).all()):
        return None, None
    return q, sm


q_end = TS.end_config(a_w, sb_line)
R_GRID = np.round(np.arange(TS.S_PRE, 0.4001, 0.02), 3)
PSI = [math.radians(x) for x in (0, 10, -10, 20, -20, 30, -30, 45, -45, 60, -60)]
a_list = np.round(np.arange(a_w, 0.40 + 1e-9, 0.01), 3)
seed, r_prev, psi_prev, rows, fail = q_end.copy(), TS.S_PRE, 0.0, [], None
for a in a_list:
    b = TS.base(a, sb_line)
    best = None
    for psi in sorted(PSI, key=abs):
        if abs(psi) < abs(psi_prev) - 1e-9 or (psi_prev != 0 and psi * psi_prev < 0):
            continue
        for r in R_GRID[R_GRID >= r_prev - 1e-9]:
            q, sm = solve(b, float(r), psi, seed)
            if q is None or np.abs(q - seed).max() > 0.3 or sm < TS.SIG:
                continue
            lim = float(min((q - TS.LO).min(), (TS.HI - q).min()))
            cl = TS.W.clear_door(np.r_[b, q, 0.0], TS.T_H)
            if lim < 0 or any(cl[k] < TS.THRP[k] for k in ('env', 'base', 'self')):
                continue
            best = (float(r), psi, q, sm, lim, cl)
            break
        if best is not None:
            break
    if best is None:
        fail = dict(a=float(a), r_from=r_prev, psi_from_deg=round(math.degrees(psi_prev), 1)); break
    r, psi, q, sm, lim, cl = best
    rows.append(dict(a=float(a), r=r, psi_deg=round(math.degrees(psi), 1), sigma=round(float(sm), 4), lim=round(lim, 3), self=round(float(cl['self']), 4), vis=bool(TS.visible(np.r_[b, q])), q=[round(float(x), 4) for x in q]))
    seed, r_prev, psi_prev = q, r, psi
rows = rows[::-1]
Q = np.array([x['q'] for x in rows]) if rows else np.zeros((0, 6))
out = dict(axis=AX, sb_line=round(sb_line, 4), a_w=a_w, fail=fail, n=len(rows), reached_a=rows[0]['a'] if rows else None,
           schedule=[(x['a'], x['r'], x['psi_deg'], x['sigma'], x['vis']) for x in rows], sigma_min=min([x['sigma'] for x in rows] or [None]),
           joint_travel_max=round(float(np.abs(np.diff(Q, axis=0)).sum(axis=0).max()), 3) if len(Q) > 1 else None, rows=rows)
json.dump(out, open(sys.argv[4], 'w'), ensure_ascii=False, indent=1)
print('axis', AX, 'fail', fail, 'reached a', out['reached_a'], 'sigma_min', out['sigma_min'], 'travel', out['joint_travel_max'])
print('schedule (a, r, psi, sigma, vis):', out['schedule'][::4])
