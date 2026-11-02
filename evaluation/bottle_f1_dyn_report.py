#!/usr/bin/env python3
"""F1 返回段移動物的交互分析（Codex 20261103_044919；假世界與實跑共用，只讀求解紀錄）。三類分開報，不混成一項：
  tracked   被追蹤為動態：該拍障礙清單裡有動態項。
  in_solver 進入求解器：動態項在時域內離參考路徑 ≤ cbf_prune_range（_build_cbf_horizon 的距離閘沒略過未來步）。
  reaction  約束貼合／命令反應：k＝0 動態列殘差 ≤ 1e-6（貼合）；把動態項拿掉重解，底盤命令差 > 1e-3（命令反應）。重解只用紀錄輸入與手臂 hold。
分段：起動前（誤判）、返回段（其他相位的動態項也照記，不刪非預期交互）。
"""
import numpy as np

from ammr_wholebody_mpc import gmpc_wb as GW
from ammr_wholebody_mpc.se2 import from_xytheta

BIND_TOL, DU_TOL = 1e-6, 1e-3


def _dyn(obs):
    return [o for o in (obs or []) if not o.get('static', 'margin' in o)]


def analyse(rows, mpc, t_start, nav):
    """rows：求解紀錄（dict：t、obs、qp、u_prev、X_ref、xi_ref、u、eps0、resid_ns…）；mpc：導航 GMPC（手臂 hold 重解用）；t_start：障礙起動任務時刻；nav：(返回 NAV 起, 迄)。"""
    cfg = mpc.cfg
    N, dt = cfg.N, cfg.dt
    seg = {k: dict(n=0, n_tracked=0, n_in_solver=0, n_bound=0, n_react=0, t_first=None, t_last=None, du_max=0.0, eps0_max=0.0, d_center_min=9.9) for k in ('before_move', 'return_nav', 'other')}
    for e in rows:
        t = float(e['t'])
        k = 'before_move' if t < t_start else ('return_nav' if nav[0] <= t <= nav[1] else 'other')
        s = seg[k]
        s['n'] += 1
        D = _dyn(e['obs'])
        if not D:
            continue
        s['n_tracked'] += 1
        s['t_first'] = t if s['t_first'] is None else s['t_first']
        s['t_last'] = t
        X, xi = np.asarray(e['X_ref'], float), np.asarray(e['xi_ref'], float)
        p_now = np.asarray(e['qp'], float)[:2]
        dmin = 9.9
        for o in D:
            for j in range(0, N, 4):
                pk = p_now if j == 0 else X[j][:2, 2]
                dmin = min(dmin, float(np.hypot(pk[0] - (o['x'] + o.get('vx', 0.0) * j * dt), pk[1] - (o['y'] + o.get('vy', 0.0) * j * dt))))
        s['d_center_min'] = min(s['d_center_min'], round(dmin, 4))
        if cfg.cbf_prune_range <= 0.0 or dmin <= cfg.cbf_prune_range:
            s['n_in_solver'] += 1
        s['eps0_max'] = max(s['eps0_max'], float(e.get('eps0', 0.0)))
        Xn = from_xytheta(*np.asarray(e['qp'], float)[:3])
        A, l, _, _ = GW._build_cbf_horizon(cfg, Xn, X, xi, D, N)
        A = A.tocsr()
        u = np.asarray(e['u'], float)[:3]
        d0 = u - xi[0]
        for r in range(A.shape[0]):
            cols = A[r, N * 3:].nonzero()[1]
            if len(cols) and int(cols[0]) == 0:
                if float(A[r, :3] @ d0 - l[r]) <= BIND_TOL:
                    s['n_bound'] += 1
                    break
        if k == 'return_nav' or k == 'other':
            r_no = mpc.solve(obstacles=[o for o in (e['obs'] or []) if o.get('static', 'margin' in o)], X_now=Xn, X_ref_win=X, xi_ref_win=xi, xi_prev=np.asarray(e['u_prev'], float)[:3], arm=dict(mode='hold'))
            du = float(np.linalg.norm(u - np.asarray(r_no.u_opt, float)))
            s['du_max'] = max(s['du_max'], du)
            s['n_react'] += int(du > DU_TOL)
    return seg


def distances(t, Q, ca, state_fn, r_base=0.39):
    """實際軌跡到障礙表面：底盤中心、底盤圓盤。"""
    ob = np.array([state_fn(x, ca)[0] for x in t])
    d = np.hypot(Q[:, 0] - ob[:, 0], Q[:, 1] - ob[:, 1]) - ca['radius']
    i = int(np.argmin(d))
    return dict(base_center_surface=round(float(d[i]), 4), base_disc=round(float(d[i] - r_base), 4), at_t=round(float(t[i]), 2))
