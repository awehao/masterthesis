#!/usr/bin/env python3
"""F1 返回段移動物的交互分析（Codex 20261103_044919、055608；假世界與實跑共用，只讀求解紀錄）。以下各類分開報，不混成一項：
  n_tracked      動態清單存在：該拍障礙清單裡有動態項。
  n_rows_k0      k＝0 動態列保留：_build_cbf_horizon 實際建出的列；距離閘判定超出裁剪範圍時，k＝0 列仍保留。
  n_rows_future  未來步動態列保留：同上，k≥1 的列有建出（未被距離閘裁掉）。
  n_bound_k0     k＝0 列貼合：含鬆弛的列殘差絕對值 |A·δ₀＋ε₀−l| ≤ 容差。n_bound_k0_strict 用 1e-6；n_bound_k0 用單列估計的求解器容差 1e-5·(1＋|l|)。ε₀ 為紀錄值（靜態與動態兩塊的較大者）。
  n_tight_future 未來列緊：可比重解的 δ_k 對未來列 A·δ_k−l ≤ 同容差。各步鬆弛未回傳；鬆弛有懲罰，未加鬆弛殘差為負表示該列已用鬆弛而必為作用中，所以這裡不取絕對值。
  n_comparable   可比重解拍：手臂 hold（w_p＝0）；完整清單與拿掉動態項兩次重解都收斂（status＝solved）且未退回（≠brake）；完整重解重現紀錄命令（逐軸 ≤1e-6）。不可比的拍依原因分列在 not_comparable，不算命令反應。
  n_react        可比重解的命令差：底盤命令逐軸差（機體系）任一軸 > 1e-3（vx、vy 為 m/s，ωz 為 rad/s）。n_react_k0_only＝該拍沒有未來列；n_react_future＝有未來列。
  du_*_max       逐軸最大差與平移合成 du_v_max（m/s）、du_w_max（rad/s）；不混三軸。
分段：起動前（誤判）、返回 NAV 段、其他（其他相位的動態項也照記，不刪非預期交互）。手臂非 hold 的拍只統計列，不做 hold 重解。
"""
import json

import numpy as np

from ammr_wholebody_mpc import gmpc_wb as GW
from ammr_wholebody_mpc.se2 import from_xytheta

BIND_STRICT, EPS_SOLVER, DU_TOL, REPRO_TOL = 1e-6, 1e-5, 1e-3, 1e-6


def _dyn(obs):
    return [o for o in (obs or []) if not o.get('static', 'margin' in o)]


def _static(obs):
    return [o for o in (obs or []) if o.get('static', 'margin' in o)]


def load_rows(path):
    """讀 save_compact 的 npz，還原成 analyse 用的 rows。"""
    Z = np.load(path)
    keys = [k for k in Z.files if k != 'obs_json']
    return [dict({k: Z[k][i] for k in keys}, obs=json.loads(str(Z['obs_json'][i]))) for i in range(len(Z['t']))]


def _new():
    return dict(n=0, n_tracked=0, n_rows_k0=0, n_rows_future=0, n_bound_k0=0, n_bound_k0_strict=0, n_tight_future=0, n_comparable=0, not_comparable=dict(not_hold=0, unsolved=0, brake=0, not_reproduced=0),
                n_react=0, n_react_k0_only=0, n_react_future=0, t_first=None, t_last=None, du_vx_max=0.0, du_vy_max=0.0, du_v_max=0.0, du_w_max=0.0, repro_max=0.0, eps0_max=0.0, d_center_min=9.9, resid_k0_min=None)


def analyse(rows, mpc, t_start, nav):
    """rows：求解紀錄（dict：t、obs、qp、u_prev、X_ref、xi_ref、u、eps0、w_p…）；mpc：導航 GMPC（手臂 hold 重解用）；t_start：障礙起動任務時刻；nav：(返回 NAV 起, 迄)。"""
    cfg = mpc.cfg
    N, dt = cfg.N, cfg.dt
    seg = {k: _new() for k in ('before_move', 'return_nav', 'other')}
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
        eps0 = float(e.get('eps0', 0.0))
        s['eps0_max'] = max(s['eps0_max'], eps0)
        Xn = from_xytheta(*np.asarray(e['qp'], float)[:3])
        A, l, _, _ = GW._build_cbf_horizon(cfg, Xn, X, xi, D, N)
        A = A.tocsr()
        step = np.array([int(A[r, N * 3:].nonzero()[1][0]) for r in range(A.shape[0])], dtype=int)
        r0, rf = np.flatnonzero(step == 0), np.flatnonzero(step > 0)
        s['n_rows_k0'] += int(len(r0) > 0)
        s['n_rows_future'] += int(len(rf) > 0)
        u = np.asarray(e['u'], float)[:3]
        d0 = u - xi[0]
        res0 = np.array([float(A[r, :3] @ d0) + eps0 - float(l[r]) for r in r0])
        if len(res0):
            s['resid_k0_min'] = round(float(np.abs(res0).min()), 9) if s['resid_k0_min'] is None else min(s['resid_k0_min'], round(float(np.abs(res0).min()), 9))
            s['n_bound_k0_strict'] += int(np.any(np.abs(res0) <= BIND_STRICT))
            s['n_bound_k0'] += int(np.any(np.abs(res0) <= EPS_SOLVER * (1.0 + np.abs(l[r0]))))
        if float(e.get('w_p', 1.0)) != 0.0:
            s['not_comparable']['not_hold'] += 1
            continue
        kw = dict(X_now=Xn, X_ref_win=X, xi_ref_win=xi, xi_prev=np.asarray(e['u_prev'], float)[:3], arm=dict(mode='hold'))
        r_f, r_n = mpc.solve(obstacles=list(e['obs'] or []), **kw), mpc.solve(obstacles=_static(e['obs']), **kw)
        if r_f.status != 'solved' or r_n.status != 'solved':
            s['not_comparable']['unsolved'] += 1
            continue
        if r_f.accept_action == 'brake' or r_n.accept_action == 'brake':
            s['not_comparable']['brake'] += 1
            continue
        rep = float(np.abs(u - np.asarray(r_f.u_opt, float)).max())
        s['repro_max'] = max(s['repro_max'], rep)
        if rep > REPRO_TOL:
            s['not_comparable']['not_reproduced'] += 1
            continue
        s['n_comparable'] += 1
        if len(rf):
            dz = np.asarray(r_f.delta_xi_all, float)
            res_f = np.array([float(A[r, 3 * step[r]:3 * step[r] + 3] @ dz[step[r]]) - float(l[r]) for r in rf])
            s['n_tight_future'] += int(np.any(res_f <= EPS_SOLVER * (1.0 + np.abs(l[rf]))))
        du = u - np.asarray(r_n.u_opt, float)
        s['du_vx_max'] = max(s['du_vx_max'], abs(float(du[0])))
        s['du_vy_max'] = max(s['du_vy_max'], abs(float(du[1])))
        s['du_v_max'] = max(s['du_v_max'], float(np.hypot(du[0], du[1])))
        s['du_w_max'] = max(s['du_w_max'], abs(float(du[2])))
        if np.abs(du).max() > DU_TOL:
            s['n_react'] += 1
            s['n_react_future' if len(rf) else 'n_react_k0_only'] += 1
    return seg


def distances(t, Q, ca, state_fn, r_base=0.39):
    """實際軌跡到障礙表面：底盤中心、底盤圓盤。"""
    ob = np.array([state_fn(x, ca)[0] for x in t])
    d = np.hypot(Q[:, 0] - ob[:, 0], Q[:, 1] - ob[:, 1]) - ca['radius']
    i = int(np.argmin(d))
    return dict(base_center_surface=round(float(d[i]), 4), base_disc=round(float(d[i] - r_base), 4), at_t=round(float(t[i]), 2))
