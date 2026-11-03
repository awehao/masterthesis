#!/usr/bin/env python3
"""L 型走廊返回段求解未收斂的有界離線診斷（Codex 20261103_085730；不接控制器、不跑假世界、不動接受與停止規則）。
對兩個失敗拍與各自前兩拍，用紀錄的輸入照原設定重解，迭代上限比較 8000／20000／50000（只有上限不同），報：狀態、實際迭代數、原始／對偶殘差、耗時；後處理前的解與最終命令對速度框、加速度框、輪級、CBF、Shield 列的違反量。
假世界沒存掃描，Shield 列用當拍前最近 0.1 s 格點時刻的紀錄位姿重新射線，是近似；所以先和紀錄值比對，報重現差。迭代分布：既有紀錄沒有迭代數，不為此重解整趟。
    python3 evaluation/bottle_f1_lchan_osqp_diag.py
"""
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import linprog

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.argv = ['test_bottle_f1_task.py', 'nominal']
import test_bottle_f1_task as T                         # noqa: E402
import bottle_f1_dyn_report as FD                       # noqa: E402
import bottle_f1_lchan_scene as LS                      # noqa: E402
import bottle_nav_chain as NC                           # noqa: E402
import bottle_s1_cand as SC                             # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import bottle_w1_shrow_ctrl as SR                       # noqa: E402
from ammr_wholebody_mpc import gmpc_wb as GW            # noqa: E402
from ammr_wholebody_mpc.se2 import from_xytheta         # noqa: E402

REAL = GW.osqp.OSQP
LOG, CUR = [], dict(max_iter=8000)


class Spy(REAL):
    """只換主問題（手臂 hold：max_iter 8000）的迭代上限，並留下問題與結果；其餘設定不動。"""

    def setup(self, *a, **k):
        self._main = k.get('max_iter') == 8000
        if self._main:
            k['max_iter'] = CUR['max_iter']
            self._k = {n: k[n] for n in ('P', 'q', 'A', 'l', 'u', 'eps_abs', 'eps_rel', 'polish', 'max_iter')}
        return super().setup(*a, **k)

    def solve(self, *a, **k):
        r = super().solve(*a, **k)
        if self._main:
            i = r.info
            LOG.append(dict(status=str(i.status), iter=int(i.iter), pri_res=float(i.prim_res), dua_res=float(i.dual_res), run_ms=float(i.run_time) * 1e3, x=np.array(r.x, float), **self._k))
        return r


GW.osqp.OSQP = Spy
c, _ = T.build_ctrl()
cfg = c.mpc_nav.cfg
DT = cfg.dt
AGE_PAD = SR.AGE_PAD
SH = c.sh_rows
CASES = (('v1_test2', 'runs/lchan_fake_f1_lchannel_v1/s1_cbf_r0.npz', 'experiments/20261103_f1_lchannel_scene_v1_mismatch.json', 88.70),
         ('v2_test3', 'runs/lchan_fake_f1_lchannel_v2_test3/s1_cbf_r0.npz', 'experiments/20261103_f1_lchannel_scene_v2.json', 78.00))


def violations(u, xi_prev, A_row=None, rows=None):
    u = np.asarray(u, float)
    W = GW.wheel_matrix(cfg)
    wl, al = cfg.wheel_radius * cfg.wheel_w_max, cfg.wheel_radius * cfg.wheel_a_max * DT
    out = dict(vel=float(np.maximum(np.maximum(u - cfg.u_max, cfg.u_min - u), 0.0).max()),
               acc=float(np.maximum(np.abs(u - xi_prev) - np.asarray(cfg.a_max) * DT, 0.0).max()),
               wheel_vel=float(max(np.abs(W @ u).max() - wl, 0.0)), wheel_acc=float(max(np.abs(W @ (u - xi_prev)).max() - al, 0.0)))
    if rows is not None:
        out['shield'] = float(max((rows[0] @ u - rows[1]).max(), 0.0))
    return out


def cbf_resid(Xn, X, xi, obs, delta0, eps0, delta_all=None, eps_all=None, ns=None):
    A, l, _, _ = GW._build_cbf_horizon(cfg, Xn, X, xi, list(obs), ns)
    A = A.tocsr()
    N, Nm = cfg.N, cfg.N * 3
    out = dict(k0_noslack=None, k0_slack=None, all_noslack=None, all_slack=None)
    k0 = np.array([r for r in range(A.shape[0]) if A[r, :3].nnz], int)
    if len(k0):
        v = A[k0][:, :3] @ np.asarray(delta0, float) - l[k0]
        out['k0_noslack'] = float(v.min())
        out['k0_slack'] = float((v + float(eps0)).min())
    if delta_all is not None:
        z = np.zeros(A.shape[1])
        z[:Nm] = delta_all
        out['all_noslack'] = float((A[:, :Nm] @ z[:Nm] - l).min())
        if eps_all is not None:
            z[Nm:Nm + len(eps_all)] = eps_all
            out['all_slack'] = float((A @ z - l).min())
    return out


RES = dict(note='假世界沒存掃描，Shield 列是近似重建；既有紀錄沒有迭代數分布', cases={})
for nm, f, scn, t0 in CASES:
    Z = np.load(os.path.join(HERE, f))
    t = Z['t']
    i0 = int(np.argmin(np.abs(t - t0)))
    S_ = LS.load(os.path.join(HERE, scn), strict=False)
    scene = SC.scene_boxes() + [(c_ + np.array([S_['table2_x'], -1.0, 0.0]), sz) for n_, c_, sz in G.table_boxes() if n_ != 'table_top'] + [(c_, s_) for _, c_, s_ in S_['walls']]
    tick_out = []
    for i in (i0 - 2, i0 - 1, i0):
        e = dict(t=float(t[i]), obs=json.loads(str(Z['obs_json'][i])), qp=Z['qp'][i], u_prev=Z['u_prev'][i], X_ref=Z['X_ref'][i], xi_ref=Z['xi_ref'][i], u=Z['u'][i], eps0=float(Z['eps0'][i]), resid_ns=float(Z['resid_ns'][i]), resid_s=float(Z['resid_s'][i]))
        Xn = from_xytheta(*np.asarray(e['qp'], float)[:3])
        ts = np.floor(e['t'] * 10.0 + 1e-6) / 10.0
        j = int(np.argmin(np.abs(t - ts)))
        q_s = Z['qp'][j]
        rr = NC.ray_boxes(q_s[:2], float(q_s[2]), scene, G.LIDAR_Z)
        rows = SR.shield_rows(SH, SR.scan_points(SH, rr), (e['t'] - ts) + AGE_PAD)
        xi0, up = np.asarray(e['xi_ref'], float)[0], np.asarray(e['u_prev'], float)[:3]
        rec = dict(t=round(e['t'], 2), xy=np.round(e['qp'][:2], 3).tolist(), yaw=round(float(e['qp'][2]), 3), logged=dict(u=np.asarray(e['u'], float)[:3].tolist(), eps0=e['eps0'], resid_noslack=e['resid_ns'], resid_slack=e['resid_s']), n_obs=len(e['obs']), n_dyn=len(FD._dyn(e['obs'])), shield_rows=int(rows[2]['n_rows']), runs={})
        for mi in (8000, 20000, 50000):
            CUR['max_iter'] = mi
            LOG.clear()
            r = c.mpc_nav.solve(obstacles=list(e['obs']), X_now=Xn, X_ref_win=e['X_ref'], xi_ref_win=e['xi_ref'], xi_prev=up, arm=dict(mode='hold'), u0_rows=rows[:2])
            L = LOG[-1]
            if mi == 8000:
                rec_L8000 = L
            Nm = cfg.N * 3
            ns = len(L['x']) - Nm
            d_all = L['x'][:Nm]
            eps_all = L['x'][Nm:]
            u_raw = xi0 + d_all[:3]
            Ax = L['A'] @ L['x']
            prim = float(np.maximum(np.maximum(L['l'] - Ax, Ax - L['u']), 0.0).max())
            u_fin = np.asarray(r.u_opt, float)
            obj = 0.5 * float(L['x'] @ (L['P'] @ L['x'])) + float(L['q'] @ L['x'])
            rec['runs'][str(mi)] = dict(status=L['status'], iter=L['iter'], pri_res=L['pri_res'], dua_res=L['dua_res'], run_ms=round(L['run_ms'], 2), prim_viol_all_rows=prim, objective=obj, n_var=int(len(L['x'])), n_con=int(L['A'].shape[0]), nnz_A=int(L['A'].nnz), eps=L['eps_abs'], polish=bool(L['polish']), gmpc_status=str(r.status), accept=str(r.accept_action), accept_scale=float(r.accept_scale),
                                        raw=dict(u=u_raw.tolist(), viol=violations(u_raw, up, rows=rows[:2]), cbf=cbf_resid(Xn, e['X_ref'], e['xi_ref'], e['obs'], d_all[:3], eps_all[0] if ns else 0.0, d_all, eps_all if ns else None, ns or cfg.N)),
                                        final=dict(u=u_fin.tolist(), viol=violations(u_fin, up, rows=rows[:2]), cbf=cbf_resid(Xn, e['X_ref'], e['xi_ref'], e['obs'], u_fin - xi0, float(r.eps0), None, None, ns or cfg.N), eps0=float(r.eps0)),
                                        repro_u_err=float(np.abs(u_fin - np.asarray(e['u'], float)[:3]).max()))
        # 同一個 QP 用手臂主動分支既有的對偶作用集法求一次（只讀比較，不接控制器）：看問題有沒有解、最佳解要多少鬆弛
        CUR['max_iter'] = 8000
        L0 = rec_L8000
        t_g = time.perf_counter()
        gx, gy, gi = GW.qp_dual_active(L0['P'], L0['q'], L0['A'], L0['l'], L0['u'])
        ms_g = (time.perf_counter() - t_g) * 1e3
        rec['gi'] = dict(why=gi.get('why'), steps=gi.get('steps', gi.get('s')), ms=round(ms_g, 2), solved=gx is not None)
        if gx is not None:
            Nm_ = cfg.N * 3
            Ax_ = L0['A'] @ gx
            ug_ = xi0 + gx[:3]
            rec['gi'].update(u=ug_.tolist(), prim_viol_all_rows=float(np.maximum(np.maximum(L0['l'] - Ax_, Ax_ - L0['u']), 0.0).max()), objective=0.5 * float(gx @ (L0['P'] @ gx)) + float(L0['q'] @ gx), eps0=float(gx[Nm_]), eps_max=float(gx[Nm_:].max()), eps_nonzero=[[int(k_), float(v_)] for k_, v_ in enumerate(gx[Nm_:]) if v_ > 1e-6][:12], viol=violations(ug_, up, rows=rows[:2]), du_from_prev=(ug_ - up).tolist(), acc_budget=(np.asarray(cfg.a_max) * DT).tolist())
        # 第一拍硬限制（速度框、加速度框、輪級、Shield）下，k＝0 CBF 列最少還差多少：線性規劃，值 > 0 表示第一拍一定要用鬆弛
        A0, l0, _, _ = GW._build_cbf_horizon(cfg, Xn, e['X_ref'], e['xi_ref'], list(e['obs']), cfg.N)
        A0 = A0.tocsr()
        k0_ = np.array([r_ for r_ in range(A0.shape[0]) if A0[r_, :3].nnz], int)
        Ak, lk = A0[k0_][:, :3].toarray(), l0[k0_]
        W_ = GW.wheel_matrix(cfg)
        wl_, al_ = cfg.wheel_radius * cfg.wheel_w_max, cfg.wheel_radius * cfg.wheel_a_max * DT
        Aub = [np.c_[-Ak, -np.ones(len(Ak))]]                 # −A·δ − s ≤ −l
        bub = [-lk]
        for M_, b_ in ((W_, wl_ - W_ @ xi0), (-W_, wl_ + W_ @ xi0), (W_, al_ - W_ @ (xi0 - up)), (-W_, al_ + W_ @ (xi0 - up)), (rows[0], rows[1] - rows[0] @ xi0)):
            Aub.append(np.c_[M_, np.zeros(len(M_))])
            bub.append(b_)
        amax_ = np.asarray(cfg.a_max) * DT
        bd = [(max(float(cfg.u_min[a_]), float(up[a_] - amax_[a_])) - float(xi0[a_]), min(float(cfg.u_max[a_]), float(up[a_] + amax_[a_])) - float(xi0[a_])) for a_ in range(3)] + [(0.0, None)]
        lp = linprog([0.0, 0.0, 0.0, 1.0], A_ub=np.vstack(Aub), b_ub=np.concatenate(bub), bounds=bd, method='highs')
        rec['k0_need'] = dict(status=int(lp.status), slack_min=None if lp.x is None else float(lp.x[3]), n_k0_rows=int(len(k0_)))
        # 這拍相對前一拍新增的靜態項，與第一拍命令沿用前一筆時各列的缺口
        prev_ = json.loads(str(Z['obs_json'][i - 1]))
        key = lambda o_: (round(float(o_['x']), 3), round(float(o_['y']), 3))   # noqa: E731
        new_ = [o_ for o_ in e['obs'] if key(o_) not in {key(p_) for p_ in prev_}]
        p_ = np.asarray(e['qp'], float)[:2]
        rec['obs_new'] = [dict(x=round(float(o_['x']), 3), y=round(float(o_['y']), 3), radius=o_.get('radius'), margin=o_.get('margin'), d=round(float(np.hypot(o_['x'] - p_[0], o_['y'] - p_[1])), 3)) for o_ in new_]
        rec['obs_gone'] = [dict(x=round(float(o_['x']), 3), y=round(float(o_['y']), 3)) for o_ in prev_ if key(o_) not in {key(q_) for q_ in e['obs']}]
        gap_ = Ak @ (up - xi0) - lk
        rec['k0_gap_at_prev_u'] = dict(min=float(gap_.min()), n_neg=int((gap_ < 0).sum()))
        rec['on_scan_grid'] = bool(abs(e['t'] * 10.0 - round(e['t'] * 10.0)) < 1e-6)
        tick_out.append(rec)
    RES['cases'][nm] = dict(file=f, scene=S_['name'], ticks=tick_out)
print(json.dumps(RES, ensure_ascii=False, default=float))
json.dump(RES, open(os.path.join(HERE, 'results', 'bottle', 'f1_lchan_osqp_diag.json'), 'w'), ensure_ascii=False, indent=1, default=float)
