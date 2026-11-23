#!/usr/bin/env python3
"""gmpc_wb.qp_dual_active（Goldfarb–Idnani）單元測試：對照 OSQP 精確解，並驗不可行判定、等式列、無作用列、P 非正定、數值縮放差異大的問題。
    python3 evaluation/test_qp_dual_active.py
"""
import os
import sys

import numpy as np
import osqp
from scipy import sparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc.gmpc_wb import qp_dual_active as G, qp_kkt_check as CK     # noqa: E402

n_ok = 0


def chk(name, c, info=''):
    global n_ok
    if not c:
        raise AssertionError(f'{name} {info}')
    n_ok += 1


def osqp_ref(P, q, A, l, u):
    p = osqp.OSQP()
    p.setup(P=P, q=q, A=A, l=l, u=u, verbose=False, eps_abs=1e-9, eps_rel=1e-9, polish=True, max_iter=400000, scaling=25)
    return p.solve()


def f(P, q, x):
    return 0.5 * x @ (P @ x) + q @ x


rng = np.random.default_rng(7)
worst = 0.0
for trial in range(60):
    n, m = int(rng.integers(5, 60)), int(rng.integers(5, 200))
    M = rng.normal(size=(n, n))
    d = 10.0 ** rng.uniform(-1, 6, n) if trial % 2 else np.ones(n)     # 奇數題：對角跨 7 個數量級（同 CBF slack）
    P = sparse.csc_matrix(M.T @ M * 1e-2 + np.diag(d))
    q = rng.normal(size=n) * 10.0
    Ad = rng.normal(size=(m, n))
    x0 = rng.normal(size=n) * 0.1                                      # 保證可行
    z0 = Ad @ x0
    l, u = z0 - rng.uniform(0.0, 0.2, m), z0 + rng.uniform(0.0, 0.2, m)
    l[rng.uniform(size=m) < 0.2] = -np.inf
    u[rng.uniform(size=m) < 0.2] = np.inf
    A = sparse.csc_matrix(Ad)
    x, y, info = G(P, q, A, l, u)
    chk(f'隨機題 {trial} 解出', x is not None, str(info))
    z = Ad @ x
    chk(f'隨機題 {trial} 原始可行', np.all(z >= l - 1e-8) and np.all(z <= u + 1e-8))
    r = osqp_ref(P, q, A, l, u)
    if r.info.status == 'solved':
        gap = f(P, q, x) - f(P, q, r.x)
        worst = max(worst, abs(gap) / max(1.0, abs(f(P, q, r.x))))
        chk(f'隨機題 {trial} 目標不劣於 OSQP', gap <= 1e-6 * max(1.0, abs(f(P, q, r.x))), str(gap))

P2, q2 = sparse.identity(3, format='csc'), np.array([1.0, -2.0, 0.5])
A2 = sparse.csc_matrix(np.eye(3))
x, _, info = G(P2, q2, A2, np.full(3, -np.inf), np.full(3, np.inf))
chk('無約束＝−P⁻¹q', np.allclose(x, -q2), str(x))
x, _, info = G(P2, q2, A2, np.full(3, -10.0), np.full(3, 10.0))
chk('約束不作用', info['n_active'] == 0 and np.allclose(x, -q2))
A3 = sparse.csc_matrix(np.array([[1.0, 1.0, 0.0]]))
x, y, info = G(P2, q2, A3, np.array([1.0]), np.array([1.0]))
chk('等式列（l＝u）', abs(x[0] + x[1] - 1.0) < 1e-9, str(x))
A4 = sparse.csc_matrix(np.array([[1.0, 0, 0], [1.0, 0, 0]]))
x, y, info = G(P2, q2, A4, np.array([1.0, 2.0]), np.array([np.inf, np.inf]))
chk('重複方向取較緊者', abs(x[0] - 2.0) < 1e-9, str(x))
x, y, info = G(P2, q2, A4, np.array([2.0, -np.inf]), np.array([np.inf, 1.0]))
chk('矛盾約束回不可行', x is None and info['why'] == 'infeasible', str(info))
A5 = sparse.csc_matrix(np.array([[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0]]))
x, y, info = G(P2, q2, A5, np.array([0.5, -np.inf, -np.inf]), np.array([np.inf, -0.5, 0.0]))
chk('正反列同時作用', x is not None and abs(x[0] - 0.5) < 1e-9, str(info))
Pn = sparse.csc_matrix(np.diag([1.0, -1.0, 1.0]))
x, y, info = G(Pn, q2, A2, np.full(3, -1.0), np.full(3, 1.0))
chk('P 非正定不回傳解', x is None and info['why'] == 'P_not_pd', str(info))
# qp_kkt_check 負測：好解通過；各類破壞都被擋下（不接受可疑解）
Pg = sparse.csc_matrix(np.diag([2.0, 3.0]))
qg = np.array([-4.0, 1.0])
Ag = sparse.csc_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))
lg, ug = np.array([-np.inf, -1.0]), np.array([1.0, np.inf])
xg, yg, ig = G(Pg, qg, Ag, lg, ug)
chk('KKT 好解通過', ig['ok'] is True and CK(Pg, qg, Ag, lg, ug, xg, yg)['ok'], str(ig))
chk('KKT 上界作用乘子為正', yg[0] > 0.0, str(yg))
bad = dict(nan_x=(xg * np.array([np.nan, 1.0]), yg, 'nonfinite'), inf_y=(xg, yg * np.array([np.inf, 1.0]), 'nonfinite'), none_x=(None, yg, 'nonfinite'),
           primal=(xg + np.array([1e-3, 0.0]), yg, 'primal'), stat=(xg - np.array([0.0, 1e-3]), yg, 'stationarity'),
           comp=(np.array([0.5, xg[1]]), np.array([yg[0] - 3.0 * 0.5 * 0 - 2.0 * (0.5 - xg[0]), yg[1]]), 'complementarity'))
for nm, (bx, by, want) in bad.items():
    r_ = CK(Pg, qg, Ag, lg, ug, bx, by)
    chk(f'KKT 破壞 {nm} 被擋', r_['ok'] is False and r_['why'] == want, str(r_))
r_ = CK(Pg, qg + np.array([2.0 * yg[0], 0.0]) * 0, Ag, lg, ug, xg, np.array([yg[0], -1e-3 * 0 + 1e-3]))
chk('KKT 無界側乘子被擋', r_['ok'] is False, str(r_))
print(f'qp_dual_active {n_ok} 項：全過；對 OSQP 相對目標差峰 {worst:.1e}')
