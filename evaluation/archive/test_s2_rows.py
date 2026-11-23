#!/usr/bin/env python3
"""S2 新約束最小作用模式驗收（Codex 20261101_003315）：點 Jacobian 對有限差分、含第一個未來步、違反新列不得採用、不可行走停止。再跑名目一次。
    python3 evaluation/test_s2_rows.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bottle_s1_cand as SC                             # noqa: E402
import bottle_s1_ctrl as S1                             # noqa: E402
import bottle_s1_groups_check as CK                     # noqa: E402
import bottle_s2_check as S2                            # noqa: E402
import test_bottle_s0b as T                             # noqa: E402
from ammr_wholebody_mpc import gmpc_wb as GW            # noqa: E402

K = T.K
fc = GW.FastChain.of(K, 'link_tcp')
rng = np.random.default_rng(3)
worst = 0.0
for _ in range(40):
    q = np.r_[rng.uniform(-1, 1, 2), rng.uniform(-3, 3), rng.uniform(-2.5, 2.5, 6)]     # 偏航非零
    off = rng.uniform(-0.15, 0.15, 3)                                                   # TCP 偏移非零
    Tc, J = fc.fk_jac(q)
    p, Jp = GW.point_jac(Tc, J, off)
    Jn = np.zeros((3, 9))
    for i in range(9):
        d = np.zeros(9)
        d[i] = 1e-6
        Jn[:, i] = (GW.point_jac(*fc.fk_jac(q + d), off)[0] - GW.point_jac(*fc.fk_jac(q - d), off)[0]) / 2e-6
    worst = max(worst, float(np.abs(Jp - Jn).max()))
assert worst < 1e-7, worst
print(f'ok 點 Jacobian 對有限差分 誤差峰 {worst:.1e}（含偏航與 TCP 偏移）')

# 含第一個未來步：列出 solve 內取樣的 k
src = open(GW.__file__, encoding='utf-8').read()
assert "for k in range(0, N, bo.get('stride', 2)):" in src
print('ok 預測取樣含第一個未來控制步')

# 違反新列不得採用：讓 GI 解不含新列的問題，回傳核對須擋下
g0 = GW.qp_dual_active
n_trip = []


def g_drop(P, q, A, l, u, **k):
    x, y, i = g0(P, q, A[:-n_trip[0]] if n_trip else A, l[:-n_trip[0]] if n_trip else l, u[:-n_trip[0]] if n_trip else u, **k)
    if n_trip and y is not None:
        y = np.r_[y, np.zeros(n_trip[0])]
    return x, y, i


c = SC.make_lidar(body_boxes=[S2.S2_BOX])
sv = GW.GMPC.solve


def sv2(self, *a, **k):
    arm = k.get('arm') or {}
    bo = arm.get('body_obst') if isinstance(arm, dict) else None
    return sv(self, *a, **k)


f = T.Fake(c, x0=-0.2)
c.t = 0.0
c.st.phase, c.st.t_phase, c.st.grasp_ok, c.blending = 'CARRY', 0.0, True, True
q9 = np.r_[0.45, 0.0, 0.0, T.Q_READY]
c.st.T_carry_rel = T.K.fk(np.r_[0.0, 0.0, 0.0, T.Q_READY], 'link_tcp')
c.judge.T_rel = np.eye(4)
box = (T.K.fk(q9, 'link_tcp')[:3, 3] + np.array([0.12, 0.0, 0.0]), np.array([0.06, 0.4, 0.4]))
c.body_boxes = [box]
GW.qp_dual_active = lambda P, q, A, l, u, **k: g0(P, q, A[:-1], l[:-1], u[:-1], **k)[:2] + (dict(ok=True, solver='gi'),) if A.shape[0] > 700 else g0(P, q, A, l, u, **k)
GW.qp_dual_active = g0
print('ok 違反列測試見下')

# 不可行走停止：邊距大到不可能滿足
c = SC.make_lidar(body_boxes=[S2.S2_BOX], d_body=0.8)
f = CK.fake(c, 32.0)
r = c.report()
print('不可行案', r['stop'], '停止有效', r['stop_effect'], r['gi_fail'])
assert str((r['stop'] or {}).get('why')).startswith('wb_fallback') and r['success'] is False and (r['stop_effect'] or {}).get('ok') is True
print('ok 不可行 ⇒ 走停止、不報成功、停止有效')

# 違反新列：實際手法——包住 qp_dual_active，把解換成未加新列的解
class Tamper:
    pass


c = SC.make_lidar(body_boxes=[S2.S2_BOX])
real = GW.qp_dual_active


def tamper(P, q, A, l, u, **k):
    m = (np.isfinite(l) & ~np.isfinite(u))
    x, y, i = real(P, q, A, l, u, **k)
    return x, y, i


hits = []
orig_solve = GW.GMPC.solve


def wrapped(self, *a, **k):
    r_ = orig_solve(self, *a, **k)
    return r_


info = {}


def fake_gi(P, q, A, l, u, **k):
    x, y, i = real(P, q, A, l, u, **k)
    if x is not None and info.get('on'):
        z = A @ x
        act = np.where((z - l < 1e-9) & ~np.isfinite(u))[0]
        if act.size:                                            # 把解推出下界（模擬求解器回傳違反新列的解）
            x = x - 1e-3 * np.asarray(A[act[-1]].todense()).ravel() / max(float(np.linalg.norm(A[act[-1]].toarray())) ** 2, 1e-9)
            hits.append(1)
    return x, y, i


GW.qp_dual_active = fake_gi
f = T.Fake(c, x0=SC.X0)
f.q[3:] = np.array(SC.Q_TUCK) + T.D_TRUE


def fault(t, s):
    info['on'] = s.c.st.phase == 'CARRY'
    return (s.c._trigger_stop('sim_limit') if t >= 29.0 else None) or {}


f.run(32.0, fault)
GW.qp_dual_active = real
print('違反注入次數', len(hits), '第一阻點', (c.judge.stop or {}).get('why'))
assert len(hits) > 0 and str((c.judge.stop or {}).get('why')).startswith('wb_fallback'), str(c.judge.stop)
print('ok 回傳解違反新列 ⇒ 不採用、走停止')

c = SC.make_lidar(body_boxes=[S2.S2_BOX])
f = CK.fake(c, 50.0)
r = CK.summary('S2 名目', c, f)
print(' ', S2.box_min(f, S2.S2_BOX), '列峰', r['body_rows_max'], '成功', r['success'], S1.exit_code(r))
