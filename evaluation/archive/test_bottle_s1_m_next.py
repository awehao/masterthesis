#!/usr/bin/env python3
"""M 組一拍續行保護的針對性測試（Codex 20261101_001552）：當下停止點通過、續行一拍停止點跨線時必須拒絕；再跑 M 名目一次。
    python3 evaluation/test_bottle_s1_m_next.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bottle_s1_cand as SC                             # noqa: E402
import bottle_s1_ctrl as S1                             # noqa: E402
import bottle_s1_groups as GR                           # noqa: E402
import bottle_s1_groups_check as CK                     # noqa: E402
import test_bottle_s0b as T                             # noqa: E402

KW = dict(cls=GR.S1MotM, kp=4.0)


def case(d_rest, d_next):
    c = SC.make_lidar(**KW)
    c.p1.mu_post = 0.01
    c.t, c.st.phase, c.T_b_now = 20.0, 'INSERT', T.bottle_T()
    q = np.r_[-0.2, 0.0, 0.0, T.Q_READY]
    u = np.r_[0.05, 0.0, 0.0, 0.1, -0.1, 0.1, 0.0, 0.0, 0.0]
    c.base_plan, c.u_arm_plan, c.u_last = np.stack([u[:3], u[:3]]), np.stack([u[3:], u[3:]]), u
    qp, q_rest, q_next = c._stop_rest(q)
    moved = float(np.abs(q_next - q_rest).max())
    tag = {id(qp): 0.009, id(q_rest): d_rest, id(q_next): d_next}
    c._stop_rest = lambda q_: (qp, q_rest, q_next)
    c._bottle_clear = lambda q9, Tb: tag.get(id(q9), 0.010)
    c._clear_check(T.K.fk(np.r_[0.0, 0.0, 0.0, T.Q_READY], 'link_tcp') * 0 + np.r_[np.c_[np.eye(3), [0.0, 0.42, 0.70]], [[0, 0, 0, 1]]], q)
    return c, moved


c, moved = case(0.007, 0.0052)
assert moved > 1e-3, f'下一拍停止點未前推 {moved}'
assert (c.judge.stop or {}).get('why') == 'clearance_bottle', str(c.judge.stop)
print(f'ok 當下 7.0 通過、下一拍 5.2 跨 5.5 線 ⇒ 拒絕；下一拍前推 {moved * 1e3:.2f} mrad／mm')
c, _ = case(0.007, 0.0060)
assert c.judge.stop is None, str(c.judge.stop)
print('ok 下一拍 6.0 未跨線 ⇒ 不拒絕')
c = SC.make_lidar(**KW)
c.p1.mu_post = 0.01
f = CK.fake(c, 45.0)
CK.summary('M 名目（kp 4、mu 0.01）', c, f)
print('瓶淨空', {k: None if v is None else round(v * 1e3, 2) for k, v in c.report()['clear_bottle'].items() if k in ('meas_min', 'pred_min', 'next_min')}, '成功', c.report()['success'], S1.exit_code(c.report()))
