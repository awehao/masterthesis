#!/usr/bin/env python3
"""瓶子對照 C1：**MotM 式移植基線**（Codex 20261031_165248 §3；不稱完整重現原方法）。
以 wg4b_b1_core（單步全身 QP）為底，只改兩處：
  1. 任務項速度目標 e ＝ v_ff ＋ kp·clip(誤差)：v_ff 為參考軌跡的**世界座標**末端線／角速度（與 J_w 輸出、rot_error 同座標；
     J_b＝J_w·B(θ) 只把本體輸入映到世界末端速度，末端輸出不是本體座標 ⇒ v_ff 不再旋轉）。
  2. 速度限制改用 cfg.vbox()（與 MPC 組同一實際限制）；成本正規化（w_vref 的 1/vmax²、阻尼權重）保留既定尺度。
名目姿態改用瓶子任務的備妥姿態（B1Params.q_pref 由呼叫端給），不用抽屜姿態。
未重現的原方法元件（Burgess-Limerick 等 Reactive MotM）：其 QP 的可操作度最大化項、關節限位避讓的原式、PBS 切換門檻與原平台速度介面；
底盤以定速沿路徑（不含其 STAA* 避障與通過位姿選擇）；不做延遲補償（原方法以當下狀態反應式控制）。
"""
from __future__ import annotations

import contextlib
import io
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))
import wg4b_b1_core as B1                                              # noqa: E402
from ammr_wholebody_mpc.arm_pregrasp import rot_error                 # noqa: E402
from ammr_wholebody_mpc.wgmpc_core import body_to_world, so3_log      # noqa: E402


def ref_twist(T_fn, t, h=1e-3):
    """參考位姿的**世界座標**線／角速度（中央差分）：v＝(p(t+h)−p(t−h))/2h，ω＝R(t)·log(R(t−h)ᵀR(t+h))/2h。"""
    Ta, Tb, T0 = T_fn(t - h), T_fn(t + h), T_fn(t)
    v = (Tb[:3, 3] - Ta[:3, 3]) / (2 * h)
    w = T0[:3, :3] @ so3_log(Ta[:3, :3].T @ Tb[:3, :3]) / (2 * h)
    return np.r_[v, w]


def task_terms_ff(K, q0, T_des, v_ff, tcp, p: B1.B1Params):
    T = K.fk(q0, tcp)
    e_p = T_des[:3, 3] - T[:3, 3]
    e_r = rot_error(T[:3, :3], T_des[:3, :3])
    np_, nr_ = float(np.linalg.norm(e_p)), float(np.linalg.norm(e_r))
    if np_ > p.v_task:
        e_p = e_p * (p.v_task / np_)
    if nr_ > p.w_task:
        e_r = e_r * (p.w_task / nr_)
    e = np.asarray(v_ff, float) + np.concatenate([p.kp * e_p, p.kp * e_r])
    return e, K.jacobian(q0, tcp)


def build_constraints_vbox(q0, s0, u_prev, cfg):
    A, l, h, blocks = B1.build_constraints(q0, s0, u_prev, cfg)
    vb = cfg.vbox()
    for name, r0, r1 in blocks:
        if name == 'velocity':
            l[r0:r1], h[r0:r1] = -vb, vb
    return A, l, h, blocks


def solve_c1(K, q0, s0, u_prev, T_des, v_ff, cfg, p: B1.B1Params) -> B1.B1Result:
    """同 wg4b_b1_core.solve_b1 的流程與接受規則，只換任務項（含前饋）與速度框（vbox）。"""
    import osqp
    from scipy import sparse
    t0 = time.monotonic()
    q0 = np.asarray(q0, float)
    if s0 is None or len(s0) != 6 or not np.isfinite(np.asarray(s0, float)).all():
        return B1.B1Result(ok=False, reason='no_setpoint', timing_ms={'total': 0.0})
    if not (np.isfinite(q0).all() and np.isfinite(np.asarray(u_prev, float)).all() and np.isfinite(np.asarray(v_ff, float)).all()):
        return B1.B1Result(ok=False, reason='nonfinite_input', timing_ms={'total': 0.0})
    e, Jw = task_terms_ff(K, q0, T_des, v_ff, cfg.tcp, p)
    Jb = Jw @ body_to_world(float(q0[2]))
    H, g, info = B1.build_objective(Jb, e, q0[B1.ARM], cfg, p)
    A, l, h, blocks = build_constraints_vbox(q0, s0, u_prev, cfg)
    t1 = time.monotonic()
    m = osqp.OSQP()
    with contextlib.redirect_stdout(io.StringIO()):
        m.setup(P=sparse.csc_matrix(np.triu((H + H.T) / 2.0)), q=-g, A=sparse.csc_matrix(A), l=l, u=h, verbose=False,
                eps_abs=p.eps_abs, eps_rel=p.eps_rel, max_iter=p.max_iter, polish=p.polish, warm_starting=False)
        r = m.solve()
    t2 = time.monotonic()
    st = str(r.info.status)
    tm = {'build': (t1 - t0) * 1e3, 'qp': (t2 - t1) * 1e3, 'total': (t2 - t0) * 1e3}
    if st != 'solved' or r.x is None or not np.isfinite(r.x).all():
        return B1.B1Result(ok=False, reason=f'osqp_{st}', timing_ms=tm, b1=info)
    u = np.asarray(r.x, float)
    res = B1.residuals(A, l, h, blocks, u)
    rmax = max(res.values())
    if rmax > p.r_tol:
        return B1.B1Result(ok=False, reason='residual', max_residual=rmax, timing_ms=tm, b1=info)
    info.update(e=[float(x) for x in e], residual_by_block=res)
    return B1.B1Result(ok=True, reason='ok', u0=u, U=u[None, :], max_residual=rmax, timing_ms=tm, b1=info)


__all__ = ['ref_twist', 'task_terms_ff', 'build_constraints_vbox', 'solve_c1', 'math']
