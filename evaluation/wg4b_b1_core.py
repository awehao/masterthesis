#!/usr/bin/env python3
"""WG4-B 的 B1：由凍結原型 baseline_B_frozen_20260909（wholebody_pregrasp.py --solver qp）**改編的單步 QP 新基線**。

規格：evaluation/results/wg4b/experiment_spec_WG4B.yaml（draft-2b）。每週期以目前狀態的全身 Jacobian 解一次 QP，
**沒有增廣動態預測模型**（只有單步運動學 q⁺ = q + dt·u），沒有 SQP、終端權重、延遲補償、偏移估計。

決策變數 u ∈ R⁹ 為**本體座標**（與 P 相同：[vx_b, vy_b, ω, q̇₁..q̇₆]），J_b = J_w · B(θ)。

    min_u  ‖J_b u − e‖²                                 任務項（凍結原型的誤差定義與截斷）
         + μ²‖u_arm − v_post‖²                          凍結原型的姿態回復項（有 q_nom 時整個關閉）
         + λ²‖u ⊘ w‖²                                   凍結原型的阻尼
         + w_vref‖(u_base − v_ref) ⊘ vmax_base‖²        協同：與 P 的 coord_cost 同一式子
         + w_qn‖q_arm + dt·u_arm − q_nom‖²              協同：單步運動學形式

二次式寫成 ½uᵀHu − gᵀu（整體除以 2，與凍結原型同一慣例；極小點不變）。

限制全部由**目前狀態**建立，數值取自 P 的 WGMPCConfig（不搬增廣模型的限制建構）：
速度框、加速度框（對實際套用的 u_prev）、實測與設定點關節限位（LITE6_SAFE ± joint_margin）、輪速、輪加速度。
OSQP 每輪冷啟動；只接受 solved，並以未縮放原始限制核對殘差（同 P 的接受規則）。
"""
from __future__ import annotations

import contextlib
import io
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE                 # noqa: E402
from ammr_wholebody_mpc.arm_pregrasp import rot_error                # noqa: E402
from ammr_wholebody_mpc.wgmpc_core import body_to_world              # noqa: E402
from ammr_wholebody_mpc.wgmpc_core_sp import coord_params            # noqa: E402

NU = 9
BASE = slice(0, 3)
ARM = slice(3, 9)
# 抽屜任務的展開完成參考構型（evaluation/drawer_wholebody_node.py:199 的 Q_GRASP；
# mt_b1 各趟 wholebody.json 的 q_grasp 同值）。不用 pregrasp_reference（舊箱體案例）。
Q_PREF_DRAWER = (-0.0321, 0.9177, 1.4853, -0.3580, -1.0325, -1.3813)
Q_PREF_SOURCE = 'evaluation/drawer_wholebody_node.py:199 Q_GRASP（抽屜展開完成參考構型）'


@dataclass
class B1Params:
    """凍結原型的數值（baseline_B_frozen_20260909.md）；只有 kp 是事前登錄的可調量。"""
    kp: float = 1.0                 # kp_p = kp_r 同值；候選順序 1.0 → 2.0 → 0.5
    v_task: float = 0.10            # 任務項位置誤差截斷（m）
    w_task: float = 0.5             # 任務項姿態誤差截斷（rad）
    base_weight: float = 3.0
    damping: float = 0.06           # λ
    mu_post: float = 0.03           # μ
    kp_post: float = 1.2
    k_limit: float = 1.5
    limit_margin: float = 0.35
    q_pref: tuple = Q_PREF_DRAWER
    q_pref_source: str = Q_PREF_SOURCE
    eps_abs: float = 1e-7
    eps_rel: float = 1e-7
    max_iter: int = 50000
    polish: bool = True
    r_tol: float = 1e-6

    def record(self) -> dict:
        d = asdict(self)
        d['q_pref'] = [float(v) for v in self.q_pref]
        return d


@dataclass
class B1Result:
    """欄位名與 WGMPCResult 對齊，節點的紀錄程式不必分支。"""
    ok: bool
    reason: str = ''
    u0: np.ndarray | None = None
    U: np.ndarray | None = None
    # SQP 相容欄位：B1 不是 SQP ⇒ 明確標為不適用（不當成「收斂一次」）
    sqp_converged: bool | None = None
    sqp_stop_reason: str = 'not_applicable'
    n_sqp_used: int | None = None
    max_residual: float = float('nan')
    timing_ms: dict = field(default_factory=dict)
    b1: dict = field(default_factory=dict)      # μ 開關、成本分項、OSQP 狀態、各區塊殘差


def mu_state(cfg) -> tuple[bool, str]:
    """(μ 項是否開啟, 原因)。協同設定已由節點的 _apply_coord 寫進 cfg（過期 ⇒ 啟動值）。"""
    wv, vr, wq, qn = coord_params(cfg)
    if wq > 0.0 and qn is not None:
        return False, 'qnom_active'
    return True, 'no_qnom'


def task_terms(K, q0, T_des, tcp, p: B1Params):
    """凍結原型的任務項：e = [kp·clip(e_p) ; kp·clip(e_r)]、J_w（世界座標 6×9）。"""
    T = K.fk(q0, tcp)
    e_p = T_des[:3, 3] - T[:3, 3]
    e_r = rot_error(T[:3, :3], T_des[:3, :3])
    np_, nr_ = float(np.linalg.norm(e_p)), float(np.linalg.norm(e_r))
    if np_ > p.v_task:
        e_p = e_p * (p.v_task / np_)
    if nr_ > p.w_task:
        e_r = e_r * (p.w_task / nr_)
    e = np.concatenate([p.kp * e_p, p.kp * e_r])
    return e, K.jacobian(q0, tcp)


def build_objective(Jb, e, q_arm, cfg, p: B1Params):
    """回傳 (H, g, info)，目標 ½uᵀHu − gᵀu。"""
    w = np.ones(NU)
    w[BASE] = p.base_weight
    H = Jb.T @ Jb + (p.damping ** 2) * np.diag(1.0 / (w * w))
    g = Jb.T @ e
    mu_on, why = mu_state(cfg)
    v_post = None
    if mu_on:
        lo, hi = LITE6_SAFE.lower, LITE6_SAFE.upper
        m = p.limit_margin
        push = (np.maximum(0.0, m - (q_arm - lo))
                - np.maximum(0.0, m - (hi - q_arm))) / m
        v_post = (p.kp_post * (np.asarray(p.q_pref, float) - q_arm)
                  + p.k_limit * push)
        H[ARM, ARM] += (p.mu_post ** 2) * np.eye(6)
        g[ARM] += (p.mu_post ** 2) * v_post
    wv, vr, wq, qn = coord_params(cfg)
    if wv > 0.0:
        D = np.diag(1.0 / cfg.vmax()[BASE] ** 2)
        H[BASE, BASE] += wv * D
        g[BASE] += wv * (D @ vr)
    if wq > 0.0:
        dt = cfg.dt
        H[ARM, ARM] += wq * dt * dt * np.eye(6)
        g[ARM] += wq * dt * (qn - q_arm)
    info = dict(mu_on=bool(mu_on), mu_reason=why, dt=float(cfg.dt),
                w_vref=float(wv), w_qn=float(wq),
                v_post=None if v_post is None else [float(x) for x in v_post])
    return H, g, info


def build_constraints(q0, s0, u_prev, cfg):
    """l ≤ A u ≤ h，全部由目前狀態建立。回傳 (A, l, h, blocks)；blocks = [(名稱, 起列, 終列)]。"""
    dt = cfg.dt
    vmax, amax = cfg.vmax(), cfg.amax()
    up = np.asarray(u_prev, float)
    lo_j = np.asarray(LITE6_SAFE.lower, float) + cfg.joint_margin
    hi_j = np.asarray(LITE6_SAFE.upper, float) - cfg.joint_margin
    rows, lo, hi, blocks = [], [], [], []

    def add(name, A, l, h):
        r0 = len(rows)
        rows.extend(np.atleast_2d(A))
        lo.extend(np.ravel(l))
        hi.extend(np.ravel(h))
        blocks.append((name, r0, len(rows)))
    I = np.eye(NU)
    add('velocity', I, -vmax, vmax)
    add('acceleration', I, up - amax * dt, up + amax * dt)
    Sa = np.zeros((6, NU))
    Sa[:, ARM] = dt * np.eye(6)
    q_arm = np.asarray(q0, float)[ARM]
    add('measured_position', Sa, lo_j - q_arm, hi_j - q_arm)
    s = np.asarray(s0, float).reshape(6)
    add('setpoint_position', Sa, lo_j - s, hi_j - s)
    W = cfg.wheel_matrix()
    Wf = np.zeros((4, NU))
    Wf[:, BASE] = W
    wlim = cfg.wheel_radius * cfg.wheel_w_max
    alim = cfg.wheel_radius * cfg.wheel_a_max * dt
    add('wheel_speed', Wf, np.full(4, -wlim), np.full(4, wlim))
    off = W @ up[BASE]
    add('wheel_accel', Wf, off - alim, off + alim)
    return (np.asarray(rows, float), np.asarray(lo, float),
            np.asarray(hi, float), blocks)


def residuals(A, l, h, blocks, u):
    """各區塊的未縮放違反量（≥ 0）。"""
    Au = A @ u
    v = np.maximum(0.0, np.maximum(l - Au, Au - h))
    return {n: float(v[a:b].max()) if b > a else 0.0 for n, a, b in blocks}


def solve_b1(K, q0, s0, u_prev, T_des, cfg, p: B1Params) -> B1Result:
    """一個控制週期。輸入非有限或 s0 缺失 ⇒ 拒絕（不以實測角代替設定點）。"""
    import osqp
    from scipy import sparse
    t0 = time.monotonic()
    q0 = np.asarray(q0, float)
    if s0 is None or len(s0) != 6 or not np.isfinite(np.asarray(s0, float)).all():
        return B1Result(ok=False, reason='no_setpoint',
                        timing_ms={'total': 0.0}, b1={'dt': float(cfg.dt)})
    if not (np.isfinite(q0).all() and np.isfinite(np.asarray(u_prev, float)).all()):
        return B1Result(ok=False, reason='nonfinite_input',
                        timing_ms={'total': 0.0}, b1={'dt': float(cfg.dt)})
    e, Jw = task_terms(K, q0, T_des, cfg.tcp, p)
    Jb = Jw @ body_to_world(float(q0[2]))
    H, g, info = build_objective(Jb, e, q0[ARM], cfg, p)
    A, l, h, blocks = build_constraints(q0, s0, u_prev, cfg)
    t1 = time.monotonic()
    m = osqp.OSQP()
    with contextlib.redirect_stdout(io.StringIO()):
        m.setup(P=sparse.csc_matrix(np.triu((H + H.T) / 2.0)), q=-g,
                A=sparse.csc_matrix(A), l=l, u=h, verbose=False,
                eps_abs=p.eps_abs, eps_rel=p.eps_rel, max_iter=p.max_iter,
                polish=p.polish, warm_starting=False)
        r = m.solve()
    t2 = time.monotonic()
    st = str(r.info.status)
    info.update(osqp_status=st, osqp_iter=int(r.info.iter),
                e=[float(x) for x in e])
    tm = {'build': (t1 - t0) * 1e3, 'qp': (t2 - t1) * 1e3,
          'total': (t2 - t0) * 1e3}
    if st != 'solved' or r.x is None or not np.isfinite(r.x).all():
        return B1Result(ok=False, reason=f'osqp_{st}', timing_ms=tm, b1=info)
    u = np.asarray(r.x, float)
    res = residuals(A, l, h, blocks, u)
    info['residual_by_block'] = res
    rmax = max(res.values())
    if rmax > p.r_tol:
        return B1Result(ok=False, reason='residual', max_residual=rmax,
                        timing_ms=tm, b1=info)
    info['u_raw'] = [float(x) for x in u]       # 未捨入，供逐位元重播
    return B1Result(ok=True, reason='ok', u0=u, U=u[None, :],
                    max_residual=rmax, timing_ms=tm, b1=info)
