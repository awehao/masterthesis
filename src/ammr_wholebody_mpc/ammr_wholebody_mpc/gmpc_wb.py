# gmpc_wb（2026-10-31；Howard：已封存的導航 GMPC 不動，新的銜接複製一份過去做）：gmpc.py 的**原樣複本**（封存版 git a10568326）。
#   之後只在本檔擴充手臂設定點、末端任務與全身安全；gmpc.py／gmpc_node.py 與既有導航跑法一律不改。
#   手臂保持且操作關閉時，須以構造案例證明與 gmpc.py 底盤問題等價（Codex 20261031 同步回覆 §3）。
"""SE(2) Geometric Model-Predictive Controller (offline prototype).

Decision variable
-----------------
    z = [δξ_0; δξ_1; ...; δξ_{N-1}] ∈ R^{3N}
    δξ_k = u_k - ξ_ref(k)   (deviation from reference body twist at step k)

Error state and dynamics (multi-point linearisation along the reference)
------------------------------------------------------------------------
    e_0 = log(X_ref(0)^{-1} · X_now)^vee    (current geodesic error)
    e_{k+1} = A_d(k) · e_k + dt · δξ_k
    A_d(k) = I - dt · ad(ξ_ref(k))

This is the key fix vs. the previous (x,y,θ) MPC attempt (memory
`project_mpc_lessons`):

    - Linearisation is now around ξ_ref(k) *at every step k* of the horizon,
      not just around the current state. So a horizon that needs to turn
      30° / 60° / ... is still linearised against the *reference* twist that
      already turns at that rate — the error correction stays in the
      small-perturbation regime.
    - The SE(2) error e never wraps. log_ recovers θ ∈ (-π, π] and the
      translation parts are body-frame, so a robot rotated past ±π still
      sees a smooth error vector pointing back toward the reference.

Prediction matrices (condensed form)
------------------------------------
    E = [e_1; e_2; ...; e_N] = Φ · e_0 + Γ · z
    Φ ∈ R^{3N × 3},  Γ ∈ R^{3N × 3N}

Cost
----
    J = Σ_{k=1}^{N-1} e_k^T Q  e_k + e_N^T Q_f e_N + Σ_{k=0}^{N-1} δξ_k^T R δξ_k
      = (1/2) z^T P z + q^T z + const
    P = 2 · (Γ^T · Q̄ · Γ + R̄)
    q = 2 · Γ^T · Q̄ · Φ · e_0
    Q̄ = blkdiag(Q,...,Q, Q_f)   (N blocks, last is terminal)
    R̄ = blkdiag(R,...,R)        (N blocks)

Constraints
-----------
    Velocity:      u_min ≤ ξ_ref(k) + δξ_k ≤ u_max     for k = 0..N-1
    Acceleration:  |u_k - u_{k-1}| ≤ a_max · dt        (with u_{-1} := ξ_prev)

Stacked into OSQP standard form  l ≤ A_total z ≤ u.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np
import osqp
from scipy import sparse

from .se2 import ad, geodesic_error


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# （gmpc_wb）手臂作用模式：設定點模型預測＋末端任務線性化（單次線性化，非 SQP）
# ---------------------------------------------------------------------------
ARM_N = 6


def arm_prediction(q0, s0, N, dt, model):
    """設定點模型（wgmpc_core_sp.ArmSetpointModel.compose(dt) 的 P,Q,G,h）的線性預測。
    回傳 (Mq, cq, Ms, cs)：q_k＝Mq[k]·U＋cq[k]、s_k＝Ms[k]·U＋cs[k]（k＝1..N；U＝u_a 堆疊 6N）。"""
    P, Q, G, h, _ = model.compose(dt)
    n = ARM_N * N
    Mq, Ms = np.zeros((N, ARM_N, n)), np.zeros((N, ARM_N, n))
    cq, cs = np.zeros((N, ARM_N)), np.zeros((N, ARM_N))
    q_M, q_c = np.zeros((ARM_N, n)), np.asarray(q0, float).copy()
    s_M, s_c = np.zeros((ARM_N, n)), np.asarray(s0, float).copy()
    for k in range(N):
        Uk = np.zeros((ARM_N, n))
        Uk[:, ARM_N * k:ARM_N * (k + 1)] = np.eye(ARM_N)
        q_M, q_c = P[:, None] * q_M + Q[:, None] * s_M + G[:, None] * Uk, P * q_c + Q * s_c + h
        s_M, s_c = s_M + dt * Uk, s_c.copy()
        Mq[k], cq[k], Ms[k], cs[k] = q_M, q_c, s_M, s_c
    return Mq, cq, Ms, cs


class FastChain:
    """（gmpc_wb）同一條運動鏈一次前向掃描得 FK 與幾何 Jacobian，取代 K.fk＋K.jacobian 的重複計算。
    數值與 wholebody_kinematics 相同（誤差 <1e-12，test_gmpc_wb_arm 核對）；不改共用檔。相鄰固定關節先合併。"""
    _cache = {}

    @classmethod
    def of(cls, K, link):
        key = (id(K), link)
        c = cls._cache.get(key)
        if c is None:
            c = cls._cache[key] = cls(K, link)
        return c

    def __init__(self, K, link):
        self.n = len(K.dof_names)
        self.steps = []                                 # (T_origin, axis, 類型, dof 索引)
        pend = np.eye(4)
        for jn in K.chain(link):
            j = K.joints[jn]
            O = np.eye(4)
            O[:3, :3], O[:3, 3] = j.origin_rot, j.origin_xyz
            pend = pend @ O
            if jn in K.dof_names:
                ax = np.asarray(j.axis, float)
                self.steps.append((pend, ax / (np.linalg.norm(ax) if j.jtype != 'prismatic' else 1.0), ax, j.jtype, K.dof_names.index(jn)))
                pend = np.eye(4)
        self.tail = pend

    def fk_jac(self, q):
        T = np.eye(4)
        cols = []
        for O, k, ax, jt, i in self.steps:
            Tj = T @ O
            Rj = Tj[:3, :3]
            B = np.eye(4)
            if jt == 'prismatic':
                axw = Rj @ ax
                B[:3, 3] = ax * q[i]
            else:
                axw = Rj @ ax
                sn, cs = math.sin(q[i]), math.cos(q[i])
                K_ = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
                B[:3, :3] += sn * K_ + (1.0 - cs) * (K_ @ K_)
            cols.append((i, jt, axw, Tj[:3, 3]))
            T = Tj @ B
        T = T @ self.tail
        pe = T[:3, 3]
        J = np.zeros((6, self.n))
        for i, jt, a, p in cols:
            if jt == 'prismatic':
                J[:3, i] = a
            else:
                r = pe - p
                J[0, i], J[1, i], J[2, i] = a[1] * r[2] - a[2] * r[1], a[2] * r[0] - a[0] * r[2], a[0] * r[1] - a[1] * r[0]
                J[3:, i] = a
        return T, J


def point_jac(Tc, J, off):
    """TCP 剛連點（TCP 座標 off）的世界位置與位置 Jacobian：v_p＝v＋ω×r＝v−[r]×ω。"""
    rw = Tc[:3, :3] @ np.asarray(off, float)
    Sk = np.array([[0.0, -rw[2], rw[1]], [rw[2], 0.0, -rw[0]], [-rw[1], rw[0], 0.0]])
    return Tc[:3, 3] + rw, J[:3] - Sk @ J[3:]


def qp_kkt_check(P, q, A, l, u, x, y, tol_p=1e-8, tol_d=1e-6, tol_c=1e-6):
    """在原問題上核對解：有限值、原始可行、對偶可行（y>0 只在上界作用、y<0 只在下界作用）、站立性、互補。
    回傳 dict(ok, why, 各殘差)。門檻為原單位或相對 max(1,‖q‖∞)／max(1,‖y‖∞)；不通過的解不得採用。"""
    if x is None or y is None or not (np.isfinite(x).all() and np.isfinite(y).all()):
        return dict(ok=False, why='nonfinite')
    z = A @ x
    prim = float(np.max(np.maximum(l - z, z - u))) if z.size else 0.0
    stat = float(np.abs(P @ x + q + A.T @ y).max())
    gap_u, gap_l = np.where(np.isfinite(u), u - z, np.inf), np.where(np.isfinite(l), z - l, np.inf)
    yp, ym = np.maximum(y, 0.0), np.maximum(-y, 0.0)
    dual = float(max(np.max(yp[~np.isfinite(u)], initial=0.0), np.max(ym[~np.isfinite(l)], initial=0.0)))      # 無界側不得有乘子
    comp = float(max(np.max(yp * np.where(np.isfinite(gap_u), np.abs(gap_u), 0.0), initial=0.0), np.max(ym * np.where(np.isfinite(gap_l), np.abs(gap_l), 0.0), initial=0.0)))
    sq, sy = max(1.0, float(np.abs(q).max())), max(1.0, float(np.abs(y).max()))
    out = dict(prim=prim, stat=stat, dual=dual, comp=comp)
    why = 'primal' if prim > tol_p else 'stationarity' if stat > tol_d * sq else 'dual' if dual > 0.0 else 'complementarity' if comp > tol_c * sy else None
    return dict(out, ok=why is None, why=why)


def qp_dual_active(P, q, A, l, u, tol=1e-9, max_steps=800):
    """Goldfarb–Idnani 對偶作用集法：min ½xᵀPx＋qᵀx s.t. l ≤ Ax ≤ u（P 正定）。實驗用，尚未定版。
    耦合問題的主求解器：CBF slack 懲罰使 P 對角跨 0.2–2e8，OSQP 在作用集轉變時要數千至數萬次迭代（max 381 ms、D1 到迭代上限），本實作於該批難題 max 5.1 ms。
    回傳 (x, y, info)；x 為 None 表示未得解或未過 qp_kkt_check，一律交由 OSQP 備援，不接受可疑解。回傳 infeasible 只代表本實作未找到解，不作不可行證明。"""
    from scipy.linalg import cho_factor, cho_solve
    Pd = P.toarray()
    try:
        cf = cho_factor(Pd)
    except np.linalg.LinAlgError:
        return None, None, dict(ok=False, why='P_not_pd', solver='gi')
    As = A.tocsr()
    x = -cho_solve(cf, q)
    act, sgn, Nr, PiN, mu = [], [], [], [], np.zeros(0)
    for step in range(max_steps):
        z = As @ x
        vl, vu = l - z, z - u
        il, iu = int(np.argmax(vl)), int(np.argmax(vu))
        if max(vl[il], vu[iu]) <= tol:
            break
        i, sg = (il, 1.0) if vl[il] >= vu[iu] else (iu, -1.0)         # 列寫成 n·x ≥ b
        n = sg * np.asarray(As[i].todense()).ravel()
        b = sg * (l[i] if sg > 0 else u[i])
        Hn = cho_solve(cf, n)
        up = 0.0
        while True:
            if act:
                N_, Pi = np.array(Nr), np.array(PiN).T
                try:
                    r = np.linalg.solve(N_ @ Pi, N_ @ Hn)
                except np.linalg.LinAlgError:
                    return None, None, dict(ok=False, why='singular', steps=step, solver='gi')
                zd = Hn - Pi @ r
            else:
                r, zd = np.zeros(0), Hn
            nz = float(n @ zd)
            t2 = (b - float(n @ x)) / nz if nz > 1e-12 * float(n @ Hn) else float('inf')   # 相對容差：P 對角到 1e8 時 n·P⁻¹n 本身就小，絕對容差會把合法步長判成 0
            t1, drop = float('inf'), None
            r_tol = 1e-12 * max(float(np.abs(r).max()), 1e-300) if r.size else 0.0
            for j, rj in enumerate(r):
                if rj > r_tol and mu[j] / rj < t1:
                    t1, drop = mu[j] / rj, j
            t = min(t1, t2)
            if not np.isfinite(t):
                return None, None, dict(ok=False, why='infeasible', steps=step, solver='gi')
            if np.isfinite(t2):
                x = x + t * zd
            mu = mu - t * r
            up += t
            if t2 <= t1:
                act.append(i); sgn.append(sg); Nr.append(n); PiN.append(Hn); mu = np.r_[mu, up]
                break
            act.pop(drop); sgn.pop(drop); Nr.pop(drop); PiN.pop(drop); mu = np.delete(mu, drop)
    else:
        return None, None, dict(ok=False, why='steps', steps=max_steps, solver='gi')
    y = np.zeros(A.shape[0])
    for i, sg, m in zip(act, sgn, mu):
        y[i] = -sg * m
    ck = qp_kkt_check(P, q, A, l, u, x, y)
    if not ck['ok'] and act:                            # 作用集固定做殘差精修：剛性座標（P 對角 1e8）上 x 差 1e-8 就是站立性殘差 1；不改作用集、不放寬驗收
        N_ = np.array(Nr)
        b_ = np.array([sg * (l[i] if sg > 0 else u[i]) for i, sg in zip(act, sgn)])
        S_ = N_ @ cho_solve(cf, N_.T)
        for _ in range(3):
            rd, rp = Pd @ x + q - N_.T @ mu, N_ @ x - b_
            dm = np.linalg.lstsq(S_, N_ @ cho_solve(cf, rd) - rp, rcond=None)[0]
            x = x + cho_solve(cf, N_.T @ dm - rd)
            mu = mu + dm
        y = np.zeros(A.shape[0])
        for i, sg, m in zip(act, sgn, mu):
            y[i] = -sg * m
        ck = qp_kkt_check(P, q, A, l, u, x, y)
    if not ck['ok']:
        return None, None, dict(ck, ok=False, why='verify:' + str(ck['why']), steps=step, solver='gi')
    return x, y, dict(ck, steps=step, n_active=len(act), solver='gi')


def arm_stop_sequence(arm, N, dt, tol=1e-9):
    """退回用手臂序列：自 u_arm_prev 以 a_arm·dt 有界減速到 0，並核速度框、加速度框、預測實測角與設定點限位（±margin）。
    零速度命令不保證實體手臂立即停止，也可能違反對上一筆的加速度限制（Codex 20261031_183817），所以不直接給零。
    回傳 (U (N,6), check)；check['verified'] False ⇒ 呼叫端不得當成可行序列。"""
    up = np.asarray(arm['u_arm_prev'], float).copy()
    da, va = float(arm['a_arm']) * dt, float(arm['v_arm'])
    U = np.zeros((N, ARM_N))
    u = up
    for k in range(N):
        u = u + np.clip(-u, -da, da)
        U[k] = u
    Mq, cq, Ms, cs = arm_prediction(arm['q_arm'], arm['s_arm'], N, dt, arm['model'])
    v = U.reshape(-1)
    lo = np.asarray(arm['lo'], float) + arm['margin']
    hi = np.asarray(arm['hi'], float) - arm['margin']
    viol = {'velocity': float(np.abs(U).max() - va),
            'acceleration': float(np.abs(np.diff(np.vstack([up, U]), axis=0)).max() - da),
            'measured_position': max(float(max((lo - (Mq[k] @ v + cq[k])).max(), ((Mq[k] @ v + cq[k]) - hi).max())) for k in range(N)),
            'setpoint_position': max(float(max((lo - (Ms[k] @ v + cs[k])).max(), ((Ms[k] @ v + cs[k]) - hi).max())) for k in range(N))}
    ok = all(x <= tol for x in viol.values()) and bool(np.isfinite(U).all())
    return U, dict(verified=bool(ok), violations=viol, reaches_zero=bool(np.abs(U[-1]).max() <= tol))


def _rot_err_world(R_cur, R_des):
    Re = R_cur.T @ R_des
    c = float(np.clip((np.trace(Re) - 1.0) * 0.5, -1.0, 1.0))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    w = np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]]) * (th / (2.0 * math.sin(th)))
    return R_cur @ w


def _pose_from_err(X_ref, e):
    from .se2 import exp_
    X = X_ref @ exp_(e)
    return np.array([X[0, 2], X[1, 2], math.atan2(X[1, 0], X[0, 0])])


def _pose_jac(X_ref, e, h=1e-7):
    """∂(x, y, θ)/∂e 於 e（中央差分；θ 差取 wrap）。"""
    J = np.zeros((3, 3))
    for i in range(3):
        d = np.zeros(3); d[i] = h
        a, b = _pose_from_err(X_ref, e + d), _pose_from_err(X_ref, e - d)
        dd = a - b
        dd[2] = math.atan2(math.sin(dd[2]), math.cos(dd[2]))
        J[:, i] = dd / (2 * h)
    return J


@dataclass
class GMPCConfig:
    N      : int                          # horizon length (number of input steps)
    dt     : float                        # time-step in seconds
    u_min  : np.ndarray                   # (3,) lower velocity limit (vx,vy,ω)
    u_max  : np.ndarray                   # (3,) upper velocity limit
    a_max  : np.ndarray                   # (3,) per-axis acceleration magnitude limit
    Q      : np.ndarray                   # (3,3) running state weight
    R      : np.ndarray                   # (3,3) input deviation weight
    Qf     : np.ndarray                   # (3,3) terminal state weight
    # Input-increment (Δu) smoothness weight. Penalises Δu_k = u_k - u_{k-1} in
    # the cost: Σ Δu_k^T S Δu_k. A SOFT cost -> silky-smooth cruising, yet the
    # optimiser will still accept a hard acceleration burst when the CBF demands
    # it (safety preserved). Default 0 = disabled (identical to old behaviour).
    S      : np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    # （gmpc_wb，2026-11-01 候選）輸入二階差分權重：Σ Δ²u_kᵀ S2 Δ²u_k，Δ²u_k = u_k − 2u_{k−1} + u_{k−2}，
    # u_{−1}＝xi_prev、u_{−2}＝xi_prev2（預測起點之前兩筆已套用／在途命令，由呼叫端給）。預設 0＝不加任何項（問題與先前相同）。
    S2     : np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    # （gmpc_wb，2026-11-01 候選）底盤位置硬界限 ((x_lo, x_hi), (y_lo, y_hi))，世界座標：時域第 1..N 步的預測位置（一階：p_ref,k＋R_ref,k·e_k[:2]，
    # e＝Φ·e0＋Γ·z，e0 為在途前推後的初態）都須在矩形內。None＝不加任何列（問題與先前相同）。不是完整操作可行域。
    base_box: tuple | None = None
    # ---- Wheel-speed coupling ---------------------------------------------
    # u_min/u_max/a_max form an axis-aligned BOX, but the chassis's feasible set
    # is not a box: the two translation axes drive different wheel pairs and so
    # saturate independently, while rotation drives all four and competes with
    # both. Inverting the omni Jacobian gives r·ω = W·u with
    #
    #     W = [[0, 1, L], [-1, 0, L], [0, -1, L], [1, 0, L]]
    #
    # so |v_x ± L·φ̇| ≤ r·ω_max and |v_y ± L·φ̇| ≤ r·ω_max. With L·φ̇_max exactly
    # equal to r·ω_max, spinning at full rate leaves ZERO translation authority
    # -- a fact the box cannot express. Measured on a 19-trial batch, 7.4% of
    # control cycles asked for wheel speeds beyond the hardware, worst case 2.0x,
    # and gz's VelocityControl executes them anyway because it has no actuator
    # model. Those results did not represent the robot.
    wheel_coupling : bool  = False        # off = old box-only behaviour
    # 'segment' = 原本的單一 λ 線段縮放（預設，行為一位元未改）。
    # 'project' = 投影到同一組輪級集合，避免「一列貼邊就凍結三個自由度」。
    wheel_fit_mode : str   = 'segment'
    wheel_radius   : float = 0.05         # r   [m]
    wheel_base_L   : float = 0.245        # L   [m]
    wheel_w_max    : float = 5.55         # ω_max [rad/s]
    wheel_a_max    : float = 125.0        # α_max [rad/s²]
    # Verify the COMMAND that actually leaves solve(), not just the solver's
    # status. 'maximum iterations reached' is accepted as usable (see the solve
    # tail), but the point OSQP returns then carries no feasibility guarantee,
    # and the per-axis box clip below it cannot see the wheel rows at all: the
    # coupling set cuts the corners off the box, so a command inside the box can
    # still ask for 2x the hardware wheel speed. Measured by injecting a
    # non-converged solution into this solver: u = (0.2775, 0.2775, 1.1327) left
    # the guard and implied 11.10 rad/s against ω_max = 5.55.
    #
    # Off reproduces the old behaviour exactly, so a run can be replayed as it
    # was recorded; on, the command is pulled back along the segment to ξ_prev
    # (both wheel sets are convex and contain ξ_prev, so a single λ satisfies
    # both) until it fits. A converged solve already satisfies the rows to
    # eps_abs, so λ = 1 and nothing is touched.
    wheel_enforce_output : bool = True
    # （gmpc_wb，2026-11-01 候選）輸出端輪級後處理的數值邊界容差 [m/s，同 fit_to_wheels 的 r·ω 單位]。0＝原行為。
    # >0：原始命令的輪速 |W·u| ≤ r·ω_max＋tol 且輪加速度 |W·(u−ξ_prev)| ≤ r·α_max·dt＋tol 時視為可行、不做後處理。
    # 起因（bottle_w1_dyn2b_g1）：QP 解與上一筆都貼在輪速邊界（殘差 1e-11～1e-9 rad/s），線段縮放的 (w_lim − b) 因捨入為負，λ 塌成 0。
    # 超出容差的真正違反仍照原線段縮放／投影，回傳被標成後處理改動。
    wheel_fit_boundary_tol : float = 0.0
    # （gmpc_wb，2026-11-02 候選，Codex 20261102_184223）零速度框夾限的數值容差 (平移 m/s, 角速度 rad/s)。None＝原行為（逐位元相同）。
    # 只在底盤三軸上下限皆為零（停車求解器）時生效：可用且收斂的解、未經加速度夾限／輪級縮放或投影／煞停後備（accept_action＝as_is），
    # 回傳命令恰為速度框夾限結果，且夾回量逐軸 ≤ 容差 ⇒ seq_consistent 視為「數值容差內一致」（arm_check['seq_tol'] 記原值、回傳值、修正量）。
    # 起因（R-B2 假世界進 HOLD 那拍）：底盤 δ0＝(7.4e-11, 2.3e-10, −6.4e-12) 被 [0, 0] 夾回 0，atol 1e-12 比對判成後處理改動而退回。
    # 不是逐位元或精確同一序列；超出容差、非零速度框、其他後處理照原規則判不一致。
    zero_box_clip_tol : Optional[Tuple[float, float]] = None
    # ---- Control Barrier Function safety filter ----------------------------
    cbf_alpha        : float = 1.0        # decay rate α in  ḣ + α·h ≥ 0
    cbf_safe_margin  : float = 0.30       # extra clearance added to obstacle radius [m]
    cbf_slack_weight : float = 1.0e4      # ρ in cost ρ·ε² ; large = CBF near-hard
    cbf_eps0_scale   : float = 100.0      # ε_0 penalty multiplier (near-hard at k=0)
    # Separate slack for STATIC (wall) constraints, penalised this many times
    # harder than the dynamic slack.
    #
    # Why: with a single ε_k per step, every obstacle at that step shares one
    # relaxation, so when the robot is pinched between a moving obstacle and a
    # wall the QP can only relax BOTH by the same amount -- it cannot express
    # "rather give up dynamic clearance than touch the wall". The two buffers are
    # not equivalent: a dynamic obstacle carries margin 0.38 against a 0.30 m
    # robot (0.08 m of give) while a wall point carries 0.33 (only 0.03 m), so an
    # equal 0.05 m relaxation is still safe on the dynamic side but 0.02 m INTO
    # the wall. Giving the static rows their own, much more expensive slack makes
    # the QP eat the dynamic buffer first, which is what stops the controller
    # squeezing itself against walls while dodging.
    #
    # 1.0 reproduces the old shared-slack behaviour exactly (same weight for
    # both blocks), so this is a strict generalisation.
    cbf_static_slack_scale : float = 1.0
    # Hard barrier at k = 0 for KNOWN STATIC geometry only.
    #
    # The earlier cbf_hard_k0 pinned the step-0 slack outright, but the slack is
    # SHARED across every obstacle at that step, so it hardened the dynamic rows
    # too -- including surface points that can appear the instant perception
    # resolves them. That gave 13-31 'primal infeasible' events per trial. Walls
    # are different: they come from a stationary occupancy map, need no velocity
    # prediction, cannot appear suddenly, and the residual contacts measured on
    # bigarena are entirely door posts. Hardening only that class keeps the
    # dynamic side recoverable. Forces the slack split so the two classes stop
    # sharing one epsilon.
    cbf_hard_k0_static : bool = False
    # ---- Spatio-temporal cost field (proactive detour) ---------------------
    # Soft Gaussian barrier evaluated at the PREDICTED obstacle position for
    # every horizon step, added to the QP cost. Where the hard CBF only reacts
    # once the barrier is about to be violated, this puts a gradient on the cost
    # metres before the encounter, so the controller drifts aside early instead
    # of correcting hard and late.
    #
    #   C_k(p) = W * exp(-||p(k) - o_i(k)||^2 / (2 sigma_k^2))
    #   o_i(k) = o_i(0) + v_i * k * dt          (same CV prediction the CBF uses)
    #   sigma_k = sigma0 * (1 + growth * k)     (prediction gets less certain)
    #
    # st_weight = 0 disables it and reproduces the previous solver bit-for-bit,
    # so it can be A/B'd against (and combined with) the CBF independently.
    st_weight  : float = 0.0      # W; 0 = off
    st_sigma0  : float = 0.6      # m, ~ robot radius + obstacle radius
    st_growth  : float = 0.02     # sigma grows this fraction per horizon step
    # ---- Prediction-uncertainty growth in the CBF keep-out ------------------
    # The horizon CBF trusts o(k) = o(0) + v k dt equally at every k, but that
    # extrapolation degrades: an obstacle that reverses makes the far-horizon
    # prediction point the wrong way entirely. Measured on this benchmark, 43%
    # of the >90 deg heading reversals happen within 3 s of the ping-pong
    # obstacle turning around (median gap 3.3 s against ~9.8 s if the two were
    # unrelated), i.e. the controller commits to a prediction and then has to
    # undo it. Growing the keep-out with k makes far-future constraints
    # conservative and lets the near-term, still-reliable ones dominate:
    #     r_eff(k) = r_i + m_i * (1 + cbf_margin_growth * k)
    # 0.0 keeps the constant margin the validated configuration uses.
    cbf_margin_growth : float = 0.0
    # Extra keep-out proportional to how fast the gap is CLOSING [s]:
    #     r_eff += cbf_vel_margin_gain * max(0, closing_rate)
    # A fixed margin has to be sized for the worst approach and then costs that
    # width everywhere; at 0.75 m the doorways became infeasible and contacts
    # went UP. This spends the margin only when something is actually coming at
    # the robot. Braking distance sets the scale: 0.52 m/s against a_y = 0.6
    # needs 0.225 m, and the perception path adds ~0.3 s of lag. 0.0 reproduces
    # the fixed-margin behaviour every earlier result used.
    cbf_vel_margin_gain : float = 0.0
    # ---- Constraint pruning -------------------------------------------------
    # The row count is n_obs x N and n_obs is the number of SURFACE POINTS, not
    # objects: measured on the 20 m floor it runs 8 points (160 rows) at the
    # median and 40 points (800 rows) at the peak. OSQP is a first-order ADMM
    # solver, and at those sizes -- with gain scheduling swinging the cost
    # matrix as the robot closes in -- its p99 solve time reaches 425 ms against
    # a 50 ms control period, overrunning in 9% of cycles.
    #
    # Most of those rows carry no information. A point 3 m away cannot be
    # violated within a 1 s horizon, and the far half of the horizon is
    # predicted with constant velocity, so its precision is spurious anyway.
    #
    #   cbf_prune_range   drop a point whose closest approach to the reference
    #                     over the whole horizon exceeds this [m]. 0 = keep all.
    #   cbf_near_steps    steps from 0 that always keep every point.
    #   cbf_far_stride    beyond that, keep only every n-th step. 1 = keep all.
    #
    # k = 0 is never dropped: it is the only row that constrains the command
    # actually about to be sent.
    cbf_prune_range : float = 0.0
    cbf_near_steps  : int   = 6
    cbf_far_stride  : int   = 1
    # Pin the k = 0 slack to zero -> hard barrier at the current state.
    cbf_hard_k0     : bool  = False
    # ---- Forward-progress preference ---------------------------------------
    # Linear reward on velocity projected along the reference tangent:
    #     J -= prog_weight * sum_k  t_ref(k) . R(theta_k) v_body(k)
    # When the CBF forces the robot off the reference, the QP has no reason to
    # prefer deviating FORWARD over deviating BACKWARD -- both cost the same in
    # tracking error, so the choice falls out of the obstacle geometry alone and
    # the robot happily gives ground. Measured against MPPI on the same
    # scenario, that shows up as 7.7 reversals per run versus 3.2, and 0.75% of
    # travel spent moving backwards. A linear term breaks the tie toward
    # progress without touching feasibility: it only enters q, stays convex, and
    # cannot override the CBF, which is a hard constraint.
    # 0.0 = no preference (the validated behaviour).
    prog_weight : float = 0.0
    # ---- Gain scheduling (danger-aware Q/slack) ----------------------------
    # When min_h is small (robot close to an obstacle's safety zone), we
    #   • drop Q (tracking) so the controller stops fighting safety
    #   • multiply the slack penalty so CBF becomes effectively hard
    # When min_h is large, weights stay at their nominal values.
    cbf_danger_thresh : float = 0.5       # h above this → no scaling
    cbf_Q_min_scale   : float = 0.2       # Q is multiplied by this when fully danger
    cbf_slack_max_scale : float = 100.0   # slack penalty multiplied by this when danger


@dataclass
class GMPCResult:
    u_opt          : np.ndarray           # (3,) applied body twist for next step
    delta_xi_all   : np.ndarray           # (N, 3) full optimal sequence (for warm-start / debug)
    e0             : np.ndarray           # (3,) current geodesic error
    solve_time_s   : float                # wall-time of OSQP solve()
    status         : str                  # OSQP status string
    cbf_active     : int   = 0            # number of CBF constraints applied (info)
    min_h          : float = float('inf') # smallest barrier value across obstacles
    # Entry diagnostics. min_h alone cannot say whether a constraint was absent
    # or merely satisfied, and it is published only when cbf_active > 0, so a
    # bag can hold a stale value from an earlier cycle. These are computed from
    # the SAME obstacle list and pose the QP used, in the same call.
    n_obs_in       : int   = 0            # obstacles handed to solve()
    n_static_in    : int   = 0            # of those, tagged static (wall points)
    min_h_static   : float = float('inf') # min h at k=0 over static obstacles
    min_h_dynamic  : float = float('inf') # min h at k=0 over dynamic obstacles
    # Barrier-condition audit (see the post-solve block in solve()).
    eps0              : float = 0.0            # slack actually used at k = 0
    cbf_resid_noslack : float = float('inf')   # min over rows of  A z - l
    cbf_resid_slack   : float = float('inf')   # min over rows of  A z + eps - l
    # Output acceptance audit (see wheel_enforce_output).
    accept_action     : str   = 'as_is'   # as_is | acc_clipped | wheel_scaled | brake
    # 出口修正後，對**回傳的命令**重算的殘差。舊欄位 cbf_resid_* 描述的是
    # 求解器的解，出口修正後兩者可能不同（實測 0.177 -> 0.452），因此分開回報。
    cbf_resid_post_noslack : float = float('nan')
    cbf_resid_post_slack   : float = float('nan')
    cbf_resid_evaluated_on : str   = 'solved'   # solved | returned
    cmd_id                 : int   = -1
    accept_scale      : float = 1.0       # λ applied along the segment to ξ_prev
    wheel_w_cmd_max   : float = 0.0       # worst |ω| the RETURNED command implies
    # （gmpc_wb）手臂：arm_mode None＝未使用手臂（與封存版相同）；'hold'＝手臂速度序列固定 0（u_arm_seq (N,6) 全零）
    arm_mode          : str | None = None
    u_arm_seq         : np.ndarray | None = None
    u0_rows_viol      : float = float('nan')    # （候選）給了 u0_rows 時：回傳命令（後處理後）對 A·u ≤ b 的最大違反；未給＝NaN
    arm_check         : dict | None = None      # active：回傳序列的原單位最大違反、有限值、seq_consistent（全身序列一致，退回時恆 False）、fallback 原因、arm_decel_valid（退回：手臂減速限制驗證）、base_as_is（退回：hold 底盤未被改動）、failed（手臂減速未驗證 ⇒ 呼叫端須終止）


# ---------------------------------------------------------------------------
# Output acceptance: does the command that leaves solve() fit the wheels?
# ---------------------------------------------------------------------------

def zero_box_clip_ok(cfg: 'GMPCConfig', u_opt, u_raw, accept_action: str, usable: bool, status: str):
    """零速度框夾限的數值容差判定（GMPCConfig.zero_box_clip_tol）。回 None＝不適用或不通過（照原規則）；否則回紀錄 dict。"""
    tol = cfg.zero_box_clip_tol
    if tol is None or accept_action != 'as_is' or not usable or status != 'solved':
        return None
    lo, hi = np.asarray(cfg.u_min, float), np.asarray(cfg.u_max, float)
    if np.any(lo != 0.0) or np.any(hi != 0.0):
        return None
    u_opt, u_raw = np.asarray(u_opt, float), np.asarray(u_raw, float)
    if not (np.isfinite(u_raw).all() and np.array_equal(u_opt, np.clip(u_raw, lo, hi))):
        return None
    corr = np.abs(u_opt - u_raw)
    if corr[0] > tol[0] or corr[1] > tol[0] or corr[2] > tol[1]:
        return None
    return dict(label='數值容差內一致', raw=u_raw.tolist(), returned=u_opt.tolist(), corr_lin=float(max(corr[0], corr[1])), corr_ang=float(corr[2]), tol=[float(tol[0]), float(tol[1])])


def wheel_matrix(cfg: 'GMPCConfig') -> np.ndarray:
    """W such that r·ω = W·u for the four wheels, in the same row order the
    coupling constraints are built with."""
    L = cfg.wheel_base_L
    return np.array([[0.0,  1.0, L],
                     [-1.0, 0.0, L],
                     [0.0, -1.0, L],
                     [1.0,  0.0, L]])


def wheel_speeds(u, cfg: 'GMPCConfig') -> np.ndarray:
    """The four wheel speeds [rad/s] a body twist u implies."""
    return (wheel_matrix(cfg) @ np.asarray(u, dtype=float)) / cfg.wheel_radius


def wheels_within(u, xi_prev, cfg: 'GMPCConfig', dt: float, tol: float) -> bool:
    """（gmpc_wb 候選）命令是否在兩組輪級界內（容差 tol，r·ω 單位 m/s）：|W·u| ≤ r·ω_max＋tol 且 |W·(u−ξ_prev)| ≤ r·α_max·dt＋tol。"""
    W = wheel_matrix(cfg)
    u, xp = np.asarray(u, dtype=float), np.asarray(xi_prev, dtype=float)
    return bool(np.max(np.abs(W @ u)) <= cfg.wheel_radius * cfg.wheel_w_max + tol
                and np.max(np.abs(W @ (u - xp))) <= cfg.wheel_radius * cfg.wheel_a_max * dt + tol)


def fit_to_wheels(u, xi_prev, cfg: 'GMPCConfig', dt: float, tol: float = 1e-6):
    """Largest λ ∈ [0, 1] with u_λ = ξ_prev + λ(u - ξ_prev) inside BOTH wheel
    sets, and that u_λ.

    Both  |W·u| ≤ r·ω_max  and  |W·(u - ξ_prev)| ≤ r·α_max·dt  are convex and
    contain ξ_prev (it is the previously applied command, which left this same
    guard), so moving back along the segment can only help and one λ settles
    both. Each row gives a closed-form bound, so this costs four comparisons,
    not an iteration.

    Returns (u_fitted, λ). λ = 1.0 means the command was already feasible --
    which is the case for every converged solve, since OSQP satisfied the same
    rows to eps_abs.
    """
    u  = np.asarray(u, dtype=float)
    xp = np.asarray(xi_prev, dtype=float)
    W  = wheel_matrix(cfg)
    w_lim = cfg.wheel_radius * cfg.wheel_w_max          # r·ω_max      [m/s]
    a_lim = cfg.wheel_radius * cfg.wheel_a_max * dt     # r·α_max·dt   [m/s]

    b = W @ xp                      # wheel-space speed at ξ_prev
    d = W @ (u - xp)                # direction toward the requested command

    # If ξ_prev is itself outside the speed set the segment gives us nothing to
    # retreat along -- that can only happen on the first cycle after a state
    # reset or with the flag switched on mid-run, so aim at zero instead.
    if np.max(np.abs(b)) > w_lim + tol:
        b = np.zeros_like(b)
        d = W @ u
        xp = np.zeros_like(xp)

    lam = 1.0
    for bi, di in zip(b, d):
        if abs(di) > 1e-12:
            hi = (w_lim - bi) / di if di > 0 else (-w_lim - bi) / di
            lam = min(lam, max(hi, 0.0))
            lam = min(lam, a_lim / abs(di))
        elif abs(bi) > w_lim + tol:
            lam = 0.0
    lam = float(min(max(lam, 0.0), 1.0))
    return xp + lam * (u - xp), lam


def project_to_wheels(u, xi_prev, cfg: 'GMPCConfig', dt: float,
                      tol: float = 1e-9):
    """Closest point to `u` inside BOTH wheel sets. Returns (v, ok).

    Why this exists, when `fit_to_wheels` already enforces the same two sets:
    that one retreats along the SEGMENT ξ_prev → u with a single scalar λ, so
    ONE row that sits on the boundary and points outward collapses λ to 0 for
    ALL THREE degrees of freedom -- and λ = 0 means "hold the previous
    command", not "stop".

    Measured on `demo_heading_054503`: 229 of 300 solves (76.3%) returned a
    command bit-identical to the previous one, every one tagged
    `wheel_scaled`, with the same (+0.1887, -0.1191, -0.3626) held for over
    1.1 s while the QP was asking to REDUCE the yaw rate. The chassis kept
    turning past its heading target because a command it had already finished
    with could not be withdrawn.

    The feasible set is {v : |W v| <= r*w_max, |W (v - xi_prev)| <= r*a_max*dt}
    -- an intersection of half-spaces, convex, and (when xi_prev is itself
    feasible) non-empty. Projecting onto it keeps EVERY limit exactly as it
    was and lets the components that were coming back inside actually move.
    This is not a relaxation: the returned point satisfies the same rows the
    segment search checked, and the function verifies that before returning.

    `ok=False` means the projection did not produce a feasible point -- the
    caller must then fall back to the segment result rather than trust this.
    """
    u = np.asarray(u, dtype=float)
    xp = np.asarray(xi_prev, dtype=float)
    W = wheel_matrix(cfg)
    w_lim = cfg.wheel_radius * cfg.wheel_w_max
    a_lim = cfg.wheel_radius * cfg.wheel_a_max * dt
    # **投影集合要含全部四族限制**，不只輪級。先前只放了輪速與輪加速度，
    # 結果投影出來的解可以違反逐軸加速度框：反例是請求轉速變化 −0.1000，
    # 投影後變成 −0.105894，超過 a_max[2]·dt = 2×0.05 = 0.1。
    #   1. 輪速      |W v| ≤ r·ω_max
    #   2. 輪加速度  |W (v − ξ_prev)| ≤ r·α_max·dt
    #   3. 逐軸速度  u_min ≤ v ≤ u_max
    #   4. 逐軸加速度 |v − ξ_prev| ≤ a_max·dt
    I3 = np.eye(3)
    A = sparse.csc_matrix(np.vstack([W, W, I3, I3]))
    b_acc = W @ xp
    ax = np.asarray(cfg.a_max, float) * float(dt)
    lo = np.concatenate([np.full(4, -w_lim), -a_lim + b_acc,
                         np.asarray(cfg.u_min, float), xp - ax])
    hi = np.concatenate([np.full(4, w_lim), a_lim + b_acc,
                         np.asarray(cfg.u_max, float), xp + ax])
    P = sparse.csc_matrix(np.eye(3))
    q = -u
    try:
        pr = osqp.OSQP()
        pr.setup(P=P, q=q, A=A, l=lo, u=hi, verbose=False,
                 eps_abs=1e-9, eps_rel=1e-9, polish=False)
        r = pr.solve()
    except Exception:                                   # pragma: no cover
        return xp, False
    if r.info.status_val not in (1, 2) or r.x is None or not np.all(
            np.isfinite(r.x)):
        return xp, False
    v = np.asarray(r.x, dtype=float)
    # **自己驗一次**：解算器回報成功不等於點在集合裡
    wv = W @ v
    if np.max(np.abs(wv)) > w_lim + 1e-6:
        return xp, False
    if np.max(np.abs(wv - b_acc)) > a_lim + 1e-6:
        return xp, False
    if np.any(v < np.asarray(cfg.u_min, float) - 1e-6) or \
            np.any(v > np.asarray(cfg.u_max, float) + 1e-6):
        return xp, False
    if np.max(np.abs(v - xp) - ax) > 1e-6:
        return xp, False
    return v, True


# ---------------------------------------------------------------------------
# CBF: single-step velocity-layer Control Barrier Function for circular obstacles
# ---------------------------------------------------------------------------
# For each circular obstacle i at world position (ox, oy) with radius r_i, define
#     h_i(p) = ||p - o_i||² - (r_i + d_safe)²
# with the chassis at world position p = (px, py), body orientation θ.
# Because the chassis is a velocity-layer holonomic system,
#     ṗ_world = R(θ) · v_body,         v_body = (vx, vy)
# so
#     ḣ_i = 2 (p - o_i)ᵀ · R(θ) · v_body
# (h does NOT depend on θ for a point-mass obstacle, so the row for ω is zero.)
#
# Velocity CBF condition  ḣ + α·h ≥ 0  becomes a linear inequality in u_0:
#     [ 2 (p - o_i)ᵀ R(θ),  0 ] · u_0  ≥  -α · h_i
# We apply this to the *first* input  u_0 = ξ_ref(0) + δξ_0  only, which is the
# standard single-step CBF-QP "safety filter" (Ames et al. 2017). Receding-horizon
# re-solving every dt keeps the chassis safe at every control step. The constraint
# row touches only the first 3 columns of the decision vector z.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Prediction matrices
# ---------------------------------------------------------------------------

def _build_prediction(A_d: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Construct Φ and Γ for E = Φ·e_0 + Γ·z.

    A_d : (N, n, n) — discrete error A matrices, one per step
    """
    N, n, _ = A_d.shape
    m = n
    Phi   = np.zeros((N * n, n))
    Gamma = np.zeros((N * n, N * m))

    # Row block 0: e_1 = A_d[0]·e_0 + dt·δξ_0
    Phi[0:n, :]            = A_d[0]
    Gamma[0:n, 0:m]        = dt * np.eye(n)

    for i in range(1, N):
        # Φ[i] = A_d[i] · Φ[i-1]   (prefix-product update)
        Phi[i*n:(i+1)*n, :] = A_d[i] @ Phi[(i-1)*n:i*n, :]
        # Γ[i, j<i] = A_d[i] · Γ[i-1, j]
        Gamma[i*n:(i+1)*n, 0:i*m] = A_d[i] @ Gamma[(i-1)*n:i*n, 0:i*m]
        # Γ[i, i] = dt · I (new input enters)
        Gamma[i*n:(i+1)*n, i*m:(i+1)*m] = dt * np.eye(n)

    return Phi, Gamma


def _build_Q_bar(Q: np.ndarray, Qf: np.ndarray, N: int) -> np.ndarray:
    """Block-diagonal weight Q̄ = diag(Q, Q, ..., Q, Q_f), N blocks total."""
    n = Q.shape[0]
    Qb = np.zeros((N * n, N * n))
    for k in range(N - 1):
        Qb[k*n:(k+1)*n, k*n:(k+1)*n] = Q
    Qb[(N-1)*n:N*n, (N-1)*n:N*n] = Qf
    return Qb


def _build_R_bar(R: np.ndarray, N: int) -> np.ndarray:
    """Block-diagonal input weight R̄ = diag(R, ..., R), N blocks."""
    m = R.shape[0]
    Rb = np.zeros((N * m, N * m))
    for k in range(N):
        Rb[k*m:(k+1)*m, k*m:(k+1)*m] = R
    return Rb


def _build_smoothness_terms(S: np.ndarray, xi_ref_win: np.ndarray,
                            xi_prev: np.ndarray, N: int):
    """P/q additions for the input-increment cost  Σ_{k=0}^{N-1} Δu_k^T S Δu_k.

    With u_k = ξ_ref(k) + δξ_k and the decision vector z = [δξ_0,…,δξ_{N-1}]:
        Δu_0 = δξ_0           + (ξ_ref(0) - ξ_prev)        (u_{-1} = ξ_prev)
        Δu_k = δξ_k - δξ_{k-1}+ (ξ_ref(k) - ξ_ref(k-1)),  k ≥ 1
    Stack as  Δu = D·z + c, so the cost = (Dz+c)^T S̄ (Dz+c) contributes
        P += 2·Dᵀ S̄ D ,   q += 2·Dᵀ S̄ c        (OSQP form ½zᵀPz + qᵀz).
    The reference-difference offset c also smooths the FIRST step from the
    robot's actually-applied velocity ξ_prev (no jump at the seam).
    """
    m = S.shape[0]
    D = np.zeros((N * m, N * m))
    c = np.zeros(N * m)
    for k in range(N):
        D[k*m:(k+1)*m, k*m:(k+1)*m] = np.eye(m)
        if k == 0:
            c[0:m] = xi_ref_win[0] - xi_prev
        else:
            D[k*m:(k+1)*m, (k-1)*m:k*m] = -np.eye(m)
            c[k*m:(k+1)*m] = xi_ref_win[k] - xi_ref_win[k-1]
    S_bar = np.zeros((N * m, N * m))
    for k in range(N):
        S_bar[k*m:(k+1)*m, k*m:(k+1)*m] = S
    DtS = D.T @ S_bar
    return 2.0 * (DtS @ D), 2.0 * (DtS @ c)


def _err_jac(e_bar, h=1e-7):
    """（gmpc_wb 候選）M＝∂log(exp(ē)·exp(ε))/∂ε 於 ε＝0（中央差分；θ 取 wrap）：名目誤差 ε 對真參考誤差的一階對應。"""
    from .se2 import exp_, log_
    E = exp_(np.asarray(e_bar, float))
    M = np.zeros((3, 3))
    for i in range(3):
        d = np.zeros(3)
        d[i] = h
        a, b = log_(E @ exp_(d)), log_(E @ exp_(-d))
        dd = a - b
        dd[2] = math.atan2(math.sin(dd[2]), math.cos(dd[2]))
        M[:, i] = dd / (2 * h)
    return M


def _build_box_rows(box, X_ref_win, Phi, Gamma, e0, N):
    """（gmpc_wb 候選）底盤位置硬界限列：第 i＋1 步（i＝0..N−1，含第一未來步）的世界位置一階預測
    p ≈ p_ref,i+1 ＋ R_ref,i+1 · (Φ·e0 ＋ Γ·z)[3i:3i+2]，要求 x、y 各在 box 內。回傳 (A (2N×3N), l, u)。"""
    (xlo, xhi), (ylo, yhi) = box
    E0 = Phi @ e0
    rows, lo, hi = [], [], []
    for i in range(N):
        Xr = X_ref_win[i + 1]
        R, p = Xr[:2, :2], Xr[:2, 2]
        rows.append(R @ Gamma[3 * i:3 * i + 2, :])
        c = p + R @ E0[3 * i:3 * i + 2]
        lo += [xlo - c[0], ylo - c[1]]
        hi += [xhi - c[0], yhi - c[1]]
    return sparse.csc_matrix(np.vstack(rows)), np.array(lo), np.array(hi)


FACE_OPPOSE = 0.5                                       # 同原規則的 0.5：n·d̂ < −0.5 視為迎面


def _face_pref(P3, dd3, c, h_, dP, d_safe, slope, pv, stage):
    """（gmpc_wb 候選 face_rule='pref'）三維障礙分隔面的選面偏好，行進方向只作偏好、不硬排除整個軸。六面固定候選：
    背面一律不選：第 0 步本體均點在該面的對側面之外，且在另兩軸都落在盒的範圍內（要到達該面必須穿過障礙）。
    req＝d_safe − slope·切向距離 − 有號距離（同列的斜坡需求），取名目時域最大值；n·d̂＝法向與時域行進方向的餘弦（靜止時 0）。
    stage 'primary'：非迎面（n·d̂ ≥ −0.5）的面中，前一拍的面仍被名目滿足（req ≤ 0）就沿用；否則取 req 最小者（同分取 n·d̂ 大者）。
    stage 'fallback'（primary 求解不可行時，由 solve 重解一次）：名目已滿足的面（可含迎面，作為減速／保持分隔），前一拍優先，否則 n·d̂ 最大；
      都不滿足則沿用前一拍（非背面），再不然取 req 最小者。兩次都不可行 ⇒ 照原路徑停止。pv：前一拍採用的 (軸, 號) 或 None。"""
    p0 = P3[0].mean(axis=0)
    nd = float(np.linalg.norm(dP))
    cand = []
    for a_ in range(3):
        oth = [i for i in range(3) if i != a_]
        for sg_ in (-1.0, 1.0):
            if (-sg_ * (p0[a_] - c[a_]) > h_[a_]) and all(abs(float(p0[i] - c[i])) <= h_[i] for i in oth):
                continue
            rq = d_safe - slope * np.linalg.norm(np.maximum(dd3[:, :, oth], 0.0), axis=2) - (sg_ * (P3[:, :, a_] - c[a_]) - h_[a_])
            cand.append(dict(f=(a_, sg_), req=float(rq.max()), m_all=float((sg_ * (P3[:, :, a_] - c[a_]) - h_[a_]).min()),
                             nd=0.0 if nd < 0.02 else float(sg_ * dP[a_] / nd)))
    by = {x['f']: x for x in cand}
    if stage == 'primary':
        nonf = [x for x in cand if x['nd'] >= -FACE_OPPOSE]
        if pv in by and by[pv]['nd'] >= -FACE_OPPOSE and by[pv]['req'] <= 0.0:
            return pv
        if nonf:
            return min(nonf, key=lambda x: (x['req'], -x['nd']))['f']
    V = [x for x in cand if x['req'] <= 0.0]
    if pv in by and by[pv]['req'] <= 0.0:
        return pv
    if V:
        return max(V, key=lambda x: (x['nd'], x['m_all']))['f']
    if pv in by:
        return pv
    return min(cand, key=lambda x: x['req'])['f']


def _build_jerk_terms(S2: np.ndarray, xi_ref_win: np.ndarray, xi_prev: np.ndarray, xi_prev2: np.ndarray, N: int):
    """（gmpc_wb 候選）輸入二階差分成本 Σ_{k=0}^{N-1} Δ²u_kᵀ S2 Δ²u_k 的 P／q 增量，寫法同 _build_smoothness_terms。
    u_k = ξ_ref(k) + δξ_k，u_{-1} = xi_prev，u_{-2} = xi_prev2：
        Δ²u_0 = δξ_0                   + (ξ_ref(0) − 2ξ_prev + ξ_prev2)
        Δ²u_1 = δξ_1 − 2δξ_0            + (ξ_ref(1) − 2ξ_ref(0) + ξ_prev)
        Δ²u_k = δξ_k − 2δξ_{k-1} + δξ_{k-2} + (ξ_ref(k) − 2ξ_ref(k-1) + ξ_ref(k-2)),  k ≥ 2
    """
    m = S2.shape[0]
    D = np.zeros((N * m, N * m))
    c = np.zeros(N * m)
    I = np.eye(m)
    for k in range(N):
        D[k*m:(k+1)*m, k*m:(k+1)*m] = I
        if k >= 1:
            D[k*m:(k+1)*m, (k-1)*m:k*m] = -2.0 * I
        if k >= 2:
            D[k*m:(k+1)*m, (k-2)*m:(k-1)*m] = I
        um1 = xi_ref_win[k - 1] if k >= 1 else xi_prev
        um2 = xi_ref_win[k - 2] if k >= 2 else (xi_prev if k == 1 else xi_prev2)
        c[k*m:(k+1)*m] = xi_ref_win[k] - 2.0 * um1 + um2
    S_bar = np.kron(np.eye(N), S2)
    DtS = D.T @ S_bar
    return 2.0 * (DtS @ D), 2.0 * (DtS @ c)


# ---------------------------------------------------------------------------
# Constraints (velocity + acceleration)
# ---------------------------------------------------------------------------

_A_CACHE = {}


def _constraint_matrix(cfg):
    """（gmpc_wb）約束矩陣只由 N 與輪距決定，每週期相同；lil 逐格賦值佔求解時間最大宗，快取後上下界仍每週期重算。"""
    key = (cfg.N, bool(cfg.wheel_coupling), float(cfg.wheel_base_L) if cfg.wheel_coupling else 0.0)
    A = _A_CACHE.get(key)
    if A is not None:
        return A
    N, m = cfg.N, 3
    Nm = N * m
    A_acc = sparse.lil_matrix((Nm, Nm))
    A_acc[0:m, 0:m] = sparse.eye(m)
    for k in range(1, N):
        A_acc[k*m:(k+1)*m, (k-1)*m:k*m] = -sparse.eye(m)
        A_acc[k*m:(k+1)*m, k*m:(k+1)*m] =  sparse.eye(m)
    blocks = [sparse.eye(Nm), A_acc.tocsr()]
    if cfg.wheel_coupling:
        L = cfg.wheel_base_L
        W = np.array([[0.0,  1.0, L],
                      [-1.0, 0.0, L],
                      [0.0, -1.0, L],
                      [1.0,  0.0, L]])
        nw = 4
        A_wv = sparse.lil_matrix((N * nw, Nm))
        A_wa = sparse.lil_matrix((N * nw, Nm))
        for k in range(N):
            r0, r1 = k * nw, (k + 1) * nw
            A_wv[r0:r1, k*m:(k+1)*m] = W
            A_wa[r0:r1, k*m:(k+1)*m] = W
            if k > 0:
                A_wa[r0:r1, (k-1)*m:k*m] = -W
        blocks += [A_wv.tocsr(), A_wa.tocsr()]
    A = sparse.vstack(blocks, format='csc')
    _A_CACHE[key] = A
    return A


def _build_constraints(cfg: GMPCConfig,
                       xi_ref_win: np.ndarray,
                       xi_prev: np.ndarray):
    """Build (A, l, u) for OSQP standard form  l ≤ A·z ≤ u.

    A is sparse, m_cons × (N·3) where m_cons = 2N·3 (velocity + acceleration).
    """
    N, dt = cfg.N, cfg.dt
    m = 3
    Nm = N * m

    # ---- Velocity: identity on z ---------------------------------------
    lb_vel = np.zeros(Nm)
    ub_vel = np.zeros(Nm)
    for k in range(N):
        lb_vel[k*m:(k+1)*m] = cfg.u_min - xi_ref_win[k]
        ub_vel[k*m:(k+1)*m] = cfg.u_max - xi_ref_win[k]

    # ---- Acceleration: first row uses ξ_prev, rest are δξ_k - δξ_{k-1} -
    # row 0:    δξ_0           ∈ [-a·dt - ξ_ref(0) + ξ_prev,  +a·dt - ξ_ref(0) + ξ_prev]
    # row k>0:  δξ_k - δξ_{k-1} ∈ [-a·dt - ξ_ref(k) + ξ_ref(k-1), +a·dt - ξ_ref(k) + ξ_ref(k-1)]
    lb_acc = np.zeros(Nm)
    ub_acc = np.zeros(Nm)

    lb_acc[0:m] = -cfg.a_max * dt - xi_ref_win[0] + xi_prev
    ub_acc[0:m] =  cfg.a_max * dt - xi_ref_win[0] + xi_prev

    for k in range(1, N):
        lb_acc[k*m:(k+1)*m] = -cfg.a_max * dt - xi_ref_win[k] + xi_ref_win[k-1]
        ub_acc[k*m:(k+1)*m] =  cfg.a_max * dt - xi_ref_win[k] + xi_ref_win[k-1]

    lbs    = [lb_vel, lb_acc]
    ubs    = [ub_vel, ub_acc]

    # ---- Wheel-speed coupling: 4 rows per step, on top of the box ---------
    # r·ω = W·u, so |W·u_k| ≤ r·ω_max and |W·(u_k - u_{k-1})| ≤ r·α_max·dt.
    # These do not replace the box -- the box still carries per-axis limits the
    # user may set tighter than the wheels allow -- they cut the corners off it.
    if cfg.wheel_coupling:
        L = cfg.wheel_base_L
        W = np.array([[0.0,  1.0, L],
                      [-1.0, 0.0, L],
                      [0.0, -1.0, L],
                      [1.0,  0.0, L]])
        w_lim = cfg.wheel_radius * cfg.wheel_w_max          # r·ω_max  [m/s]
        a_lim = cfg.wheel_radius * cfg.wheel_a_max * dt     # r·α_max·dt

        nw = 4
        lb_wv = np.zeros(N * nw); ub_wv = np.zeros(N * nw)
        lb_wa = np.zeros(N * nw); ub_wa = np.zeros(N * nw)

        for k in range(N):
            r0, r1 = k * nw, (k + 1) * nw
            off = W @ xi_ref_win[k]
            lb_wv[r0:r1] = -w_lim - off
            ub_wv[r0:r1] =  w_lim - off

            if k == 0:
                d_ref = W @ (xi_ref_win[0] - xi_prev)
            else:
                d_ref = W @ (xi_ref_win[k] - xi_ref_win[k-1])
            lb_wa[r0:r1] = -a_lim - d_ref
            ub_wa[r0:r1] =  a_lim - d_ref

        lbs    += [lb_wv, lb_wa]
        ubs    += [ub_wv, ub_wa]

    A_total  = _constraint_matrix(cfg)
    lb_total = np.concatenate(lbs)
    ub_total = np.concatenate(ubs)
    return A_total, lb_total, ub_total


def _psd_project_2x2(H: np.ndarray) -> np.ndarray:
    """Clip the eigenvalues of a symmetric 2x2 matrix at zero.

    A Gaussian bump is CONCAVE near its peak, so its Hessian is negative
    definite there and only turns positive in the radial direction beyond one
    sigma. Adding it raw to P makes the QP non-convex and OSQP either fails or
    returns nonsense. Projecting onto the PSD cone (the Gauss-Newton style
    approximation used throughout nonlinear MPC) keeps the exact gradient while
    guaranteeing the quadratic term can only ever help convexity. Closed form
    for 2x2, so the cost is negligible.
    """
    a, b, d = H[0, 0], H[0, 1], H[1, 1]
    tr, det = a + d, a * d - b * b
    disc = max(0.0, tr * tr / 4.0 - det) ** 0.5
    l1, l2 = tr / 2.0 + disc, tr / 2.0 - disc
    if l1 <= 0.0:
        return np.zeros((2, 2))
    if l2 >= 0.0:
        return H
    # keep only the positive eigenpair
    if abs(b) > 1e-12:
        v = np.array([l1 - d, b])
    else:
        v = np.array([1.0, 0.0]) if a >= d else np.array([0.0, 1.0])
    v = v / np.linalg.norm(v)
    return l1 * np.outer(v, v)


def _build_spacetime_cost(cfg: GMPCConfig, X_ref_win, Phi, Gamma, e0, obstacles):
    """Quadratic model of the spatio-temporal cost field, in OSQP form.

    The horizon position is p(k) = p_ref(k) + R_ref(k) e_xy(k), and the geodesic
    error follows e = Phi e0 + Gamma z, so the world-frame deviation is affine
    in the decision variable:

        dp_k = A_k z + b_k ,   A_k = R_ref(k) S_k Gamma ,  b_k = R_ref(k) S_k Phi e0

    Expanding C_k to second order about p_ref(k) and substituting gives a rank-2
    update per step, so nothing is added to the decision dimension:

        P += sum_k A_k^T H_k A_k ,   q += sum_k A_k^T (g_k + H_k b_k)

    Note e_xy is expressed in the REFERENCE frame, hence the R_ref rotation;
    dropping it would put the gradient in the wrong direction whenever the
    reference heading is not zero.
    """
    N, dt, m = cfg.N, cfg.dt, 3
    Nm = N * m
    P_add = np.zeros((Nm, Nm))
    q_add = np.zeros(Nm)
    if not obstacles or cfg.st_weight <= 0.0:
        return P_add, q_add

    # (1) SCALE. st_weight IS the peak cost of one obstacle's bump, directly
    # comparable to the per-step tracking cost e^T Q e. That reference is small:
    # a 0.1 m deviation against Q = 15 costs only 0.15, so a sensible st_weight
    # is order 1, not order 100.
    #
    # An earlier version multiplied by sigma^2 * trace(Q)/2 to "anchor it to the
    # tracking weight", which put W = 30 at an effective 200 -- three orders of
    # magnitude above the cost it was meant to sit beside. The field then simply
    # overrode tracking: in the 2D sandbox the robot swung wide (path 27.6 ->
    # 48.9 m, clearance +0.45 m) and never reached the goal, and clamping only
    # the gradient turned that into oscillation instead (5226 deg/m). Scale
    # first, clamp second.
    # (2) CLAMPS on both derivatives. Clamping only the Hessian is not enough:
    # the GRADIENT lands in q, which is what actually drives the solution. With
    # W=30 the normalised weight reaches ~200, so |grad| peaks near 200 while the
    # tracking q is single digits -- the field then dominates completely and the
    # robot behaves like a pure repulsive-field escaper. Measured in the 2D
    # sandbox: clearance ballooned to +0.45 m, path length went 27.6 -> 48.9 m,
    # heading change 192 -> 1389 deg/m, and it never reached the goal. Capping
    # both derivatives at the tracking scale keeps the field an INFLUENCE rather
    # than an override, which is the whole point of layering it under the CBF.
    # Clamp each step's derivatives at the scale of the tracking term evaluated
    # at a typical 0.1 m deviation, so one very close obstacle cannot swamp the
    # objective even if st_weight is set too high.
    q_ref = float(np.max(np.abs(cfg.Q))) * 0.01
    h_cap = 2.0 * q_ref
    g_cap = 2.0 * q_ref

    b_all = Phi @ e0                                   # (Nm,)
    for k in range(N):
        rows = slice(k * m, k * m + 2)                 # xy part of block k
        R_k = X_ref_win[k][:2, :2]
        A_k = R_k @ Gamma[rows, :]                     # (2, Nm)
        # Linearise about the PREDICTED position, not the reference position.
        # p(k) = p_ref(k) + R_k (Phi e0 + Gamma z)_xy = p_lin(k) + A_k z, so
        # expanding about p_lin absorbs the offset exactly and leaves no b term.
        # Expanding about p_ref instead evaluates the gradient wherever the
        # REFERENCE is, which is fine while the robot tracks it but wrong as soon
        # as it deviates -- and it deviates most while avoiding. Measured in
        # closed loop, the push direction was correct 94.7% of the time at
        # 1.2-1.8 m but only 47.5% at under 0.8 m, i.e. a coin flip exactly when
        # it matters, because the reference there sits on the far side of the
        # obstacle from the robot.
        p_lin = X_ref_win[k][:2, 2] + R_k @ b_all[rows]
        sig = cfg.st_sigma0 * (1.0 + cfg.st_growth * k)
        s2 = sig * sig
        g_tot = np.zeros(2)
        H_tot = np.zeros((2, 2))
        for obs in obstacles:
            o = np.array([float(obs['x']) + float(obs.get('vx', 0.0)) * k * dt,
                          float(obs['y']) + float(obs.get('vy', 0.0)) * k * dt])
            d = p_lin - o
            c = cfg.st_weight * math.exp(-float(d @ d) / (2.0 * s2))
            if c < 1e-9:                               # negligible this far out
                continue
            g_tot += c * (-d / s2)
            H_tot += c * (np.outer(d, d) / (s2 * s2) - np.eye(2) / s2)
        if not H_tot.any() and not g_tot.any():
            continue
        gn = float(np.linalg.norm(g_tot))
        if gn > g_cap:                                 # clamp, keep direction
            g_tot = g_tot * (g_cap / gn)
        H_psd = _psd_project_2x2(0.5 * (H_tot + H_tot.T))
        nrm = float(np.linalg.norm(H_psd, 2))
        if nrm > h_cap:
            H_psd *= h_cap / nrm
        P_add += A_k.T @ H_psd @ A_k
        q_add += A_k.T @ g_tot
    return P_add, q_add


# ---------------------------------------------------------------------------
# CBF row builder — full-horizon receding CBF with per-step soft slack
# ---------------------------------------------------------------------------
#
# For each obstacle i and each prediction step k = 0, 1, ..., N-1:
#
#   Predicted obstacle position (constant-velocity assumption):
#       o_i(k) = o_i(0) + v_i · k · dt
#
#   Robot linearisation point:
#       k = 0  → use X_now (the actual current pose)
#       k ≥ 1  → use X_ref_win[k] (the reference path nominal pose)
#
#   Barrier value at step k:
#       h_i(k) = ‖ p(k) − o_i(k) ‖² − ( r_i + d_safe )²
#
#   Velocity-layer CBF (continuous form):
#       ḣ_i(k) + α · h_i(k)  ≥  −ε_k
#       where  ḣ_i(k) = 2 (p(k) − o_i(k))ᵀ ( R(θ(k)) · v_body(k) − v_obs(k) )
#
#   Substituting  v_body(k) = ξ_ref(k) + δξ_k  and rearranging into the form
#       [grad_body(k)] · δξ_k  +  ε_k  ≥  b(k)
#   we obtain a *linear* row in (δξ_k, ε_k):
#
#       grad_body(k) = 2 (p(k) − o_i(k))ᵀ · R(θ(k))      ∈ R^{1×2}
#       b(k)         = 2 (p(k) − o_i(k))ᵀ · v_obs(k)
#                    − α · h_i(k)
#                    − [grad_body(k), 0] · ξ_ref(k)
#
# ε_0 is **hard-wired to 0**:  the first step is the safety-critical "now"
# step; allowing any slack there would let the controller crash in the next
# 50 ms tick. ε_k for k ≥ 1 carry an L2 penalty  ρ · ε_k²  so the QP only uses
# them when the look-ahead horizon is genuinely infeasible.

def _build_cbf_horizon(cfg          : GMPCConfig,
                       X_now        : np.ndarray,
                       X_ref_win    : np.ndarray,
                       xi_ref_win   : np.ndarray,
                       obstacles    ,
                       slack_dim    : int):
    """Build full-horizon CBF rows for the augmented decision

        z = [δξ_0, δξ_1, ..., δξ_{N-1}, ε_1, ε_2, ..., ε_{N-1}]

    where the ε's live in columns [Nm : Nm + slack_dim], slack_dim = N − 1.

    Returns
    -------
    A_cbf  : sparse (n_obs · N, Nm + slack_dim)
    l_cbf  : (n_obs · N,)
    u_cbf  : (n_obs · N,)   all +∞
    h_now  : (n_obs,)       current-step barrier values (for /gmpc/min_h)
    """
    n_obs = len(obstacles)
    N     = cfg.N
    Nm    = N * 3
    dt    = cfg.dt
    if n_obs == 0:
        return (sparse.csr_matrix((0, Nm + slack_dim)),
                np.zeros(0), np.zeros(0), np.zeros(0))

    # Which horizon steps to emit rows for. k=0 always; near steps always;
    # beyond that, every cbf_far_stride-th step.
    stride = max(1, int(cfg.cbf_far_stride))
    steps = [k for k in range(N)
             if k <= cfg.cbf_near_steps or (k - cfg.cbf_near_steps) % stride == 0]
    rows_per_obs = len(steps)

    n_rows = n_obs * rows_per_obs
    A      = np.zeros((n_rows, Nm + slack_dim))
    l_vec  = np.zeros(n_rows)
    u_vec  = np.full(n_rows, np.inf)
    h_now  = np.zeros(n_obs)
    keep_rows = []          # rows actually filled, after the distance gate
    # gmpc_node tags wall points from the map-based static-CBF with their own
    # (smaller) 'margin'; everything else is a tracked dynamic obstacle.
    is_static = [bool(o.get('static', 'margin' in o)) for o in obstacles]

    for i, obs in enumerate(obstacles):
        ox, oy = float(obs['x']),   float(obs['y'])
        vox    = float(obs.get('vx', 0.0))
        voy    = float(obs.get('vy', 0.0))
        # Per-obstacle safety margin: static wall points carry a smaller margin
        # ('margin' key) than dynamic obstacles so the robot can thread narrow
        # passages without the static keep-out boxing it in. Falls back to the
        # global cfg.cbf_safe_margin when no per-obstacle margin is given.
        m_i = float(obs.get('margin', cfg.cbf_safe_margin))
        static_i = is_static[i]

        # Distance gate: the closest this point comes to the reference over the
        # whole horizon. Beyond cbf_prune_range it cannot be violated, so its
        # rows are pure cost. h_now is still computed below, so /gmpc/min_h and
        # the danger gain-scheduling see every point regardless.
        if cfg.cbf_prune_range > 0.0:
            far = True
            for k in range(0, N, 4):
                p_k = (X_now if k == 0 else X_ref_win[k])[:2, 2]
                if (math.hypot(p_k[0] - (ox + vox * k * dt),
                               p_k[1] - (oy + voy * k * dt))
                        <= cfg.cbf_prune_range):
                    far = False
                    break
        else:
            far = False

        for j, k in enumerate(steps):
            if far and k != 0:
                continue
            # Robot linearisation pose at step k
            if k == 0:
                X_k = X_now
            else:
                X_k = X_ref_win[k]
            R_k = X_k[:2, :2]
            p_k = X_k[:2, 2]

            # Predicted obstacle position at step k (constant velocity)
            ox_k = ox + vox * k * dt
            oy_k = oy + voy * k * dt
            diff = p_k - np.array([ox_k, oy_k])

            # Keep-out grows with the horizon step for MOVING obstacles only:
            # a wall point has v = 0, so its predicted position carries no
            # extrapolation error and inflating it would just squeeze corridors.
            r_eff = float(obs['radius']) + m_i * (
                1.0 if static_i else 1.0 + cfg.cbf_margin_growth * k)

            # ... and grows again with how fast the gap is CLOSING.
            #
            # A fixed keep-out has to be sized for the worst approach, which
            # then squeezes every corridor: at 0.75 m measured arrival dropped
            # and contacts rose, because a 2.0 m doorway leaves only 0.5 m of
            # feasible corridor and the QP falls back on slack. Sizing it by
            # the closing rate instead spends the margin only where it is
            # needed. The amount comes from braking distance: at a_y = 0.6 and
            # a closing rate of 0.52 m/s that is 0.52^2/(2*0.6) = 0.225 m, and
            # the perception path adds about 0.3 s of lag = 0.16 m. Measured
            # over 18 contacts, acceleration sat at its limit for 26% of the
            # 1.5 s before impact (83% in the worst trial) -- the barrier was
            # not being ignored, it was being asked for more deceleration than
            # the base can produce.
            #
            # Only the CLOSING component counts: an obstacle crossing sideways
            # needs no extra room, and inflating it would cost corridor width
            # for nothing. Static points are excluded for the same reason the
            # horizon growth excludes them.
            if not static_i and cfg.cbf_vel_margin_gain > 0.0:
                n2 = float(diff @ diff)
                if n2 > 1e-9:
                    u = diff / math.sqrt(n2)          # obstacle -> robot
                    v_rob = R_k @ xi_ref_win[k][:2]   # reference twist, world
                    closing = float((np.array([vox, voy]) - v_rob) @ u)
                    if closing > 0.0:
                        r_eff += cfg.cbf_vel_margin_gain * closing

            # Barrier value
            h_k = float(diff @ diff - r_eff * r_eff)
            if k == 0:
                h_now[i] = h_k

            # Linearised CBF row on δξ_k
            grad_body = 2.0 * (diff @ R_k)                # shape (2,)
            A_row_3   = np.array([grad_body[0], grad_body[1], 0.0])

            row = i * rows_per_obs + j
            keep_rows.append(row)
            A[row, 3*k:3*k + 3] = A_row_3
            # Slack column. Each step has its own ε_k (including k=0, whose
            # penalty is 100x higher so the "now" constraint is near-hard while
            # the QP stays feasible if the robot has drifted inside the zone).
            # When slack_dim == 2N the step's slacks are split in two blocks:
            # [0:N] for dynamic obstacles, [N:2N] for static wall points, the
            # latter penalised cbf_static_slack_scale times harder so a pinched
            # robot gives up dynamic clearance instead of hitting the wall.
            A[row, Nm + k + (N if (is_static[i] and slack_dim >= 2 * N) else 0)] = 1.0

            # Lower bound from rearranged CBF inequality
            l_vec[row] = (
                2.0 * (diff[0] * vox + diff[1] * voy)
                - cfg.cbf_alpha * h_k
                - A_row_3 @ xi_ref_win[k]
            )

    if cfg.cbf_prune_range > 0.0 and len(keep_rows) < n_rows:
        idx = np.array(sorted(keep_rows), dtype=int)
        A, l_vec, u_vec = A[idx], l_vec[idx], u_vec[idx]
    return sparse.csr_matrix(A), l_vec, u_vec, h_now


# ---------------------------------------------------------------------------
# Top-level solver class
# ---------------------------------------------------------------------------

class GMPC:
    """One-shot GMPC solver (rebuilds QP each call).

    Suitable for the offline prototype — correctness first, OSQP warm-start
    + matrix-update optimisation can come later when porting to ROS2.
    """

    def __init__(self, cfg: GMPCConfig):
        self.cfg = cfg

    def solve(self,
              X_now      : np.ndarray,
              X_ref_win  : np.ndarray,
              xi_ref_win : np.ndarray,
              xi_prev    : np.ndarray,
              obstacles  : list | None = None,
              arm        : dict | None = None,
              xi_prev2   : np.ndarray | None = None,
              cost_ref   : tuple | None = None,
              u0_rows    : tuple | None = None,
              ) -> GMPCResult:
        """
        obstacles : optional list of dicts {x, y, radius} in world frame.
                    If non-empty, a CBF inequality is added per obstacle on δξ_0.
        arm       : （gmpc_wb）None ⇒ 與封存 gmpc.py 完全相同的問題。{'mode': 'hold'} ⇒ 手臂速度序列固定為 0：依結構**消去**這些
                    等式為零的決策變數（r2，Codex 20261031：不交 120 個零變數給 OSQP），QP 與封存版相同；結果另帶 arm_mode、u_arm_seq＝0。
                    r1（已撤）曾附加 u_a 並以等式固定，底盤區塊逐元素等價但未收斂案求解路徑不同。
                    {'mode': 'active', 'K', 'tcp', 'q_arm', 's_arm', 'u_arm_prev', 'model', 'T_seq'(N+1,4,4), 'w_p', 'w_r', 'w_a', 'w_s',
                     'v_arm', 'a_arm', 'lo', 'hi', 'margin', 'u_arm_nom'(可省, N×6)} ⇒ 決策末端加 u_a（N×6）：設定點模型預測（實測角＋設定點）、
                    每步 TCP 誤差在名目點（底盤沿參考＋e_nom、手臂 u_arm_nom）單次線性化，底盤增量 δe 經 X_ref·exp 轉世界位移後與手臂
                    共用同一全身 Jacobian；手臂速度框、對上一筆的加速度框、實測角與設定點限位（±margin）。底盤原成本／限制／CBF 不動。
        """
        if arm is not None and arm.get('mode') == 'active' and not arm.get('_inner'):
            # （gmpc_wb）手臂作用模式不得削弱封存導航的安全判定：先解與封存版相同的底盤問題（hold，已證逐位元相同）。
            #   底盤不可行（煞停）⇒ 照封存版煞停、手臂歸零；耦合問題未完全收斂（status ≠ solved）⇒ 退回 hold 底盤解、手臂保持。
            #   動機：硬 k0 過近案封存版判 primal infeasible 煞停，耦合問題卻在迭代上限回傳朝障礙前進的命令（CBF 殘差 −0.55）。
            import dataclasses as _dc
            r_hold = self.solve(X_now, X_ref_win, xi_ref_win, xi_prev, obstacles, arm={'mode': 'hold'}, xi_prev2=xi_prev2, cost_ref=cost_ref, u0_rows=u0_rows)
            N_ = self.cfg.N
            def _fb(why, coupled_status=None, qp=None):
                U_s, ck = arm_stop_sequence(arm, N_, self.cfg.dt)
                # ck 只驗手臂限制；回傳底盤來自 hold 問題（可能已被煞停或後處理改過），與手臂減速序列不是同一問題的解 ⇒ seq_consistent 一律 False，兩者分列
                return _dc.replace(r_hold, arm_mode='active', u_arm_seq=U_s,
                                   arm_check=dict(fallback=why, coupled_status=coupled_status, finite=True, seq_consistent=False, arm_decel_valid=bool(ck['verified']),
                                                  base_as_is=bool(r_hold.accept_action == 'as_is'), failed=not ck['verified'], qp=qp,
                                                  max_violation=max(ck['violations'].values()), fallback_check=ck,
                                                  note='退回 hold 底盤解；手臂有界減速序列（未驗證通過 ⇒ failed，呼叫端須終止）'))
            if r_hold.accept_action == 'brake':
                return _fb('base_brake')
            bo0 = arm.get('body_obst')
            pref = bool(bo0) and bo0.get('face_rule') == 'pref'
            arm_i = dict(arm, _inner=True, body_obst=dict(bo0, face_stage='primary', _picks={})) if pref else dict(arm, _inner=True)
            r = self.solve(X_now, X_ref_win, xi_ref_win, xi_prev, obstacles, arm=arm_i, xi_prev2=xi_prev2, cost_ref=cost_ref, u0_rows=u0_rows)
            face_stage = 'primary'
            if pref and r.status != 'solved':                          # （候選）選面主選不可行 ⇒ 退回已滿足面重解一次（固定預算 2 次），仍不可行照原路徑停止
                arm_i = dict(arm, _inner=True, body_obst=dict(bo0, face_stage='fallback', _picks={}))
                r = self.solve(X_now, X_ref_win, xi_ref_win, xi_prev, obstacles, arm=arm_i, xi_prev2=xi_prev2, cost_ref=cost_ref, u0_rows=u0_rows)
                face_stage = 'fallback'
            if r.status != 'solved':
                return _fb('coupled_not_solved', r.status, (r.arm_check or {}).get('qp'))
            if pref:
                bo0.setdefault('face_prev', {}).update(arm_i['body_obst'].get('_picks', {}))
                r.arm_check = dict(r.arm_check or {}, face_stage=face_stage)
            ck = r.arm_check or {}
            # 實際回傳命令核對：有限值、手臂限制原單位違反、底盤後處理未改動；任一不過即退回，不只記錄
            if not (np.isfinite(np.asarray(r.u_opt, float)).all() and r.u_arm_seq is not None and np.isfinite(r.u_arm_seq).all() and ck.get('finite') is True):
                return _fb('returned_nonfinite', r.status)
            if not (ck.get('max_violation', float('inf')) <= arm.get('r_tol', 1e-6)):
                return _fb('arm_limit_violation', r.status)
            if ck.get('seq_consistent') is not True:
                return _fb('base_postprocess_changed', r.status)
            r.arm_check = dict(ck, fallback=None, failed=False, hold_status=r_hold.status)
            return r
        cfg = self.cfg
        N, dt = cfg.N, cfg.dt
        n = 3

        # 1. Current error
        e0 = geodesic_error(X_ref_win[0], X_now)

        # 2. Per-step error A matrices  (multi-point linearisation)
        A_d = np.zeros((N, n, n))
        I3  = np.eye(n)
        for k in range(N):
            A_d[k] = I3 - dt * ad(xi_ref_win[k])

        # 3. Prediction
        Phi, Gamma = _build_prediction(A_d, dt)

        # 4. Cost weights — with danger-aware gain scheduling.
        #    Compute min_h NOW (cheap loop over obstacles, no QP needed) and
        #    interpolate between nominal and "panic" weights.
        danger = 0.0                            # 0 = safe, 1 = at boundary
        if obstacles:
            p_now = X_now[:2, 2]
            h_min_for_scaling = float('inf')
            for obs in obstacles:
                diff = p_now - np.array([float(obs['x']), float(obs['y'])])
                r_eff = float(obs['radius']) + float(obs.get('margin', cfg.cbf_safe_margin))
                h_i = float(diff @ diff - r_eff * r_eff)
                if h_i < h_min_for_scaling:
                    h_min_for_scaling = h_i
            if h_min_for_scaling < cfg.cbf_danger_thresh:
                # ramp from 0 (h = thresh) to 1 (h ≤ 0)
                danger = max(0.0, 1.0 - max(0.0, h_min_for_scaling) / cfg.cbf_danger_thresh)

        Q_scale     = 1.0 - (1.0 - cfg.cbf_Q_min_scale)   * danger
        slack_scale = 1.0 + (cfg.cbf_slack_max_scale - 1.0) * danger
        slack_weight_eff = cfg.cbf_slack_weight * slack_scale
        Q_eff   = Q_scale * cfg.Q
        Qf_eff  = Q_scale * cfg.Qf

        Q_bar = _build_Q_bar(Q_eff, Qf_eff, N)
        R_bar = _build_R_bar(cfg.R, N)

        if cost_ref is None:
            P_dense = 2.0 * (Gamma.T @ Q_bar @ Gamma + R_bar)
            q_vec   = 2.0 * Gamma.T @ Q_bar @ Phi @ e0
        else:
            # （候選）沿名目軌跡重線性化：X_ref_win／xi_ref_win 是線性化用的名目（動態、CBF、窗口、輪級、手臂 Jacobian 都繞它），
            # 成本仍對真參考 cost_ref＝(X_true (N+1,3,3), xi_true (N+1,3))：e_true,k ≈ ē_k ＋ M_k ε_k，ē_k＝log(X_true,k⁻¹·X̄_k)，
            # M_k＝∂log(exp(ē_k)·exp(ε))/∂ε（數值）；輸入成本 R 對 (u − ξ_true)＝δ ＋ (ū − ξ_true)。
            Xt, xt = cost_ref
            Mb = np.zeros((N * n, N * n))
            eb = np.zeros(N * n)
            for k in range(N):
                ek_bar = geodesic_error(np.asarray(Xt[k + 1], float), np.asarray(X_ref_win[k + 1], float))
                Mb[k * n:(k + 1) * n, k * n:(k + 1) * n] = _err_jac(ek_bar)
                eb[k * n:(k + 1) * n] = ek_bar
            G2 = Mb @ Gamma
            c2 = Mb @ (Phi @ e0) + eb
            P_dense = 2.0 * (G2.T @ Q_bar @ G2 + R_bar)
            q_vec   = 2.0 * G2.T @ Q_bar @ c2 + 2.0 * R_bar @ (np.asarray(xi_ref_win, float)[:N] - np.asarray(xt, float)[:N]).ravel()
        # Forward-progress preference: reward velocity along the reference
        # tangent so that, when forced aside, the QP prefers to give way in the
        # direction of travel rather than backwards.
        if cfg.prog_weight > 0.0:
            for k in range(N):
                t_w = X_ref_win[k][:2, :2] @ np.array([1.0, 0.0])   # tangent
                nrm = float(np.linalg.norm(t_w))
                if nrm < 1e-9:
                    continue
                t_b = X_ref_win[k][:2, :2].T @ (t_w / nrm)          # body frame
                q_vec[3 * k:3 * k + 2] -= cfg.prog_weight * t_b
        # Spatio-temporal cost field — proactive, soft, and independent of the
        # hard CBF so the two can be switched on and off separately.
        if cfg.st_weight > 0.0 and obstacles:
            P_st, q_st = _build_spacetime_cost(
                cfg, X_ref_win, Phi, Gamma, e0, obstacles)
            P_dense = P_dense + P_st
            q_vec   = q_vec + q_st
        # Input-increment (Δu) smoothness cost — soft penalty on control jerk.
        if np.any(cfg.S):
            P_s, q_s = _build_smoothness_terms(cfg.S, xi_ref_win, xi_prev, N)
            P_dense = P_dense + P_s
            q_vec   = q_vec + q_s
        # （候選）輸入二階差分成本：S2＝0 時整段跳過，問題與先前逐位元相同；S2 非零卻沒給前兩筆命令 ⇒ 拒絕，不默默降成一階
        if np.any(cfg.S2):
            if xi_prev2 is None:
                raise ValueError('S2 非零需要 xi_prev2（預測起點前第二筆已套用／在途命令）')
            P_j, q_j = _build_jerk_terms(cfg.S2, xi_ref_win, xi_prev, xi_prev2, N)
            P_dense = P_dense + P_j
            q_vec   = q_vec + q_j
        # Symmetrise to absorb floating-point asymmetry (OSQP wants symmetric P)
        P_dense = 0.5 * (P_dense + P_dense.T)

        # 5. Constraints (velocity + acceleration)
        A_total, lb_total, ub_total = _build_constraints(cfg, xi_ref_win, xi_prev)
        # 5a.（候選）底盤位置硬界限：只在 δξ 欄，之後的鬆弛欄與手臂欄補零時一併補上；hold 與耦合問題都含
        if cfg.base_box is not None:
            A_box, l_box, u_box = _build_box_rows(cfg.base_box, X_ref_win, Phi, Gamma, e0, N)
            A_total  = sparse.vstack([A_total, A_box], format='csc')
            lb_total = np.concatenate([lb_total, l_box])
            ub_total = np.concatenate([ub_total, u_box])
        # 5a'.（候選）第一拍命令接受列 A·u_0 ≤ b（u_0＝ξ_ref,0＋δξ_0，只在 δξ_0 三欄）：None＝不加，問題與先前逐位元相同
        if u0_rows is not None:
            A_u0, b_u0 = np.asarray(u0_rows[0], float).reshape(-1, 3), np.asarray(u0_rows[1], float).ravel()
            if len(b_u0):
                A_r = np.zeros((len(b_u0), N * 3))
                A_r[:, :3] = A_u0
                A_total  = sparse.vstack([A_total, sparse.csc_matrix(A_r)], format='csc')
                lb_total = np.concatenate([lb_total, np.full(len(b_u0), -np.inf)])
                ub_total = np.concatenate([ub_total, b_u0 - A_u0 @ np.asarray(xi_ref_win[0], float)])

        # 5b. Full-horizon CBF safety filter — receding QP with per-step slack.
        #
        #     z = [δξ_0, δξ_1, ..., δξ_{N-1},  ε_0, ε_1, ..., ε_{N-1}]
        #                                       ^^^^^^^^^^^^^^^^^^^^^^^^^
        #     ε_0 has 100× higher penalty (cbf_eps0_scale × slack_weight_eff)
        #     so the QP treats it as "near-hard". This lets the solver still
        #     find a solution when the robot is already inside the safety
        #     zone (avoiding primal-infeasible crashes that would otherwise
        #     freeze the robot forever).
        Nm          = N * n
        cbf_active  = 0
        min_h       = float('inf')
        n_obs_in = n_static_in = 0
        min_h_static = min_h_dynamic = float('inf')
        A_cbf_keep = l_cbf_keep = None
        eps0 = 0.0
        resid_noslack = resid_slack = float('inf')
        n_slack     = 0
        if obstacles:
            # One slack per step for dynamic rows, and (when the static block is
            # priced differently) a second per-step slack for the wall rows.
            split_slack = (cfg.cbf_static_slack_scale != 1.0
                           or cfg.cbf_hard_k0_static) and \
                any(o.get('static', 'margin' in o) for o in obstacles)
            n_slack = 2 * N if split_slack else N
            A_cbf, l_cbf, u_cbf, h_now = _build_cbf_horizon(
                cfg, X_now, X_ref_win, xi_ref_win, obstacles, n_slack,
            )
            # Same static/dynamic split _build_cbf_horizon uses, so the entry
            # diagnostic and the rows cannot disagree about what a point is.
            stat_mask = np.array(
                [bool(o.get('static', 'margin' in o)) for o in obstacles], bool)
            n_obs_in    = len(obstacles)
            n_static_in = int(stat_mask.sum())
            if len(h_now):
                if stat_mask.any():
                    min_h_static = float(np.min(h_now[stat_mask]))
                if (~stat_mask).any():
                    min_h_dynamic = float(np.min(h_now[~stat_mask]))
            if A_cbf.shape[0] > 0:
                cbf_active = A_cbf.shape[0]
                min_h      = float(np.min(h_now))
                # Kept for the post-solve audit: h < 0 only says the robot is
                # inside the keep-out, NOT that the CBF inequality was broken --
                # it can be recovering. Proving the slack let it through needs
                # the row residual with and without epsilon.
                A_cbf_keep, l_cbf_keep = A_cbf, l_cbf

                # P,q augmented with slack block (diagonal L2 penalty).
                # Step k=0 gets cbf_eps0_scale × the nominal penalty so that
                # the "now" CBF is effectively hard, while k ≥ 1 use the
                # standard danger-aware slack weight.
                P_aug = np.zeros((Nm + n_slack, Nm + n_slack))
                P_aug[:Nm, :Nm] = P_dense
                # Block 0 (dynamic) at the nominal weight, block 1 (static walls,
                # only present when split_slack) scaled up so relaxing a wall
                # constraint costs far more than relaxing a dynamic one.
                for blk in range(n_slack // N):
                    w = slack_weight_eff * (cfg.cbf_static_slack_scale if blk else 1.0)
                    base = Nm + blk * N
                    P_aug[base, base] = 2.0 * w * cfg.cbf_eps0_scale
                    for s in range(1, N):
                        P_aug[base + s, base + s] = 2.0 * w
                P_dense = P_aug
                q_vec   = np.concatenate([q_vec, np.zeros(n_slack)])

                # Pad existing constraint rows with zeros for slack columns
                A_total = sparse.hstack(
                    [A_total, sparse.csc_matrix((A_total.shape[0], n_slack))],
                    format='csc',
                )

                # CBF rows (already shaped to Nm + n_slack columns)
                A_total  = sparse.vstack([A_total, A_cbf], format='csc')
                lb_total = np.concatenate([lb_total, l_cbf])
                ub_total = np.concatenate([ub_total, u_cbf])

                # ε_k ≥ 0  for k = 1..N-1
                eps_rows = sparse.lil_matrix((n_slack, Nm + n_slack))
                for s in range(n_slack):
                    eps_rows[s, Nm + s] = 1.0
                A_total  = sparse.vstack([A_total, eps_rows.tocsr()], format='csc')
                lb_total = np.concatenate([lb_total, np.zeros(n_slack)])
                eps_ub = np.full(n_slack, np.inf)
                # cbf_hard_k0: forbid ANY relaxation of the k = 0 constraint by
                # pinning its slack to zero, making the barrier hard at the
                # current state. k >= 1 keeps its slack, because those rows are
                # linearised about the REFERENCE trajectory (see the X_k
                # selection above) and pinning a constraint built at a pose the
                # robot will not occupy would refuse feasible commands.
                #
                # The cost is defined and unavoidable: when the barrier cannot
                # be satisfied at all -- which the feasibility boundary says
                # happens once an obstacle closes faster than the chassis can
                # retreat -- OSQP reports primal infeasible and the solver falls
                # through to the emergency-brake path (u = 0). Against a mover
                # still approaching, stopping is not a solution. So this trades
                # "shallow violation" for "stop and be hit", and only a
                # measurement can say which is better here.
                if cfg.cbf_hard_k0:
                    eps_ub[0] = 0.0                       # dynamic block, k = 0
                    if n_slack >= 2 * N:
                        eps_ub[N] = 0.0                   # static block, k = 0
                elif cfg.cbf_hard_k0_static and n_slack >= 2 * N:
                    # Walls only. The dynamic block keeps its slack, so a mover
                    # the perception has only just resolved can still be given
                    # way to rather than making the whole QP infeasible.
                    eps_ub[N] = 0.0
                ub_total = np.concatenate([ub_total, eps_ub])

        # 5c. （gmpc_wb）手臂區塊：附加在決策向量最後（δξ、鬆弛之後），原有列不動，只補零欄＋手臂列。
        if arm is not None and arm.get('mode') not in ('hold', 'active'):  # active 只在外層安全檢查後以 _inner 進入
            raise ValueError(f"gmpc_wb：arm mode {arm.get('mode')!r} 未實作（hold／active）")
        # hold：手臂速度序列≡0，依結構消去，不改 QP（見 docstring r2）
        n_arm, arm_rows, arm_info = 0, 0, {}
        t_arm0 = time.perf_counter()
        if arm is not None and arm['mode'] == 'active':
            nz = P_dense.shape[0]
            n_arm = ARM_N * N
            Kk, tcp = arm['K'], arm.get('tcp', 'link_tcp')
            Mq, cq, Ms, cs = arm_prediction(arm['q_arm'], arm['s_arm'], N, dt, arm['model'])
            U0 = np.zeros(n_arm) if arm.get('u_arm_nom') is None else np.asarray(arm['u_arm_nom'], float).reshape(-1)
            T_seq = np.asarray(arm['T_seq'], float)
            if T_seq.shape != (N + 1, 4, 4) or not np.isfinite(T_seq).all():
                raise ValueError(f'arm T_seq 形狀須為 ({N + 1},4,4) 且有限，收到 {T_seq.shape}')
            e_nom = (Phi @ e0).reshape(N, n)
            Wt = np.diag([arm['w_p']] * 3 + [arm['w_r']] * 3)
            Pzz = np.zeros((nz + n_arm, nz + n_arm))
            qz = np.zeros(nz + n_arm)
            rel_base = bool(arm.get('tcp_rel_base'))      # True：T_seq 是底盤座標目標，TCP 任務與底盤解耦（持瓶相對底盤）；世界座標代價會讓 QP 以為動底盤能消除誤差
            for k in range(N):
                xb = np.zeros(3) if rel_base else _pose_from_err(X_ref_win[k + 1], e_nom[k])
                qa = Mq[k] @ U0 + cq[k]
                qf = np.r_[xb, qa]
                Tc, J = FastChain.of(Kk, tcp).fk_jac(qf)
                Td = T_seq[k + 1]
                ek = np.r_[Td[:3, 3] - Tc[:3, 3], _rot_err_world(Tc[:3, :3], Td[:3, :3])]
                L = np.zeros((6, nz + n_arm))
                if not rel_base:
                    Bk = _pose_jac(X_ref_win[k + 1], e_nom[k])              # ∂(x,y,θ)/∂e（X_ref·exp(e)；含 SE(2) 右 Jacobian）
                    L[:, :Nm] = J[:, :3] @ Bk @ Gamma[k * n:(k + 1) * n, :]
                L[:, nz:] = J[:, 3:] @ Mq[k]
                r0 = ek + L[:, nz:] @ U0                                  # e_k ≈ r0 − L·z
                Pzz += 2.0 * L.T @ Wt @ L
                qz += -2.0 * L.T @ Wt @ r0
            va = float(arm['v_arm'])
            Ra = np.eye(n_arm) * (float(arm['w_a']) / va ** 2)
            D = np.eye(n_arm) - np.eye(n_arm, k=-ARM_N)
            Sa = np.eye(n_arm) * (float(arm['w_s']) / va ** 2)
            f0 = np.zeros(n_arm); f0[:ARM_N] = -np.asarray(arm['u_arm_prev'], float)
            Pzz[nz:, nz:] += 2.0 * (Ra + D.T @ Sa @ D)
            qz[nz:] += 2.0 * D.T @ Sa @ f0
            P_new = np.zeros((nz + n_arm, nz + n_arm)); P_new[:nz, :nz] = P_dense
            P_dense = P_new + Pzz
            q_vec = np.concatenate([q_vec, np.zeros(n_arm)]) + qz
            lo_j, hi_j = np.asarray(arm['lo'], float) + arm['margin'], np.asarray(arm['hi'], float) - arm['margin']
            aa = float(arm['a_arm']) * dt
            up = np.asarray(arm['u_arm_prev'], float)
            rows_A = [np.eye(n_arm), D]
            rows_l = [np.full(n_arm, -va), np.r_[up - aa, np.full(n_arm - ARM_N, -aa)]]
            rows_u = [np.full(n_arm, va), np.r_[up + aa, np.full(n_arm - ARM_N, aa)]]
            for k in range(N):
                rows_A += [Mq[k], Ms[k]]
                rows_l += [lo_j - cq[k], lo_j - cs[k]]
                rows_u += [hi_j - cq[k], hi_j - cs[k]]
            A_arm = np.vstack(rows_A)
            arm_rows = A_arm.shape[0]
            A_total = sparse.vstack([sparse.hstack([A_total, sparse.csc_matrix((A_total.shape[0], n_arm))]),
                                     sparse.hstack([sparse.csc_matrix((arm_rows, nz)), sparse.csc_matrix(A_arm)])], format='csc')
            lb_total = np.concatenate([lb_total, np.concatenate(rows_l)])
            ub_total = np.concatenate([ub_total, np.concatenate(rows_u)])
            arm_info = dict(nz=nz, A=A_arm, l=np.concatenate(rows_l), u=np.concatenate(rows_u), Mq=Mq, cq=cq, U0=U0)
            bo = arm.get('body_obst')
            if bo:                                              # 三維障礙：TCP 剛連點（夾爪、持瓶）的預測距離約束，底盤與手臂都在列內，耦合問題自己分配誰讓路
                r_ob, l_ob, planes, by_body, row_body = [], [], [], {}, []
                k_idx = list(range(0, N, bo.get('stride', 2)))  # 含第一個未來控制步
                preds = [(np.r_[_pose_from_err(X_ref_win[k + 1], e_nom[k]), Mq[k] @ U0 + cq[k]], _pose_jac(X_ref_win[k + 1], e_nom[k]) @ Gamma[k * n:(k + 1) * n, :], Mq[k]) for k in k_idx]
                # 每個本體是一組包圍盒角點（各用自己連桿的 FK／Jacobian），共用一個分隔面
                for body in bo['bodies']:
                    fc, pts, bname = FastChain.of(Kk, body['link']), body['pts'], body.get('name', body['link'])
                    fj = [fc.fk_jac(qf_) for qf_, _, _ in preds]
                    pj = [[point_jac(T_, J_, off) for off in pts] for T_, J_ in fj]
                    P3 = np.array([[p_ for p_, _ in row_] for row_ in pj])      # (步, 點, 3)
                    for c, sz in bo['boxes']:
                        h_ = np.asarray(sz, float) / 2.0
                        dd3 = np.abs(P3 - c) - h_
                        if float(np.linalg.norm(np.maximum(dd3, 0.0), axis=2).min()) > bo['d_act']:
                            continue
                        dP = P3[-1].mean(axis=0) - P3[0].mean(axis=0)    # 行進中不選迎面的面：迎面代表停在前面，側面才是繞過；靜止時六面都可選
                        axes = [a_ for a_ in range(3) if float(np.linalg.norm(dP)) < 0.02 or abs(float(dP[a_])) <= 0.5 * float(np.linalg.norm(dP))]
                        _, ax, sg = max(((float((sg_ * (P3[:, :, a_] - c[a_]) - h_[a_]).min()), a_, sg_) for a_ in axes for sg_ in (-1.0, 1.0)))
                        if bo.get('face_rule') == 'pref':               # （候選，預設 None＝上面原規則）行進方向改為選面偏好，不硬排除
                            fkey = (bname, round(float(c[0]), 4), round(float(c[1]), 4), round(float(c[2]), 4))
                            ax, sg = _face_pref(P3, dd3, c, h_, dP, float(bo['d_safe']), float(bo['slope']), (bo.get('face_prev') or {}).get(fkey), bo.get('face_stage', 'primary'))
                            bo.setdefault('_picks', {})[fkey] = (ax, sg)  # 只有被採用的求解才寫回 face_prev（由 solve 外層）
                        ff = (bo.get('force_face') or {}).get(bname)    # （診斷，預設 None）指定本體採用某一面，只供逐面可行性核對
                        if ff is not None:
                            ax, sg = int(ff[0]), float(ff[1])
                        tang = [i for i in range(3) if i != ax]
                        planes.append(f"{'+' if sg > 0 else '-'}{'xyz'[ax]}")
                        if bo.get('diag') is not None:                  # （診斷，預設 None）六面表：法向、第 0 步／全時域最小有號餘裕、原規則是否排除、需求最大值
                            tab = []
                            for a_ in range(3):
                                for sg_ in (-1.0, 1.0):
                                    tg_ = [i for i in range(3) if i != a_]
                                    rq = float(bo['d_safe']) - float(bo['slope']) * np.linalg.norm(np.maximum(dd3[:, :, tg_], 0.0), axis=2) - (sg_ * (P3[:, :, a_] - c[a_]) - h_[a_])
                                    tab.append(dict(face=f"{'+' if sg_ > 0 else '-'}{'xyz'[a_]}", m0=float((sg_ * (P3[0, :, a_] - c[a_]) - h_[a_]).min()),
                                                    m_all=float((sg_ * (P3[:, :, a_] - c[a_]) - h_[a_]).min()), excluded=a_ not in axes, req_max=float(rq.max()),
                                                    n_rows=int((rq >= -0.05).sum()), n_dot_dP=float(sg_ * dP[a_] / max(float(np.linalg.norm(dP)), 1e-9))))
                            bo['diag'].append(dict(body=bname, box=[np.asarray(c, float).tolist(), np.asarray(sz, float).tolist()], dP=dP.tolist(), chosen=planes[-1], forced=ff is not None,
                                                   p0_mean=P3[0].mean(axis=0).tolist(), faces=tab))
                        for j, (_, BG, Mk) in enumerate(preds):
                            for i_p, (_, Jp) in enumerate(pj[j]):
                                # 斜坡邊界：允許深入量隨切向距離線性放寬，約束連續收緊；到箱角的真實距離是 d_safe／√(1＋slope²)，d_safe 由呼叫端補償
                                req = float(bo['d_safe']) - float(bo['slope']) * float(np.linalg.norm(np.maximum(dd3[j, i_p, tang], 0.0))) - float(sg * (P3[j, i_p, ax] - c[ax]) - h_[ax])
                                if req < -0.05:
                                    continue
                                row = np.zeros(nz + n_arm)
                                row[:Nm] = sg * (Jp[ax, :3] @ BG)
                                row[nz:] = sg * (Jp[ax, 3:] @ Mk)
                                r_ob.append(row)
                                l_ob.append(req + float(row[nz:] @ U0))
                                by_body[bname] = by_body.get(bname, 0) + 1
                                row_body.append(bname)
                if r_ob:
                    Aob = np.array(r_ob)
                    lob = np.array(l_ob)
                    A_total = sparse.vstack([A_total, sparse.csc_matrix(Aob)], format='csc')
                    lb_total = np.concatenate([lb_total, lob])
                    ub_total = np.concatenate([ub_total, np.full(len(lob), np.inf)])
                    arm_info['ob'] = (Aob, lob)
                arm_info['planes'], arm_info['by_body'], arm_info['row_body'] = planes, by_body, row_body

        # 6. OSQP solve（耦合問題先用對偶作用集法，未得解才走 OSQP；底盤問題維持封存設定）
        t0 = time.perf_counter()
        res, qp_info = None, None
        P_sp = sparse.csc_matrix(P_dense)
        if n_arm:
            qp_info = dict(solver='osqp', gi_why='disabled', gi_ms=0.0, osqp_ms=None, asm_ms=None)
            t_asm = time.perf_counter() - t_arm0
            if arm.get('qp_solver', 'gi') == 'gi':
                t_gi = time.perf_counter()
                gx, gy, gi = qp_dual_active(P_sp, q_vec, A_total, lb_total, ub_total)
                qp_info = dict(solver='gi' if gx is not None else 'osqp', gi_why=gi.get('why'), gi_ms=(time.perf_counter() - t_gi) * 1e3, osqp_ms=None, gi_steps=gi.get('steps'), asm_ms=t_asm * 1e3)
                if gx is not None:
                    res = SimpleNamespace(x=gx, y=gy, info=SimpleNamespace(status='solved'))
        if res is None:
            t_os = time.perf_counter()
            prob = osqp.OSQP()
            # 耦合問題備援比照 W-GMPC：QP 容差須緊於殘差核對門檻 1e-6（eps 1e-5 實測手臂限制殘差 6.3e-6）
            prob.setup(P=P_sp, q=q_vec, A=A_total, l=lb_total, u=ub_total,
                       verbose=False,
                       eps_abs=1e-7 if n_arm else 1e-5, eps_rel=1e-7 if n_arm else 1e-5,        # looser tolerance for speed
                       polish=bool(n_arm), max_iter=20000 if n_arm else 8000)        # more iter for hard cases
            res = prob.solve()
            if n_arm:
                qp_info['osqp_ms'] = (time.perf_counter() - t_os) * 1e3
        solve_time = time.perf_counter() - t0

        status = res.info.status
        # 'maximum iterations reached' often returns a usable (sub-optimal) solution.
        # We accept it rather than emergency-braking, which would freeze the robot.
        # Only true infeasibility falls through to brake.
        usable = status in ('solved', 'solved inaccurate',
                            'maximum iterations reached')
        # A usable STATUS is not a usable POINT. OSQP can report max-iterations
        # and hand back a vector carrying NaN/inf, or one shorter than the
        # decision vector; either would propagate straight into cmd_vel through
        # the arithmetic below, because np.clip passes NaN through unchanged.
        # Check the object before trusting the label.
        if usable:
            _sol = np.asarray(res.x) if res.x is not None else None
            if (_sol is None or _sol.size < Nm + n_arm + (n_slack if n_arm else 0)
                    or not np.all(np.isfinite(_sol))):
                usable = False
                status = f'{status} (rejected: non-finite or short solution)'
        if not usable:
            delta = np.zeros((N, n))
            delta[0] = -xi_ref_win[0]
        else:
            sol = np.asarray(res.x)
            delta = sol[:Nm].reshape(N, n)
            # Post-solve audit of the CBF rows. h < 0 only says the robot is
            # inside the keep-out; it does NOT say the barrier condition was
            # broken, because a robot already inside but moving out still
            # satisfies h_dot + alpha h >= 0. What proves the slack let it
            # through is the row residual: A z - l < 0 without epsilon while
            # A z + eps - l >= 0 with it.
            if A_cbf_keep is not None and A_cbf_keep.shape[0] > 0:
                z_full = np.zeros(Nm + n_slack)
                z_full[:len(sol)] = sol[:Nm + n_slack]
                r_slack = A_cbf_keep @ z_full - l_cbf_keep
                r_none  = A_cbf_keep[:, :Nm] @ z_full[:Nm] - l_cbf_keep
                resid_slack   = float(np.min(r_slack))
                resid_noslack = float(np.min(r_none))
                if len(sol) >= Nm + n_slack:
                    eps_v = sol[Nm:Nm + n_slack]
                    # eps_0 is the step-0 slack; with split_slack there is one
                    # per block and the larger of the two is what mattered.
                    eps0 = float(max(eps_v[0], eps_v[N]) if n_slack >= 2 * N
                                 else eps_v[0])

        u_arm_seq, arm_check = (None if arm is None else np.zeros((N, 6))), None
        if n_arm:
            if usable:
                ua = np.asarray(res.x)[arm_info['nz']:arm_info['nz'] + n_arm]
                r_arm = arm_info['A'] @ ua
                viol = float(max(np.max(arm_info['l'] - r_arm), np.max(r_arm - arm_info['u'])))
                if 'ob' in arm_info:                  # 三維障礙列也進回傳核對：違反走既有退回路徑
                    r_ob_ = arm_info['ob'][0] @ np.asarray(res.x)[:arm_info['nz'] + n_arm] - arm_info['ob'][1]
                    viol = max(viol, float(np.max(-r_ob_)))
                    arm_info['bind'] = {}                 # 實際貼到作用邊界的列（殘差 ≤1e-6），與「列有加入」分開記
                    for b_, g_ in zip(arm_info['row_body'], r_ob_):
                        if g_ <= 1e-6:
                            arm_info['bind'][b_] = arm_info['bind'].get(b_, 0) + 1
                u_arm_seq = ua.reshape(N, ARM_N)
                arm_check = dict(max_violation=viol, finite=bool(np.isfinite(ua).all()), rows=int(arm_rows), qp=qp_info, ob_rows=int(len(arm_info['ob'][1])) if 'ob' in arm_info else 0, ob_planes=arm_info.get('planes'), ob_by_body=arm_info.get('by_body'), ob_bind=arm_info.get('bind'))
            else:
                u_arm_seq = np.zeros((N, ARM_N))
                arm_check = dict(max_violation=float('nan'), finite=False, rows=int(arm_rows), qp=qp_info, note='底盤煞停路徑：手臂命令歸零（未求得序列）')
        u_opt = xi_ref_win[0] + delta[0]
        u_raw = u_opt
        u_opt = np.clip(u_opt, cfg.u_min, cfg.u_max)
        accept_action, accept_scale = ('as_is' if usable else 'brake'), 1.0
        # Final feasibility guard, applied ONLY to solutions the solver did not
        # fully converge on. 'maximum iterations reached' usually carries a
        # usable sub-optimal point, but it is not guaranteed feasible: with the
        # spatio-temporal cost field an ill-conditioned P pushed OSQP to its
        # iteration cap and the point it returned violated the acceleration box
        # (|du/dt| = 1.07 against a_max = 0.80), so the chassis received a step
        # change. A converged solve already satisfies the box by construction, and
        # the emergency-brake path deliberately commands zero, so neither is
        # touched here -- keeping this narrow is what stops the guard from
        # altering the validated behaviour.
        if usable and status != 'solved':
            u_opt = np.clip(u_opt, xi_prev - cfg.a_max * dt,
                            xi_prev + cfg.a_max * dt)
            u_opt = np.clip(u_opt, cfg.u_min, cfg.u_max)
            accept_action = 'acc_clipped'

        # Wheel-space acceptance of the FINAL command. The per-axis clips above
        # cannot see the coupling rows -- v_x, v_y and φ̇ each sit inside their
        # own interval while the wheel they share is asked for twice ω_max -- so
        # this is the only check between a non-converged QP and the chassis.
        # It runs on the emergency-brake path too, where u = 0 trivially passes.
        if cfg.wheel_coupling and cfg.wheel_enforce_output and not (
                cfg.wheel_fit_boundary_tol > 0.0 and wheels_within(u_opt, xi_prev, cfg, dt, cfg.wheel_fit_boundary_tol)):
            u_fit, lam = fit_to_wheels(u_opt, xi_prev, cfg, dt)
            # λ ≈ 0 代表「沿用上一筆」，不是「停止」。這在上一筆正好貼著
            # 輪速邊界時會連續發生，把已經該收回的轉速一直送出去。
            # 投影模式只在這個退化情形接手，而且**驗過才用**；
            # 驗不過就退回線段結果，限制一條都不放寬。
            # **門檻改成「只要線段會改動命令就投影」，不是只救 λ≈0。**
            # 原本 λ < 1e-6 只擋住「完全凍結」，近乎凍結照樣發生：實測
            # 這一趟 NAV 區間仍有 17 輪 λ < 0.05、最小 5.38e-6，
            # sim 30.30 s 的 λ = 0.001757 只接受新命令變化量的 0.176%，
            # 轉速維持 −0.1457 rad/s 繼續往原方向轉，隨後 30.43 s 出現
            # 最大橫向偏差 76.19 mm。「輸出不再逐位元等於上一筆」不足以
            # 說問題解決了。
            #
            # 把門檻再調大是治標；線段縮放在輪速邊界上本來就不適合 ——
            # 它只有一個旋鈕，而投影每次都回傳**集合裡離要求最近的點**，
            # 在 λ=1 的情形兩者相同，所以沒有理由只在退化時才用它。
            _projected = False
            if getattr(cfg, 'wheel_fit_mode', 'segment') == 'project' \
                    and lam < 1.0 - 1e-9:
                _v, _ok = project_to_wheels(u_opt, xi_prev, cfg, dt)
                if _ok:
                    u_fit, _projected = _v, True
            # **寫回要用明確旗標。** 先前把 λ 設成 NaN 表示「不是線段縮放」，
            # 但 `NaN < 1.0` 永遠是 False ⇒ 投影成功反而跳過寫回，回傳的是
            # **投影前**的命令。改善因此不能歸因於投影，只是剛好沒走到卡住
            # 的線段輸出。由 test_wheel_fit_stuck.py 的 E 組（核 solve() 的
            # 實際回傳）釘住。
            if _projected:
                u_opt = u_fit
                accept_scale = float('nan')      # 不是縮放比例，不冒稱
                accept_action = 'wheel_projected'
            elif lam < 1.0 - 1e-9:
                u_opt, accept_scale = u_fit, lam
                accept_action = 'wheel_scaled'
        w_cmd_max = (float(np.max(np.abs(wheel_speeds(u_opt, cfg))))
                     if cfg.wheel_coupling else 0.0)

        # 對**回傳的命令**重算 CBF 殘差。用的是同一組 A_cbf_keep / l_cbf_keep，
        # 不重寫障壁公式。z' 只改 delta[0]，其餘沿用求解器的解。
        resid_post_ns = resid_post_s = float('nan')
        evaluated_on = 'solved'
        if (A_cbf_keep is not None and A_cbf_keep.shape[0] > 0 and usable):
            try:
                sol = np.asarray(res.x)
                z2 = np.zeros(Nm + n_slack)
                z2[:len(sol)] = sol[:Nm + n_slack]
                z2[:n] = u_opt - xi_ref_win[0]          # 回傳命令對應的 delta[0]
                resid_post_s = float(np.min(A_cbf_keep @ z2 - l_cbf_keep))
                resid_post_ns = float(np.min(
                    A_cbf_keep[:, :Nm] @ z2[:Nm] - l_cbf_keep))
                evaluated_on = 'returned'
            except Exception:
                pass

        if arm_check is not None:                     # 後處理若改了底盤命令，手臂序列不再與回傳底盤同解（不宣稱整串可行）
            arm_check['seq_consistent'] = bool(accept_action == 'as_is' and np.allclose(u_opt, xi_ref_win[0] + delta[0], atol=1e-12, rtol=0))
            if not arm_check['seq_consistent'] and cfg.zero_box_clip_tol is not None:
                zt_ = zero_box_clip_ok(cfg, u_opt, u_raw, accept_action, usable, status)
                if zt_ is not None:
                    arm_check['seq_consistent'], arm_check['seq_tol'] = True, zt_
        u0_viol = float('nan')
        if u0_rows is not None:                       # 對回傳命令（含煞停、加速度夾、輪級後處理之後）核對；呼叫端決定違反時停止
            A_u0, b_u0 = np.asarray(u0_rows[0], float).reshape(-1, 3), np.asarray(u0_rows[1], float).ravel()
            u0_viol = float(np.max(A_u0 @ u_opt - b_u0)) if len(b_u0) else -np.inf
        return GMPCResult(u_opt=u_opt, delta_xi_all=delta, u0_rows_viol=u0_viol,
                          e0=e0, solve_time_s=solve_time, status=status,
                          cbf_active=cbf_active, min_h=min_h,
                          n_obs_in=n_obs_in, n_static_in=n_static_in,
                          min_h_static=min_h_static, min_h_dynamic=min_h_dynamic,
                          eps0=eps0, cbf_resid_noslack=resid_noslack,
                          cbf_resid_slack=resid_slack,
                          accept_action=accept_action, accept_scale=accept_scale,
                          wheel_w_cmd_max=w_cmd_max,
                          cbf_resid_post_noslack=resid_post_ns,
                          cbf_resid_post_slack=resid_post_s,
                          cbf_resid_evaluated_on=evaluated_on,
                          arm_mode=None if arm is None else arm.get('mode'),
                          u_arm_seq=u_arm_seq, arm_check=arm_check)


# ---------------------------------------------------------------------------
# Self-test: solve a trivial problem (X = X_ref ⇒ δξ = 0)
# ---------------------------------------------------------------------------

def _selftest():
    import numpy as np
    from .se2 import from_xytheta

    cfg = GMPCConfig(
        N=20, dt=0.05,
        u_min=np.array([-0.20, -0.25, -0.8]),
        u_max=np.array([ 0.35,  0.25,  0.8]),
        a_max=np.array([ 1.5,   1.0,   2.0]),
        Q =np.diag([10.0, 10.0, 5.0]),
        R =np.diag([ 0.5,  0.5, 0.2]),
        Qf=np.diag([50.0, 50.0, 25.0]),
    )
    mpc = GMPC(cfg)

    # Case 1: zero error, ξ_ref = 0 (rest). Expect u_opt ≈ 0.
    X_ref_win  = np.tile(np.eye(3), (cfg.N + 1, 1, 1))
    xi_ref_win = np.zeros((cfg.N + 1, 3))
    res = mpc.solve(X_now=np.eye(3),
                    X_ref_win=X_ref_win,
                    xi_ref_win=xi_ref_win,
                    xi_prev=np.zeros(3))
    assert res.status in ('solved', 'solved inaccurate'), res.status
    assert np.linalg.norm(res.u_opt) < 1e-6, f'expected u≈0, got {res.u_opt}'

    # Case 2: ξ_ref = (0.3, 0, 0) (steady forward), zero error.
    # Expect δξ ≈ 0 and u_opt ≈ (0.3, 0, 0).
    xi_ref_win = np.tile([0.30, 0.0, 0.0], (cfg.N + 1, 1))
    # Recompute reference poses consistent with this xi_ref:
    from .se2 import exp_
    X_ref_win = [np.eye(3)]
    for k in range(cfg.N):
        X_ref_win.append(X_ref_win[-1] @ exp_(xi_ref_win[k] * cfg.dt))
    X_ref_win = np.array(X_ref_win)
    res = mpc.solve(X_now=X_ref_win[0],            # on the trajectory
                    X_ref_win=X_ref_win,
                    xi_ref_win=xi_ref_win,
                    xi_prev=np.array([0.30, 0.0, 0.0]))
    assert res.status in ('solved', 'solved inaccurate'), res.status
    err = np.linalg.norm(res.u_opt - np.array([0.30, 0.0, 0.0]))
    assert err < 1e-4, f'on-trajectory case, u_opt deviates: {res.u_opt}'

    # Case 3: small lateral offset, ξ_ref = forward.
    # Expect a non-zero correction (some vy) without blowing through limits.
    X_now = from_xytheta(0.0, 0.1, 0.0)            # 10cm to the left of ref
    res = mpc.solve(X_now=X_now,
                    X_ref_win=X_ref_win,
                    xi_ref_win=xi_ref_win,
                    xi_prev=np.array([0.30, 0.0, 0.0]))
    assert res.status in ('solved', 'solved inaccurate'), res.status
    # body-frame error should be (0, 0.1, 0) → controller should push -vy
    assert res.u_opt[1] < 0.0, f'expected negative vy correction, got {res.u_opt}'
    assert cfg.u_min[1] <= res.u_opt[1] <= cfg.u_max[1]

    print(f'gmpc.py self-test: OK (last solve {res.solve_time_s*1e3:.2f} ms)')


if __name__ == '__main__':
    _selftest()
