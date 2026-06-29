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

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import osqp
from scipy import sparse

from .se2 import ad, geodesic_error


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

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
    # ---- Control Barrier Function safety filter ----------------------------
    cbf_alpha        : float = 1.0        # decay rate α in  ḣ + α·h ≥ 0
    cbf_safe_margin  : float = 0.30       # extra clearance added to obstacle radius [m]
    cbf_slack_weight : float = 1.0e4      # ρ in cost ρ·ε² ; large = CBF near-hard
    cbf_eps0_scale   : float = 100.0      # ε_0 penalty multiplier (near-hard at k=0)
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


# ---------------------------------------------------------------------------
# Constraints (velocity + acceleration)
# ---------------------------------------------------------------------------

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
    A_vel = sparse.eye(Nm)
    lb_vel = np.zeros(Nm)
    ub_vel = np.zeros(Nm)
    for k in range(N):
        lb_vel[k*m:(k+1)*m] = cfg.u_min - xi_ref_win[k]
        ub_vel[k*m:(k+1)*m] = cfg.u_max - xi_ref_win[k]

    # ---- Acceleration: first row uses ξ_prev, rest are δξ_k - δξ_{k-1} -
    # row 0:    δξ_0           ∈ [-a·dt - ξ_ref(0) + ξ_prev,  +a·dt - ξ_ref(0) + ξ_prev]
    # row k>0:  δξ_k - δξ_{k-1} ∈ [-a·dt - ξ_ref(k) + ξ_ref(k-1), +a·dt - ξ_ref(k) + ξ_ref(k-1)]
    A_acc = sparse.lil_matrix((Nm, Nm))
    lb_acc = np.zeros(Nm)
    ub_acc = np.zeros(Nm)

    A_acc[0:m, 0:m] = sparse.eye(m)
    lb_acc[0:m] = -cfg.a_max * dt - xi_ref_win[0] + xi_prev
    ub_acc[0:m] =  cfg.a_max * dt - xi_ref_win[0] + xi_prev

    for k in range(1, N):
        A_acc[k*m:(k+1)*m, (k-1)*m:k*m] = -sparse.eye(m)
        A_acc[k*m:(k+1)*m, k*m:(k+1)*m] =  sparse.eye(m)
        lb_acc[k*m:(k+1)*m] = -cfg.a_max * dt - xi_ref_win[k] + xi_ref_win[k-1]
        ub_acc[k*m:(k+1)*m] =  cfg.a_max * dt - xi_ref_win[k] + xi_ref_win[k-1]

    A_total  = sparse.vstack([A_vel, A_acc.tocsr()], format='csc')
    lb_total = np.concatenate([lb_vel, lb_acc])
    ub_total = np.concatenate([ub_vel, ub_acc])
    return A_total, lb_total, ub_total


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

    n_rows = n_obs * N
    A      = np.zeros((n_rows, Nm + slack_dim))
    l_vec  = np.zeros(n_rows)
    u_vec  = np.full(n_rows, np.inf)
    h_now  = np.zeros(n_obs)

    for i, obs in enumerate(obstacles):
        ox, oy = float(obs['x']),   float(obs['y'])
        vox    = float(obs.get('vx', 0.0))
        voy    = float(obs.get('vy', 0.0))
        # Per-obstacle safety margin: static wall points carry a smaller margin
        # ('margin' key) than dynamic obstacles so the robot can thread narrow
        # passages without the static keep-out boxing it in. Falls back to the
        # global cfg.cbf_safe_margin when no per-obstacle margin is given.
        r_eff  = float(obs['radius']) + float(obs.get('margin', cfg.cbf_safe_margin))

        for k in range(N):
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

            # Barrier value
            h_k = float(diff @ diff - r_eff * r_eff)
            if k == 0:
                h_now[i] = h_k

            # Linearised CBF row on δξ_k
            grad_body = 2.0 * (diff @ R_k)                # shape (2,)
            A_row_3   = np.array([grad_body[0], grad_body[1], 0.0])

            row = i * N + k
            A[row, 3*k:3*k + 3] = A_row_3
            # Slack column — every step gets its own ε_k (including k=0).
            # ε_0 has a 100× higher penalty in the cost (set in solve()), so
            # it behaves "near-hard" but the QP stays feasible even when the
            # robot has accidentally drifted into the safety zone.
            A[row, Nm + k] = 1.0

            # Lower bound from rearranged CBF inequality
            l_vec[row] = (
                2.0 * (diff[0] * vox + diff[1] * voy)
                - cfg.cbf_alpha * h_k
                - A_row_3 @ xi_ref_win[k]
            )

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
              ) -> GMPCResult:
        """
        obstacles : optional list of dicts {x, y, radius} in world frame.
                    If non-empty, a CBF inequality is added per obstacle on δξ_0.
        """
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

        P_dense = 2.0 * (Gamma.T @ Q_bar @ Gamma + R_bar)
        q_vec   = 2.0 * Gamma.T @ Q_bar @ Phi @ e0
        # Input-increment (Δu) smoothness cost — soft penalty on control jerk.
        if np.any(cfg.S):
            P_s, q_s = _build_smoothness_terms(cfg.S, xi_ref_win, xi_prev, N)
            P_dense = P_dense + P_s
            q_vec   = q_vec + q_s
        # Symmetrise to absorb floating-point asymmetry (OSQP wants symmetric P)
        P_dense = 0.5 * (P_dense + P_dense.T)

        # 5. Constraints (velocity + acceleration)
        A_total, lb_total, ub_total = _build_constraints(cfg, xi_ref_win, xi_prev)

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
        n_slack     = 0
        if obstacles:
            n_slack = N                         # one slack per step including k=0
            A_cbf, l_cbf, u_cbf, h_now = _build_cbf_horizon(
                cfg, X_now, X_ref_win, xi_ref_win, obstacles, n_slack,
            )
            if A_cbf.shape[0] > 0:
                cbf_active = A_cbf.shape[0]
                min_h      = float(np.min(h_now))

                # P,q augmented with slack block (diagonal L2 penalty).
                # Step k=0 gets cbf_eps0_scale × the nominal penalty so that
                # the "now" CBF is effectively hard, while k ≥ 1 use the
                # standard danger-aware slack weight.
                P_aug = np.zeros((Nm + n_slack, Nm + n_slack))
                P_aug[:Nm, :Nm] = P_dense
                P_aug[Nm, Nm] = 2.0 * slack_weight_eff * cfg.cbf_eps0_scale
                for s in range(1, n_slack):
                    P_aug[Nm + s, Nm + s] = 2.0 * slack_weight_eff
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
                ub_total = np.concatenate([ub_total, np.full(n_slack, np.inf)])

        # 6. OSQP solve
        P_sp = sparse.csc_matrix(P_dense)
        prob = osqp.OSQP()
        prob.setup(P=P_sp, q=q_vec, A=A_total, l=lb_total, u=ub_total,
                   verbose=False,
                   eps_abs=1e-5, eps_rel=1e-5,        # looser tolerance for speed
                   polish=False, max_iter=8000)        # more iter for hard cases
        import time
        t0 = time.perf_counter()
        res = prob.solve()
        solve_time = time.perf_counter() - t0

        status = res.info.status
        # 'maximum iterations reached' often returns a usable (sub-optimal) solution.
        # We accept it rather than emergency-braking, which would freeze the robot.
        # Only true infeasibility falls through to brake.
        usable = status in ('solved', 'solved inaccurate',
                            'maximum iterations reached')
        if not usable:
            delta = np.zeros((N, n))
            delta[0] = -xi_ref_win[0]
        else:
            sol = np.asarray(res.x)
            delta = sol[:Nm].reshape(N, n)

        u_opt = xi_ref_win[0] + delta[0]
        u_opt = np.clip(u_opt, cfg.u_min, cfg.u_max)

        return GMPCResult(u_opt=u_opt, delta_xi_all=delta,
                          e0=e0, solve_time_s=solve_time, status=status,
                          cbf_active=cbf_active, min_h=min_h)


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
