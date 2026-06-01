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

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import osqp
from scipy import sparse

from se2 import ad, geodesic_error


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


@dataclass
class GMPCResult:
    u_opt          : np.ndarray           # (3,) applied body twist for next step
    delta_xi_all   : np.ndarray           # (N, 3) full optimal sequence (for warm-start / debug)
    e0             : np.ndarray           # (3,) current geodesic error
    solve_time_s   : float                # wall-time of OSQP solve()
    status         : str                  # OSQP status string


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
              ) -> GMPCResult:
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

        # 4. Cost weights
        Q_bar = _build_Q_bar(cfg.Q, cfg.Qf, N)
        R_bar = _build_R_bar(cfg.R, N)

        P_dense = 2.0 * (Gamma.T @ Q_bar @ Gamma + R_bar)
        # Symmetrise to absorb floating-point asymmetry (OSQP wants symmetric P)
        P_dense = 0.5 * (P_dense + P_dense.T)
        q_vec   = 2.0 * Gamma.T @ Q_bar @ Phi @ e0

        # 5. Constraints
        A_total, lb_total, ub_total = _build_constraints(cfg, xi_ref_win, xi_prev)

        # 6. OSQP solve
        P_sp = sparse.csc_matrix(P_dense)
        prob = osqp.OSQP()
        prob.setup(P=P_sp, q=q_vec, A=A_total, l=lb_total, u=ub_total,
                   verbose=False,
                   eps_abs=1e-6, eps_rel=1e-6,
                   polish=True, max_iter=4000)
        import time
        t0 = time.perf_counter()
        res = prob.solve()
        solve_time = time.perf_counter() - t0

        status = res.info.status
        if status not in ('solved', 'solved inaccurate'):
            # Fall back to reference twist (no correction) — flag in status
            delta = np.zeros((N, n))
        else:
            delta = np.asarray(res.x).reshape(N, n)

        u_opt = xi_ref_win[0] + delta[0]
        u_opt = np.clip(u_opt, cfg.u_min, cfg.u_max)

        return GMPCResult(u_opt=u_opt, delta_xi_all=delta,
                          e0=e0, solve_time_s=solve_time, status=status)


# ---------------------------------------------------------------------------
# Self-test: solve a trivial problem (X = X_ref ⇒ δξ = 0)
# ---------------------------------------------------------------------------

def _selftest():
    import numpy as np
    from se2 import from_xytheta

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
    from se2 import exp_
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
