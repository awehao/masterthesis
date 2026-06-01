"""Discrete kinematics for the simulated Omni chassis (offline prototype).

The continuous model is purely kinematic:

    Ẋ(t) = X(t) · ξ̂(t),    ξ = (vx, vy, ω) ∈ se(2)   (body twist)

For a zero-order-hold input ξ_k held over [t_k, t_k+dt], the exact discretisation
is the group exponential update:

    X_{k+1} = X_k · exp(ξ_k · dt)

This is what the GMPC linearisation is fitted *against*, so using exp_ here
keeps the simulator self-consistent with the QP model (no integration error).
"""

from __future__ import annotations

import numpy as np

from se2 import exp_


def step(X: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """One zero-order-hold step on SE(2)."""
    return X @ exp_(u * dt)


def clamp_twist(u: np.ndarray,
                u_min: np.ndarray,
                u_max: np.ndarray) -> np.ndarray:
    """Element-wise saturation of (vx, vy, ω) to the chassis limits.

    Used as a safety net in case the QP returns something slightly outside
    bounds due to numerical slack; in normal operation it is a no-op.
    """
    return np.clip(u, u_min, u_max)


if __name__ == '__main__':
    # Sanity check: applying constant forward twist for 1 sec at vx=0.3
    # should land at x=0.3, y=0, θ=0 (consistent with trajectories.straight).
    X = np.eye(3)
    u = np.array([0.30, 0.0, 0.0])
    dt = 0.05
    for _ in range(20):                      # 20 × 0.05 = 1 s
        X = step(X, u, dt)
    from se2 import to_xytheta
    q = to_xytheta(X)
    assert abs(q[0] - 0.30) < 1e-12, q
    assert abs(q[1]) < 1e-12 and abs(q[2]) < 1e-12
    print('kinematics.py self-test: OK')
