"""Reference trajectories for the offline GMPC validation set.

Each trajectory is defined by a *body twist* schedule  ξ_ref(t) = (vx, vy, ω).
We integrate the kinematic identity  Ẋ = X · ξ̂  to produce the reference pose
sequence X_ref(k) at sample times t_k = k·dt.

Five trajectories chosen to expose different failure modes of past MPC attempts
(see memory `project_mpc_lessons`):

    1. straight    — pure forward (vx only): trivial baseline
    2. lateral     — pure side-slip (vy only): only achievable by holonomic base
    3. diagonal    — vx+vy simultaneously: tests Omni decomposition
    4. s_curve     — vx const, ω sinusoidal: tests ω≠0 linearization
    5. yaw_wrap    — pure spin past ±π: directly attacks the wrap-around bug
                     that broke the previous (x,y,θ)-state MPC

All trajectories start at the identity pose X0 = (0, 0, 0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import numpy as np

from se2 import exp_, to_xytheta


# ---------------------------------------------------------------------------
# Trajectory container
# ---------------------------------------------------------------------------

@dataclass
class ReferenceTrajectory:
    """Pre-sampled reference trajectory in SE(2)."""
    name        : str
    dt          : float
    t           : np.ndarray          # (N,)
    xi_ref      : np.ndarray          # (N, 3) body twists at each step
    X_ref       : np.ndarray          # (N, 3, 3) SE(2) matrices

    @property
    def N(self) -> int:
        return self.X_ref.shape[0]

    def xy_theta(self) -> np.ndarray:
        """Return reference as (N, 3) array of (x, y, θ) — for plotting."""
        return np.array([to_xytheta(X) for X in self.X_ref])


def from_twist_fn(name: str,
                  twist_fn: Callable[[float], np.ndarray],
                  duration: float,
                  dt: float) -> ReferenceTrajectory:
    """Build a ReferenceTrajectory by integrating a body-twist schedule.

    The schedule is sampled with zero-order-hold:
        X_{k+1} = X_k · exp(ξ_ref(t_k) · dt)
    so that the discrete reference is *consistent* with the SE(2) kinematics
    used by the simulator and the linearised QP.
    """
    N = int(round(duration / dt)) + 1
    t = np.arange(N) * dt
    xi_ref = np.zeros((N, 3))
    X_ref  = np.zeros((N, 3, 3))
    X_ref[0] = np.eye(3)
    for k in range(N):
        xi_ref[k] = np.asarray(twist_fn(t[k]), dtype=float)
    for k in range(N - 1):
        X_ref[k + 1] = X_ref[k] @ exp_(xi_ref[k] * dt)
    return ReferenceTrajectory(name=name, dt=dt, t=t,
                               xi_ref=xi_ref, X_ref=X_ref)


# ---------------------------------------------------------------------------
# Twist schedules
# ---------------------------------------------------------------------------

def _straight(_t):     return (0.30, 0.0, 0.0)
def _lateral(_t):      return (0.0,  0.20, 0.0)
def _diagonal(_t):     return (0.20, 0.15, 0.0)
def _s_curve(t):       return (0.25, 0.0, 0.6 * np.sin(2.0 * np.pi * 0.10 * t))
def _yaw_wrap(_t):     return (0.0,  0.0, 0.40)  # 25 s × 0.4 rad/s ≈ 10 rad ≈ 1.6 turns


# ---------------------------------------------------------------------------
# Public catalogue
# ---------------------------------------------------------------------------

def all_trajectories(dt: float = 0.05) -> List[ReferenceTrajectory]:
    """Return the canonical 5-trajectory test set."""
    return [
        from_twist_fn('01_straight',  _straight,  duration=6.0,  dt=dt),
        from_twist_fn('02_lateral',   _lateral,   duration=6.0,  dt=dt),
        from_twist_fn('03_diagonal',  _diagonal,  duration=6.0,  dt=dt),
        from_twist_fn('04_s_curve',   _s_curve,   duration=12.0, dt=dt),
        from_twist_fn('05_yaw_wrap',  _yaw_wrap,  duration=25.0, dt=dt),
    ]


# ---------------------------------------------------------------------------
# Self-test: integrate, summarise, and verify a few basic properties
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print(f'{"name":<14} {"N":>5} {"T[s]":>7} '
          f'{"x_end":>9} {"y_end":>9} {"θ_end[rad]":>11}')
    for tr in all_trajectories():
        q = to_xytheta(tr.X_ref[-1])
        assert np.all(np.isfinite(tr.X_ref)),  f'{tr.name}: non-finite X'
        assert np.all(np.isfinite(tr.xi_ref)), f'{tr.name}: non-finite ξ'
        print(f'{tr.name:<14} {tr.N:>5d} {tr.t[-1]:>7.2f} '
              f'{q[0]:>9.3f} {q[1]:>9.3f} {q[2]:>11.3f}')

    # Sanity: straight-line trajectory should end at (vx · T, 0, 0)
    tr = from_twist_fn('chk', _straight, duration=6.0, dt=0.05)
    q = to_xytheta(tr.X_ref[-1])
    assert abs(q[0] - 0.30 * 6.0) < 1e-9, f'straight x-end: {q[0]}'
    assert abs(q[1]) < 1e-12
    assert abs(q[2]) < 1e-12

    # Sanity: yaw wrap-around end angle wraps into (-π, π]
    tr = from_twist_fn('chk', _yaw_wrap, duration=25.0, dt=0.05)
    q = to_xytheta(tr.X_ref[-1])
    assert -np.pi < q[2] <= np.pi, f'wrapped θ out of branch: {q[2]}'

    print('trajectories.py self-test: OK')
