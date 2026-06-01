"""SE(2) Lie group toolkit for the offline GMPC prototype.

Conventions
-----------
- Pose X ∈ SE(2)            represented either as a 3-vector q = (x, y, θ)
                            or as a 3x3 homogeneous matrix.
- Body twist ξ = (vx, vy, ω) ∈ se(2). All twists are expressed in the body
  frame of the chassis (this matches an Omni base whose cmd_vel is body-frame).
- Group operation is matrix multiplication: X1 · X2.
- Body twist convention for kinematics: Ẋ = X · ξ̂.

Operators
---------
- hat(ξ)    : se(2) vector → 3x3 algebra matrix
- vee(Ξ)    : 3x3 algebra matrix → se(2) vector
- exp_(ξ)   : se(2) → SE(2)
- log_(X)   : SE(2) → se(2)
- inv(X)    : group inverse
- compose   : X1 · X2
- Ad(X)     : Adjoint matrix Ad_X ∈ R^{3x3}
- ad(ξ)     : lowercase adjoint (Lie bracket as matrix), ad(ξ1)·ξ2 = [ξ1, ξ2]
- geodesic_error(X_ref, X) : e = log(X_ref^{-1} · X)^vee  (body-frame error)

All routines are pure numpy, no ROS dependency. Validated by exp/log round-trip
and a few hand-computed adjoint cases (see tests at bottom of file).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Conversions between (x, y, θ) vector and 3x3 homogeneous matrix
# ---------------------------------------------------------------------------

def from_xytheta(x: float, y: float, theta: float) -> np.ndarray:
    """Build a 3x3 SE(2) matrix from (x, y, θ)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, x],
                     [s,  c, y],
                     [0,  0, 1]], dtype=float)


def to_xytheta(X: np.ndarray) -> np.ndarray:
    """Extract (x, y, θ) from an SE(2) matrix."""
    return np.array([X[0, 2], X[1, 2], np.arctan2(X[1, 0], X[0, 0])])


# ---------------------------------------------------------------------------
# hat / vee
# ---------------------------------------------------------------------------

def hat(xi: np.ndarray) -> np.ndarray:
    """se(2) vector ξ=(vx, vy, ω) → 3x3 algebra matrix.

    ξ̂ = [[ 0, -ω, vx],
         [ ω,  0, vy],
         [ 0,  0,  0]]
    """
    vx, vy, w = float(xi[0]), float(xi[1]), float(xi[2])
    return np.array([[0.0, -w, vx],
                     [w,  0.0, vy],
                     [0.0, 0.0, 0.0]])


def vee(Xi_hat: np.ndarray) -> np.ndarray:
    """3x3 algebra matrix → ξ=(vx, vy, ω)."""
    return np.array([Xi_hat[0, 2], Xi_hat[1, 2], Xi_hat[1, 0]])


# ---------------------------------------------------------------------------
# exp / log on SE(2)
# ---------------------------------------------------------------------------

_SMALL = 1e-8


def _V(theta: float) -> np.ndarray:
    """Translation Jacobian V(θ) used in SE(2) exp/log.

    For ξ = (v, ω) ∈ se(2), exp(ξ̂) = [R(ω), V(ω)·v; 0, 1].
    V(ω) = (1/ω) · [[ sinω,  -(1-cosω)], [ (1-cosω),  sinω]]
    For |ω| < eps, V ≈ I + (ω/2)·J.
    """
    if abs(theta) < _SMALL:
        # Taylor: V = I + (θ/2)·J + O(θ²)
        return np.array([[1.0, -theta / 2.0],
                         [theta / 2.0, 1.0]])
    s, c = np.sin(theta), np.cos(theta)
    return np.array([[s / theta,        -(1.0 - c) / theta],
                     [(1.0 - c) / theta,  s / theta]])


def exp_(xi: np.ndarray) -> np.ndarray:
    """se(2) → SE(2). Closed-form."""
    vx, vy, w = float(xi[0]), float(xi[1]), float(xi[2])
    R = np.array([[np.cos(w), -np.sin(w)],
                  [np.sin(w),  np.cos(w)]])
    p = _V(w) @ np.array([vx, vy])
    X = np.eye(3)
    X[:2, :2] = R
    X[:2, 2]  = p
    return X


def log_(X: np.ndarray) -> np.ndarray:
    """SE(2) → se(2). Closed-form, returns ξ=(vx, vy, ω)."""
    theta = np.arctan2(X[1, 0], X[0, 0])
    p = X[:2, 2]
    # Invert V(θ) — closed form is V^{-1} = (1/2) [[A, B], [-B, A]] with
    # A = θ·sin(θ)/(2·(1-cos(θ))), B = θ/2
    if abs(theta) < _SMALL:
        # V^{-1} ≈ I - (θ/2)·J  (since V ≈ I + (θ/2)·J)
        Vinv = np.array([[1.0,  theta / 2.0],
                         [-theta / 2.0, 1.0]])
    else:
        half = theta / 2.0
        cot = half / np.tan(half)   # θ/2 · cot(θ/2)
        Vinv = np.array([[cot,  half],
                         [-half, cot]])
    v = Vinv @ p
    return np.array([v[0], v[1], theta])


# ---------------------------------------------------------------------------
# Group ops
# ---------------------------------------------------------------------------

def inv(X: np.ndarray) -> np.ndarray:
    """SE(2) group inverse (cheaper than np.linalg.inv)."""
    R = X[:2, :2]
    p = X[:2, 2]
    Xi = np.eye(3)
    Xi[:2, :2] = R.T
    Xi[:2, 2]  = -R.T @ p
    return Xi


def compose(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """X1 · X2."""
    return X1 @ X2


# ---------------------------------------------------------------------------
# Adjoint representations
# ---------------------------------------------------------------------------

def Ad(X: np.ndarray) -> np.ndarray:
    """Big Adjoint Ad_X ∈ R^{3x3} satisfying X · exp(ξ̂) · X⁻¹ = exp((Ad_X · ξ)^).

    For X = [R, p; 0, 1] and twist order (vx, vy, ω):
        Ad_X = [[ R,   -J·p ],
                [ 0,    1   ]]
    where J = [[0,-1],[1,0]] is the SO(2) generator. The −J·p block accounts
    for how a pure rotation about the body origin moves a frame translated by p.

    Sign verified by the conjugation identity (test 5 in _selftest).
    """
    R = X[:2, :2]
    p = X[:2, 2]
    J = np.array([[0.0, -1.0], [1.0, 0.0]])
    A = np.zeros((3, 3))
    A[:2, :2] = R
    A[:2,  2] = -J @ p
    A[2,   2] = 1.0
    return A


def ad(xi: np.ndarray) -> np.ndarray:
    """Lowercase adjoint matrix ad(ξ) ∈ R^{3x3} such that ad(ξ1)·ξ2 = [ξ1, ξ2].

    For ξ = (vx, vy, ω):
        ad(ξ) = [[ 0,  -ω,  vy],
                 [ ω,   0, -vx],
                 [ 0,   0,   0]]
    Derived from ξ̂1·ξ̂2 - ξ̂2·ξ̂1; verified in tests below.
    """
    vx, vy, w = float(xi[0]), float(xi[1]), float(xi[2])
    return np.array([[0.0, -w,   vy],
                     [w,  0.0, -vx],
                     [0.0, 0.0, 0.0]])


# ---------------------------------------------------------------------------
# Geodesic error (the core quantity GMPC controls)
# ---------------------------------------------------------------------------

def geodesic_error(X_ref: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Body-frame geometric error e = log(X_ref^{-1} · X)^vee.

    Properties used by GMPC:
      - e = 0 iff X = X_ref
      - No wrap-around (θ component is recovered by atan2 then logged)
      - Linearized dynamics:  ė = -ad(ξ_ref) · e + (ξ - ξ_ref)
        (this is what we discretize for the QP)
    """
    return log_(inv(X_ref) @ X)


# ===========================================================================
# Self-tests — run via `python3 se2.py`
# ===========================================================================

def _approx(a, b, tol=1e-9):
    return np.allclose(a, b, atol=tol, rtol=tol)


def _selftest():
    rng = np.random.default_rng(0)

    # 1. hat/vee round-trip
    for _ in range(20):
        xi = rng.normal(size=3)
        assert _approx(vee(hat(xi)), xi)

    # 2a. exp/log vector round-trip — only valid when ω ∈ (-π, π]
    #     (log returns the shortest geodesic; larger ω wraps. This is by design.)
    for theta_scale in (1e-6, 0.1, 1.0, 3.0):
        for _ in range(20):
            xi = rng.normal(size=3)
            xi[2] *= theta_scale
            # Force ω into the principal branch (-π, π] for round-trip test
            xi[2] = (xi[2] + np.pi) % (2 * np.pi) - np.pi
            X = exp_(xi)
            xi_back = log_(X)
            assert _approx(xi_back, xi, tol=1e-8), \
                f'exp/log mismatch at θ-scale {theta_scale}: {xi} -> {xi_back}'

    # 2b. Group round-trip: exp(log(X)) = X always (even when log unwraps ω)
    for _ in range(20):
        xi = rng.normal(size=3) * np.array([1.0, 1.0, 5.0])   # large ω OK here
        X = exp_(xi)
        assert _approx(exp_(log_(X)), X, tol=1e-9)

    # 3. Group inverse: X · X^{-1} = I
    for _ in range(20):
        X = exp_(rng.normal(size=3))
        assert _approx(compose(X, inv(X)), np.eye(3))

    # 4. ad(ξ1)·ξ2 == vee(ξ̂1·ξ̂2 - ξ̂2·ξ̂1)
    for _ in range(20):
        xi1 = rng.normal(size=3)
        xi2 = rng.normal(size=3)
        lhs = ad(xi1) @ xi2
        rhs = vee(hat(xi1) @ hat(xi2) - hat(xi2) @ hat(xi1))
        assert _approx(lhs, rhs)

    # 5. Adjoint identity: X · exp(ξ̂) · X^{-1} = exp((Ad_X · ξ)^hat)
    for _ in range(20):
        X  = exp_(rng.normal(size=3))
        xi = rng.normal(size=3) * 0.3
        lhs = X @ exp_(xi) @ inv(X)
        rhs = exp_(Ad(X) @ xi)
        assert _approx(lhs, rhs, tol=1e-7)

    # 6a. Geodesic error vanishes on identical poses
    X = from_xytheta(1.2, -0.4, 0.7)
    assert _approx(geodesic_error(X, X), np.zeros(3))

    # 6b. With X_ref at θ=0, body frame = world frame: pure +x offset ⇒ e_vx only
    Xref = from_xytheta(1.2, -0.4, 0.0)
    e = geodesic_error(Xref, from_xytheta(1.5, -0.4, 0.0))
    assert _approx(e, np.array([0.3, 0.0, 0.0])), f'got {e}'

    # 6c. Same world offset but X_ref rotated: error rotates into body frame
    Xref = from_xytheta(1.2, -0.4, 0.7)
    e = geodesic_error(Xref, from_xytheta(1.5, -0.4, 0.7))
    expected = np.array([0.3 * np.cos(0.7), -0.3 * np.sin(0.7), 0.0])
    assert _approx(e, expected), f'got {e}, expected {expected}'

    print('se2.py self-test: OK')


if __name__ == '__main__':
    _selftest()
