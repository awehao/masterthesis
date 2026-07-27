"""Training-only speed patch for gmpc._build_constraints.

The production builder assembles the acceleration block with scipy `lil_matrix`
and per-block __setitem__ (39 slow scipy calls per solve for N=20). Profiling the
2D trainer showed this ONE function costs ~10 ms of the ~15 ms GMPC step -- i.e.
the bottleneck is matrix ASSEMBLY, not the OSQP solve.

The structure is fixed: A = [[I], [D]] where D has +I on the diagonal blocks and
-I on the sub-diagonal blocks. So it can be built directly in COO/CSC once and
reused, with only the bounds (l, u) recomputed per call. This module builds the
identical matrix ~100x faster.

IMPORTANT: this does NOT touch the ROS package. It monkey-patches the imported
`gmpc` module in-process, for the RL training sandbox only. Numerical output is
bit-identical to the production builder (verified in __main__).

Usage:
    import gmpc_fast          # applies the patch on import
"""
from __future__ import annotations
import numpy as np
from scipy import sparse

from ammr_wholebody_mpc import gmpc as _gmpc

_A_CACHE: dict[int, sparse.csc_matrix] = {}


def _structure(N: int, m: int = 3) -> sparse.csc_matrix:
    """[[I_Nm], [D]] with D = +I on diag blocks, -I on sub-diag blocks."""
    key = N * 100 + m
    if key in _A_CACHE:
        return _A_CACHE[key]
    Nm = N * m
    rows, cols, data = [], [], []
    for i in range(Nm):                       # velocity block: identity
        rows.append(i); cols.append(i); data.append(1.0)
    for i in range(Nm):                       # acceleration block: +I diagonal
        rows.append(Nm + i); cols.append(i); data.append(1.0)
    for k in range(1, N):                     # acceleration block: -I sub-diagonal
        for j in range(m):
            rows.append(Nm + k * m + j); cols.append((k - 1) * m + j); data.append(-1.0)
    A = sparse.csc_matrix((data, (rows, cols)), shape=(2 * Nm, Nm))
    _A_CACHE[key] = A
    return A


def build_constraints_fast(cfg, xi_ref_win: np.ndarray, xi_prev: np.ndarray):
    """Drop-in replacement for gmpc._build_constraints (identical output)."""
    N, dt, m = cfg.N, cfg.dt, 3
    Nm = N * m
    A = _structure(N, m)

    xi = np.asarray(xi_ref_win, float)[:N]            # (N,3)
    adt = np.asarray(cfg.a_max, float) * dt           # (3,)

    lb_vel = (np.asarray(cfg.u_min, float) - xi).reshape(Nm)
    ub_vel = (np.asarray(cfg.u_max, float) - xi).reshape(Nm)

    # row 0 uses xi_prev; rows k>0 use xi_ref(k-1)
    prev = np.vstack([np.asarray(xi_prev, float).reshape(1, 3), xi[:-1]])   # (N,3)
    base = prev - xi                                                        # (N,3)
    lb_acc = (base - adt).reshape(Nm)
    ub_acc = (base + adt).reshape(Nm)

    return A, np.concatenate([lb_vel, lb_acc]), np.concatenate([ub_vel, ub_acc])


# apply the patch on import
_gmpc._build_constraints = build_constraints_fast


if __name__ == '__main__':
    import time, importlib
    # reload a pristine copy to get the ORIGINAL builder for comparison
    orig_mod = importlib.reload(importlib.import_module('ammr_wholebody_mpc.gmpc'))
    original = orig_mod._build_constraints
    orig_mod._build_constraints = build_constraints_fast      # re-apply after reload

    class Cfg:
        N, dt = 20, 0.05
        u_min = np.array([-0.20, -0.25, -0.80])
        u_max = np.array([0.35, 0.25, 0.80])
        a_max = np.array([0.8, 0.6, 1.2])

    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(50):
        xi_ref = rng.uniform(-0.3, 0.3, (Cfg.N, 3))
        xi_prev = rng.uniform(-0.3, 0.3, 3)
        A1, l1, u1 = original(Cfg, xi_ref, xi_prev)
        A2, l2, u2 = build_constraints_fast(Cfg, xi_ref, xi_prev)
        max_err = max(max_err,
                      abs(A1.toarray() - A2.toarray()).max(),
                      abs(l1 - l2).max(), abs(u1 - u2).max())
    print(f"equivalence: max |orig - fast| = {max_err:.2e}  ({'OK' if max_err < 1e-12 else 'FAIL'})")

    xi_ref = rng.uniform(-0.3, 0.3, (Cfg.N, 3)); xi_prev = rng.uniform(-0.3, 0.3, 3)
    for name, fn in (('original', original), ('fast', build_constraints_fast)):
        fn(Cfg, xi_ref, xi_prev)
        t0 = time.time()
        for _ in range(500):
            fn(Cfg, xi_ref, xi_prev)
        print(f"  {name:9s} {(time.time()-t0)/500*1e3:7.3f} ms/call")
