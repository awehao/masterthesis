"""Link-level velocity safety filter for the whole body.

Takes a desired generalised velocity and returns the nearest one that does not
drive any arm link into an obstacle:

    n_i^T J_{p_i}(q) v  <=  alpha_i (d_i - d_stop_i)

for each of the twelve detection points, together with the joint velocity box
and the joint position limits. This is the link-level analogue of the Phase 1
chassis shield, and it exists because the chassis version cannot cover the arm:
its barrier row is n^T(v + omega J r), which for a disc reduces to n^T v, and
the arm is neither a disc nor confined to the scan plane.

What the constraint says, and what it does not
----------------------------------------------
The row bounds the APPROACH speed of one point toward one surface. It does not
bound tangential motion and does not bound retreat -- those have
n^T J v <= 0 and satisfy any non-negative right-hand side automatically. A
filter that slowed them down would be broken, and the acceptance test checks
exactly that.

d_stop depends on the commanded velocity through the braking term, which would
make the constraint nonlinear in v. As in Phase 1 it is evaluated once from the
INPUT command and then held fixed for the projection, so each cycle solves a
linear feasibility problem. The residual is measured afterwards rather than
assumed away.

Degradation, which is the part that has to be got right
------------------------------------------------------
    status OK        use d_i as given.
    status STALE     the obstacle pose is old, so it may already be closer than
                     d_i says. The distance is shrunk by how far an obstacle
                     could have travelled in that time, and the whole command is
                     speed-capped. Ignoring a stale row and continuing to
                     approach is the one behaviour that must not happen.
    status NODATA    no distance at all for that point: nothing can be
                     constrained, so the only sound response is a global cap.
    occluded = 1     the distance itself stays valid -- with a known scene model
                     it does not depend on line of sight -- so the row is kept
                     unchanged. What is added is a separate, tighter cap on
                     motion INTO that direction, because something not in the
                     model could be hiding there. Dropping the row here would
                     throw away a good constraint; treating the direction as
                     clear would invent clearance that was never measured.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .arm_limits import LITE6_SAFE

STATUS_OK, STATUS_UNKNOWN, STATUS_STALE, STATUS_NODATA = 0, 1, 2, 3


@dataclass
class DetectionPoint:
    """One row's worth of input, matching arm_link_distance's PointCloud2."""
    frame: str
    p: np.ndarray                # position, report frame
    n: np.ndarray                # unit vector toward the nearest surface
    d: float                     # distance to that surface
    status: int = STATUS_OK
    age: float = 0.0
    occluded: bool = False


@dataclass
class SafetyConfig:
    alpha: float = 2.0           # 1/s, barrier relaxation
    d0: float = 0.05             # m, standoff at zero speed
    tau: float = 0.15            # s, sense + control + actuation latency
    a_brake: float = 1.0         # m/s^2 at the link point (see note below)
    eps: float = 0.03            # m, geometry + measurement allowance
    dt: float = 0.05             # s, control period

    # Degradation
    stale_obstacle_speed: float = 0.30   # m/s an unseen obstacle may close at
    stale_speed_cap: float = 0.05        # m/s cap while any row is stale
    nodata_speed_cap: float = 0.05       # m/s cap while any row has no distance
    blind_approach_cap: float = 0.03     # m/s toward an occluded direction

    proj_iters: int = 8
    fallback_iters: int = 30

    # Generalised velocity box, |v_k| <= vmax_k, in the DOF order of the
    # kinematics object. Base from the wheel Jacobian; arm from
    # arm_limits.LITE6_SAFE, whose velocity is the manual-verified 180 deg/s
    # rather than a number retyped here.
    vmax: np.ndarray = field(default_factory=lambda: np.concatenate(
        [[0.2775, 0.2775, 1.1327], LITE6_SAFE.max_velocity]))

    # Acceleration box, |v - v_prev| <= amax * dt. The arm's 1145 deg/s^2 is
    # manual-verified but was in no constraint anywhere until now: the URDF has
    # no acceleration field, so nothing enforced it. A filter that respects
    # position and velocity but not acceleration will happily command a step
    # change the hardware cannot follow.
    amax: np.ndarray = field(default_factory=lambda: np.concatenate(
        [[6.25, 6.25, 25.51], LITE6_SAFE.max_acceleration]))

    # Joint POSITION limits used for planning. The URDF carries the same values,
    # but reading them from arm_limits keeps one documented source with a stated
    # provenance instead of two that can drift apart.
    use_arm_limits_positions: bool = True


@dataclass
class SafetyResult:
    v: np.ndarray
    n_rows: int
    n_active: int                # rows the input command violated
    max_resid_before: float
    max_resid_after: float
    iters: int
    fallback: bool
    unresolved: bool
    speed_cap: float             # inf when no cap was applied
    runtime_s: float


def _rows_from_points(K, q, pts, cfg, v_in):
    """Barrier rows A v <= b, plus the per-row bookkeeping."""
    A, b = [], []
    cap = np.inf
    for pt in pts:
        if pt.status == STATUS_NODATA:
            cap = min(cap, cfg.nodata_speed_cap)
            continue
        J = K.jacobian(q, pt.frame)[:3]          # 3 x n, linear part
        row = pt.n @ J                            # 1 x n, approach speed
        d_eff = pt.d
        if pt.status == STATUS_STALE:
            # The obstacle could have closed in during the gap. Shrink the
            # distance by that much rather than trusting a stale number.
            d_eff = pt.d - pt.age * cfg.stale_obstacle_speed
            cap = min(cap, cfg.stale_speed_cap)
        v_app = max(0.0, float(row @ v_in))
        d_stop = (cfg.d0 + v_app * cfg.tau
                  + v_app * v_app / (2.0 * max(cfg.a_brake, 1e-3)) + cfg.eps)
        A.append(row)
        b.append(cfg.alpha * (d_eff - d_stop))
        if pt.occluded:
            # Keep the model row above; add a tighter cap on approaching into
            # a direction nothing has actually observed.
            A.append(row)
            b.append(cfg.blind_approach_cap)
    return A, b, cap


def _joint_limit_rows(K, q, cfg):
    """Do not command a joint past its position limit within one step."""
    lo, hi = K.joint_limits()
    n = len(K.dof_names)
    if cfg.use_arm_limits_positions and n >= 9:
        # Arm joints are the last six of the generalised coordinate vector.
        lo = np.concatenate([lo[:n - 6], LITE6_SAFE.lower])
        hi = np.concatenate([hi[:n - 6], LITE6_SAFE.upper])
    A, b = [], []
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        A.append(e.copy())
        b.append(max(0.0, (hi[i] - q[i]) / cfg.dt))
        A.append(-e)
        b.append(max(0.0, (q[i] - lo[i]) / cfg.dt))
    return A, b


def _box_rows(cfg, n, cap, v_prev=None):
    A, b = [], []
    vmax = np.minimum(cfg.vmax[:n], cap if np.isfinite(cap) else cfg.vmax[:n])
    if v_prev is not None:
        # Acceleration is a box AROUND the previous command, so it tightens the
        # velocity box asymmetrically rather than replacing it.
        step = cfg.amax[:n] * cfg.dt
        hi = np.minimum(vmax, v_prev[:n] + step)
        lo = np.maximum(-vmax, v_prev[:n] - step)
    else:
        hi, lo = vmax, -vmax
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        A.append(e.copy()); b.append(float(hi[i]))
        A.append(-e);       b.append(float(-lo[i]))
    return A, b


def _project(A, b, v, iters):
    """Successive projection onto the most-violated half-space.

    Bounded iterations give a bounded runtime, not a guarantee that every
    constraint holds at the end -- projecting onto one row can re-break
    another. The caller measures the residual and degrades if it is still
    positive, exactly as the Phase 1 shield does.
    """
    used = 0
    for _ in range(iters):
        r = A @ v - b
        k = int(np.argmax(r))
        if r[k] <= 1e-9:
            break
        a = A[k]
        nn = float(a @ a)
        if nn < 1e-12:
            break
        v = v - (r[k] / nn) * a
        used += 1
    return v, used


def filter_velocity(K, q, v_in, pts, cfg=None, v_prev=None) -> SafetyResult:
    t0 = time.perf_counter()
    cfg = cfg or SafetyConfig()
    n = len(K.dof_names)
    v_in = np.asarray(v_in, dtype=float).copy()

    Ab, bb, cap = _rows_from_points(K, q, pts, cfg, v_in)
    Aj, bj = _joint_limit_rows(K, q, cfg)
    Ax, bx = _box_rows(cfg, n, cap, v_prev)

    A = np.array(Ab + Aj + Ax) if (Ab or Aj or Ax) else np.zeros((0, n))
    b = np.array(bb + bj + bx) if len(A) else np.zeros(0)

    if len(A) == 0:
        return SafetyResult(v_in, 0, 0, 0.0, 0.0, 0, False, False, cap,
                            time.perf_counter() - t0)

    r0 = A @ v_in - b
    n_active = int((r0 > 1e-9).sum())
    v, used = _project(A, b, v_in.copy(), cfg.proj_iters)

    fallback = False
    resid = float((A @ v - b).max())
    if resid > 1e-6:
        # Weaker but always-feasible set: do not APPROACH any further. Zero is
        # inside it, so this cannot be infeasible the way the full set can.
        fallback = True
        b2 = b.copy()
        b2[:len(bb)] = np.maximum(b[:len(bb)], 0.0)
        v, used2 = _project(A, b2, v_in.copy(), cfg.fallback_iters)
        used += used2
        resid = float((A @ v - b2).max())
        if resid > 1e-3:
            v = np.zeros(n)
            resid = float((A @ v - b2).max())

    return SafetyResult(v, len(A), n_active, float(r0.max()), resid, used,
                        fallback, resid > 1e-3, cap, time.perf_counter() - t0)
