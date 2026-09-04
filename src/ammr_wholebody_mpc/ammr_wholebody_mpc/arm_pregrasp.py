"""Reach a pre-grasp pose with a fixed base, and hold it.

Scope, deliberately narrow (Phase 2 plan, stage A):

    fixed base, six arm joints only
    target pose from the scene model, not from perception
    machine safe position limits, plus velocity / acceleration / jerk
    link safety layer active, static environment included
    stop AT the pre-grasp pose -- no handle contact, no gripper motion

Three pieces, kept apart so a failure says which one failed:

    solve_ik       damped least squares on the verified 6x6 arm Jacobian
    min_jerk       a time-scaled trajectory that respects v/a/j by construction
    PregraspPlan   the two together, plus the checks that must pass before any
                   of it is executed

Why damped least squares rather than a library IK: the damping is what makes
the solve behave near singularities instead of demanding enormous joint rates,
and this arm's wrist is nested tightly enough that near-singular configurations
are ordinary. The damping factor is raised automatically as the manipulability
drops, so the solver slows down near a singularity rather than diverging.

The trajectory is min-jerk with a duration chosen so that peak velocity,
acceleration AND jerk all sit inside their limits:

    |qd|_max  = 1.875 |dq| / T
    |qdd|_max = 5.7735 |dq| / T^2
    |qddd|_max = 60 |dq| / T^3

so T is the largest of the three requirements. Generating a trajectory that
already respects the limits, rather than clipping one that does not, keeps the
safety filter free to act on obstacles instead of spending its authority
undoing the planner.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .arm_limits import LITE6_SAFE
from .wholebody_kinematics import WholeBodyKinematics

ARM_JOINTS = [f'joint{i}' for i in range(1, 7)]
TCP = 'link_tcp'


def rot_error(R_cur: np.ndarray, R_des: np.ndarray) -> np.ndarray:
    """Orientation error as a rotation vector, log(R_cur^T R_des) in world."""
    Re = R_cur.T @ R_des
    c = float(np.clip((np.trace(Re) - 1.0) * 0.5, -1.0, 1.0))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    w = np.array([Re[2, 1] - Re[1, 2],
                  Re[0, 2] - Re[2, 0],
                  Re[1, 0] - Re[0, 1]]) * (th / (2.0 * math.sin(th)))
    return R_cur @ w


@dataclass
class IKResult:
    q: np.ndarray
    ok: bool
    pos_err: float
    rot_err: float
    iters: int
    reason: str = ''
    manipulability: float = 0.0


def solve_ik(K: WholeBodyKinematics, q_seed: np.ndarray, T_des: np.ndarray,
             tol_pos: float = 1e-4, tol_rot: float = 1e-3,
             max_iters: int = 200, arm_idx=None) -> IKResult:
    """Damped least squares over the six arm joints, base held fixed."""
    idx = list(arm_idx or [K.dof_names.index(j) for j in ARM_JOINTS])
    q = np.asarray(q_seed, float).copy()
    lo, hi = K.joint_limits()
    lo = lo.copy(); hi = hi.copy()
    for k, i in enumerate(idx):
        lo[i], hi[i] = LITE6_SAFE.lower[k], LITE6_SAFE.upper[k]

    for it in range(max_iters):
        T = K.fk(q, TCP)
        ep = T_des[:3, 3] - T[:3, 3]
        er = rot_error(T[:3, :3], T_des[:3, :3])
        pe, re_ = float(np.linalg.norm(ep)), float(np.linalg.norm(er))
        J = K.jacobian(q, TCP)[:, idx]                       # 6 x 6
        w = float(math.sqrt(max(np.linalg.det(J @ J.T), 0.0)))
        if pe < tol_pos and re_ < tol_rot:
            return IKResult(q, True, pe, re_, it, '', w)
        # Damping grows as manipulability falls: near a singularity the solve
        # must slow down, not demand joint rates the arm cannot produce.
        lam = 0.01 + 0.5 * math.exp(-w / 1e-3)
        e = np.concatenate([ep, er])
        dq = J.T @ np.linalg.solve(J @ J.T + lam * lam * np.eye(6), e)
        step = float(np.abs(dq).max())
        if step > 0.2:
            dq *= 0.2 / step
        for k, i in enumerate(idx):
            q[i] = float(np.clip(q[i] + dq[k], lo[i], hi[i]))
    T = K.fk(q, TCP)
    pe = float(np.linalg.norm(T_des[:3, 3] - T[:3, 3]))
    re_ = float(np.linalg.norm(rot_error(T[:3, :3], T_des[:3, :3])))
    return IKResult(q, False, pe, re_, max_iters, 'not converged', 0.0)


def min_jerk_duration(dq: np.ndarray, lim=LITE6_SAFE, floor: float = 0.3) -> float:
    """Shortest duration whose min-jerk profile respects v, a AND j."""
    a = np.abs(np.asarray(dq, float))
    if not np.any(a > 1e-12):
        return floor
    T = floor
    with np.errstate(divide='ignore', invalid='ignore'):
        T = max(T, float(np.max(1.875 * a / lim.max_velocity)))
        T = max(T, float(np.max(np.sqrt(5.7735 * a / lim.max_acceleration))))
        T = max(T, float(np.max(np.cbrt(60.0 * a / lim.max_jerk))))
    return T


def min_jerk(q0: np.ndarray, q1: np.ndarray, T: float, t: float):
    """(q, qd, qdd) of the min-jerk profile at time t."""
    s = float(np.clip(t / T, 0.0, 1.0))
    dq = np.asarray(q1, float) - np.asarray(q0, float)
    p = 10 * s**3 - 15 * s**4 + 6 * s**5
    v = (30 * s**2 - 60 * s**3 + 30 * s**4) / T
    a = (60 * s - 180 * s**2 + 120 * s**3) / (T * T)
    return q0 + dq * p, dq * v, dq * a


@dataclass
class PregraspPlan:
    q_goal: np.ndarray
    duration: float
    ik: IKResult
    ok: bool = False
    reason: str = ''
    checks: dict = field(default_factory=dict)


def plan_pregrasp(K, q_start, T_des, self_collision_fn=None,
                  env_clearance_fn=None, min_margin: float = 0.005,
                  n_check: int = 25, n_seeds: int = 12,
                  rng=None) -> PregraspPlan:
    """IK, then validate the whole path before anything moves.

    Checked along the path rather than only at the goal: a start and a goal
    that are both clear say nothing about the arc between them, and that arc is
    what the robot actually traverses.

    Multiple IK seeds, because this is a redundant-in-practice problem: a
    6-DOF arm reaching a 6-DOF pose has discrete solution branches, and the
    branch nearest the start pose is often the one that folds the arm into
    itself. Solving once from the start pose rejected 6 of 8 starts on
    self-collision margin -- not because the pose was unreachable, but because
    the first branch found happened to be the bad one.
    """
    rng = rng or np.random.default_rng(0)
    seeds = [np.asarray(q_start, float)]
    idx0 = [K.dof_names.index(j) for j in ARM_JOINTS]
    for _ in range(max(0, n_seeds - 1)):
        s = np.asarray(q_start, float).copy()
        s[idx0] = rng.uniform(LITE6_SAFE.lower, LITE6_SAFE.upper)
        seeds.append(s)

    best = None
    for seed in seeds:
        cand = _try_plan(K, q_start, seed, T_des, self_collision_fn,
                         env_clearance_fn, min_margin, n_check)
        if cand.ok:
            return cand
        if best is None or (cand.ik.ok and not best.ik.ok):
            best = cand
    return best


def _try_plan(K, q_start, seed, T_des, self_collision_fn, env_clearance_fn,
              min_margin, n_check) -> PregraspPlan:
    ik = solve_ik(K, seed, T_des)
    idx = [K.dof_names.index(j) for j in ARM_JOINTS]
    dq = ik.q[idx] - np.asarray(q_start, float)[idx]
    T = min_jerk_duration(dq)
    plan = PregraspPlan(ik.q.copy(), T, ik)
    if not ik.ok:
        plan.reason = f'IK failed: {ik.reason} (pos {ik.pos_err:.4f} m)'
        return plan

    worst_self, worst_env = math.inf, math.inf
    for s in np.linspace(0.0, 1.0, n_check):
        qa, _, _ = min_jerk(np.asarray(q_start, float)[idx], ik.q[idx], T, s * T)
        q_full = np.asarray(q_start, float).copy()
        q_full[idx] = qa
        if self_collision_fn is not None:
            worst_self = min(worst_self, float(self_collision_fn(q_full)))
        if env_clearance_fn is not None:
            worst_env = min(worst_env, float(env_clearance_fn(q_full)))
    plan.checks = {'min_self_clearance': worst_self,
                   'min_env_clearance': worst_env,
                   'duration_s': T,
                   'manipulability': ik.manipulability}
    if worst_self < min_margin:
        plan.reason = f'self-collision margin {worst_self:.4f} < {min_margin}'
        return plan
    if worst_env < min_margin:
        plan.reason = f'environment margin {worst_env:.4f} < {min_margin}'
        return plan
    plan.ok = True
    return plan
