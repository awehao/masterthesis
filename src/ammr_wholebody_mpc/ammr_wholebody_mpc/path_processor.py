"""Convert a nav_msgs/Path into the (X_ref_win, xi_ref_win) format that the
GMPC solver expects.

Nav2 planners (NavFn, SmacPlanner2D, ...) publish a geometric path as a list
of poses without velocities. We turn this into a discrete-time SE(2) reference
trajectory by:

  1. Finding the path point closest to the robot (projection).
  2. Walking forward along the path, sampling at arclength step  v_nom · dt
     for  N+1  samples.
  3. Replacing each sample's yaw with the path direction at that point
     (more stable than trusting the planner's per-pose orientations, which
     NavFn sets to identity).
  4. Computing the body-frame reference twist between consecutive samples
     via the SE(2) log map:   ξ_ref(k) = log(X_k⁻¹ · X_{k+1}) / dt.

The result is consistent with the kinematic model used by the GMPC linearisation,
so when the robot is on the path the error is zero and δξ ≈ 0.
"""

from __future__ import annotations

from typing import Tuple

import math

import numpy as np

from .se2 import from_xytheta, inv, log_


# ---------------------------------------------------------------------------
# Quaternion → yaw (ROS uses (x, y, z, w))
# ---------------------------------------------------------------------------

def quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def path_msg_to_xyth(path_msg) -> np.ndarray:
    """nav_msgs.msg.Path → (M, 3) array of (x, y, yaw) in the path frame."""
    M = len(path_msg.poses)
    out = np.zeros((M, 3))
    for i, ps in enumerate(path_msg.poses):
        p = ps.pose.position
        o = ps.pose.orientation
        out[i, 0] = p.x
        out[i, 1] = p.y
        out[i, 2] = quaternion_to_yaw(o.x, o.y, o.z, o.w)
    return out


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def _interp_xy(path_xyth, cum_s, s):
    """Position on the path at arclength `s`, linearly interpolated."""
    M = int(path_xyth.shape[0])
    j = int(np.searchsorted(cum_s, s) - 1)
    j = max(0, min(j, M - 2))
    s0, s1 = cum_s[j], cum_s[j + 1]
    t = 0.0 if (s1 - s0) < 1e-9 else max(0.0, min(1.0, float((s - s0) / (s1 - s0))))
    return ((1.0 - t) * path_xyth[j, 0] + t * path_xyth[j + 1, 0],
            (1.0 - t) * path_xyth[j, 1] + t * path_xyth[j + 1, 1])


def build_reference_window(path_xyth : np.ndarray,
                           robot_xyth: np.ndarray,
                           N         : int,
                           dt        : float,
                           v_nom     : float,
                           yaw_lookahead: float = 0.0
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    path_xyth  : (M, 3)  path samples in *map* frame (x, y, yaw)
    robot_xyth : (3,)    current robot pose in the same frame
    N          : horizon length (number of MPC input steps)
    dt         : control period [s]
    v_nom      : nominal forward speed for arclength sampling [m/s]

    Returns
    -------
    X_ref_win  : (N+1, 3, 3) SE(2) reference poses
    xi_ref_win : (N+1, 3)    body-frame reference twists (last row mirrors prev)

    Notes
    -----
    * If the path is empty, we hold the robot's current pose with zero twist
      (robot brakes in place).
    * If the path is shorter than the horizon, later samples saturate at the
      goal and ξ_ref decays to zero — equivalent to a "stop at goal" plan.
    * Yaw is taken from the path itself, not the per-pose orientations in the
      message, which keeps the reference usable even when the planner emits
      identity quaternions.

      `yaw_lookahead` (metres) chooses HOW: 0 uses the local segment tangent,
      and anything positive uses the chord from s to s + yaw_lookahead. The
      smoother resamples at about 0.15 m, so a single-segment tangent is a
      direction estimated over 0.15 m of path -- it inherits every wiggle the
      planner leaves behind, and the robot is asked to rotate for all of them.
      A chord averages the same wiggles out while still pointing along the path.
      0 reproduces the validated configuration exactly.
    """
    M = int(path_xyth.shape[0])
    if M == 0:
        Xr = from_xytheta(*robot_xyth)
        X_ref  = np.tile(Xr, (N + 1, 1, 1))
        xi_ref = np.zeros((N + 1, 3))
        return X_ref, xi_ref
    if M == 1:
        Xr = from_xytheta(*path_xyth[0])
        X_ref  = np.tile(Xr, (N + 1, 1, 1))
        xi_ref = np.zeros((N + 1, 3))
        return X_ref, xi_ref

    # Cumulative arclength along the path
    deltas      = np.diff(path_xyth[:, :2], axis=0)
    seg_lengths = np.linalg.norm(deltas, axis=1)
    cum_s       = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_s     = float(cum_s[-1])

    # Closest-point projection (cheap — paths are typically <1000 pts)
    dists        = np.linalg.norm(path_xyth[:, :2] - robot_xyth[:2], axis=1)
    i_close      = int(np.argmin(dists))
    s_start      = float(cum_s[i_close])

    # Arclength targets for the N+1 horizon samples
    step      = v_nom * dt
    targets_s = np.minimum(s_start + np.arange(N + 1) * step, total_s)

    # Interpolate each target back to (x, y, yaw)
    sample_xyth = np.zeros((N + 1, 3))
    for k, s in enumerate(targets_s):
        j  = int(np.searchsorted(cum_s, s) - 1)
        j  = max(0, min(j, M - 2))
        s0 = cum_s[j]
        s1 = cum_s[j + 1]
        if s1 - s0 < 1e-9:
            t = 0.0
        else:
            t = float((s - s0) / (s1 - s0))
            t = max(0.0, min(1.0, t))

        sample_xyth[k, 0] = (1.0 - t) * path_xyth[j, 0] + t * path_xyth[j + 1, 0]
        sample_xyth[k, 1] = (1.0 - t) * path_xyth[j, 1] + t * path_xyth[j + 1, 1]

        # Yaw from the path: local tangent, or a look-ahead chord (see docstring)
        if yaw_lookahead > 1e-6:
            ax, ay = _interp_xy(path_xyth, cum_s, s)
            bx, by = _interp_xy(path_xyth, cum_s,
                                min(s + yaw_lookahead, total_s))
            dx, dy = bx - ax, by - ay
            if dx * dx + dy * dy < 1e-9:
                # at (or past) the end of the path the chord collapses; fall
                # back to the chord BEHIND us so the goal approach keeps a
                # sensible heading instead of snapping to the last tangent
                ax, ay = _interp_xy(path_xyth, cum_s,
                                    max(0.0, total_s - yaw_lookahead))
                dx, dy = bx - ax, by - ay
        else:
            dx = path_xyth[j + 1, 0] - path_xyth[j, 0]
            dy = path_xyth[j + 1, 1] - path_xyth[j, 1]
        if dx * dx + dy * dy > 1e-9:
            sample_xyth[k, 2] = float(np.arctan2(dy, dx))
        else:
            sample_xyth[k, 2] = path_xyth[j, 2]

    # Build SE(2) matrices
    X_ref_win = np.array([from_xytheta(*p) for p in sample_xyth])    # (N+1, 3, 3)

    # Reference twists via log map of relative pose
    xi_ref_win = np.zeros((N + 1, 3))
    for k in range(N):
        xi_ref_win[k] = log_(inv(X_ref_win[k]) @ X_ref_win[k + 1]) / dt
    xi_ref_win[-1] = xi_ref_win[-2] if N > 0 else np.zeros(3)

    return X_ref_win, xi_ref_win


# ---------------------------------------------------------------------------
# Self-test: no ROS dependency, runs from CLI
# ---------------------------------------------------------------------------

def _selftest():
    # Build a straight-line path from (0,0) to (5,0), spaced 0.05m
    M = 101
    path_xyth = np.zeros((M, 3))
    path_xyth[:, 0] = np.linspace(0.0, 5.0, M)
    # yaw left as zero — should be overwritten by tangent (atan2(0,Δx) = 0)

    # Robot starts at (0, 0.1, 0): 10cm off-track to +y
    robot_xyth = np.array([0.0, 0.1, 0.0])
    N, dt, v_nom = 20, 0.05, 0.30

    X_ref, xi_ref = build_reference_window(path_xyth, robot_xyth, N, dt, v_nom)

    # Sanity checks
    assert X_ref.shape  == (N + 1, 3, 3)
    assert xi_ref.shape == (N + 1, 3)

    # Reference should be on the x-axis (y=0) and yaw=0
    for k in range(N + 1):
        y_k     = X_ref[k, 1, 2]
        theta_k = np.arctan2(X_ref[k, 1, 0], X_ref[k, 0, 0])
        assert abs(y_k) < 1e-9,     f'k={k}: y={y_k}'
        assert abs(theta_k) < 1e-9, f'k={k}: θ={theta_k}'

    # Twist should be ≈ (v_nom, 0, 0) all the way (path much longer than horizon)
    expected = np.array([v_nom, 0.0, 0.0])
    for k in range(N):
        assert np.allclose(xi_ref[k], expected, atol=1e-9), \
            f'k={k}: ξ={xi_ref[k]}, expected {expected}'

    # Now test path-end saturation: short path, large horizon
    short_path = np.array([[0.0, 0.0, 0.0],
                           [0.1, 0.0, 0.0]])
    X_ref, xi_ref = build_reference_window(short_path, np.zeros(3), N=10, dt=0.05, v_nom=0.30)
    # After saturation, last samples should all sit at the path end
    assert np.allclose(X_ref[-1, :2, 2], [0.1, 0.0])
    # ξ at the very end should be (near-)zero since both ends saturate
    assert np.linalg.norm(xi_ref[-2]) < 1e-9, f'end twist: {xi_ref[-2]}'

    print('path_processor.py self-test: OK')


if __name__ == '__main__':
    _selftest()


def blend_reference(X_old, xi_old, X_new, xi_new, alpha):
    """Cross-fade two horizon references. alpha=0 -> old, 1 -> new.

    Positions and twists interpolate linearly; HEADING must not, because a
    straight average of two angles either side of +-pi swings the long way round
    and would hand the controller a 2*pi rotation to chase. Interpolating the
    signed shortest difference keeps the fade on the short arc.

    Why fade at all: the global planner republishes every 3 s, and a quarter of
    those updates move the path in front of the robot by more than 0.25 m
    (measured p90 0.44 m, max 1.82 m, reference heading p90 26 deg). Adopting
    one instantly is a step input, and the controller answers it at 0.96 of
    a_max, saturating 88.5% of the time in the following 0.3 s against 13.5%
    otherwise. Spreading the same change over a second removes the step without
    discarding the new plan.
    """
    a = float(np.clip(alpha, 0.0, 1.0))
    n = min(len(X_old), len(X_new))
    out = np.empty((n, 3, 3))
    for k in range(n):
        px = (1.0 - a) * X_old[k][0, 2] + a * X_new[k][0, 2]
        py = (1.0 - a) * X_old[k][1, 2] + a * X_new[k][1, 2]
        th_o = math.atan2(X_old[k][1, 0], X_old[k][0, 0])
        th_n = math.atan2(X_new[k][1, 0], X_new[k][0, 0])
        d = (th_n - th_o + math.pi) % (2.0 * math.pi) - math.pi
        out[k] = from_xytheta(px, py, th_o + a * d)
    xi = (1.0 - a) * np.asarray(xi_old)[:n] + a * np.asarray(xi_new)[:n]
    return out, xi
