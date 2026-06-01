"""Closed-loop simulation driver for the offline GMPC prototype.

For each reference trajectory:
  1. start the robot at a *non-zero initial offset* so the controller has to
     converge (otherwise tracking is trivial),
  2. at every step k:  read window X_ref[k..k+N], xi_ref[k..k+N], solve GMPC,
     apply optimal u to the kinematic model, log state / input / solve_time,
  3. compute and report:
        - per-axis and norm tracking RMSE
        - mean / max / p95 OSQP solve time
        - infeasible-step count

Logs are saved to logs/<trajectory_name>.npz for plotting.

Usage:
    python3 run.py                 # run all 5 trajectories
    python3 run.py 04_s_curve      # run a single one
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np

from se2          import from_xytheta, geodesic_error, to_xytheta
from kinematics   import step as kin_step
from trajectories import all_trajectories, ReferenceTrajectory
from gmpc         import GMPC, GMPCConfig


# ---------------------------------------------------------------------------
# Tunable test setup (single source of truth — keep numbers reproducible)
# ---------------------------------------------------------------------------

INITIAL_OFFSET_XY  = (0.15, -0.10)   # m   start 0.15m forward / 0.10m right of ref
INITIAL_OFFSET_TH  = 0.30            # rad start rotated +0.3 rad relative to ref

HORIZON_N = 20                       # MPC horizon (steps)

VEL_MIN = np.array([-0.20, -0.25, -0.80])
VEL_MAX = np.array([ 0.35,  0.25,  0.80])
ACC_MAX = np.array([ 1.5,   1.0,   2.0])

Q_DIAG  = np.array([10.0, 10.0,  5.0])   # state weight (e_vx, e_vy, e_ω)
R_DIAG  = np.array([ 0.5,  0.5,  0.2])   # input deviation weight
QF_DIAG = np.array([50.0, 50.0, 25.0])   # terminal


# ---------------------------------------------------------------------------
# Run a single trajectory
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    name           : str
    N_steps        : int
    dt             : float
    # arrays
    t              : np.ndarray            # (N_steps+1,)
    X_actual_xyth  : np.ndarray            # (N_steps+1, 3)  - actual pose history
    X_ref_xyth     : np.ndarray            # (N_steps+1, 3)  - reference pose
    u_applied      : np.ndarray            # (N_steps, 3)    - applied body twist
    xi_ref         : np.ndarray            # (N_steps, 3)    - reference twist
    e              : np.ndarray            # (N_steps+1, 3)  - body-frame error
    solve_time_ms  : np.ndarray            # (N_steps,)
    status         : list                  # per-step OSQP status
    # scalar metrics
    rmse_pos       : float                 # sqrt(mean(e_vx²+e_vy²))    [m]
    rmse_yaw       : float                 # sqrt(mean(e_ω²))            [rad]
    max_pos_err    : float                 # max(sqrt(e_vx²+e_vy²))      [m]
    max_yaw_err    : float                 # max(|e_ω|)                  [rad]
    solve_mean_ms  : float
    solve_max_ms   : float
    solve_p95_ms   : float
    n_infeasible   : int


def _initial_pose(tr: ReferenceTrajectory) -> np.ndarray:
    """Start pose = X_ref(0) composed with a body-frame offset."""
    X0_ref = tr.X_ref[0]                              # identity in our setup
    # Body-frame offset
    offset = from_xytheta(INITIAL_OFFSET_XY[0],
                         INITIAL_OFFSET_XY[1],
                         INITIAL_OFFSET_TH)
    return X0_ref @ offset


def run_trajectory(tr: ReferenceTrajectory) -> RunResult:
    cfg = GMPCConfig(
        N=HORIZON_N, dt=tr.dt,
        u_min=VEL_MIN, u_max=VEL_MAX, a_max=ACC_MAX,
        Q =np.diag(Q_DIAG),
        R =np.diag(R_DIAG),
        Qf=np.diag(QF_DIAG),
    )
    mpc = GMPC(cfg)

    N_total = tr.N - 1 - cfg.N        # steps we can run while keeping a full horizon
    if N_total <= 0:
        raise ValueError(f'Trajectory {tr.name} shorter than horizon')

    X = _initial_pose(tr)
    u_prev = np.zeros(3)

    Xs            = [X.copy()]
    us            = []
    es            = [geodesic_error(tr.X_ref[0], X)]
    solve_times   = []
    statuses      = []

    for k in range(N_total):
        X_ref_win  = tr.X_ref[k : k + cfg.N + 1]
        xi_ref_win = tr.xi_ref[k : k + cfg.N + 1]
        res = mpc.solve(X, X_ref_win, xi_ref_win, u_prev)

        X       = kin_step(X, res.u_opt, tr.dt)
        u_prev  = res.u_opt
        Xs.append(X.copy())
        us.append(res.u_opt.copy())
        es.append(geodesic_error(tr.X_ref[k + 1], X))
        solve_times.append(res.solve_time_s * 1e3)
        statuses.append(res.status)

    # ---- summarise ----
    Xs            = np.array(Xs)
    us            = np.array(us)
    es            = np.array(es)
    solve_times   = np.array(solve_times)

    X_actual_xyth = np.array([to_xytheta(X) for X in Xs])
    X_ref_xyth    = tr.xy_theta()[:N_total + 1]

    e_pos = np.linalg.norm(es[:, :2], axis=1)
    rmse_pos  = float(np.sqrt(np.mean(e_pos ** 2)))
    rmse_yaw  = float(np.sqrt(np.mean(es[:, 2] ** 2)))
    max_pos   = float(np.max(e_pos))
    max_yaw   = float(np.max(np.abs(es[:, 2])))

    return RunResult(
        name=tr.name, N_steps=N_total, dt=tr.dt,
        t              = tr.t[:N_total + 1],
        X_actual_xyth  = X_actual_xyth,
        X_ref_xyth     = X_ref_xyth,
        u_applied      = us,
        xi_ref         = tr.xi_ref[:N_total],
        e              = es,
        solve_time_ms  = solve_times,
        status         = statuses,
        rmse_pos       = rmse_pos,
        rmse_yaw       = rmse_yaw,
        max_pos_err    = max_pos,
        max_yaw_err    = max_yaw,
        solve_mean_ms  = float(np.mean(solve_times)),
        solve_max_ms   = float(np.max(solve_times)),
        solve_p95_ms   = float(np.percentile(solve_times, 95)),
        n_infeasible   = sum(1 for s in statuses
                             if s not in ('solved', 'solved inaccurate')),
    )


# ---------------------------------------------------------------------------
# Save + summary
# ---------------------------------------------------------------------------

def save(result: RunResult, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'{result.name}.npz'
    np.savez(
        path,
        name           = result.name,
        N_steps        = result.N_steps,
        dt             = result.dt,
        t              = result.t,
        X_actual_xyth  = result.X_actual_xyth,
        X_ref_xyth     = result.X_ref_xyth,
        u_applied      = result.u_applied,
        xi_ref         = result.xi_ref,
        e              = result.e,
        solve_time_ms  = result.solve_time_ms,
        rmse_pos       = result.rmse_pos,
        rmse_yaw       = result.rmse_yaw,
        max_pos_err    = result.max_pos_err,
        max_yaw_err    = result.max_yaw_err,
        solve_mean_ms  = result.solve_mean_ms,
        solve_max_ms   = result.solve_max_ms,
        solve_p95_ms   = result.solve_p95_ms,
        n_infeasible   = result.n_infeasible,
    )
    return path


def print_table(results):
    head = (f'{"trajectory":<13} {"N":>5} {"RMSE_xy[m]":>11} {"RMSE_yaw[rad]":>14} '
            f'{"max_xy[m]":>10} {"max_yaw[rad]":>13} '
            f'{"solve_mean[ms]":>15} {"solve_p95[ms]":>14} {"solve_max[ms]":>14} {"infeas":>7}')
    print(head)
    print('-' * len(head))
    for r in results:
        print(f'{r.name:<13} {r.N_steps:>5} '
              f'{r.rmse_pos:>11.4f} {r.rmse_yaw:>14.4f} '
              f'{r.max_pos_err:>10.4f} {r.max_yaw_err:>13.4f} '
              f'{r.solve_mean_ms:>15.3f} {r.solve_p95_ms:>14.3f} '
              f'{r.solve_max_ms:>14.3f} {r.n_infeasible:>7d}')


def main():
    selected = sys.argv[1] if len(sys.argv) > 1 else None
    out_dir  = Path(__file__).parent / 'logs'

    results = []
    for tr in all_trajectories(dt=0.05):
        if selected and tr.name != selected:
            continue
        print(f'Running {tr.name} ...')
        r = run_trajectory(tr)
        save(r, out_dir)
        results.append(r)

    if results:
        print()
        print_table(results)


if __name__ == '__main__':
    main()
