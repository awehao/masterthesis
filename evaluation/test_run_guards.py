"""Regression tests for the failure paths, not the success path.

The review that prompted these found the same shape of hole in six places: a
step that works is checked, and the step that fails is assumed not to happen.
Each test below reproduces one failure with the real code, so a later change
that reopens the hole fails here instead of in a run.

    python3 evaluation/test_run_guards.py
"""
import json
import math
import sys
import types

import numpy as np

sys.path.insert(0, '/home/howardchen/masterthesis/src/ammr_wholebody_mpc')
sys.path.insert(0, '/home/howardchen/masterthesis/evaluation')

from ammr_wholebody_mpc import gmpc as G                   # noqa: E402
from ammr_wholebody_mpc.gmpc import GMPC, GMPCConfig       # noqa: E402

FAILS = []


def check(name, ok, detail=''):
    print(f'  {"PASS" if ok else "FAIL"}  {name}' + (f'   {detail}' if detail else ''))
    if not ok:
        FAILS.append(name)


# ---------------------------------------------------------------------------
# 1. A non-converged QP must not be allowed to command impossible wheel speeds
# ---------------------------------------------------------------------------
# gmpc accepts 'maximum iterations reached' as usable rather than freezing the
# robot. That is a deliberate trade, but the point OSQP returns then satisfies
# nothing, and the per-axis box clip underneath it cannot see the wheel-coupling
# rows. Inject exactly such a point and check what reaches the chassis.

def cfg_like_node():
    """The parameters gmpc_node actually declares, so the test constrains the
    configuration that runs, not a toy one."""
    return GMPCConfig(
        N=20, dt=0.05,
        u_min=np.array([-0.2775, -0.2775, -1.1327]),
        u_max=np.array([0.2775, 0.2775, 1.1327]),
        a_max=np.array([1.5, 1.0, 2.0]),
        Q=np.diag([10.0, 10.0, 5.0]),
        R=np.diag([0.5, 0.5, 0.2]),
        Qf=np.diag([50.0, 50.0, 25.0]),
        wheel_coupling=True,
    )


class _StuckOSQP:
    """OSQP that always reports max-iterations and returns `x`."""

    def __init__(self, x, status='maximum iterations reached'):
        self._x, self._status = x, status

    def __call__(self):
        return self

    def setup(self, **kw):
        self._n = kw['P'].shape[0]

    def solve(self):
        x = np.zeros(self._n)
        v = np.asarray(self._x, dtype=float)
        x[:v.size] = v
        return types.SimpleNamespace(x=x,
                                     info=types.SimpleNamespace(status=self._status))


def solve_with(fake, cfg, xi_prev=np.zeros(3)):
    X_ref = np.tile(np.eye(3), (cfg.N + 1, 1, 1))
    xi_ref = np.zeros((cfg.N + 1, 3))
    saved, G.osqp = G.osqp, types.SimpleNamespace(OSQP=fake)
    try:
        return GMPC(cfg).solve(X_now=np.eye(3), X_ref_win=X_ref,
                               xi_ref_win=xi_ref, xi_prev=np.asarray(xi_prev, float))
    finally:
        G.osqp = saved


print('\n-- QP output acceptance --')
cfg = cfg_like_node()
w_lim = cfg.wheel_w_max

# The corner where every axis sits on its own limit: the box is satisfied, the
# shared wheel is not. This is the geometry the whole guard exists for.
corner = np.array([0.2775, 0.2775, 1.1327])
check('the box corner really is wheel-infeasible',
      float(np.max(np.abs(G.wheel_speeds(corner, cfg)))) > w_lim,
      f'|ω|max={float(np.max(np.abs(G.wheel_speeds(corner, cfg)))):.4f} rad/s '
      f'vs limit {w_lim:.2f}')

# The single-shot case is NOT the exposure: from rest the acceleration clip
# (a_max·dt = 0.075, 0.05, 0.1) caps one cycle far below the wheel limit on its
# own. What the clip cannot do is bound the MAGNITUDE -- it is a rate limit, so
# a QP that keeps missing its iteration cap ramps 0.075 m/s per cycle straight
# into the corner. Run the loop the controller actually runs, feeding each
# command back as ξ_prev, and watch where it ends up.
def ramp(enforce, cycles=40):
    c = cfg_like_node()
    c.wheel_enforce_output = enforce
    u = np.zeros(3)
    worst, actions = 0.0, set()
    for _ in range(cycles):
        r = solve_with(_StuckOSQP(corner), c, xi_prev=u)
        u = r.u_opt
        actions.add(r.accept_action)
        worst = max(worst, float(np.max(np.abs(G.wheel_speeds(u, c)))))
    return worst, u, actions

worst_off, u_off, _ = ramp(False)
check('WITHOUT the guard a sustained non-converged QP ramps past the wheels',
      worst_off > w_lim,
      f'|ω|max reached {worst_off:.4f} rad/s, u={np.round(u_off, 4)}')

worst_on, u_on, acts_on = ramp(True)
check('WITH the guard the command never leaves the wheel envelope',
      worst_on <= w_lim + 1e-6,
      f'|ω|max reached {worst_on:.4f} rad/s, u={np.round(u_on, 4)}')
check('and the guard reports that it acted',
      'wheel_scaled' in acts_on, f'actions={sorted(acts_on)}')
check('the guard still lets the robot move (it is not a disguised brake)',
      float(np.linalg.norm(u_on[:2])) > 0.05,
      f'|v|={float(np.linalg.norm(u_on[:2])):.4f} m/s')

# fit_to_wheels as a unit: λ is the largest feasible step, not a safe guess.
u_fit, lam = G.fit_to_wheels(corner, np.zeros(3), cfg, cfg.dt)
check('λ is maximal -- 1% more would violate',
      float(np.max(np.abs(G.wheel_speeds(u_fit, cfg)))) <= w_lim + 1e-6
      and float(np.max(np.abs(G.wheel_speeds(corner * lam * 1.01, cfg)))) > w_lim,
      f'λ={lam:.6f}')

# A non-finite solution must brake, not propagate: np.clip passes NaN through.
res_nan = solve_with(_StuckOSQP(np.array([np.nan, 0.0, 0.0])), cfg)
check('a NaN solution brakes instead of reaching cmd_vel',
      np.all(np.isfinite(res_nan.u_opt)) and np.allclose(res_nan.u_opt, 0.0),
      f'u={res_nan.u_opt}, status={res_nan.status!r}')

# A converged solve must be left alone -- the guard is not allowed to alter the
# validated behaviour of the configuration that has been benchmarked.
res_ok = solve_with(_StuckOSQP(np.zeros(3 * cfg.N), status='solved'), cfg)
check('a converged, feasible solve is untouched',
      res_ok.accept_action == 'as_is' and res_ok.accept_scale == 1.0,
      f'action={res_ok.accept_action}')

# ---------------------------------------------------------------------------
# 2. A trial that did not arrive must not carry an arrival time
# ---------------------------------------------------------------------------
from isaac_result import arrival_fields                     # noqa: E402

print('\n-- trial result fields --')

# The OFF run this came from: it timed out, but first_plan_t and the stop time
# both existed, so the old expression produced a completion time regardless.
to = arrival_fields('task_timeout', first_plan_t=10.0, stop_sim_t=190.0,
                    motion_start_t=12.0)
check('a timeout carries no arrival_time_s',
      to['arrival_time_s'] is None and to['arrived'] is False,
      f'arrival_time_s={to["arrival_time_s"]}')
check('but the elapsed time is still recorded, under its own name',
      abs(to['elapsed_until_stop_s'] - 180.0) < 1e-9,
      f'elapsed_until_stop_s={to["elapsed_until_stop_s"]}')

ar = arrival_fields('goal_reached_truth', first_plan_t=10.0, stop_sim_t=95.0,
                    motion_start_t=12.0, dist_goal=0.21, arrive_tol=0.30)
check('an arrival keeps its arrival time',
      ar['arrived'] and abs(ar['arrival_time_s'] - 85.0) < 1e-9,
      f'arrival_time_s={ar["arrival_time_s"]}')
check('and its motion-elapsed time',
      abs(ar['motion_elapsed_s'] - 83.0) < 1e-9,
      f'motion_elapsed_s={ar["motion_elapsed_s"]}')

th = arrival_fields('thermal_abort', first_plan_t=10.0, stop_sim_t=50.0,
                    motion_start_t=12.0)
check('a thermal abort is flagged as not a navigation outcome',
      th['scoreable'] is False and th['arrival_time_s'] is None,
      f'scoreable={th["scoreable"]}')

try:
    arrival_fields('goal_reached_truth', 10.0, 95.0, 12.0,
                   dist_goal=0.62, arrive_tol=0.30)
    ok = False
except ValueError:
    ok = True
check('a label that contradicts the geometry raises instead of being averaged',
      ok)

miss = arrival_fields('goal_reached_truth', first_plan_t=None,
                      stop_sim_t=95.0, motion_start_t=None)
check('a missing first-plan time yields None, not a bogus number',
      miss['arrival_time_s'] is None, f'{miss["arrival_time_s"]}')

print('\n' + ('FAILURES: ' + ', '.join(FAILS) if FAILS else 'all guard tests pass'))
sys.exit(1 if FAILS else 0)
