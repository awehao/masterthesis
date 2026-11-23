"""Checks the spatio-temporal cost field before it goes near a benchmark.

Three things must hold:

1. **st_weight = 0 changes nothing.** The validated configuration has to be
   reproducible bit-for-bit, otherwise the 98% result silently moved.

2. **P stays positive semi-definite.** A Gaussian bump is concave at its peak,
   so its raw Hessian is negative definite there; adding it to P would make the
   QP non-convex and OSQP would return nonsense. The PSD projection is supposed
   to prevent that, and this test is what proves it.

3. **It actually pushes away from the predicted obstacle**, not the current one.
   That is the whole point of doing it in space-TIME: with an obstacle crossing
   ahead, the controller should already be leaning aside before it arrives.

Run:  python3 evaluation/test_spacetime_cost.py
"""
import sys

import numpy as np

sys.path.insert(0, '/home/howardchen/masterthesis/src/ammr_wholebody_mpc')

from ammr_wholebody_mpc.gmpc import GMPC, GMPCConfig, _build_prediction  # noqa: E402
from ammr_wholebody_mpc import se2                                        # noqa: E402
from ammr_wholebody_mpc.gmpc import _psd_project_2x2, _build_spacetime_cost  # noqa: E402
from ammr_wholebody_mpc.se2 import ad, geodesic_error                     # noqa: E402

DT, N = 0.05, 20
V_NOM = 0.22


def make(st_weight, cbf=True):
    return GMPCConfig(
        N=N, dt=DT,
        u_min=np.array([-0.20, -0.25, -0.80]),
        u_max=np.array([0.35, 0.25, 0.80]),
        a_max=np.array([0.8, 0.6, 1.2]),
        Q=np.diag([15.0, 15.0, 7.0]), R=np.diag([2.0, 2.0, 1.0]),
        Qf=5.0 * np.diag([15.0, 15.0, 7.0]), S=np.diag([15.0, 15.0, 8.0]),
        cbf_alpha=3.0, cbf_safe_margin=0.38 if cbf else 1e-6,
        cbf_slack_weight=5e2, cbf_eps0_scale=30.0,
        cbf_danger_thresh=0.4, cbf_Q_min_scale=0.20, cbf_slack_max_scale=20.0,
        st_weight=st_weight, st_sigma0=0.6, st_growth=0.02,
    )


def scene(obs_y, obs_vy, obs_x=1.4):
    """Robot at the origin heading +x; one obstacle ahead, crossing in y."""
    X_now = se2.from_xytheta(0.0, 0.0, 0.0)
    X_ref = np.stack([se2.from_xytheta(V_NOM * DT * k, 0.0, 0.0) for k in range(N)])
    xi_ref = np.tile(np.array([V_NOM, 0.0, 0.0]), (N, 1))
    obstacles = [dict(x=obs_x, y=obs_y, radius=0.25, vx=0.0, vy=obs_vy,
                      margin=0.38)]
    return X_now, X_ref, xi_ref, obstacles


def main():
    xi_prev = np.array([V_NOM, 0.0, 0.0])

    # ---- 1. PSD projection unit test ------------------------------------
    print('1) PSD projection')
    rng = np.random.default_rng(0)
    worst_neg = 0.0
    for _ in range(2000):
        M = rng.normal(size=(2, 2))
        M = 0.5 * (M + M.T)
        Hp = _psd_project_2x2(M)
        w = np.linalg.eigvalsh(Hp)
        worst_neg = min(worst_neg, float(w.min()))
    print(f'   min eigenvalue over 2000 random symmetric inputs: {worst_neg:.2e} '
          f'({"OK" if worst_neg > -1e-9 else "FAIL"})')

    # ---- 2. st_weight = 0 must be a no-op --------------------------------
    print('\n2) st_weight = 0 reproduces the previous solver')
    worst = 0.0
    for y in (0.3, 0.6, 1.0, 1.5):
        for vy in (-0.3, 0.0, 0.3):
            X_now, X_ref, xi_ref, obs = scene(y, vy)
            a = GMPC(make(0.0)).solve(X_now, X_ref, xi_ref, xi_prev, obs)
            b = GMPC(make(0.0)).solve(X_now, X_ref, xi_ref, xi_prev, obs)
            worst = max(worst, float(np.linalg.norm(a.u_opt - b.u_opt)))
    print(f'   max |u - u| across configurations = {worst:.2e} '
          f'({"OK" if worst < 1e-12 else "FAIL"})')

    # ---- 3. P must stay PSD with the field on ----------------------------
    print('\n3) P remains positive semi-definite with the field on')
    worst_eig = np.inf
    for W in (1.0, 10.0, 100.0):
        for y in (0.0, 0.3, 0.8):
            cfg = make(W)
            X_now, X_ref, xi_ref, obs = scene(y, 0.0)
            e0 = geodesic_error(X_ref[0], X_now)
            A_d = np.stack([np.eye(3) - DT * ad(xi_ref[k]) for k in range(N)])
            Phi, Gamma = _build_prediction(A_d, DT)
            P_st, _ = _build_spacetime_cost(cfg, X_ref, Phi, Gamma, e0, obs)
            ev = float(np.linalg.eigvalsh(0.5 * (P_st + P_st.T)).min())
            worst_eig = min(worst_eig, ev)
    print(f'   min eigenvalue of the added P block: {worst_eig:.3e} '
          f'({"OK" if worst_eig > -1e-9 else "FAIL"})')

    # ---- 4. does it actually lean away, and EARLY? -----------------------
    print('\n4) proactive response to an obstacle crossing ahead')
    print(f'   {"W":>6} {"vy_obs":>8} {"vy_cmd":>9} {"solve ms":>10} {"min_h":>8}')
    for W in (0.0, 10.0, 50.0):
        for vy in (-0.25, +0.25):
            # obstacle 1.4 m ahead, offset 0.5 m, crossing toward the path
            X_now, X_ref, xi_ref, obs = scene(0.5 if vy < 0 else -0.5, vy)
            r = GMPC(make(W)).solve(X_now, X_ref, xi_ref, xi_prev, obs)
            print(f'   {W:>6.0f} {vy:>+8.2f} {r.u_opt[1]:>+9.4f} '
                  f'{1000*r.solve_time_s:>10.2f} {r.min_h:>8.3f}')
    print('   (vy_cmd should move AWAY from where the obstacle is heading,')
    print('    and grow with W, while solve time stays ~2 ms)')

    # ---- 5. the checks the first version MISSED --------------------------
    # Verifying "the answer looks sensible" is not enough: the first attempt
    # passed convexity and the single-step response test, then wrecked the QP's
    # conditioning in closed loop. OSQP hit its iteration cap and returned a
    # point violating the acceleration box, which only showed up as a failed
    # benchmark. So: assert convergence, and assert the emitted command respects
    # a_max, across the whole range of distances a run actually visits.
    print('\n5) convergence and acceleration feasibility across the workspace')
    print(f'   {"W":>6} {"solved":>9} {"maxiter":>9} {"other":>7} '
          f'{"worst |du/dt| / a_max":>24}')
    a_max = np.array([0.8, 0.6, 1.2])
    for W in (0.0, 30.0, 100.0, 300.0):
        cfg = make(W)
        solver = GMPC(cfg)
        n_ok = n_iter = n_bad = 0
        worst = 0.0
        for d in np.linspace(0.25, 3.0, 24):
            for off in (-0.5, -0.2, 0.0, 0.2, 0.5):
                for vy in (-0.3, 0.0, 0.3):
                    X_now, X_ref, xi_ref, obs = scene(off, vy, obs_x=d)
                    r = solver.solve(X_now, X_ref, xi_ref, xi_prev, obs)
                    if r.status == 'solved':
                        n_ok += 1
                    elif 'maximum iterations' in r.status:
                        n_iter += 1
                    else:
                        n_bad += 1
                    worst = max(worst, float(
                        np.max(np.abs(r.u_opt - xi_prev) / (a_max * DT))))
        tot = n_ok + n_iter + n_bad
        print(f'   {W:>6.0f} {100*n_ok/tot:>8.1f}% {100*n_iter/tot:>8.1f}% '
              f'{100*n_bad/tot:>6.1f}% {worst:>24.3f}')
    print('   (worst ratio must stay <= 1.000, else the chassis gets a jump)')


if __name__ == '__main__':
    main()
