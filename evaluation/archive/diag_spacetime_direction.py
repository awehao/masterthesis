"""Does the spatio-temporal cost field push AWAY from obstacles, in closed loop?

The single-step unit test only ever placed the obstacle straight ahead of a
robot whose reference heading was zero, i.e. R_ref = I. That is exactly the case
that cannot catch a rotation mistake: if the reference-frame rotation in

    p(k) = p_ref(k) + R_ref(k) e_xy(k)

were dropped, transposed, or taken from the wrong frame, the sign would only
show up once the reference heading is non-zero. In gz the field made the robot
collide (clearance -0.100 m) while the numerics were healthy, which is what a
wrong push direction looks like.

This replays the real 2D scenario, and at every step solves the QP twice, with
and without the field, then projects the difference onto the direction away from
the nearest obstacle:

    align = (u_ST - u_base) . away_hat        > 0 means it pushes AWAY (correct)

Run:  python3 evaluation/diag_spacetime_direction.py
"""
import sys

import numpy as np

sys.path.insert(0, '/home/howardchen/masterthesis/rl_smoothness')
sys.path.insert(0, '/home/howardchen/masterthesis/src/ammr_wholebody_mpc')

from env_real import RealAvoidEnv                     # noqa: E402
from ammr_wholebody_mpc.gmpc import GMPC              # noqa: E402
from ammr_wholebody_mpc import se2                    # noqa: E402

W_TEST = 30.0
MAX_STEPS = 1500


def main():
    env = RealAvoidEnv(seed=0, lag_beta=0.0, max_steps=5000)

    import copy
    cfg_st = copy.deepcopy(env.cfg)
    cfg_st.st_weight = W_TEST
    gmpc_st = GMPC(cfg_st)

    align, mags, headings, dists = [], [], [], []
    done, n = False, 0
    while not done and n < MAX_STEPS:
        X_ref, xi_ref = env._ref_window()
        X_now = se2.from_xytheta(*env.pose)
        obs = env._obstacles_for_cbf()
        dyn = [o for o in obs if not o.get('margin')]      # dynamic only
        if dyn:
            r0 = env.gmpc.solve(X_now, X_ref, xi_ref, env.xi_prev, obs)
            r1 = gmpc_st.solve(X_now, X_ref, xi_ref, env.xi_prev, obs)
            du = r1.u_opt - r0.u_opt                        # body-frame twist
            p = env.pose[:2]
            th = env.pose[2]
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            dv_world = R @ du[:2]                           # world-frame velocity change
            near = min(dyn, key=lambda o: np.hypot(o['x'] - p[0], o['y'] - p[1]))
            away = p - np.array([near['x'], near['y']])
            d = np.linalg.norm(away)
            if d > 1e-6 and np.linalg.norm(dv_world) > 1e-6:
                align.append(float(dv_world @ (away / d)))
                mags.append(float(np.linalg.norm(dv_world)))
                headings.append(float(X_ref[0][:2, :2][1, 0]))   # sin(theta_ref)
                dists.append(float(d))
        _, _, done, info = env.step(None)
        n += 1

    align = np.array(align); mags = np.array(mags)
    headings = np.array(headings); dists = np.array(dists)
    print(f"samples with a dynamic obstacle in range: {len(align)} / {n} steps")
    if len(align) == 0:
        print("no samples; nothing to say")
        return
    print(f"\npush magnitude |du|: median {np.median(mags):.4f}  max {mags.max():.4f} m/s")
    print(f"alignment with 'away from obstacle':")
    print(f"  median {np.median(align):+.4f}   mean {align.mean():+.4f}")
    print(f"  pushes AWAY  (align > 0): {100*(align > 0).mean():5.1f}%")
    print(f"  pushes TOWARD(align < 0): {100*(align < 0).mean():5.1f}%")

    print(f"\nby reference heading (sin theta_ref) -- a rotation bug shows up here:")
    for lo, hi in ((-1.01, -0.5), (-0.5, -0.1), (-0.1, 0.1), (0.1, 0.5), (0.5, 1.01)):
        m = (headings >= lo) & (headings < hi)
        if m.sum() > 5:
            print(f"  sin(th_ref) in [{lo:+.1f},{hi:+.1f}): n={m.sum():4d}  "
                  f"away {100*(align[m] > 0).mean():5.1f}%   "
                  f"median align {np.median(align[m]):+.4f}")

    print(f"\nby distance to the obstacle:")
    for lo, hi in ((0.0, 0.8), (0.8, 1.2), (1.2, 1.8), (1.8, 3.1)):
        m = (dists >= lo) & (dists < hi)
        if m.sum() > 5:
            print(f"  d in [{lo:.1f},{hi:.1f}) m: n={m.sum():4d}  "
                  f"away {100*(align[m] > 0).mean():5.1f}%   "
                  f"|du| median {np.median(mags[m]):.4f}")


if __name__ == '__main__':
    main()
