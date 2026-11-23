"""Verify the split static/dynamic CBF slack.

Two things must hold before this goes anywhere near a benchmark:

1. **Backwards compatibility.** With cbf_static_slack_scale = 1.0 the solver must
   reproduce the old shared-slack behaviour bit-for-bit, otherwise the validated
   98% configuration has silently changed.

2. **It actually does the intended thing.** Put the robot in the pinch the change
   is meant to fix -- a dynamic obstacle on one side, a wall on the other, close
   enough that the QP cannot satisfy both -- and check that raising the scale
   shifts the violation off the wall and onto the dynamic buffer.

Run:  python3 evaluation/test_split_slack.py
"""
import sys

import numpy as np

sys.path.insert(0, '/home/howardchen/masterthesis/src/ammr_wholebody_mpc')

from ammr_wholebody_mpc.gmpc import GMPC, GMPCConfig      # noqa: E402
from ammr_wholebody_mpc import se2                        # noqa: E402

DT, N = 0.05, 20
ROBOT_R = 0.30
DYN_MARGIN, STATIC_MARGIN = 0.38, 0.33


def make(scale):
    return GMPCConfig(
        N=N, dt=DT,
        u_min=np.array([-0.20, -0.25, -0.80]),
        u_max=np.array([0.35, 0.25, 0.80]),
        a_max=np.array([0.8, 0.6, 1.2]),
        Q=np.diag([15.0, 15.0, 7.0]), R=np.diag([2.0, 2.0, 1.0]),
        Qf=5.0 * np.diag([15.0, 15.0, 7.0]),
        S=np.diag([15.0, 15.0, 8.0]),
        cbf_alpha=3.0, cbf_safe_margin=DYN_MARGIN,
        cbf_slack_weight=5e2, cbf_eps0_scale=30.0,
        cbf_danger_thresh=0.4, cbf_Q_min_scale=0.20, cbf_slack_max_scale=20.0,
        cbf_static_slack_scale=scale,
    )


def scene(gap):
    """Robot heading +x, a moving obstacle closing from the left-front and a wall
    just to the right. `gap` sets how tight the corridor is."""
    pose = (0.0, 0.0, 0.0)
    X_now = se2.from_xytheta(*pose)
    X_ref = np.stack([se2.from_xytheta(0.22 * DT * k, 0.0, 0.0) for k in range(N)])
    xi_ref = np.tile(np.array([0.22, 0.0, 0.0]), (N, 1))
    # Keep-out radii: dynamic 0.25+0.38 = 0.63, wall 0.05+0.33 = 0.38. With the
    # two on opposite sides at +-gap the feasible band is
    #   y in (-gap+0.38, gap-0.63)
    # which is EMPTY once 2*gap < 1.01, i.e. gap < 0.505. Below that the QP has
    # to spend slack, which is the only regime where splitting it can matter.
    obstacles = [
        dict(x=0.40, y=gap, radius=0.25, vx=-0.30, vy=0.0, margin=DYN_MARGIN,
             static=False),
        dict(x=0.40, y=-gap, radius=0.05, vx=0.0, vy=0.0, margin=STATIC_MARGIN,
             static=True),
    ]
    return X_now, X_ref, xi_ref, obstacles


def clearances(u, obstacles, pose=(0.0, 0.0, 0.0)):
    """Surface-to-surface clearance after one step of applying u."""
    X = se2.from_xytheta(*pose) @ se2.exp_(np.asarray(u) * DT)
    p = X[:2, 2]
    out = {}
    for o in obstacles:
        d = np.hypot(p[0] - o['x'], p[1] - o['y'])
        out['static' if o['static'] else 'dynamic'] = d - o['radius'] - ROBOT_R
    return out


def main():
    xi_prev = np.array([0.22, 0.0, 0.0])

    # ---- 1. backwards compatibility -------------------------------------
    print('1) scale = 1.0 must reproduce the shared-slack solver')
    worst = 0.0
    rng = np.random.default_rng(0)
    for _ in range(40):
        gap = float(rng.uniform(0.30, 1.0))
        X_now, X_ref, xi_ref, obs = scene(gap)
        a = GMPC(make(1.0)).solve(X_now, X_ref, xi_ref, xi_prev, obs)
        # reference path: no 'static' key anywhere -> split never triggers
        obs_plain = [{k: v for k, v in o.items() if k != 'static'} for o in obs]
        b = GMPC(make(1.0)).solve(X_now, X_ref, xi_ref, xi_prev, obs_plain)
        worst = max(worst, float(np.linalg.norm(a.u_opt - b.u_opt)))
    print(f'   max |u_split_off - u_shared| = {worst:.2e}  '
          f'({"OK" if worst < 1e-6 else "FAIL"})')

    # ---- 2. does it move the violation off the wall? ---------------------
    print('\n2) pinched between a mover and a wall: where does the slack go?')
    print(f'   {"gap":>5} {"scale":>7} {"dyn clr":>9} {"wall clr":>9} {"min_h":>8}')
    for gap in (0.50, 0.45, 0.40, 0.35):
        row = {}
        for scale in (1.0, 10.0):
            X_now, X_ref, xi_ref, obs = scene(gap)
            r = GMPC(make(scale)).solve(X_now, X_ref, xi_ref, xi_prev, obs)
            c = clearances(r.u_opt, obs)
            row[scale] = c
            print(f'   {gap:>5.2f} {scale:>7.1f} {c["dynamic"]:>9.3f} '
                  f'{c["static"]:>9.3f} {r.min_h:>8.3f}')
        d_wall = row[10.0]['static'] - row[1.0]['static']
        d_dyn = row[10.0]['dynamic'] - row[1.0]['dynamic']
        verdict = ('wall clearance improved' if d_wall > 1e-4 else
                   'no shift' if abs(d_wall) <= 1e-4 else 'wall got WORSE')
        print(f'   {"":>5} {"->":>7} wall {d_wall:+.4f} m, dyn {d_dyn:+.4f} m'
              f'   [{verdict}]')


if __name__ == '__main__':
    main()
