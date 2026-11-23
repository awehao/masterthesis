"""Offline oracle: analytic closest distance between link meshes and obstacle
primitives, with the regression tests that pin it down.

This is NOT in the control loop and is not meant to be. It costs 11.6 s at the
median per configuration against a 50 ms period -- about 1850x slower than the
sampling path it checks, and that is before the 52.2 ms of end-to-end latency it
would have to fit inside. Its job is to be right, not fast:

    compute the true minimum distance between a link's collision mesh and an
    obstacle, so the online sampling approximation can be shown never to
    overstate the clearance it has.

The online path uses certified dense surface sampling and pays about 13 mm of
conservatism for it. That figure is measured here, which is what makes it a
documented geometric margin rather than an unknown error.

    python3 evaluation/verify_analytic_oracle.py            # geometry only
    python3 evaluation/verify_analytic_oracle.py --oracle   # + compare online
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np

sys.path.insert(0, 'src/ammr_wholebody_mpc')
sys.path.insert(0, 'evaluation')

from ammr_wholebody_mpc.arm_detection_points import (  # noqa: E402
    Obstacle, _iso, _rpy_to_rot)
from ammr_wholebody_mpc.arm_link_geometry import (  # noqa: E402
    closest_link_to_obstacle, link_collision_tris, nearest_points,
    primitive_sdf, sample_links_certified)
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402
import verify_pregrasp as VP  # noqa: E402

PASS = []


def check(name, cond, detail=''):
    PASS.append(bool(cond))
    print(f"  {'OK ' if cond else '★  '}{name}{('   ' + detail) if detail else ''}")


def box(size, xyz=(0, 0, 0), rpy=(0, 0, 0)):
    o = Obstacle(name='b', model='', kind='box')
    o.size = np.array(size, float)
    o.T_world_link = _iso(_rpy_to_rot(*rpy), np.array(xyz, float))
    o.T_link_collision = np.eye(4)
    return o


def cyl(r, h, xyz=(0, 0, 0), rpy=(0, 0, 0)):
    o = Obstacle(name='c', model='', kind='cylinder')
    o.radius, o.height = r, h
    o.T_world_link = _iso(_rpy_to_rot(*rpy), np.array(xyz, float))
    o.T_link_collision = np.eye(4)
    return o


def sph(r, xyz=(0, 0, 0)):
    o = Obstacle(name='s', model='', kind='sphere')
    o.radius = r
    o.T_world_link = _iso(np.eye(3), np.array(xyz, float))
    o.T_link_collision = np.eye(4)
    return o


def tri(*pts):
    return np.array([list(pts)], float)


def geometry_tests():
    """Each case has an answer that can be written down without running the
    code being tested, which is the only kind of test worth having here."""
    B = box((1.0, 1.0, 1.0))            # faces at +-0.5 on each axis

    r = closest_link_to_obstacle(tri((0.8, 0, 0), (0.8, 0.1, 0), (0.8, 0, 0.1)),
                                 B, tol=1e-5)
    check('面：距離 0.3', abs(r['d'] - 0.3) < 1e-4,
          f"d={r['d']:.6f}  gap={r['gap']:.2e}")
    check('面：法向指向障礙物', abs(r['n'][0] + 1) < 1e-6, str(np.round(r['n'], 4)))
    check('面：witness 落在障礙表面', abs(r['surface'][0] - 0.5) < 1e-6,
          f"s*={np.round(r['surface'], 4)}")

    r = closest_link_to_obstacle(
        tri((0.8, 0, 0.8), (0.85, 0.05, 0.8), (0.8, 0.05, 0.85)), B, tol=1e-5)
    e = math.sqrt(2) * 0.3
    check('稜邊：兩面等距', abs(r['d'] - e) < 1e-3, f"d={r['d']:.6f} 期望 {e:.6f}")

    r = closest_link_to_obstacle(
        tri((0.8, 0.8, 0.8), (0.85, 0.8, 0.8), (0.8, 0.85, 0.8)), B, tol=1e-5)
    e = math.sqrt(3) * 0.3
    check('角點：三面等距', abs(r['d'] - e) < 1e-3, f"d={r['d']:.6f} 期望 {e:.6f}")

    r = closest_link_to_obstacle(tri((0.0, 0, 0), (0.1, 0, 0), (0.0, 0.1, 0)),
                                 B, tol=1e-5)
    check('穿透：有號距離為負', r['d'] < 0, f"d={r['d']:.6f}")
    sdf, _, nn = primitive_sdf(B, np.array([[0.0, 0, 0]]))
    check('穿透：法向指向更深（退離方向正確）',
          float(nn[0] @ np.array([1.0, 0, 0])) < 0 or abs(nn[0][0]) > 0.99,
          f"sdf={sdf[0]:.4f}  n={np.round(nn[0], 3)}")

    C = cyl(0.3, 1.0)
    r = closest_link_to_obstacle(
        tri((0.6, 0, 0.5), (0.65, 0.05, 0.5), (0.6, 0.05, 0.55)), C, tol=1e-5)
    rr = float(np.hypot(r['surface'][0], r['surface'][1]))
    check('圓柱側面／端面接縫', abs(rr - 0.3) < 2e-3 and abs(r['surface'][2] - 0.5) < 2e-3,
          f"s* r={rr:.4f} z={r['surface'][2]:.4f}")

    r = closest_link_to_obstacle(
        tri((1.0, 0, 0), (1.05, 0.05, 0), (1.0, 0.05, 0.05)), sph(0.4), tol=1e-5)
    check('球：距離 0.6', abs(r['d'] - 0.6) < 1e-3, f"d={r['d']:.6f}")

    rng = np.random.default_rng(0)
    T0 = tri((0.8, 0.1, 0.2), (0.9, 0.2, 0.15), (0.85, 0.05, 0.3))
    d0 = closest_link_to_obstacle(T0, B, tol=1e-6)['d']
    worst = 0.0
    for _ in range(20):
        R = _rpy_to_rot(*rng.uniform(-3, 3, 3))
        t = rng.uniform(-2, 2, 3)
        B2 = box((1., 1., 1.))
        B2.T_world_link = _iso(R, t)
        d = closest_link_to_obstacle((T0 @ R.T) + t, B2, tol=1e-6)['d']
        worst = max(worst, abs(d - d0))
    check('SE(3) 不變性', worst < 1e-5, f'20 組，最大變化 {worst:.3e} m')

    Ts = np.array([[[0.8, 0, 0.8], [0.85, 0.05, 0.8], [0.8, 0.05, 0.85]],
                   [[0.801, 0, 0.8], [0.851, 0.05, 0.8], [0.801, 0.05, 0.85]],
                   [[1.5, 0, 0], [1.55, 0.05, 0], [1.5, 0.05, 0.05]]], float)
    r = closest_link_to_obstacle(Ts, B, tol=1e-5, eps_face=0.002)
    check('近等距面全部保留、遠的排除',
          len(r['active']) >= 2 and 2 not in r['active'], f"active={r['active']}")


def oracle_vs_sampling(urdf, world, n_pose=10, seed=31):
    """The online sampling lower bound must never exceed the analytic truth."""
    xml = open(urdf).read()
    K = WholeBodyKinematics.from_urdf_string(xml)
    obs = VP.obstacles_from_world(world)
    TR = link_collision_tris(xml)
    S = sample_links_certified(xml, rho_target=0.015, tol=0.001)
    n = len(K.dof_names)
    idx = [K.dof_names.index(f'joint{i}') for i in range(1, 7)]
    rng = np.random.default_rng(seed)
    viol, tot, slack = 0, 0, []
    for _ in range(n_pose):
        q = np.zeros(n)
        q[:3] = [17.35, 14.25, math.pi]
        q[idx] = rng.uniform(-1.4, 1.4, 6)
        near = [o for o in obs
                if np.linalg.norm(o.T_world_link[:2, 3] - q[:2]) < 3.0]
        samp = {}
        for p_ in nearest_points(K, q, near, S):
            samp[p_.link] = min(samp.get(p_.link, 1e9), p_.d - p_.rho)
        for nm, tl in TR.items():
            if nm not in samp:
                continue
            T = K.fk(q, nm)
            W = (tl @ T[:3, :3].T) + T[:3, 3]
            best = min((closest_link_to_obstacle(W, o, tol=5e-4)['d']
                        for o in near), default=np.inf)
            tot += 1
            s_ = best - samp[nm]
            slack.append(s_)
            viol += int(s_ < -1e-9)
    sl = np.array(slack)
    print(f"\n  oracle vs 線上取樣（{tot} 組 連桿x姿態）")
    check('取樣下界從未高於解析真值', viol == 0, f'{viol}/{tot}')
    print(f"     解析 − 取樣下界：最小 {sl.min()*1000:+.3f} mm  "
          f"中位 {np.median(sl)*1000:+.3f}  最大 {sl.max()*1000:+.3f} mm")
    print("     中位即線上方法付出的保守量，應接近 rho")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--urdf', default='evaluation/results/wholebody_expanded.urdf')
    ap.add_argument('--world', default='src/ammr_bringup/worlds/random_room.sdf')
    ap.add_argument('--oracle', action='store_true',
                    help='also run the slow comparison against the online path')
    a = ap.parse_args()
    print('  ── 解析最近點幾何測試')
    geometry_tests()
    if a.oracle:
        oracle_vs_sampling(a.urdf, a.world)
    print(f"\n  {sum(PASS)}/{len(PASS)} 通過")
    return 0 if all(PASS) else 1


if __name__ == '__main__':
    sys.exit(main())
