"""Is max_rows_per_link safe, or does it drop constraints that bind?

The distance node keeps only the nearest few samples of each link's band. That
was justified on one pose where the output happened to be identical, which
proves nothing: a sample further away can still produce the binding row,
because the row is n^T J v and both the normal and the Jacobian differ from
point to point. A point 10 mm further from the obstacle whose normal lines up
with the motion constrains the command harder than a nearer one facing away.

So the test is not whether the two outputs look alike. It is whether the
truncated solution SATISFIES THE FULL ROW SET:

    solve with the cap, then evaluate the uncapped rows at that solution

Any row that comes out positive is a constraint the cap discarded and the
motion then broke. Random poses and random commanded velocities, because a
single velocity direction exercises one corner of the constraint geometry.

    python3 evaluation/verify_row_truncation.py [--n 300] [--cap 8]
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np

sys.path.insert(0, 'src/ammr_wholebody_mpc')
sys.path.insert(0, 'evaluation')

from ammr_wholebody_mpc.arm_limits import LITE6_SAFE  # noqa: E402
from ammr_wholebody_mpc.arm_link_geometry import (  # noqa: E402
    nearest_points, sample_links_certified)
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402
from ammr_wholebody_mpc.wholebody_safety_filter import (  # noqa: E402
    STATUS_OK, DetectionPoint, SafetyConfig, _rows_from_points,
    filter_velocity)
import verify_pregrasp as VP  # noqa: E402


def to_pts(NP):
    return [DetectionPoint(frame=p.link, p=p.world, n=p.n, d=p.d,
                           status=STATUS_OK, offset=p.local, rho=p.rho)
            for p in NP]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--urdf', default='evaluation/results/wholebody_expanded.urdf')
    ap.add_argument('--world', default='src/ammr_bringup/worlds/random_room.sdf')
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--cap', type=int, default=8)
    ap.add_argument('--seed', type=int, default=909)
    a = ap.parse_args()

    xml = open(a.urdf).read()
    K = WholeBodyKinematics.from_urdf_string(xml)
    obs = VP.obstacles_from_world(a.world)
    S = sample_links_certified(xml, rho_target=0.015, tol=0.001)
    cfg = SafetyConfig()
    n = len(K.dof_names)
    idx = [K.dof_names.index(f'joint{i}') for i in range(1, 7)]
    rng = np.random.default_rng(a.seed)
    lo6, hi6 = LITE6_SAFE.lower, LITE6_SAFE.upper
    vmax6 = LITE6_SAFE.max_velocity
    boxes = [o for o in obs if o.kind == 'box']

    dv, resid_gap, missed, n_full, n_cap = [], [], [], [], []
    worst = None
    trials = 0
    for _ in range(a.n):
        box = boxes[int(rng.integers(len(boxes)))]
        c = box.T_world_link[:3, 3]
        face = np.array([1.0, 0.0, 0.0]) if rng.random() < 0.5 else np.array([0.0, 1.0, 0.0])
        face = face * (1.0 if rng.random() < 0.5 else -1.0)
        half = 0.5 * float(box.size @ np.abs(face))
        p_base = c + face * (half + float(rng.uniform(0.45, 0.80)))
        q = np.zeros(n)
        q[:3] = [p_base[0], p_base[1], math.atan2(-face[1], -face[0])]
        q[idx] = rng.uniform(lo6 * 0.9, hi6 * 0.9)
        near = [o for o in obs
                if np.linalg.norm(o.T_world_link[:2, 3] - q[:2]) < 3.0]
        if not near:
            continue
        v_in = np.zeros(n)
        v_in[idx] = rng.uniform(-1, 1, 6) * vmax6
        v_prev = np.zeros(n)
        v_prev[idx] = rng.uniform(-0.3, 0.3, 6) * vmax6

        full = to_pts(nearest_points(K, q, near, S, k_max=10 ** 6))
        capped = to_pts(nearest_points(K, q, near, S, k_max=a.cap))
        if not full:
            continue
        trials += 1
        n_full.append(len(full))
        n_cap.append(len(capped))

        r_c = filter_velocity(K, q, v_in, capped, cfg, v_prev=v_prev, dt=cfg.dt)
        r_f = filter_velocity(K, q, v_in, full, cfg, v_prev=v_prev, dt=cfg.dt)
        dv.append(float(np.abs(r_c.v - r_f.v).max()))

        # both solutions against the SAME full row set
        A, b, _ = _rows_from_points(K, q, full, cfg, v_in)
        if A:
            A = np.array(A); b = np.array(b)
            res_c = A @ r_c.v - b
            res_f = A @ r_f.v - b
            g = float(res_c.max() - res_f.max())      # capping's own cost
            resid_gap.append(g)
            # rows the capped solution breaks that the uncapped one does not
            missed.append(int(((res_c > 1e-6) & (res_f <= 1e-6)).sum()))
            if worst is None or g > worst[0]:
                worst = (g, missed[-1], len(full), len(capped))

    dv = np.array(dv); rg = np.array(resid_gap); ms = np.array(missed)
    print(f'  {trials} 組隨機姿態／速度   cap={a.cap}')
    print(f'  列數：完整 中位 {np.median(n_full):.0f} 最大 {max(n_full)}   '
          f'截斷 中位 {np.median(n_cap):.0f} 最大 {max(n_cap)}')
    print()
    print('  ── 截斷造成的額外殘差 = res(截斷解) − res(完整解)，對同一組完整列')
    print(f'     最大                {rg.max()*1000:+9.4f} mm/s')
    print(f'     p95                 {np.percentile(rg,95)*1000:+9.4f} mm/s')
    print(f'     變差的週期數        {int((rg > 1e-6).sum())} / {len(rg)}')
    print(f'     截斷才被破壞的列數  單週期最多 {int(ms.max())}   總計 {int(ms.sum())}')
    print()
    print('  ── 輸出速度差（截斷 vs 完整）')
    print(f'     最大 {dv.max():.3e}   p95 {np.percentile(dv,95):.3e}   '
          f'中位 {np.median(dv):.3e}')
    if worst:
        g, m, nf, nc = worst
        print(f'     最壞：額外殘差 {g*1000:+.4f} mm/s，多壞 {m} 列，'
              f'完整 {nf} 列截成 {nc} 列')
    ok = (rg.max() <= 1e-6)
    print(f'\n  {"通過：截斷未丟掉任何會作用的約束" if ok else "★ 未通過：截斷會丟掉作用中的約束"}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
