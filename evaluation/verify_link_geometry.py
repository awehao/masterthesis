"""Acceptance for the whole-link geometric safety representation.

Two things have to be shown, and they are shown separately because they are
guaranteed by different arguments.

DISTANCE.  The row uses d(s) - rho, where s is the sampled surface point with
the smallest distance. Because the distance-to-obstacle field is 1-Lipschitz
and some sample lies within rho of the true closest point,

    d(s) - rho  <=  d_true

is an inequality, not an estimate. The test confirms it holds and reports how
much slack is left, which is the price paid for discretisation.

VELOCITY.  This one is NOT guaranteed by the same argument and must not be
claimed to be. The row is built at s, but the true closest point x* can be
anywhere on the link -- rho bounds the DISTANCE error, not the separation
between s and x*. On a rigid link two points differ in velocity by
|omega| |x - s|, which is bounded by the link's length, not by rho. So the
approach rate the row constrains can be smaller than the true one. The test
measures that difference directly, over random poses, random obstacle
directions and random generalised velocities, and reports the worst case.

The twelve fixed detection points are measured alongside, on the same samples,
so the improvement is a comparison rather than an assertion.

    python3 evaluation/verify_link_geometry.py <expanded.urdf> [--n 400]
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
    ARM_LINKS, _fps, _load_stl_tris, _surface_points, nearest_points,
    obstacle_distances, sample_links)
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402
import verify_pregrasp as VP  # noqa: E402

DETECT = ['detect0_1', 'detect0_2', 'detect1', 'detect2_1', 'detect2_2',
          'detect2_3', 'detect3_1', 'detect3_2', 'detect4_1', 'detect4_2',
          'detect5', 'detect6']
OWNER = {'detect0_1': 'link_base', 'detect0_2': 'link_base', 'detect1': 'link1',
         'detect2_1': 'link2', 'detect2_2': 'link2', 'detect2_3': 'link2',
         'detect3_1': 'link3', 'detect3_2': 'link3', 'detect4_1': 'link4',
         'detect4_2': 'link4', 'detect5': 'link5', 'detect6': 'uflite_gripper_link'}


def dense_reference(xml, rho_ref=0.004):
    """A much finer sampling than the barrier uses, standing in for the true
    surface.

    Its own covering radius matters: a sampled minimum OVER-estimates the true
    one by at most rho_ref, so a slack smaller than rho_ref proves nothing. The
    first run capped out at 9.22 mm while the smallest slack was 4.53 mm, which
    left the distance claim undecided exactly where it mattered. The cap is
    lifted here so the reference is finer than the margin being checked.
    """
    return sample_links(xml, rho_target=rho_ref, n_ref=200000, seed=11, cap=3000)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('urdf')
    ap.add_argument('--world', default='src/ammr_bringup/worlds/random_room.sdf')
    ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--rho', type=float, default=0.015)
    ap.add_argument('--k', type=int, default=6,
                    help='sampled points constrained per link')
    ap.add_argument('--seed', type=int, default=0,
                    help='draws poses, obstacle directions and velocities; use '
                         'a seed that took no part in fitting the margin')
    ap.add_argument('--knn', action='store_true',
                    help='use the old k-nearest selection instead of the band')
    ap.add_argument('--ev', type=float, default=0.0165,
                    help='empirical velocity error margin being checked, m/s')
    a = ap.parse_args()

    xml = open(a.urdf).read()
    K = WholeBodyKinematics.from_urdf_string(xml)
    obs_all = VP.obstacles_from_world(a.world)
    n = len(K.dof_names)
    idx = [K.dof_names.index(f'joint{i}') for i in range(1, 7)]

    print('  建立取樣…')
    S = sample_links(xml, rho_target=a.rho)
    R = dense_reference(xml)
    print(f'    屏障取樣 {sum(len(s.points) for s in S.values())} 點，'
          f'最大 rho {max(s.rho for s in S.values())*1000:.2f} mm')
    print(f'    參考取樣 {sum(len(s.points) for s in R.values())} 點，'
          f'最大 rho {max(s.rho for s in R.values())*1000:.2f} mm\n')

    rng = np.random.default_rng(a.seed)
    lo6, hi6 = LITE6_SAFE.lower, LITE6_SAFE.upper
    vmax6 = LITE6_SAFE.max_velocity

    # place the base near a box each trial so the arm actually has something
    # close by; a barrier tested only in free space proves nothing
    boxes = [o for o in obs_all if o.kind == 'box']

    dist_slack = []          # d_true - (d_s - rho); must be >= 0
    dist_viol = 0
    old_slack = []           # d_true - d_detect ; negative = the old rows were optimistic
    vel_gap = []             # true approach rate - rate the row constrains
    vel_d = []               # the true distance each of those was measured at
    vel_viol = 0
    worst = None

    for t in range(a.n):
        box = boxes[int(rng.integers(len(boxes)))]
        c = box.T_world_link[:3, 3]
        face = np.array([1.0, 0.0, 0.0]) if rng.random() < 0.5 else np.array([0.0, 1.0, 0.0])
        face = face * (1.0 if rng.random() < 0.5 else -1.0)
        half = 0.5 * float(box.size @ np.abs(face))
        off = float(rng.uniform(0.45, 0.80))
        p_base = c + face * (half + off)
        q = np.zeros(n)
        q[:3] = [p_base[0], p_base[1], math.atan2(-face[1], -face[0])]
        q[idx] = rng.uniform(lo6 * 0.9, hi6 * 0.9)
        near = [o for o in obs_all
                if np.linalg.norm(o.T_world_link[:2, 3] - q[:2]) < 3.0]
        if not near:
            continue

        # ---- distance -------------------------------------------------
        NP = nearest_points(K, q, near, S, k_per_link=a.k, band=not a.knn)
        by_link = {}
        for np_ in NP:
            by_link.setdefault(np_.link, []).append(np_)
        for lk, group in by_link.items():
            np_ = group[0]                    # nearest, for the distance test
            T = K.fk(q, np_.link)
            W = (R[np_.link].points @ T[:3, :3].T) + T[:3, 3]
            d_ref, v_ref, _ = obstacle_distances(W, near)
            k_ref = int(np.argmin(d_ref))
            d_true = float(d_ref[k_ref])
            slack = d_true - (np_.d - np_.rho)   # signed on both sides
            dist_slack.append(slack)
            if slack < -1e-9:
                dist_viol += 1

            # ---- velocity ---------------------------------------------
            v = np.zeros(n)
            v[idx] = rng.uniform(-1, 1, 6) * vmax6
            # the binding row is whichever of this link's rows constrains the
            # approach rate most tightly, since all of them are imposed
            # each row also carries an |omega| rho allowance, so the rate it
            # actually permits is the constrained rate plus that allowance
            rate_row = max(
                float(g.n @ (K.jacobian(q, g.link, offset=g.local)[:3] @ v))
                + g.rho * float(np.linalg.norm(
                    K.jacobian(q, g.link, offset=g.local)[3:] @ v))
                for g in group)
            x_true_local = R[np_.link].points[k_ref]
            Jt = K.jacobian(q, np_.link, offset=x_true_local)[:3]
            n_true = v_ref[k_ref] / max(abs(d_true), 1e-9)
            rate_true = float(n_true @ (Jt @ v))        # what actually happens
            gap = rate_true - rate_row
            vel_gap.append(gap)
            vel_d.append(d_true)
            if gap > 1e-9:
                vel_viol += 1
                if worst is None or gap > worst[0]:
                    om = K.jacobian(q, np_.link, offset=np_.local)[3:] @ v
                    worst = (gap, np_.link, d_true, np_.d,
                             float(np.linalg.norm(np_.world - ((T[:3, :3] @ x_true_local) + T[:3, 3]))),
                             float(np.linalg.norm(om)))

        # ---- the old twelve points, same pose -------------------------
        for fr in DETECT:
            p = K.fk(q, fr)[:3, 3]
            d_pt, _, _ = obstacle_distances(p.reshape(1, 3), near)
            link = OWNER[fr]
            T = K.fk(q, link)
            W = (R[link].points @ T[:3, :3].T) + T[:3, 3]
            d_ref, _, _ = obstacle_distances(W, near)
            old_slack.append(float(d_ref.min()) - float(d_pt[0]))

    ds = np.array(dist_slack); vg = np.array(vel_gap); os_ = np.array(old_slack)
    vd = np.array(vel_d)
    print(f'  {a.n} 組隨機姿態／障礙物方向，{len(ds)} 個連桿列\n')
    print('  ── 距離（d_s − rho ≤ d_true 必須成立）')
    print(f'     違反次數            {dist_viol} / {len(ds)}')
    print(f'     最小裕度            {ds.min()*1000:+8.3f} mm')
    print(f'     中位裕度            {np.median(ds)*1000:+8.3f} mm')
    print(f'     最大裕度（浪費量）  {ds.max()*1000:+8.3f} mm')
    print()
    print('  ── 舊的 12 個固定偵測點，同樣姿態')
    print(f'     d_true − d_detect   最小 {os_.min()*1000:+8.3f} mm   '
          f'中位 {np.median(os_)*1000:+8.3f} mm')
    print(f'     樂觀（負值）次數    {int((os_ < -1e-9).sum())} / {len(os_)}'
          f'   最樂觀 {os_.min()*1000:+.1f} mm')
    print()
    print('  ── 速度（真實接近速率 − 該列所約束的速率）')
    print(f'     樂觀次數            {vel_viol} / {len(vg)}'
          f'   ({100*vel_viol/max(len(vg),1):.1f}%)')
    print(f'     最大樂觀量          {vg.max()*1000:+8.2f} mm/s')
    print(f'     95 百分位           {np.percentile(vg,95)*1000:+8.2f} mm/s')
    # Split by how far the link actually was. The normal is v / d, so as d goes
    # to zero it is the difference of two nearly coincident points divided by
    # nothing: the direction stops being defined long before the geometry does.
    # A barrier is not meaningful once a link is already touching, so the two
    # regimes are reported apart rather than averaged into one misleading number.
    for lab, m in (('d_true > 20 mm', vd > 0.020),
                   ('5 < d_true <= 20 mm', (vd > 0.005) & (vd <= 0.020)),
                   ('d_true <= 5 mm（已近接觸，法向病態）', vd <= 0.005)):
        if m.sum() == 0:
            continue
        g_ = vg[m]
        print(f'     {lab:34} {int((g_ > 1e-9).sum()):>4}/{int(m.sum()):<4} '
              f'最大 {g_.max()*1000:+8.2f} mm/s   p95 {np.percentile(g_,95)*1000:+7.2f}')
    far = vd > 0.020
    if far.sum():
        gf = vg[far]
        ok = gf.max() <= a.ev + 1e-12
        print()
        print(f'  ── 對照經驗速度誤差餘裕 e_v = {a.ev*1000:.1f} mm/s'
              f'（seed {a.seed}，未參與估計）')
        print(f'     d > 20 mm 的最大樂觀量 {gf.max()*1000:+8.2f} mm/s   '
              f'{"<= e_v ✓" if ok else "★ 超過 e_v"}')
        print(f'     超過 e_v 的列數        {int((gf > a.ev).sum())} / {int(far.sum())}')
    if worst:
        g, lk, dt, dsp, sep, om = worst
        print(f'     最壞：{lk}  真實距離 {dt:.4f} m  取樣報 {dsp:.4f} m')
        print(f'           取樣點與真正最近點相隔 {sep*1000:.1f} mm，'
              f'該連桿角速度 {om:.3f} rad/s')
        print(f'           |omega| x 間隔 = {om*sep*1000:.2f} mm/s'
              f'（速度差的剛體上界）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
