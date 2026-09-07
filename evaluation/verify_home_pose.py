"""Certified self-collision lower bound for one candidate initial pose.

A sampled minimum between two point clouds is an OVER-estimate of the true
distance between the surfaces they came from: the closest pair of surface
points can fall between samples on either side. The bound that holds is

    d_lower = d(S_A, S_B) - rho_A - rho_B

with each rho the certified covering radius of that link's samples. Both are
subtracted because either surface can be the one whose closest point was
missed.

That matters here. The candidate pose was picked by a sampled estimate of about
21 mm, and two covering radii of roughly 14 mm each consume more than that --
so the estimate that chose the pose cannot confirm it. Densifying only the
pairs that come out tightest is cheaper than densifying everything.

The chassis is included. Its collision geometry is a cylinder rather than a
mesh, so it is tessellated here and certified by the same branch and bound as
the arm links; leaving it out would check the arm against itself and call that
a clearance result.

    python3 evaluation/verify_home_pose.py [--pose a,b,c,d,e,f]
"""
from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, 'src/ammr_wholebody_mpc')
sys.path.insert(0, 'evaluation')

from ammr_wholebody_mpc.arm_link_geometry import (  # noqa: E402
    _fps, _surface_points, certified_covering_radius, link_collision_tris,
    sample_links_certified)
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402

ARM = [f'joint{i}' for i in range(1, 7)]


def cylinder_tris(radius, length, origin, seg=48, rings=6):
    """Triangulate a URDF cylinder so it can go through the same certification."""
    th = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    zs = np.linspace(-length / 2, length / 2, rings)
    ring = np.stack([radius * np.cos(th), radius * np.sin(th)], 1)
    tris = []
    for k in range(rings - 1):
        a = np.c_[ring, np.full(seg, zs[k])]
        b = np.c_[ring, np.full(seg, zs[k + 1])]
        for i in range(seg):
            j = (i + 1) % seg
            tris += [[a[i], a[j], b[i]], [a[j], b[j], b[i]]]
    for z, sgn in ((zs[0], -1), (zs[-1], 1)):
        c = np.array([0.0, 0.0, z])
        r = np.c_[ring, np.full(seg, z)]
        for i in range(seg):
            j = (i + 1) % seg
            tris += [[c, r[i], r[j]][::sgn]]
    T = np.array(tris, float)
    return (T @ origin[:3, :3].T) + origin[:3, 3]


def chassis_samples(xml, rho_target=0.015, tol=0.001):
    root = ET.fromstring(re.sub(r'<!--.*?-->', '', xml, flags=re.S))
    for link in root.findall('link'):
        if link.get('name') != 'base_link':
            continue
        for col in link.findall('collision'):
            g = col.find('geometry')
            cy = g.find('cylinder') if g is not None else None
            if cy is None:
                continue
            o = col.find('origin')
            xyz = np.array([float(x) for x in
                            (o.get('xyz', '0 0 0') if o is not None else '0 0 0').split()])
            T = np.eye(4); T[:3, 3] = xyz
            tris = cylinder_tris(float(cy.get('radius')), float(cy.get('length')), T)
            cloud = _surface_points(tris, 60000, np.random.default_rng(0))
            n = 8
            while n <= 1200:
                sel = _fps(cloud, n, np.random.default_rng(1))
                c = certified_covering_radius(tris, cloud[sel], tol=tol)
                if c['rho'] <= rho_target:
                    return cloud[sel].copy(), float(c['rho']), tris
                n = int(np.ceil(n * 1.6))
            sel = _fps(cloud, 1200, np.random.default_rng(1))
            c = certified_covering_radius(tris, cloud[sel], tol=tol)
            return cloud[sel].copy(), float(c['rho']), tris
    return None, None, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--urdf', default='evaluation/results/wholebody_expanded.urdf')
    ap.add_argument('--pose', default='0.774,-0.348,0.840,-0.919,-0.902,-1.452')
    ap.add_argument('--rho', type=float, default=0.015)
    ap.add_argument('--tight', default='',
                    help='comma-separated links to sample finer')
    ap.add_argument('--tight-rho', dest='trho', type=float, default=0.004,
                    help='covering radius target for those links, m')
    a = ap.parse_args()
    qa = np.array([float(x) for x in a.pose.split(',')])

    xml = open(a.urdf).read()
    K = WholeBodyKinematics.from_urdf_string(xml)
    n = len(K.dof_names)
    idx = [K.dof_names.index(j) for j in ARM]
    q = np.zeros(n); q[idx] = qa

    S = sample_links_certified(xml, rho_target=a.rho, tol=0.001)
    tight = [t for t in a.tight.split(',') if t]
    if tight:
        # Only the links whose pairs came out tightest. Densifying everything
        # buys nothing for the pairs that already have metres of clearance and
        # costs the certification time on every one of them.
        fine = sample_links_certified(xml, rho_target=a.trho, tol=0.0005,
                                      links=tuple(t for t in tight
                                                  if t != 'base_link'),
                                      cap=4000)
        S.update(fine)
        print(f'  加密 {list(fine)} 到 rho <= {a.trho*1e3:.1f} mm')
    cp, crho, _ = chassis_samples(
        xml, rho_target=a.trho if 'base_link' in tight else a.rho)
    pts = {k: (v.points, v.rho) for k, v in S.items()}
    if cp is not None:
        pts['base_link'] = (cp, crho)
    print(f'  取樣：{len(pts)} 個 link，'
          f'最大認證 rho {max(r for _, r in pts.values())*1000:.2f} mm')

    adj, rigid = set(), {}

    def find(x):
        while rigid.get(x, x) != x:
            x = rigid[x]
        return x
    for j in K.joints.values():
        adj.add(frozenset((j.parent, j.child)))
        if j.jtype not in ('revolute', 'prismatic', 'continuous'):
            x, y = find(j.parent), find(j.child)
            if x != y:
                rigid[x] = y
    names = list(pts)
    pairs = [(x, y) for i, x in enumerate(names) for y in names[i + 1:]
             if frozenset((x, y)) not in adj and find(x) != find(y)
             and frozenset((x, y)) != frozenset(('uflite_finger1', 'uflite_finger2'))]

    W = {}
    for nm in names:
        T = K.fk(q, nm)
        W[nm] = (pts[nm][0] @ T[:3, :3].T) + T[:3, 3]
    res = []
    for x, y in pairs:
        d = float(cKDTree(W[x]).query(W[y], k=1)[0].min())
        res.append((d - pts[x][1] - pts[y][1], d, x, y))
    res.sort()
    print(f'  姿態 {np.round(qa, 4).tolist()}')
    print(f'  {len(pairs)} 個非排除配對\n')
    print(f"  {'配對':46} {'取樣距離':>10} {'下界':>10}")
    for lo, d, x, y in res[:8]:
        flag = '  ★ 下界為負，此密度無法確認' if lo <= 0 else ''
        print(f"  {x + ' ↔ ' + y:46} {d*1000:9.2f}mm {lo*1000:9.2f}mm{flag}")
    worst = res[0]
    print(f'\n  最緊配對下界 {worst[0]*1000:+.2f} mm'
          f'（取樣距離 {worst[1]*1000:.2f} mm，兩側 rho 合計 '
          f'{(pts[worst[2]][1]+pts[worst[3]][1])*1000:.2f} mm）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
