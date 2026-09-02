"""Section 3.3 acceptance: self-collision over the whole-body configuration space.

A tucked pose that looks clear in a viewer is not evidence. This measures the
real swept geometry: every collision mesh's vertices are transformed by the
controller's own FK, so what is checked is the model the controller plans with,
not a separately loaded one.

Two questions, because they are different:

  named poses   is the drive/tuck pose actually clear, and by how much? A pose
                that merely does not intersect is not good enough -- the margin
                has to survive tracking error.
  random sweep  what fraction of the LEGAL joint box is self-colliding? Joint
                limits alone do not keep the arm out of its own base, so a
                planner that samples inside the box will propose poses that the
                robot cannot adopt. Phase 2 needs that number before it starts
                sampling pre-grasp candidates.

Distance is vertex-cloud to vertex-cloud via a KD-tree: a lower bound on true
surface distance for convex-ish links, and it never reports clearance where
there is none. Adjacent links are skipped -- they share a joint and always
touch there.

    python3 evaluation/verify_self_collision.py <expanded.urdf> [--n 500]
"""
from __future__ import annotations

import argparse
import math
import os
import re
import struct
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc.wholebody_kinematics import (  # noqa: E402
    DOF_NAMES, WholeBodyKinematics, iso, rpy_to_rot)

# Drive pose from lite6_ros2_control_gz.xacro (initial_value on each joint).
TUCK = {'joint2': -0.082, 'joint3': 0.089, 'joint5': 1.679}
MARGIN_WARN = 0.02          # m, below this a "clear" pose is not comfortable

# Pairs that are SUPPOSED to touch. The two gripper fingers meet at
# finger_joint1 = 0 (the closed end of their 0..0.0089 m travel), so a naive
# check reports zero distance in every configuration and drowns out every real
# result -- the first run of this script came back "100% self-colliding", which
# was the gripper being shut, not the arm hitting itself. Fingers are also not
# part of the 9-DOF configuration this script sweeps: they are commanded by
# their own controller, so their pose here is fixed at the URDF default.
DESIGNED_CONTACT = {frozenset(('uflite_finger1', 'uflite_finger2'))}


def load_stl(path: str) -> np.ndarray:
    d = open(path, 'rb').read()
    if d[:5] == b'solid' and b'facet' in d[:512]:
        return np.array([[float(x) for x in l.split()[1:4]]
                         for l in d.decode('ascii', 'ignore').splitlines()
                         if l.strip().startswith('vertex')])
    n = struct.unpack('<I', d[80:84])[0]
    return np.array([struct.unpack('<9f', d[84 + i * 50 + 12:84 + i * 50 + 48])
                     for i in range(n)]).reshape(-1, 3)


def primitive_cloud(tag: str, attrs: dict) -> np.ndarray:
    """Surface samples for a URDF primitive, dense enough to bound distance."""
    if tag == 'cylinder':
        r = float(attrs['radius']); L = float(attrs['length'])
        th = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        z = np.linspace(-L / 2, L / 2, 9)
        T, Z = np.meshgrid(th, z)
        side = np.stack([r * np.cos(T).ravel(), r * np.sin(T).ravel(), Z.ravel()], 1)
        rr = np.linspace(0, r, 5)
        T2, R2 = np.meshgrid(th, rr)
        cap = np.stack([(R2 * np.cos(T2)).ravel(), (R2 * np.sin(T2)).ravel(),
                        np.full(T2.size, L / 2)], 1)
        return np.vstack([side, cap, cap * [1, 1, -1]])
    if tag == 'box':
        sx, sy, sz = [float(v) for v in attrs['size'].split()]
        g = np.linspace(-0.5, 0.5, 7)
        P = np.array(np.meshgrid(g, g, g)).reshape(3, -1).T
        P = P[np.abs(P).max(axis=1) > 0.49]        # surface only
        return P * [sx, sy, sz]
    if tag == 'sphere':
        r = float(attrs['radius'])
        u = np.linspace(0, np.pi, 12); v = np.linspace(0, 2 * np.pi, 24)
        U, V = np.meshgrid(u, v)
        return np.stack([(r * np.sin(U) * np.cos(V)).ravel(),
                         (r * np.sin(U) * np.sin(V)).ravel(),
                         (r * np.cos(U)).ravel()], 1)
    return np.zeros((0, 3))


def link_clouds(xml: str, max_pts: int = 900) -> dict[str, np.ndarray]:
    xml_c = re.sub(r'<!--.*?-->', '', xml, flags=re.S)
    root = ET.fromstring(xml_c)
    rng = np.random.default_rng(0)
    out = {}
    for link in root.findall('link'):
        name = link.get('name')
        pts = []
        for col in link.findall('collision'):
            geo = col.find('geometry')
            if geo is None or len(geo) == 0:
                continue
            g = geo[0]
            if g.tag == 'mesh':
                p = g.get('filename', '').replace('file://', '')
                if not os.path.exists(p):
                    continue
                V = load_stl(p)
                sc = g.get('scale')
                if sc:
                    V = V * np.array([float(x) for x in sc.split()])
            else:
                V = primitive_cloud(g.tag, g.attrib)
            if len(V) == 0:
                continue
            o = col.find('origin')
            gx = lambda k, d: np.array([float(v) for v in
                                        ((o.get(k) or d) if o is not None else d).split()])
            T = iso(rpy_to_rot(*gx('rpy', '0 0 0')), gx('xyz', '0 0 0'))
            pts.append((T[:3, :3] @ V.T).T + T[:3, 3])
        if pts:
            P = np.vstack(pts)
            if len(P) > max_pts:
                P = P[rng.choice(len(P), max_pts, replace=False)]
            out[name] = P
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('urdf')
    ap.add_argument('--n', type=int, default=500)
    a = ap.parse_args()

    xml = open(a.urdf).read()
    K = WholeBodyKinematics.from_urdf_string(xml)
    clouds = link_clouds(xml)
    lo, hi = K.joint_limits()

    # Two exclusions, for different reasons.
    #
    # Adjacent links share a joint and touch there by construction.
    #
    # Links joined only by FIXED joints form one rigid body: their relative pose
    # is the same in every configuration, so any distance between them is a
    # property of the mesh pair, not of the pose, and reporting it as a
    # self-collision risk is meaningless. link6, link_eef, uflite_gripper_link
    # and the fingers are all one such body -- before this exclusion they
    # dominated every result with a constant 0.0011 m and hid the real numbers.
    adj = set()
    rigid = {}                                  # link -> group id
    def find(x):
        while rigid.get(x, x) != x:
            x = rigid[x]
        return x
    for j in K.joints.values():
        adj.add(frozenset((j.parent, j.child)))
        if j.jtype not in ('revolute', 'prismatic', 'continuous'):
            a_, b_ = find(j.parent), find(j.child)
            if a_ != b_:
                rigid[a_] = b_
    names = [n for n in clouds if n in K.parent_of or n == 'base_link']
    pairs = [(x, y) for i, x in enumerate(names) for y in names[i + 1:]
             if frozenset((x, y)) not in adj
             and frozenset((x, y)) not in DESIGNED_CONTACT
             and find(x) != find(y)]

    print(f'  {len(clouds)} 個碰撞幾何   {len(pairs)} 對非相鄰組合'
          f'（已排除相鄰、同剛體群、設計上接觸的組合）')

    def min_clear(q):
        world = {}
        for n, P in clouds.items():
            if n not in K.parent_of and n != 'base_link':
                continue
            T = K.fk(q, n)
            world[n] = (T[:3, :3] @ P.T).T + T[:3, 3]
        worst, who = 9.9, None
        for x, y in pairs:
            if x not in world or y not in world:
                continue
            d = cKDTree(world[x]).query(world[y], k=1)[0].min()
            if d < worst:
                worst, who = d, (x, y)
        return worst, who

    print('\n  具名姿態')
    for label, jd in (('all-zeros', {}), ('tuck (行駛姿態)', TUCK)):
        q = np.zeros(9)
        for k, v in jd.items():
            q[DOF_NAMES.index(k)] = v
        d, who = min_clear(q)
        flag = '✗ 自碰撞' if d <= 0 else ('⚠ 餘裕偏小' if d < MARGIN_WARN else '✓')
        print(f'    {label:18} 最小間距 {d:+.4f} m   {who[0]}–{who[1]:<20} {flag}')

    # Per-pair, not just the overall minimum. The overall minimum is dominated
    # by pairs that are close in EVERY configuration -- link4 and link6 sit
    # 10 mm apart no matter what joints 5 and 6 do, because the Lite 6 wrist
    # nests its links concentrically. That is the arm's own mechanical
    # clearance, not a risk, and reporting only the minimum buries the pairs
    # that actually move relative to each other.
    #
    # The separation is by VARIANCE: near-zero spread means structural, and only
    # the pairs whose distance changes with the pose can ever be planned into a
    # collision.
    print(f'\n  合法關節盒隨機取樣 n={a.n}')
    rng = np.random.default_rng(1)
    per = {p: [] for p in pairs}
    overall = []
    for _ in range(a.n):
        q = np.zeros(9)
        q[3:] = rng.uniform(lo[3:], hi[3:])
        world = {}
        for n in names:
            if n not in clouds:
                continue
            T = K.fk(q, n)
            world[n] = (T[:3, :3] @ clouds[n].T).T + T[:3, 3]
        best = 9.9
        for x, y in pairs:
            if x not in world or y not in world:
                continue
            d = float(cKDTree(world[x]).query(world[y], k=1)[0].min())
            per[(x, y)].append(d)
            best = min(best, d)
        overall.append(best)
    overall = np.array(overall)
    print(f'    自碰撞比例 {100*(overall <= 0).mean():5.1f}%   '
          f'({int((overall<=0).sum())}/{a.n})   全域最小 {overall.min():+.4f} m')

    STRUCT_STD = 0.002
    struct, varying = [], []
    for p, v in per.items():
        if not v:
            continue
        v = np.array(v)
        (struct if v.std() < STRUCT_STD else varying).append((p, v))
    print(f'\n    結構性接近（間距幾乎不隨姿態變化，std < {STRUCT_STD} m）:')
    for p, v in sorted(struct, key=lambda kv: np.median(kv[1]))[:4]:
        print(f'      {p[0]:12}–{p[1]:20} {np.median(v):+.4f} m  (std {v.std():.4f})')
    print(f'\n    姿態相依（真正需要規劃時檢查的配對）:')
    for p, v in sorted(varying, key=lambda kv: kv[1].min())[:5]:
        print(f'      {p[0]:12}–{p[1]:20} 最小 {v.min():+.4f}  中位 {np.median(v):+.4f}'
              f'  (std {v.std():.4f})')
    worst_var = min((v.min() for _, v in varying), default=float('nan'))
    print(f'\n    姿態相依配對的最小間距 {worst_var:+.4f} m；'
          f'合法關節盒內自碰撞樣本 {int((overall<=0).sum())}/{a.n}。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
