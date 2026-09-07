"""Offline mesh-to-mesh distance, intersection and containment.

For self-collision checking only, and offline only. Not in the control loop.

Why not the sampled bound
-------------------------
The online representation subtracts a covering radius from each side:

    d_lower = d(S_A, S_B) - rho_A - rho_B

On this arm that is far too conservative. The links sit millimetres apart, and
two covering radii consume the whole gap: at rho = 5 mm targets the certified
lower bound for link4 vs link6 came out at +1.14 mm from a sampled 10.45 mm,
and for the tuck pose it went negative -- which says the density cannot decide
the question, not that the links touch.

Three separate questions
------------------------
    distance        the closest pair of surface points, and where they are
    intersection    do any two triangles actually cross
    containment     is one closed body entirely inside the other

The third does not follow from the first two. Two closed surfaces can be
disjoint as surfaces while one sits wholly inside the other, and a
distance-only check reports a comfortable clearance for a link that has been
swallowed. It is tested by ray casting from one vertex.

On what this is and is not
--------------------------
Ordinary floating-point geometry with a stated tolerance, not certified
arithmetic. Distances below about 1e-9 m are not meaningful, degenerate (zero
area) triangles are skipped, and the containment test uses a single ray with a
parity count, which is robust for the closed watertight meshes here but not in
general.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

EPS = 1e-12


def _seg_seg(p1, q1, p2, q2):
    """Squared distance between segment pairs, vectorised over rows."""
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = np.einsum('ij,ij->i', d1, d1)
    e = np.einsum('ij,ij->i', d2, d2)
    f = np.einsum('ij,ij->i', d2, r)
    c = np.einsum('ij,ij->i', d1, r)
    b = np.einsum('ij,ij->i', d1, d2)
    denom = a * e - b * b
    s = np.where(denom > EPS, np.clip((b * f - c * e) / np.where(denom > EPS, denom, 1.0),
                                      0.0, 1.0), 0.0)
    t = (b * s + f) / np.where(e > EPS, e, 1.0)
    t = np.clip(t, 0.0, 1.0)
    s = np.clip((b * t - c) / np.where(a > EPS, a, 1.0), 0.0, 1.0)
    c1 = p1 + d1 * s[:, None]
    c2 = p2 + d2 * t[:, None]
    return np.einsum('ij,ij->i', c1 - c2, c1 - c2), c1, c2


def _pt_tri(p, T):
    """Closest point on each triangle to each point, vectorised."""
    a, b, c = T[:, 0], T[:, 1], T[:, 2]
    ab, ac, ap = b - a, c - a, p - a
    d1 = np.einsum('ij,ij->i', ab, ap)
    d2 = np.einsum('ij,ij->i', ac, ap)
    bp = p - b
    d3 = np.einsum('ij,ij->i', ab, bp)
    d4 = np.einsum('ij,ij->i', ac, bp)
    cp = p - c
    d5 = np.einsum('ij,ij->i', ab, cp)
    d6 = np.einsum('ij,ij->i', ac, cp)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    denom = 1.0 / np.maximum(va + vb + vc, EPS)
    v = vb * denom
    w = vc * denom
    out = a + ab * v[:, None] + ac * w[:, None]
    # region fallbacks
    m1 = (d1 <= 0) & (d2 <= 0)
    out[m1] = a[m1]
    m2 = (d3 >= 0) & (d4 <= d3)
    out[m2] = b[m2]
    m3 = (d6 >= 0) & (d5 <= d6)
    out[m3] = c[m3]
    m4 = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    t_ = np.where(m4, d1 / np.maximum(d1 - d3, EPS), 0.0)
    out[m4] = (a + ab * t_[:, None])[m4]
    m5 = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    t_ = np.where(m5, d2 / np.maximum(d2 - d6, EPS), 0.0)
    out[m5] = (a + ac * t_[:, None])[m5]
    m6 = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    t_ = np.where(m6, (d4 - d3) / np.maximum((d4 - d3) + (d5 - d6), EPS), 0.0)
    out[m6] = (b + (c - b) * t_[:, None])[m6]
    return out


def tri_tri_distance(A, B):
    """Distance between triangle pairs A[i] and B[i], with witness points.

    Minimum over the nine edge pairs and the six vertex-to-face cases, which is
    where the closest pair of two triangles always lies.
    """
    n = len(A)
    best = np.full(n, np.inf)
    wa = np.zeros((n, 3))
    wb = np.zeros((n, 3))
    for i in range(3):
        for j in range(3):
            d2, c1, c2 = _seg_seg(A[:, i], A[:, (i + 1) % 3],
                                  B[:, j], B[:, (j + 1) % 3])
            m = d2 < best
            best[m], wa[m], wb[m] = d2[m], c1[m], c2[m]
    for i in range(3):
        q = _pt_tri(A[:, i], B)
        d2 = np.einsum('ij,ij->i', A[:, i] - q, A[:, i] - q)
        m = d2 < best
        best[m], wa[m], wb[m] = d2[m], A[m, i], q[m]
        q = _pt_tri(B[:, i], A)
        d2 = np.einsum('ij,ij->i', B[:, i] - q, B[:, i] - q)
        m = d2 < best
        best[m], wa[m], wb[m] = d2[m], q[m], B[m, i]
    return np.sqrt(np.maximum(best, 0.0)), wa, wb


def _tri_intersect(A, B):
    """Moller's separating-axis test for triangle pairs."""
    def axes(T):
        e = [T[:, 1] - T[:, 0], T[:, 2] - T[:, 1], T[:, 0] - T[:, 2]]
        nrm = np.cross(e[0], -e[2])
        return e, nrm
    ea, na = axes(A)
    eb, nb = axes(B)
    cand = [na, nb] + [np.cross(x, y) for x in ea for y in eb]
    hit = np.ones(len(A), dtype=bool)
    for ax in cand:
        L = np.linalg.norm(ax, axis=1)
        ok = L > 1e-9
        u = np.where(ok[:, None], ax / np.maximum(L, 1e-9)[:, None], 0.0)
        pa = np.einsum('ijk,ik->ij', A, u)
        pb = np.einsum('ijk,ik->ij', B, u)
        sep = ok & ((pa.min(1) > pb.max(1) + 1e-12) |
                    (pb.min(1) > pa.max(1) + 1e-12))
        hit &= ~sep
    return hit


def _inside(point, tris, votes=3):
    """Is `point` inside this closed mesh, by parity of ray crossings.

    Several rays, majority vote, and none of them axis aligned or on a body
    diagonal. A single (1,1,1) ray was tried first and it reported the centre of
    a cube as OUTSIDE: that direction leaves through the corner where three
    faces meet, and the crossing count at a shared vertex is whatever the
    floating-point comparisons happen to give. Directions that avoid the
    symmetry planes hit faces in their interiors, and taking the majority means
    one unlucky ray cannot decide the answer.
    """
    dirs = np.array([[0.3411, 0.6478, 0.6805],
                     [-0.7213, 0.2891, 0.6293],
                     [0.5119, -0.7749, 0.3699]])[:votes]
    hits = [_ray_parity(point, tris, u / np.linalg.norm(u)) for u in dirs]
    return sum(hits) * 2 > len(hits)


def _ray_parity(point, tris, d):
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    e1, e2 = b - a, c - a
    h = np.cross(d, e2)
    det = np.einsum('ij,ij->i', e1, h)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = point - a
    u = inv * np.einsum('ij,ij->i', s, h)
    q = np.cross(s, e1)
    v = inv * (d @ q.T)
    t = inv * np.einsum('ij,ij->i', e2, q)
    hit = ok & (u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1) & (t > 1e-9)
    return bool(hit.sum() % 2 == 1)


def mesh_pair(TA, TB, prefilter=6):
    """Distance, witnesses, intersection and containment for two triangle sets.

    The candidate set is narrowed by triangle centroids before any exact work:
    a KD-tree gives the nearest few centroids of B for each of A, which bounds
    the search without an AABB tree while staying honest -- the prefilter is
    widened by the largest circumradius on each side so it cannot discard the
    true closest pair.
    """
    ca, cb = TA.mean(1), TB.mean(1)
    ra = np.linalg.norm(TA - ca[:, None, :], axis=2).max()
    rb = np.linalg.norm(TB - cb[:, None, :], axis=2).max()
    tree = cKDTree(cb)
    k = min(prefilter, len(cb))
    dd, jj = tree.query(ca, k=k)
    dd = np.atleast_2d(dd.T).T
    jj = np.atleast_2d(jj.T).T
    # any centroid pair beyond (best centroid distance + ra + rb) cannot win
    cut = dd[:, 0].min() + ra + rb
    ia, jb = np.nonzero(dd <= cut)
    if len(ia) == 0:
        ia = np.arange(len(ca)); jb = np.zeros(len(ca), dtype=int)
    A = TA[ia]
    B = TB[jj[ia, jb]]
    hit = _tri_intersect(A, B)
    d, wa, wb = tri_tri_distance(A, B)
    # The edge/vertex enumeration gives the distance only for triangles that do
    # NOT cross. Two that pass through each other have their closest features
    # somewhere in the interiors, and the enumeration returns a positive number
    # for a pair that is already interpenetrating -- 0.2 m on the first test
    # case. Intersection is decided separately and overrides.
    d = np.where(hit, 0.0, d)
    k0 = int(np.argmin(d))
    contained = None
    if not hit.any():
        contained = (_inside(TA[0, 0], TB), _inside(TB[0, 0], TA))
    return dict(d=float(d[k0]), pa=wa[k0], pb=wb[k0],
                intersect=bool(hit.any()),
                contained=contained, checked=int(len(A)))
