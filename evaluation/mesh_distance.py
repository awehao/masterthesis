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
    agree = all(h == hits[0] for h in hits)
    # Disagreement means at least one ray met an edge or a vertex, or the mesh
    # is not watertight. Returning the majority would turn that into a silent
    # "no collision"; it is reported as undecided instead. Majority voting is a
    # robustness improvement, not a proof.
    return (hits[0] if agree else False), (not agree)


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


def mesh_pair(TA, TB, seed_k=4, tol=1e-9):
    """Distance, witnesses, intersection and containment for two triangle sets.

    Pruning rule
    ------------
    A pair may be discarded only when a LOWER BOUND on its distance already
    exceeds the best distance found so far. Centroid distance is not that
    bound: a large triangle's centroid can be far away while one of its edges
    is touching. What is a bound is

        |c_i - c_j| - r_i - r_j        r = circumradius about the centroid

    so the candidate set is every pair whose centroid separation is within
    (best + r_max_A + r_max_B), then filtered by the per-triangle radii. The
    first version kept the k nearest centroids of B for each triangle of A,
    which is an ordering, not a bound, and could drop the true closest pair
    outright.

    A few nearest-centroid pairs are evaluated first only to seed `best`, so
    the radius query has something to shrink around.
    """
    ca, cb = TA.mean(1), TB.mean(1)
    ra = np.linalg.norm(TA - ca[:, None, :], axis=2).max(axis=1)
    rb = np.linalg.norm(TB - cb[:, None, :], axis=2).max(axis=1)
    ta, tb = cKDTree(ca), cKDTree(cb)

    # seed an upper bound from a handful of nearest-centroid pairs
    k = min(seed_k, len(cb))
    dd, jj = tb.query(ca, k=k)
    dd = np.atleast_2d(dd.T).T
    jj = np.atleast_2d(jj.T).T
    sel = np.argsort(dd[:, 0])[:200]
    A0 = np.repeat(TA[sel], k, axis=0)
    B0 = TB[jj[sel].ravel()]
    d0, wa0, wb0 = tri_tri_distance(A0, B0)
    hit0 = _tri_intersect(A0, B0)
    d0 = np.where(hit0, 0.0, d0)
    k0 = int(np.argmin(d0))
    best, pa, pb = float(d0[k0]), wa0[k0], wb0[k0]
    intersect = bool(hit0.any())

    if not intersect and best > tol:
        # complete candidate set under the bound above
        R = best + ra.max() + rb.max()
        pairs = ta.query_ball_tree(tb, R)
        ia, jb = [], []
        for i, js in enumerate(pairs):
            for j in js:
                if np.linalg.norm(ca[i] - cb[j]) - ra[i] - rb[j] <= best:
                    ia.append(i); jb.append(j)
        if ia:
            A1, B1 = TA[ia], TB[jb]
            hit1 = _tri_intersect(A1, B1)
            d1, wa1, wb1 = tri_tri_distance(A1, B1)
            d1 = np.where(hit1, 0.0, d1)
            k1 = int(np.argmin(d1))
            if d1[k1] < best:
                best, pa, pb = float(d1[k1]), wa1[k1], wb1[k1]
            intersect = bool(hit1.any())
        n_checked = len(ia) + len(A0)
    else:
        n_checked = len(A0)

    contained, undecided = (False, False), False
    if not intersect:
        ra_, ua = _inside(TA[0, 0], TB)
        rb_, ub = _inside(TB[0, 0], TA)
        contained = (ra_, rb_)
        undecided = ua or ub
    return dict(d=best, pa=pa, pb=pb, intersect=intersect,
                contained=contained, undecided=undecided,
                checked=int(n_checked), tol=tol)


def aabb_gap(TA, TB):
    """Distance between the axis-aligned boxes of two triangle sets.

    A lower bound on every triangle pair between them, computed from six
    numbers per side. Whole link pairs metres apart are settled by this before
    a single triangle is touched.
    """
    lo_a, hi_a = TA.reshape(-1, 3).min(0), TA.reshape(-1, 3).max(0)
    lo_b, hi_b = TB.reshape(-1, 3).min(0), TB.reshape(-1, 3).max(0)
    gap = np.maximum(np.maximum(lo_a - hi_b, lo_b - hi_a), 0.0)
    return float(np.linalg.norm(gap))


def clears(TA, TB, thresh, tol=1e-9):
    """Is every triangle pair at least `thresh` apart?

    Answers the question the pose check actually asks, which is cheaper than
    the distance it was asking for. Three stages, each ending the work when it
    can:

      1  the link AABBs are already `thresh` apart          -> clear
      2  per-triangle AABB lower bounds prune to a candidate set; empty -> clear
      3  exact triangle distance on what survives, stopping at the first pair
         below the threshold

    Returns (clear, d_min_or_bound, n_exact). d is exact only when the answer
    came from stage 3; otherwise it is the lower bound that settled it.
    """
    g = aabb_gap(TA, TB)
    if g >= thresh:
        return True, g, 0
    la = TA.min(1); ha = TA.max(1)
    lb = TB.min(1); hb = TB.max(1)
    ca, cb = TA.mean(1), TB.mean(1)
    ra = np.linalg.norm(TA - ca[:, None, :], axis=2).max(axis=1)
    rb = np.linalg.norm(TB - cb[:, None, :], axis=2).max(axis=1)
    tb_ = cKDTree(cb)
    ia, jb = [], []
    # radius per triangle of A, not one global radius: a single large triangle
    # otherwise widens the query for every small one and drags in thousands of
    # pairs that its own bound would have rejected.
    rbmax = rb.max()
    for i in range(len(ca)):
        for j in tb_.query_ball_point(ca[i], thresh + ra[i] + rbmax):
            d_lo = np.linalg.norm(
                np.maximum(np.maximum(la[i] - hb[j], lb[j] - ha[i]), 0.0))
            if d_lo < thresh:
                ia.append(i); jb.append(j)
    if not ia:
        return True, thresh, 0
    A, B = TA[ia], TB[jb]
    hit = _tri_intersect(A, B)
    d, _, _ = tri_tri_distance(A, B)
    d = np.where(hit, 0.0, d)
    dmin = float(d.min())
    return bool(dmin >= thresh), dmin, len(ia)
