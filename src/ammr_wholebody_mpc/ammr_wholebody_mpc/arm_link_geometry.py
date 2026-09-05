"""Whole-link geometry for the safety barrier: sampled surfaces and the point
on each link that is actually closest to an obstacle.

Why this replaces the twelve fixed detection points
---------------------------------------------------
The barrier row

    n_i^T J_{p_i}(q) v  <=  alpha_i (d_i - d_stop_i)

was built at twelve points bolted to the arm at fixed offsets. That constrains
those twelve points and nothing else. Measured over sixty legal configurations,
the collision geometry reaches up to 0.1062 m beyond the nearest detection point
(link_base; link4 0.0823 m, link2 0.0655 m), while the zero-speed stopping
distance is 0.08 m -- so every row could report satisfied while a link was
already touching. In the 5B runs the gap came to a consistent 0.0416 m, which is
exactly the difference between the 0.0710 m the points reported and the 0.0373 m
the meshes actually had.

Inflating the reported distance by a covering radius rho_i fixes the DISTANCE,
because the distance-to-obstacle field is 1-Lipschitz:

    d_link  >=  d(p_i) - rho_i

It does not fix the VELOCITY. A point x on the same link moves at

    v_x = v_{p_i} + omega_i x (x - p_i)

so a point rho_i away can be closing faster than p_i is, by up to
|omega_i| rho_i. Keeping J_{p_i} while inflating the distance therefore still
cannot support a claim about the whole link.

What this module does instead is find, each cycle, the sampled surface point
that is genuinely closest to an obstacle, and build the row there: distance,
normal and Jacobian all taken at the point that is actually in danger. The only
error left is discretisation -- the true surface minimum can sit between
samples -- and that is bounded by the sampling covering radius rho_sample,
which is measured rather than assumed and subtracted in the row.

This is less conservative than the twelve points plus rho_i, not more: rho_i
had to cover the worst case over the whole link, while rho_sample only has to
cover the gap between neighbouring samples.
"""
from __future__ import annotations

import math
import os
import re
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from .arm_detection_points import Obstacle, _closest_local, _inv, _iso, _rpy_to_rot

ARM_LINKS = ('link_base', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6',
             'uflite_gripper_link', 'uflite_finger1', 'uflite_finger2')


@dataclass
class LinkSamples:
    """Surface samples of one link, in that link's own frame."""
    link: str
    points: np.ndarray            # (N, 3)
    rho: float                    # covering radius of these samples, m
    n_ref: int = 0                # reference points the radius was measured against
    certified: bool = False       # rho proved by branch and bound, not sampled
    rho_lower: float = 0.0        # best lower bound found; rho - this is the gap
    cert_tol: float = 0.0
    worst: np.ndarray | None = None      # where on the link the bound is attained
    mesh_sha: str = ''
    sample_sha: str = ''


@dataclass
class NearestPoint:
    """The sampled point on one link that is closest to any obstacle."""
    link: str
    local: np.ndarray             # position in the link frame -- the Jacobian offset
    world: np.ndarray
    d: float                      # distance to the nearest obstacle surface
    n: np.ndarray                 # unit vector from the point toward that surface
    rho: float                    # discretisation allowance for this link
    obstacle: str = ''
    # True when the whole [d_min, d_min + rho] band fitted under the cap, i.e.
    # when the selection provably contains a sample within rho of the true
    # closest point. False means the cap bit and the guarantee is a measurement
    # again.
    band_full: bool = False


# ------------------------------------------------------------------ meshes
def _load_stl_tris(path: str) -> np.ndarray:
    with open(path, 'rb') as f:
        head = f.read(84)
        if head[:5] == b'solid' and b'facet' in head:
            f.seek(0)
            v, cur = [], []
            for line in f:
                w = line.split()
                if w and w[0] == b'vertex':
                    cur.append([float(x) for x in w[1:4]])
                    if len(cur) == 3:
                        v.append(cur); cur = []
            return np.array(v, float)
        n = struct.unpack('<I', head[80:84])[0]
        data = f.read(50 * n)
    a = np.frombuffer(data, dtype=np.uint8).reshape(n, 50)
    return a[:, 12:48].copy().view('<f4').reshape(n, 3, 3).astype(float)


def _surface_points(tris: np.ndarray, n: int, rng) -> np.ndarray:
    """Area-weighted random points on a triangle soup, plus every vertex.

    Vertices alone leave the middles of large triangles unsampled, and a link
    whose flat side is one big triangle would then have a covering radius the
    size of that side.
    """
    a = tris[:, 1] - tris[:, 0]
    b = tris[:, 2] - tris[:, 0]
    area = 0.5 * np.linalg.norm(np.cross(a, b), axis=1)
    if area.sum() <= 0:
        return tris.reshape(-1, 3)
    idx = rng.choice(len(tris), size=n, p=area / area.sum())
    u, v = rng.random((n, 1)), rng.random((n, 1))
    flip = (u + v) > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    return np.vstack([tris[idx, 0] + u * a[idx] + v * b[idx], tris.reshape(-1, 3)])


def _fps(P: np.ndarray, k: int, rng) -> np.ndarray:
    """Farthest-point sampling: for a given k this minimises the covering
    radius, which is the quantity the barrier has to subtract."""
    k = min(k, len(P))
    sel = np.empty(k, dtype=int)
    sel[0] = int(rng.integers(len(P)))
    d = np.linalg.norm(P - P[sel[0]], axis=1)
    for i in range(1, k):
        sel[i] = int(np.argmax(d))
        d = np.minimum(d, np.linalg.norm(P - P[sel[i]], axis=1))
    return sel


def link_collision_tris(xml: str, links=ARM_LINKS) -> dict[str, np.ndarray]:
    """Collision triangles of each link, in that link's own frame."""
    root = ET.fromstring(re.sub(r'<!--.*?-->', '', xml, flags=re.S))
    out = {}
    for link in root.findall('link'):
        name = link.get('name')
        if name not in links:
            continue
        tris = []
        for col in link.findall('collision'):
            g = col.find('geometry')
            m = g.find('mesh') if g is not None else None
            if m is None:
                continue
            p = m.get('filename', '').replace('file://', '')
            if not os.path.exists(p):
                continue
            V = _load_stl_tris(p)
            sc = m.get('scale')
            if sc:
                V = V * np.array([float(x) for x in sc.split()])
            o = col.find('origin')

            def gx(k, d, o=o):
                t = (o.get(k) or d) if o is not None else d
                return np.array([float(x) for x in t.split()])
            T = _iso(_rpy_to_rot(*gx('rpy', '0 0 0')), gx('xyz', '0 0 0'))
            tris.append((V @ T[:3, :3].T) + T[:3, 3])
        if tris:
            out[name] = np.concatenate(tris)
    return out


def certified_covering_radius(tris: np.ndarray, S: np.ndarray,
                              tol: float = 0.0005,
                              max_nodes: int = 400000) -> dict:
    """Certified upper bound on

        rho = max over x on the mesh surface of  min over s in S of |x - s|

    by branch and bound on the triangles, not by sampling.

    A sampled estimate answers a different question -- the distance from one
    point cloud to another -- and can only ever miss the worst place. Measured
    against a 20,000 point reference and then validated against 60,000, the
    reported distance came out on the wrong side of the true one on 4 per cent
    of rows; raising the reference to 120,000 did not fix it, because the flaw
    is the method rather than the density.

    For a triangle T with centroid c, every x in T satisfies

        d_S(x)  <=  d_S(c) + max_{v in T} |v - c|

    since d_S is 1-Lipschitz, and the farthest point of a convex set from c is
    a vertex. That is an upper bound for the whole triangle from ONE distance
    query. d_S evaluated at any point of T is a lower bound on the maximum. So
    each node carries an interval, the node with the largest upper bound is the
    only one that can still move the answer, and splitting it at the edge
    midpoints shrinks r_T by half. Iterating until the largest upper bound and
    the best lower bound agree to `tol` certifies rho to that tolerance.

    Returns the bound, the gap actually achieved, and where the worst place is,
    so the number can be checked rather than believed.
    """
    import heapq
    tree = cKDTree(S)

    def batch(nodes):
        """(upper, lower) for a list of triangles, one query each."""
        T = np.asarray(nodes)
        c = T.mean(axis=1)
        dc, _ = tree.query(c, k=1)
        r = np.linalg.norm(T - c[:, None, :], axis=2).max(axis=1)
        return dc + r, dc, c

    ub, lb, c0 = batch(tris)
    best_lb = float(lb.max())
    k = int(np.argmax(lb))
    best_pt = c0[k].copy()
    # also evaluate the vertices once: they are free lower bounds and often the
    # worst place on a mesh with long thin triangles
    dv, _ = tree.query(tris.reshape(-1, 3), k=1)
    if dv.max() > best_lb:
        best_lb = float(dv.max())
        best_pt = tris.reshape(-1, 3)[int(np.argmax(dv))].copy()

    heap = [(-float(ub[i]), i) for i in range(len(tris))]
    heapq.heapify(heap)
    store = {i: tris[i] for i in range(len(tris))}
    nxt = len(tris)
    nodes = len(tris)

    while heap:
        top = -heap[0][0]
        if top - best_lb <= tol:
            return dict(rho=top, lower=best_lb, gap=top - best_lb,
                        worst=best_pt, nodes=nodes, certified=True, tol=tol)
        if nodes >= max_nodes:
            return dict(rho=top, lower=best_lb, gap=top - best_lb,
                        worst=best_pt, nodes=nodes, certified=False, tol=tol)
        _, i = heapq.heappop(heap)
        T = store.pop(i)
        a, b_, c_ = T
        m0, m1, m2 = 0.5 * (b_ + c_), 0.5 * (c_ + a), 0.5 * (a + b_)
        kids = np.array([[a, m2, m1], [m2, b_, m0], [m1, m0, c_], [m0, m1, m2]])
        u, l, cc = batch(kids)
        j = int(np.argmax(l))
        if l[j] > best_lb:
            best_lb, best_pt = float(l[j]), cc[j].copy()
        for t in range(4):
            store[nxt] = kids[t]
            heapq.heappush(heap, (-float(u[t]), nxt))
            nxt += 1
        nodes += 4
    return dict(rho=best_lb, lower=best_lb, gap=0.0, worst=best_pt,
                nodes=nodes, certified=True, tol=tol)


def sample_links(xml: str, rho_target: float = 0.015, links=ARM_LINKS,
                 n_ref: int = 120000, seed: int = 0,
                 cap: int = 640) -> dict[str, LinkSamples]:
    """Surface samples per link, each dense enough for its own covering radius
    to reach rho_target.

    The count is chosen per link rather than shared: the fingers reach 7 mm with
    twenty points while link_base needs about three hundred, and one number for
    all of them would either waste rows on the fingers or leave the base coarse.

    rho is measured against a reference cloud, so it is an estimate of the
    covering radius and not the covering radius itself: a surface point the
    reference happened to miss can be farther from every sample than rho says.
    That is not academic. Measuring against 20,000 reference points and then
    validating against 60,000 put the reported distance ON THE WRONG SIDE of the
    true one on about 4 per cent of rows, by up to 4.2 mm -- the reference had
    simply not looked where the gap was. n_ref is high for that reason, and a
    validation run should still use a denser reference than this one.
    """
    root = ET.fromstring(re.sub(r'<!--.*?-->', '', xml, flags=re.S))
    rng = np.random.default_rng(seed)
    out = {}
    for link in root.findall('link'):
        name = link.get('name')
        if name not in links:
            continue
        tris = []
        for col in link.findall('collision'):
            g = col.find('geometry')
            m = g.find('mesh') if g is not None else None
            if m is None:
                continue
            p = m.get('filename', '').replace('file://', '')
            if not os.path.exists(p):
                continue
            V = _load_stl_tris(p)
            sc = m.get('scale')
            if sc:
                V = V * np.array([float(x) for x in sc.split()])
            o = col.find('origin')

            def gx(k, d, o=o):
                s = (o.get(k) or d) if o is not None else d
                return np.array([float(x) for x in s.split()])
            T = _iso(_rpy_to_rot(*gx('rpy', '0 0 0')), gx('xyz', '0 0 0'))
            tris.append((V @ T[:3, :3].T) + T[:3, 3])
        if not tris:
            continue
        ref = _surface_points(np.concatenate(tris), n_ref, rng)
        lo, hi = 8, cap
        best = None
        while lo <= hi:                       # smallest count that meets the target
            mid = (lo + hi) // 2
            sel = _fps(ref, mid, np.random.default_rng(seed + 1))
            rho = float(cKDTree(ref[sel]).query(ref, k=1)[0].max())
            if rho <= rho_target:
                best = (ref[sel].copy(), rho, mid)
                hi = mid - 1
            else:
                lo = mid + 1
        if best is None:                      # target unreachable within the cap
            sel = _fps(ref, hi if hi > 0 else cap, np.random.default_rng(seed + 1))
            rho = float(cKDTree(ref[sel]).query(ref, k=1)[0].max())
            best = (ref[sel].copy(), rho, len(sel))
        out[name] = LinkSamples(link=name, points=best[0], rho=best[1],
                                n_ref=len(ref))
    return out


# --------------------------------------------------------------- distances
def _closest_local_batch(o: Obstacle, P: np.ndarray) -> np.ndarray:
    """Closest surface point of one primitive, for many query points at once.

    The scalar version in arm_detection_points is called once per point, and at
    1127 samples times the obstacles in range that came to 13.7 ms of a 50 ms
    control period -- more than the projection it feeds. Same geometry, one
    array operation instead of thousands of calls.
    """
    if o.kind == 'box':
        half = 0.5 * o.size
        cl = np.clip(P, -half, half)
        inside = np.all(np.abs(P) <= half, axis=1)
        if inside.any():
            # Inside the box the nearest surface is the nearest FACE, so the
            # clamp is a no-op and the point has to be pushed out along
            # whichever axis it is least deep in.
            Q = P[inside]
            gap = half - np.abs(Q)
            ax = np.argmin(gap, axis=1)
            r = np.arange(len(Q))
            out = Q.copy()
            out[r, ax] = np.sign(Q[r, ax]) * half[ax]
            out[r, ax] = np.where(Q[r, ax] == 0.0, half[ax], out[r, ax])
            cl[inside] = out
        return cl
    if o.kind == 'cylinder':
        r, h = o.radius, 0.5 * o.height
        rad = np.linalg.norm(P[:, :2], axis=1)
        safe = np.maximum(rad, 1e-12)
        on_side = np.stack([P[:, 0] / safe * r, P[:, 1] / safe * r,
                            np.clip(P[:, 2], -h, h)], 1)
        on_cap = np.stack([np.where(rad > r, P[:, 0] / safe * r, P[:, 0]),
                           np.where(rad > r, P[:, 1] / safe * r, P[:, 1]),
                           np.sign(np.where(P[:, 2] == 0, 1.0, P[:, 2])) * h], 1)
        d_side = np.linalg.norm(on_side - P, axis=1)
        d_cap = np.linalg.norm(on_cap - P, axis=1)
        return np.where((d_side <= d_cap)[:, None], on_side, on_cap)
    if o.kind == 'sphere':
        d = np.maximum(np.linalg.norm(P, axis=1), 1e-12)
        return P / d[:, None] * o.radius
    return P


def _inside(o: Obstacle, P: np.ndarray) -> np.ndarray:
    """Which of P are inside this primitive, in its own frame."""
    if o.kind == 'box':
        return np.all(np.abs(P) <= 0.5 * o.size, axis=1)
    if o.kind == 'cylinder':
        return ((np.linalg.norm(P[:, :2], axis=1) <= o.radius)
                & (np.abs(P[:, 2]) <= 0.5 * o.height))
    if o.kind == 'sphere':
        return np.linalg.norm(P, axis=1) <= o.radius
    return np.zeros(len(P), dtype=bool)


def obstacle_distances(P: np.ndarray, obs: list[Obstacle]) -> tuple[np.ndarray, np.ndarray, list]:
    """SIGNED distance and approach direction from each of P to the nearest
    obstacle.

    Returns (d, vec, which). d is negative by the penetration depth when the
    point is inside a primitive. vec always points along what the barrier calls
    APPROACH, so that n = vec / |d| has one meaning everywhere:

        outside   the closest surface point is ahead, so approach is toward it
        inside    the closest surface point is the way OUT, so approach is the
                  other way

    Getting this wrong is not a small error. With the unsigned version a
    penetrating point produced a vector pointing outward, the row read it as an
    approach and so constrained the escape: the arm was held IN. It showed up
    as exactly two rows out of four hundred where the true approach rate beat
    the constrained one by up to 269 mm/s, both at d < 0.4 mm and both with the
    two normals exactly antiparallel -- a normal difference of 2.000 is not
    ill-conditioning, it is a sign.
    """
    best_d = np.full(len(P), np.inf)
    best_v = np.zeros((len(P), 3))
    which = [''] * len(P)
    for o in obs:
        T = o.T_world_link @ o.T_link_collision
        Ti = _inv(T)
        loc = (P @ Ti[:3, :3].T) + Ti[:3, 3]
        surf_loc = _closest_local_batch(o, loc)
        surf = (surf_loc @ T[:3, :3].T) + T[:3, 3]
        v = surf - P
        dist = np.linalg.norm(v, axis=1)
        ins = _inside(o, loc)
        d = np.where(ins, -dist, dist)          # signed
        v = np.where(ins[:, None], -v, v)       # approach direction, either way
        m = d < best_d                          # a penetration beats any clearance
        best_d[m], best_v[m] = d[m], v[m]
        for i in np.nonzero(m)[0]:
            which[i] = o.name
    return best_d, best_v, which


def nearest_points(K, q: np.ndarray, obs: list[Obstacle],
                   samples: dict[str, LinkSamples],
                   k_per_link: int = 6, band: bool = True,
                   k_max: int = 60) -> list[NearestPoint]:
    """The sampled points of every link that could be the true closest one.

    One row per link is not enough, and "the k nearest" is a guess. There is a
    rule that is not:

        let x* be the true closest point of the link, and s_j the sample
        nearest to it, so |s_j - x*| <= rho by the covering property. Then

            d(s_j) <= d(x*) + rho <= d_min + rho

        because d(x*) is the minimum over the whole surface and d_min the
        minimum over a subset of it.

    So constraining every sample whose distance falls in [d_min, d_min + rho]
    is guaranteed to constrain a point within rho of the true closest one --
    which is exactly what the |omega| rho velocity bound needs, and what taking
    the k nearest cannot promise. Held-out testing of the k-nearest rule found
    the true approach rate exceeding the constrained one by 37.9 mm/s where the
    fitted allowance was 16.5 mm/s.

    k_max caps the band, because a link running parallel to a flat face has
    many samples at nearly the same distance and all of them would qualify. The
    cap is a cost limit and it BREAKS the guarantee when it bites, so
    band_full records whether it did. Measured over 400 rows the band holds 10
    samples at the median, 31 at the 95th percentile and 41 at the worst, so 60
    leaves it intact everywhere seen; at 14 it bit on a third of them.
    """
    out = []
    for name, S in samples.items():
        try:
            T = K.fk(q, name)
        except Exception:
            continue
        W = (S.points @ T[:3, :3].T) + T[:3, 3]
        d, v, which = obstacle_distances(W, obs)
        if band:
            dmin = float(d.min())
            sel = np.nonzero(d <= dmin + S.rho)[0]
            # Sorted by distance, so the caller can rely on the first entry
            # being the nearest. np.nonzero returns index order, and a test
            # that read the first entry as the minimum reported distance
            # violations of up to 5.02 mm that were entirely its own doing.
            sel = sel[np.argsort(d[sel])]
            full = len(sel) <= k_max
            if not full:
                sel = sel[:k_max]
        else:
            sel = np.argsort(d)[:max(1, k_per_link)]
            full = False
        for k in sel:
            dk = float(d[k])
            out.append(NearestPoint(link=name, local=S.points[k].copy(),
                                    world=W[k], d=dk,
                                    n=v[k] / max(abs(dk), 1e-9), rho=S.rho,
                                    obstacle=which[k], band_full=bool(full)))
    return out


def sample_links_certified(xml: str, rho_target: float = 0.015,
                           links=ARM_LINKS, tol: float = 0.001,
                           n_ref: int = 40000, seed: int = 0,
                           cap: int = 1200) -> dict[str, LinkSamples]:
    """Surface samples whose covering radius is PROVED, not estimated.

    Same shape as sample_links, but the radius that decides the sample count
    comes from certified_covering_radius -- branch and bound over the actual
    triangles -- rather than from the distance between two point clouds. The
    difference is not cosmetic: the sampled estimate came out 1.0 to 1.5 mm
    below the certified value on every link, and a value that is too small is
    subtracted from the barrier's distance, which is the direction that makes
    the robot think it has room it does not have.

    Each link's record carries the bound, the gap left to its own lower bound,
    where on the mesh the worst point is, and hashes of the mesh and the
    samples, so the certificate can be rechecked against the geometry it was
    computed for.
    """
    import hashlib
    tris_by_link = link_collision_tris(xml, links)
    rng = np.random.default_rng(seed)
    out = {}
    for name, tris in tris_by_link.items():
        cloud = _surface_points(tris, n_ref, rng)
        mesh_sha = hashlib.sha256(np.ascontiguousarray(tris).tobytes()).hexdigest()[:16]
        n = 8
        best = None
        while n <= cap:
            sel = _fps(cloud, n, np.random.default_rng(seed + 1))
            P = cloud[sel].copy()
            c = certified_covering_radius(tris, P, tol=tol)
            if c['rho'] <= rho_target:
                best = (P, c)
                break
            n = int(math.ceil(n * 1.6))
        if best is None:
            sel = _fps(cloud, cap, np.random.default_rng(seed + 1))
            P = cloud[sel].copy()
            best = (P, certified_covering_radius(tris, P, tol=tol))
        P, c = best
        out[name] = LinkSamples(
            link=name, points=P, rho=float(c['rho']), n_ref=len(cloud),
            certified=bool(c['certified']), rho_lower=float(c['lower']),
            cert_tol=float(c['tol']), worst=np.asarray(c['worst']),
            mesh_sha=mesh_sha,
            sample_sha=hashlib.sha256(np.ascontiguousarray(P).tobytes()).hexdigest()[:16])
    return out
