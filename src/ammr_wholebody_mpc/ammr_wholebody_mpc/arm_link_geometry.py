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
            full = len(sel) <= k_max
            if not full:
                sel = sel[np.argsort(d[sel])[:k_max]]
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
